"""知识库构建（骨架，实现计划 P1）：加载 knowledge/ 语料 → 切块 → 向量索引 → 存 PG。

这个文件做什么：
- build_index()：按 _CORPUS_ROUTES 逐文档灌库
  1. 一份语料一个 pipeline 实例、一套独立 PG 三件套（一文档一表，见下）
  2. sentence/ 成篇语料走通用切块 pipeline；qa/ 问答对走独立 pipeline
     （Q 灌库、A 做元数据）
  3. 重跑不重复 embedding：docstore 记 hash，unchanged 的文档/节点全量跳过
     （filename_as_id 保证 doc id 跨次稳定，是这套去重的前提）
  写入即持久化到 PG（PGVectorStore / PostgresDocumentStore），没有本地 persist

- 为什么一文档一套表（2026-09-25 拆分，丽姐拍的板）：
  早期全部语料混灌同一套 kb_vectors / kb_docstore，QA 问句和成篇文档混存：
  · 向量层：问句（短文本）和条目段落（长文本）相似度尺度不同，混在一个
    HNSW 索引里互相挤占 top_k，metadata source 过滤只是事后遮掩
  · BM25 层：两类文本混在同一语料里，IDF 统计互相污染
  现在 products / refund_policy / faq 各自独立三件套，检索工具 1:1 打自己的表，
  source 过滤整层删除。代价：加新主题语料要登记 _CORPUS_ROUTES（表名总得有个
  地方登记，这成本躲不掉）。

- 存储红线（项目约定）：向量库 / 文本库（docstore）/ 索引库（index_store）全走
  PostgreSQL，禁止本地文件落盘。三件套：
  PGVectorStore + PostgresDocumentStore + PostgresIndexStore

- 语料约定：knowledge/ 下一个主题一个 md。qa/ 下是问答对（Q 灌库、A 做元数据），
  sentence/ 下是成篇文档（通用切块 pipeline）。语料带 source 文件名元数据，
  检索结果能溯源到具体文件

- embedding 模型：用 DashScope 兼容端点（.env 里已有 DASHSCOPE_*），
  OpenAIEmbedding 指 api_base 过去即可；embed_dim 必须和所选模型维度一致
  （选型理由写注释里，这是练习的一部分）

关键 API（LlamaIndex core 0.12+ 风格）：
  from llama_index.core import (
      SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings,
  )
  from llama_index.vector_stores.postgres import PGVectorStore
  from llama_index.storage.docstore.postgres import PostgresDocumentStore
  from llama_index.storage.index_store.postgres import PostgresIndexStore
  # PGVectorStore.from_params(host=..., port=..., user=..., password=...,
  #                           database=..., table_name=..., embed_dim=...)
  # Settings.embed_model = ...   全局指定 embedding
  # 前置：PG 里先 CREATE EXTENSION vector（pgvector）

坑位提示：
- "要不要灌库"不用 count 判空：docstore hash 去重管这件事。count 判空会漏掉
  "语料改了需要增量灌"的场景，hash 去重不会——别再加回来
- PGVectorStore 实际落库的表名是 data_{table_name}，查数据别找错表
- PG 连接信息从 .env 的 DATABASE_URL 读，别硬编码
"""

import os
import hashlib
import re
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex, StorageContext, SimpleDirectoryReader, Settings,
)
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.schema import TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.dashscope import DashScope
from llama_index.storage.docstore.postgres import PostgresDocumentStore
from llama_index.storage.index_store.postgres import PostgresIndexStore
# 底层 KV：docstore/indexstore 的 from_uri 在 docstore 0.6 + kvstore 0.4 组合下有
# 版本 bug（透传了老接口没有的参数），所以显式用 from_params 组 KV 再注入
from llama_index.storage.kvstore.postgres import PostgresKVStore
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import create_engine, text

load_dotenv(Path(__file__).parent.parent / ".env")

_DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)


def _pg_async_url(url: str) -> str:
    """把 sync 连接串转成 asyncpg 可用的：换驱动、sslmode 改写成 ssl。

    asyncpg 不认 sslmode 查询参数；sqlalchemy 的 asyncpg 方言认 ssl=，
    且取值直接沿用 sslmode 的枚举名（ssl=false 这种布尔写法反而会报错）。
    """
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query))
    sslmode = query.pop("sslmode", None)
    if sslmode and "ssl" not in query:
        query["ssl"] = sslmode
    return urlunsplit(
        (f"{parts.scheme}+asyncpg", parts.netloc, parts.path,
         urlencode(query), parts.fragment)
    )

# 语料目录：qa/ 下是问答对语料，sentence/ 下是成篇语料
SENTENCE_DIR = Path(__file__).parent / "knowledge" / "sentence"
QA_DIR = Path(__file__).parent / "knowledge" / "qa"

# 语料路由：一个主题一套独立三件套（kb_{kind}_vectors / kb_{kind}_docstore /
# kb_{kind}_indexstore + kb_{kind}_bm25_idx），表与表之间零共享。
# kind 同时是 get_pipeline(kind) 的键，retriever.py 按它取各自的检索栈。
# 新增主题语料：把 md 放进对应目录，在这里登记一行。
_CORPUS_ROUTES = (
    ("product", SENTENCE_DIR / "products.md", "sentence"),
    ("policy", SENTENCE_DIR / "refund_policy.md", "sentence"),
    ("qa", QA_DIR / "faq.md", "qa"),
)


def _load_corpus(md_path: Path):
    """读单份语料 md，带上 source 文件名元数据。"""
    return SimpleDirectoryReader(
        input_files=[str(md_path)],
        required_exts=[".md"],  # 只收 md，防止混入临时文件
        # 自定义 file_metadata 会【替换】默认实现（默认才带 file_name/file_path），
        # 所以文件名要在这里显式补回来
        file_metadata=lambda p: {"source": Path(p).name, "file_name": Path(p).name},
        filename_as_id=True,
    ).load_data(show_progress=True)


# QA 语料行格式：一行一个问答对「Q: xxx A: xxx」，## 标题为分区名
_QA_LINE_RE = re.compile(r"^Q:\s*(?P<q>.+?)\s+A:\s*(?P<a>.+?)\s*$")


def _build_qa_nodes(doc) -> list[TextNode]:
    """把 QA 语料（faq.md）解析成「Q 灌库、A 做元数据」的节点列表。

    设计理由：用户查询和"问法"天然同分布，embedding 只针对问题，答案不进
    向量/BM25 索引（只随结果当元数据带回），检索更稳、索引更小。
    - node.text = 问题原文（Q: 前缀去掉）；metadata.answer = 答案；section = ## 分区名
    - excluded_embed_metadata_keys 排掉全部 metadata：embedding 只嵌纯问题，
      答案不进向量（QA 检索的经典姿势：查询和问法同分布）
    - node id = sha256(文件名+问题)，跨次稳定：内容没变重跑直接跳过；
      答案改了 TextNode.hash（text+metadata 的 sha256）跟着变，
      pipeline 的 UPSERTS 会删掉旧节点换新，天然支持增量改答案。
    """
    source = doc.metadata.get("source", "faq.md")
    nodes: list[TextNode] = []
    section = "未分区"
    for line in doc.text.splitlines():
        line = line.strip()
        if line.startswith("## "):
            section = line[3:].strip()
            continue
        m = _QA_LINE_RE.match(line)
        if not m:
            continue
        q, a = m.group("q"), m.group("a")
        node_id = hashlib.sha256(f"{source}::{q}".encode("utf-8")).hexdigest()
        nodes.append(TextNode(
            id_=node_id,
            text=q,
            metadata={
                "source": source,
                "file_name": source,
                "section": section,
                "answer": a,
            },
            # 关键：embedding 时 get_content(EMBED) 默认会拼接全部 metadata。
            # 全排掉，嵌入文本 = 纯问题——answer 进向量则"Q 灌库 A 做元数据"白干
            excluded_embed_metadata_keys=["answer", "source", "file_name", "section"],
        ))
    if not nodes:
        raise ValueError(f"{source} 被配置为 QA 语料，但一行 Q/A 都没解析出来，检查格式")
    return nodes


class DocIngestionPipeline:
    """单份语料的灌库 pipeline：一套独立 PG 三件套 + 自己的 BM25 索引。

    kind 决定全部表名（kb_{kind}_*），实例之间零共享——一文档一表，
    向量空间和 BM25 统计彻底隔离，不存在跨语料污染。
    """

    EMBED_DIM = 1024  # 必须与 embedding 模型维度一致：text-embedding-v4 默认 1024

    # ParadeDB pg_search BM25：索引建在自己这套向量表（data_kb_{kind}_vectors）上，
    # 随 pipeline 增删 chunk 自动维护，不需要额外的同步逻辑。
    # 分词用服务端 jieba；注意 @@@ 字符串查询侧不再分词，
    # 检索端必须先把查询 jieba 分词、空格拼接后再传（见 retriever.py）。
    # 改 tokenizer 配置不会自动重建已存在的索引，开发期需手动 DROP INDEX。
    PG_BM25_TEXT_FIELDS = '{"text": {"tokenizer": {"type": "jieba"}}}'

    def __init__(self, kind: str):
        route_type = {k: t for k, _, t in _CORPUS_ROUTES}.get(kind)
        if route_type is None:
            raise ValueError(f"未知语料 kind：{kind!r}，先在 _CORPUS_ROUTES 登记")
        self.kind = kind
        self._route_type = route_type
        # PG 三件套表名集中在这，别散到各函数里
        # 前置：库里有 CREATE EXTENSION vector（pgvector 扩展）
        self.PG_VECTOR_TABLE = f"kb_{kind}_vectors"
        self.PG_DOC_TABLE = f"kb_{kind}_docstore"
        self.PG_INDEX_TABLE = f"kb_{kind}_indexstore"
        self.PG_BM25_INDEX = f"kb_{kind}_bm25_idx"
        self._setup_models()
        self.vector_store: Optional[PGVectorStore] = None
        self.doc_store: Optional[PostgresDocumentStore] = None
        self.index_store: Optional[PostgresIndexStore] = None
        self.storage_context: Optional[StorageContext] = None
        self._initialize_storage_components()

    def _setup_models(self):
        """设置LLM和嵌入模型"""
        Settings.embed_model = OpenAIEmbedding(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            api_base=_DASHSCOPE_BASE_URL,
            model_name=os.getenv("EMBEDDING_MODEL", "text-embedding-v4"),
            # 0.14 默认 batch=100（按 OpenAI 上限调的），百炼兼容端点单批上限 10，
            # 不压回来必报 400 InternalError.Algo.InvalidParameter
            embed_batch_size=10,
        )

        # LLM 用百炼原生集成：llama_index 的 OpenAI 包装类的 metadata 属性
        # 硬编码按模型名查表（openai_modelname_to_contextsize），百炼模型名
        # 直接 ValueError，救不回来；DashScope 类原生支持 qwen 系模型和流式
        Settings.llm = DashScope(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            model_name=os.getenv("MODEL_NAME") or os.getenv("QWEN_MODEL", "qwen-plus"),
            temperature=0.3,
            incremental_output=True,  # 流式输出开关，配合 query_engine streaming=True
        )



    def _initialize_storage_components(self):
        """按 .env 的 DATABASE_URL 建本文档专属的 PG 三件套并组装 StorageContext。

        注意：在 __init__ 里调用，实例化即连库并建表（perform_setup 默认 True），
        所以 PG 必须先启动；表已存在时是 IF NOT EXISTS，不会清数据。
        """
        url = (os.getenv("DATABASE_URL") or "").strip('"\'')
        if not url:
            raise RuntimeError("DATABASE_URL 未配置（customer_service/.env）")
        # PGVectorStore 要显式给 async 连接串（asyncpg 驱动，sslmode 已改写）；
        # docstore/indexstore 的 KV 用同一对连接串
        async_url = _pg_async_url(url)

        self.vector_store = PGVectorStore(
            connection_string=url,
            async_connection_string=async_url,
            table_name=self.PG_VECTOR_TABLE,
            embed_dim=self.EMBED_DIM,
        )
        # docstore / indexstore 共用 KV 实现，各建一张 KV 表注入
        kv_doc = PostgresKVStore.from_params(
            connection_string=url,
            async_connection_string=async_url,
            table_name=self.PG_DOC_TABLE,
        )
        kv_index = PostgresKVStore.from_params(
            connection_string=url,
            async_connection_string=async_url,
            table_name=self.PG_INDEX_TABLE,
        )
        self.doc_store = PostgresDocumentStore(postgres_kvstore=kv_doc)
        self.index_store = PostgresIndexStore(postgres_kvstore=kv_index)

        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store,
            docstore=self.doc_store,
            index_store=self.index_store,
        )

    def ingest_sentence_docs(self, documents):
        """成篇语料（商品库/售后政策）：按 ## 小节整条切块 → 标题抽取 → embedding。

        MarkdownNodeParser 而不是定长 SentenceSplitter：条目（## 机械键盘 K87 ¥299）
        是一等公民，名称/价格/卖点整条自包含成一个节点，标题行留在 text 里。
        定长切块会把条目腰斩，续块没有标题，下游（答题模型、护栏质检员）无法把
        卖点归到商品头上——2026-09-24 护栏误杀 K87 报价的深层原因。
        语料侧约定见 products.md 开头：一个商品 = 一个 ## 小节。
        """
        node_parser = MarkdownNodeParser()
        title_extractor = TitleExtractor(nodes=5, node_template="请为以下文档生成一个简洁的标题: {context_str}",
                                         num_workers=5)
        pipeline = IngestionPipeline(
            transformations=[node_parser, title_extractor, Settings.embed_model],
            vector_store=self.vector_store,
            docstore=self.doc_store,
        )
        # store_doc_text=False：pipeline 只把原始 Document 的 hash 写进 docstore
        # 做跨次去重，不存全文——否则 docstore 里存的是整篇 md（实测 0.14 行为），
        # BM25 / 原文回查的粒度就和条目级切块对不上
        nodes = pipeline.run(documents=documents, store_doc_text=False)
        if nodes:
            chunk_nodes = list(nodes)
            for n in chunk_nodes:
                # 向量已进向量表，docstore 只留文本语料，别重复存 1024 维 embedding
                n.embedding = None
            # 切块 id 由内容哈希派生，重跑幂等（allow_update 默认 True）
            self.doc_store.add_documents(chunk_nodes)

    def ingest_qa_docs(self, documents):
        """QA 语料（faq.md）：解析问答对 → Q 进 text 参与 embedding/索引，A 只做元数据。

        传 nodes= 而不是 documents=：pipeline 的 _update_docstore 写的就是
        我们构造的问答节点本身（问题级粒度），不会再套一层切块。
        """
        qa_pipeline = IngestionPipeline(
            transformations=[Settings.embed_model],  # 只做 embedding，不切块不抽标题
            vector_store=self.vector_store,
            docstore=self.doc_store,
        )
        for doc in documents:
            qa_nodes = _build_qa_nodes(doc)
            qa_pipeline.run(nodes=qa_nodes)

    def ensure_bm25_index(self):
        """在自己的向量表上建 ParadeDB BM25 索引（幂等，已存在则跳过）。"""
        url = (os.getenv("DATABASE_URL") or "").strip('"\'')
        if not url:
            raise RuntimeError("DATABASE_URL 未配置（customer_service/.env）")
        engine = create_engine(url)
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_search"))
            conn.execute(text(
                f"CREATE INDEX IF NOT EXISTS {self.PG_BM25_INDEX} "
                f"ON data_{self.PG_VECTOR_TABLE} USING bm25 (id, text) "
                f"WITH (key_field='id', text_fields='{self.PG_BM25_TEXT_FIELDS}')"
            ))
        engine.dispose()

    def build(self, documents):
        """按 kind 对应的路由类型灌库，并保证 BM25 索引就位。"""
        if self._route_type == "qa":
            self.ingest_qa_docs(documents)
        else:
            self.ingest_sentence_docs(documents)
        self.ensure_bm25_index()

    def get_index(self) -> VectorStoreIndex:
        if self.vector_store is None or self.storage_context is None:
            raise RuntimeError("存储组件未初始化：_initialize_storage_components 失败")
        index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store)
        return index

    def get_storage_context(self) -> StorageContext:
        if self.storage_context is None:
            raise RuntimeError("存储组件未初始化：_initialize_storage_components 失败")
        return self.storage_context




# ===== 模块级单例：一个 kind 一个实例，整个进程每套表只连一次 PG =====
# retriever / api 等其它模块这样用：
#   from customer_service.rag.ingest import get_pipeline
#   index = get_pipeline("product").get_index()
# 不要自己 new DocIngestionPipeline()，否则会重复连库、重复灌库
_pipelines: dict[str, DocIngestionPipeline] = {}


def get_pipeline(kind: str) -> DocIngestionPipeline:
    """懒加载单例：某套表第一次用到时才实例化，之后都返回同一个对象。"""
    if kind not in _pipelines:
        _pipelines[kind] = DocIngestionPipeline(kind)  # 非法 kind 在 __init__ 拦
    return _pipelines[kind]


# ===== 模块级便捷入口：retriever.py 从这里 import，不用关心单例细节 =====
def get_index(kind: str) -> VectorStoreIndex:
    """等价于 get_pipeline(kind).get_index()。"""
    return get_pipeline(kind).get_index()


def get_storage_context(kind: str) -> StorageContext:
    """等价于 get_pipeline(kind).get_storage_context()。"""
    return get_pipeline(kind).get_storage_context()


def build_index() -> None:
    """按 _CORPUS_ROUTES 把全部语料各灌各的表（重跑幂等，hash 去重）。"""
    for kind, md_path, _route_type in _CORPUS_ROUTES:
        documents = _load_corpus(md_path)
        if documents:
            get_pipeline(kind).build(documents)



if __name__ == '__main__':
    build_index()
