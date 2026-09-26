"""检索器 → LangChain 工具的生产实现（P1）。

保留 retriever.py 作为练习骨架，本文件按老师意见修正：
- 工具只负责"找得到"，不内嵌 LLM 答题；"如实转述"交给外层 agent 的 PROMPT。
- 工厂语义：检索栈在工厂函数里只建一次，内部 @tool 闭包复用。
- BM25 走 ParadeDB pg_search：索引和语料都在 PG（各文档自己的向量表上的
  bm25 索引，随灌库自动维护），本地不再起 bm25s 内存索引；中文分词用服务端 jieba。
- 一文档一套表（ingest._CORPUS_ROUTES）：products / refund_policy / faq 各自
  独立向量表 + docstore + BM25 索引，向量空间与词频统计零共享，检索不需要
  metadata source 过滤。FAQ（qa 套）仍是独立 pipeline：Q 参与 embedding/检索，
  A 只做节点元数据，展示侧由 _format_nodes 拼回完整问答对。
- 工具签名对齐初代 Mock：search_product(keyword: str) / query_refund_policy()，
  新增 search_faq(question: str)（原 Mock 文件已删，签名约定保留）。
"""

import os
from collections.abc import Callable
from typing import Any

import jieba
from langchain.tools import tool
from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import (
    NodeWithScore,
    QueryBundle,
    TextNode,
)
from llama_index.core.storage.docstore.types import BaseDocumentStore
from llama_index.postprocessor.dashscope_rerank import DashScopeRerank
from sqlalchemy import create_engine, text

from customer_service.rag.ingest import get_index, get_storage_context

__all__ = [
    "get_product_retriever_tool",
    "get_policy_retriever_tool",
    "get_faq_retriever_tool",
]


def _pg_url() -> str:
    """.env 的 DATABASE_URL（ingest 模块已负责 load_dotenv）。"""
    return (os.getenv("DATABASE_URL") or "").strip('"\'')


# ---------------------------------------------------------------------------
# 中文分词
# ---------------------------------------------------------------------------

def _chinese_tokenizer(text: str) -> list[str]:
    """jieba 中文分词，过滤纯空白碎片。"""
    return [t.strip() for t in jieba.lcut(text) if t.strip()]


# ---------------------------------------------------------------------------
# ParadeDB pg_search BM25 检索器
# ---------------------------------------------------------------------------

class PgSearchBM25Retriever(BaseRetriever):
    """基于 ParadeDB pg_search 的 BM25 检索器（语料、索引、分词全在 PG 侧）。

    索引由 ingest 灌库时建在该文档自己的向量表（data_kb_{kind}_vectors）上，
    jieba 分词；一张表只装一份语料，BM25 词频统计不被其它文档污染，
    所以查询不需要任何来源过滤。
    实测 pg_search 0.25 的 @@@ 字符串查询**不会**对查询词再做分词：
    直接传整句中文，只有恰好等于索引词的长词才匹配得上。所以查询侧必须
    先 jieba 分词、空格拼接成 "词1 词2" 再传（解析器按 AND 组合）。
    """

    def __init__(
        self,
        table_name: str,
        top_k: int,
        connection_string: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._top_k = top_k
        self._table = table_name
        self._engine = create_engine(connection_string or _pg_url())

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        terms = " ".join(_chinese_tokenizer(query_bundle.query_str))
        if not terms:
            return []
        sql = text(f"""
            SELECT node_id, text, metadata_, paradedb.score(id) AS score
            FROM {self._table}
            WHERE text @@@ :terms
            ORDER BY paradedb.score(id) DESC
            LIMIT :k
        """)
        with self._engine.connect() as conn:
            rows = conn.execute(sql, {"terms": terms, "k": self._top_k}).all()
        return [
            NodeWithScore(
                node=TextNode(
                    id_=row.node_id,
                    text=row.text,
                    metadata=dict(row.metadata_ or {}),
                ),
                score=float(row.score),
            )
            for row in rows
        ]


# ---------------------------------------------------------------------------
# 公共检索栈构建
# ---------------------------------------------------------------------------

def _build_fusion_retriever(kind: str, top_k: int = 5) -> QueryFusionRetriever:
    """构建向量 + BM25（pg_search）的融合检索器，全程只打 kind 自己的那套表。"""
    vector_retriever = get_index(kind).as_retriever(similarity_top_k=top_k)

    bm25_retriever = PgSearchBM25Retriever(
        table_name=f"data_kb_{kind}_vectors",
        top_k=top_k,
    )

    # use_async=False 是故意的：True 时每次检索都 asyncio.run 新建/关闭一个 Windows
    # ProactorEventLoop，循环半关闭状态下还在 flush 的 asyncpg/httpx socket 写者会踩
    # 'NoneType' object has no attribute 'send'（proactor 已置空）。5 条数据的检索
    # 向量/BM25 串行即可，不值得为并发付出每个调用新建事件循环的代价
    return QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=top_k,
        mode=FUSION_MODES.RECIPROCAL_RANK,
        use_async=False,
        num_queries=1,
        retriever_weights=[0.6, 0.4],
    )


def _build_reranker(top_n: int = 5) -> DashScopeRerank:
    """构建 DashScope 重排序器（qwen3-rerank）。"""
    return DashScopeRerank(
        model="qwen3-rerank",
        top_n=top_n,
        api_key=os.getenv("DASHSCOPE_API_KEY"),
    )


def _format_nodes(nodes: list[NodeWithScore], docstore: BaseDocumentStore) -> str:
    """把检索节点格式化为带来源的纯文本，从本文档自己的 docstore 取原文兜底。

    QA 节点（metadata 带 answer）特殊渲染：text 只是问题，答案在元数据里，
    拼回「Q: ... / A: ...」整对交给外层 agent。
    """
    if not nodes:
        return "知识库中没有检索到相关信息。"

    parts: list[str] = []
    for node_with_score in nodes:
        node = node_with_score.node
        source = node.metadata.get("source", "未知来源")
        answer = node.metadata.get("answer")
        if answer is not None:
            parts.append(f"【来源：{source}】\n问：{node.text.strip()}\n答：{answer}")
            continue
        original = docstore.docs.get(node.id_)
        text = original.text.strip() if original else node.text.strip()
        parts.append(f"【来源：{source}】\n{text}")
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# 工厂函数：返回 LangChain @tool 闭包
# ---------------------------------------------------------------------------

def get_product_retriever_tool() -> Callable:
    """返回商品检索工具（签名对齐初代 Mock 的 search_product(keyword: str)）。"""
    retriever = _build_fusion_retriever("product", top_k=5)
    docstore = get_storage_context("product").docstore
    reranker = _build_reranker(top_n=5)

    @tool
    def search_product(keyword: str) -> str:
        """按关键字搜索商品库，返回匹配商品的名称、价格和卖点。"""
        nodes = retriever.retrieve(keyword)
        nodes = reranker.postprocess_nodes(nodes, query_str=keyword)
        return _format_nodes(nodes, docstore)

    return search_product


def get_policy_retriever_tool() -> Callable:
    """返回售后政策检索工具（签名对齐初代 Mock 的 query_refund_policy()）。"""
    retriever = _build_fusion_retriever("policy", top_k=5)
    docstore = get_storage_context("policy").docstore
    reranker = _build_reranker(top_n=5)
    policy_query = "退换货政策 退款流程 到账时限 换货条件"

    @tool
    def query_refund_policy() -> str:
        """查询退换货政策、退款流程、到账时限、换货条件的官方说明。
        回答任何退货/退款/换货规则类问题前，必须先调用本工具取证，禁止凭记忆作答。"""
        nodes = retriever.retrieve(policy_query)
        nodes = reranker.postprocess_nodes(nodes, query_str=policy_query)
        return _format_nodes(nodes, docstore)

    return query_refund_policy


def get_faq_retriever_tool() -> Callable:
    """返回 FAQ 检索工具（QA 套：Q 参与检索，A 在元数据里随结果整对返回）。"""
    retriever = _build_fusion_retriever("qa", top_k=5)
    docstore = get_storage_context("qa").docstore
    reranker = _build_reranker(top_n=5)

    @tool
    def search_faq(question: str) -> str:
        """按用户问题搜索常见问题库，返回最匹配的问答对（含答案）。"""
        nodes = retriever.retrieve(question)
        nodes = reranker.postprocess_nodes(nodes, query_str=question)
        return _format_nodes(nodes, docstore)

    return search_faq
