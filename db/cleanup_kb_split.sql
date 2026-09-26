-- ---------------------------------------------------------------------------
-- 一次性迁移：RAG 知识库从「全套语料混存一张表」拆成「一文档一套表」
--
-- 背景：旧结构把 faq 问答对和 products/refund_policy 成篇文档混灌进同一套
--   表（实测 data_kb_vectors 里 faq/products/refund_policy 三个 source 共存），
--   向量空间和 BM25 词频统计互相污染。
--   新结构见 customer_service/rag/ingest.py 的 _CORPUS_ROUTES：
--   kb_product_* / kb_policy_* / kb_qa_* 各自独立。
--
-- 注意：llama-index 的 PGVectorStore 和 PostgresKVStore 落库表名都带 data_
--   前缀，所以 docstore 实际表名是 data_kb_docstore，不是 kb_docstore。
--
-- 用法（PG 容器在跑的前提下，二选一）：
--   psql "$DATABASE_URL" -f db/cleanup_kb_split.sql
--   PYTHONIOENCODING=utf-8 .venv/Scripts/python -c "import psycopg2,os; \
--     from dotenv import load_dotenv; load_dotenv('customer_service/.env'); \
--     c=psycopg2.connect(os.environ['DATABASE_URL'].strip('\"\'')); c.autocommit=True; \
--     c.cursor().execute(open('db/cleanup_kb_split.sql',encoding='utf-8').read())"
-- 执行后重灌（烧 embedding 额度，自己跑）：
--   PYTHONIOENCODING=utf-8 .venv/Scripts/python -m customer_service.rag.ingest
--
-- 新表由 llama-index 在灌库时自动建（perform_setup），这里只负责清旧的。
-- ---------------------------------------------------------------------------

-- 向量表：CASCADE 连带删掉建在它上面的 BM25 索引 kb_bm25_idx
DROP TABLE IF EXISTS data_kb_vectors CASCADE;
-- docstore / indexstore 是 KV 表（indexstore 可能从未写入而不存在，IF EXISTS 兜底）
DROP TABLE IF EXISTS data_kb_docstore;
DROP TABLE IF EXISTS kb_docstore;
DROP TABLE IF EXISTS kb_indexstore;
DROP TABLE IF EXISTS data_kb_indexstore;
