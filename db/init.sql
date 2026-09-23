-- ---------------------------------------------------------------------------
-- 容器首次初始化脚本（docker-entrypoint-initdb.d）
--
-- 执行时机：仅在 PG 数据卷 tcs_pg_data 为空时、容器第一次启动时执行一次。
--          改了本文件但库已存在 → 不会重跑；需要重来就 `docker compose down -v`。
--
-- 职责边界：这里只建「扩展」这种实例级前置，业务表不放这。
--          业务表统一由 db/schema.sql 定义，走
--            .venv/Scripts/python db/apply_schema.py
--          保持单一来源，避免两处 DDL 打架。
-- ---------------------------------------------------------------------------

-- 向量类型：llama-index PGVectorStore（kb_vectors / kb_docstore / kb_indexstore）前置
CREATE EXTENSION IF NOT EXISTS vector;

-- ParadeDB BM25：rag/ingest.py 的 kb_bm25_idx 用它，分词走服务端 jieba
CREATE EXTENSION IF NOT EXISTS pg_search;

-- 模糊匹配：本来用于商品名/手机号容错查询
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- 注：pg_cron 在 ParadeDB 镜像里只预配置于默认 postgres 库，业务库启用会报错，
--     本项目未使用，故刻意不启用。
