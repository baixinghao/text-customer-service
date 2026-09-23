"""数据库连接入口：所有业务工具共用的 PG 连接池。

- 连接串来自 customer_service/.env 的 DATABASE_URL（别硬编码）
- ThreadedConnectionPool：LangChain 工具可能在不同线程执行，
  线程池内部加锁分配连接，绝不让两个线程共享一条连接
- minconn=0 懒建连：import 本模块不连库，第一次取连接时才握手，
  池内无空闲连接时按需新建，直到 maxconn 上限后 getconn 阻塞等待
"""

import os
from contextlib import contextmanager
from pathlib import Path

import psycopg2.pool
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

_POOL: "psycopg2.pool.ThreadedConnectionPool | None" = None


def _get_pool() -> "psycopg2.pool.ThreadedConnectionPool":
    """懒加载单例连接池：整个进程一个池，工具层只管用 get_conn()。"""
    global _POOL
    if _POOL is None:
        dsn = (os.getenv("DATABASE_URL") or "").strip('"\'')
        if not dsn:
            raise RuntimeError("DATABASE_URL 未配置（customer_service/.env）")
        _POOL = psycopg2.pool.ThreadedConnectionPool(
            minconn=0,   # 不预热：首次调用才连库；池空了按需新建
            maxconn=10,  # 并发上限：超出后 getconn 阻塞等归还，而不是压垮 PG
            dsn=dsn,
        )
    return _POOL


@contextmanager
def get_conn():
    """从池里借连接的上下文管理器。

    - 进入：pool.getconn()（无空闲且未达上限则新建）
    - 正常退出：commit + 归还池子（连接不断开，下一个调用者复用）
    - 异常退出：rollback + 归还（不会把脏连接塞回池里）
    """
    pool = _get_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)
