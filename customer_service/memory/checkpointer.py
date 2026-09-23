"""会话持久化：PostgresSaver 替换 InMemorySaver（P2）。

get_checkpointer() 返回进程级单例 PostgresSaver，多轮记忆从"内存级"升
"持久级"：进程重启后凭 thread_id 找回全部会话历史。也是 P3 转人工跨进程
恢复、P7 多实例 API 服务的前提。

连接生命周期（本文件的核心决策）：
- 不用 PostgresSaver.from_conn_string()——它是 @contextmanager，退出即关连接，
  return 出去就是个废 saver（本练习最大的坑）
- 传 psycopg_pool.ConnectionPool 给 PostgresSaver（官方签名 Conn 联合类型
  支持单连接或池）：saver 每次存取从池里借还连接，多线程/并发图执行安全。
  连接数上限由 max_size 卡住，不会无限新建压垮 PG
- 池配置：min_size=1；max_size=10 卡住并发上限；kwargs 透传 connect 参数
  （见下方警告）；池自带借还探活（check），死连接检出后丢弃重建
- open 时机：ConnectionPool 构造完只是后台线程开始建连，min_size 条就绪是异步的；
  open(wait=True) 同步等到就绪再返回（重复调用安全），setup() 之前必须调，
  否则首开库/冷启动时可能踩"池未就绪"的时序坑
- 关闭时机：CLI/练习进程退出即池亡，不显式 close()；将来 web 服务化
  （P7 API）时在 lifespan shutdown 里调 pool.close()
- 注意：本池是 psycopg3 的池（langgraph 只认 psycopg3），与 db.py 里
  业务工具用的 psycopg2 ThreadedConnectionPool 是两套驱动、两个池，
  不要合并
- setup() 幂等（迁移版本记在 checkpoint_migrations 表），每次获取单例都顺手调，
  表没建过就建，建过就是空转
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

load_dotenv(Path(__file__).parent.parent / ".env")

_pool: ConnectionPool | None = None
_saver: PostgresSaver | None = None


def get_checkpointer() -> PostgresSaver:
    """返回基于 PostgreSQL 的 checkpointer（懒加载单例，池进程级持有）。"""
    global _pool, _saver
    if _saver is None:
        dsn = (os.getenv("DATABASE_URL") or "").strip('"\'')
        if not dsn:
            raise RuntimeError("DATABASE_URL 未配置（customer_service/.env）")
        _pool = ConnectionPool(
            conninfo=dsn,
            min_size=1,   # 预热一条
            max_size=10,  # 并发上限：池满后借连接阻塞等待归还
            # ⚠️ autocommit / row_factory 只能走 kwargs：连接是池批量创建的，
            # 在这里池子对每个新连接调 psycopg.connect(conninfo, **kwargs)。
            # 写到别处不会生效，症状是神秘的"只读事务 / current transaction
            # is aborted"类报错（autocommit 没生效，checkpointer 的事务控制全乱）
            kwargs={
                "autocommit": True,       # checkpointer 自己做事务控制
                "row_factory": dict_row,  # saver 按 dict 取行
            },
            name="langgraph-checkpointer",
        )
        # 构造完池子在后台线程异步建连，open(wait=True) 同步等 min_size 条就绪；
        # 幂等，重复调用安全
        _pool.open(wait=True)
        _saver = PostgresSaver(_pool)
        _saver.setup()  # 幂等：首次建 checkpoints 四张表，之后空转
    return _saver
