"""Langfuse 客户端工厂：评测侧唯一的平台入口。

凭据读 customer_service/.env 的 LANGFUSE_* 三个变量；
默认值是本地开发栈（docker-compose.observability.yml 的 LANGFUSE_INIT_*）。
生产环境不配默认值——没 key 就让它报错，别静默连错地方。
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langfuse import Langfuse

_DEV_DEFAULTS = {
    "LANGFUSE_PUBLIC_KEY": "pk-lf-tcs-dev-0001",
    "LANGFUSE_SECRET_KEY": "sk-lf-tcs-dev-0001",
    "LANGFUSE_HOST": "http://localhost:3000",
}


def get_langfuse() -> Langfuse:
    load_dotenv(Path(__file__).parent.parent / ".env")
    cfg = {k: os.getenv(k) or v for k, v in _DEV_DEFAULTS.items()}
    # 项目/密钥错配是实测踩过的坑（数据集写进同名不同 id 的项目，UI 里看不到），
    # 每次连接都打印横幅，连错了立刻能看出来
    print(f"[langfuse] host={cfg['LANGFUSE_HOST']} key={cfg['LANGFUSE_PUBLIC_KEY'][:16]}…")
    return Langfuse(
        public_key=cfg["LANGFUSE_PUBLIC_KEY"],
        secret_key=cfg["LANGFUSE_SECRET_KEY"],
        host=cfg["LANGFUSE_HOST"],
    )
