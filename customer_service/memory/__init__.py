"""会话记忆子包（骨架，实现计划 P2）。

- checkpointer.py  短期记忆持久化：PostgresSaver 替换 InMemorySaver，重启不丢会话
- summary.py       长期记忆 / 会话摘要（可选）：长会话压缩，超 thread 维度的用户偏好沉淀

依赖：uv sync --extra memory。没装之前 import 本包会报 ModuleNotFoundError，属预期。
"""

from customer_service.memory.checkpointer import get_checkpointer

__all__ = ["get_checkpointer"]
