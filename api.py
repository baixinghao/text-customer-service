"""对外 API 入口（骨架，实现计划 P7）：把客服图暴露成 HTTP 服务，多渠道接入的第一步。

运行（实现后）：
    uv sync --extra api
    PYTHONIOENCODING=utf-8 .venv/Scripts/python -m uvicorn api:app --reload

这个文件该实现什么：
- POST /chat            {session_id, message} → {reply, agent}
    内部调 build_graph().invoke，session_id 直接当 thread_id 用
- GET  /sessions/{session_id}/history
    查会话历史；配 memory/checkpointer.py 的 PostgresSaver 才有意义
- POST /chat/resume     {session_id, reply}   （P3 转人工配套）
    人工坐席回复通道：Command(resume=reply) 恢复挂起的图
- 注意点：
  * 图和 checkpointer 要做成全局单例，别每个请求重建
  * 多实例部署时 InMemorySaver 会串不了台，必须先完成 P2

依赖：uv sync --extra api（fastapi + uvicorn）。没装之前本文件 import 会炸，属预期。
"""

from fastapi import FastAPI  # noqa: F401  # import 先写好，实现时取用
from pydantic import BaseModel  # noqa: F401  # 请求/响应模型用它定义

app = FastAPI(title="智能文本客服 API")


# P7 练习：定义 ChatRequest / ChatResponse 模型，实现上面三个端点
