"""API 服务（P7）：FastAPI 包装现有图，对外暴露三个端点。

- POST /chat                  跑一轮会话；图被 human_node 挂起时返回 pending_human
- GET  /sessions/{id}/history 从 PostgresSaver 拉回完整会话（消息序列）
- POST /chat/resume           人工坐席回复，恢复 interrupt（P3 人工通道的 HTTP 形态）

复用件全是现成的：
- 会话 = thread_id：PostgresSaver 全局单例（P2），重启/多实例状态都在 PG
- 图在 lifespan 里建一次全局复用，checkpointer 随图单例
- phone 从请求写入 State（工单工具的入单依据），LLM 签名里摸不到

启动：uvicorn customer_service.api:app --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pydantic import BaseModel

from customer_service import build_graph
from customer_service.memory.checkpointer import get_checkpointer

_graph = None  # lifespan 启动时建一次，进程内全局复用


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph
    _graph = build_graph()
    yield


app = FastAPI(title="客服多智能体 API", lifespan=lifespan)


class ChatRequest(BaseModel):
    session_id: str
    message: str
    phone: str = ""   # 写入 State 供工单工具用；不是必填


class ResumeRequest(BaseModel):
    session_id: str
    reply: str        # 人工坐席的回复原文


def _last_ai(messages: list) -> tuple[str, str]:
    """最后一条有内容的 AI 消息 (文本, 来源名)。"""
    for m in reversed(messages):
        if getattr(m, "type", "") == "ai" and getattr(m, "content", None):
            return str(m.content), getattr(m, "name", "") or ""
    return "", ""


def _interrupt_payload(out: dict):
    interrupts = out.get("__interrupt__") or []
    return interrupts[0].value if interrupts else None


def _respond(session_id: str, out: dict) -> dict:
    """统一的响应打包：挂起 → pending_human + payload；正常 → reply。"""
    interrupt = _interrupt_payload(out)
    if interrupt:
        return {"session_id": session_id, "status": "pending_human", "interrupt": interrupt}
    reply, name = _last_ai(out["messages"])
    return {"session_id": session_id, "status": "ok", "reply": reply, "from": name}


@app.post("/chat")
def chat(req: ChatRequest) -> dict:
    """一轮对话。active_agent 不传——checkpoint 里的值原样保留，路由不重置。"""
    payload = {"messages": [HumanMessage(content=req.message)], "user_phone": req.phone}
    out = _graph.invoke(payload, config={"configurable": {"thread_id": req.session_id}})
    return _respond(req.session_id, out)


@app.get("/sessions/{session_id}/history")
def history(session_id: str) -> dict:
    """拉回完整会话：checkpointer 最新 checkpoint 的 channel_values 直出。"""
    saver = get_checkpointer()
    tup = saver.get_tuple({"configurable": {"thread_id": session_id}})
    if tup is None:
        raise HTTPException(status_code=404, detail="session not found")
    values = tup.checkpoint.get("channel_values", {})
    messages = [
        {
            "role": getattr(m, "type", type(m).__name__),
            "name": getattr(m, "name", None),
            "content": str(getattr(m, "content", m)),
        }
        for m in (values.get("messages") or [])
    ]
    return {
        "session_id": session_id,
        "active_agent": values.get("active_agent"),
        "messages": messages,
    }


@app.post("/chat/resume")
def resume(req: ResumeRequest) -> dict:
    """恢复被 human_node 挂起的图：坐席回复经 interrupt 回流为坐席消息。"""
    out = _graph.invoke(
        Command(resume=req.reply),
        config={"configurable": {"thread_id": req.session_id}},
    )
    return _respond(req.session_id, out)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("customer_service.api:app", host="127.0.0.1", port=8000)
