"""CLI 入口：多轮对话 + 转人工 resume 分支（P3）。

- 用户消息走普通 invoke；结果里有 __interrupt__ 说明图被 human_node 挂起，
  打印 payload（转人工提示 + 最近上下文），输入坐席回复后用
  Command(resume=...) 恢复——这一步不是普通消息，别混
- thread_id 决定会话：传 --thread-id 续聊旧会话（PostgresSaver 持久化，
  跨进程可恢复）；不传则开新会话
- 输入 /exit 退出；坐席侧输入 /end 表示结束接管（恢复时原样传给图）
"""

import argparse
import uuid

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from customer_service import build_graph


def _last_ai_text(messages: list) -> tuple[str, str]:
    """取最后一条 AI 消息的 (文本, 来源标签)；人工坐席的消息单独标注。"""
    for m in reversed(messages):
        if getattr(m, "type", "") == "ai" and getattr(m, "content", None):
            label = "坐席" if getattr(m, "name", "") == "human_agent" else "客服"
            return str(m.content), label
    return "", "客服"


def run_cli(thread_id: str | None = None, phone: str = "") -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}
    print(f"[thread_id = {config['configurable']['thread_id']}] 输入 /exit 退出")

    while True:
        user_input = input("\n用户> ").strip()
        if user_input in ("/exit", "/quit"):
            break

        # user_phone 进 State：工单工具的入单依据（LLM 签名里不出现），每次带上幂等
        payload = {"messages": [HumanMessage(content=user_input)],
                   "active_agent": None, "user_phone": phone}
        while True:
            out = graph.invoke(payload, config=config)

            interrupts = out.get("__interrupt__") or []
            if not interrupts:
                reply, label = _last_ai_text(out["messages"])
                if reply:
                    print(f"\n{label}> {reply}")
                break

            # 图被挂起（转人工）：展示 payload，收坐席回复，resume 恢复
            value = interrupts[0].value
            print("\n===== 已转人工坐席 =====")
            print(value.get("recent_context", ""))
            print("========================")
            seat_reply = input("坐席> ").strip()
            if not seat_reply:
                seat_reply = "（坐席暂未回复，请稍候）"
            payload = Command(resume=seat_reply)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="客服多智能体 CLI")
    parser.add_argument("--thread-id", default=None, help="续聊已有会话；不传则开新会话")
    parser.add_argument("--phone", default="", help="用户手机号，写入 State 供工单工具使用")
    args = parser.parse_args()
    run_cli(thread_id=args.thread_id, phone=args.phone)
