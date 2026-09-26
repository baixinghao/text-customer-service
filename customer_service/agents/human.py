"""转人工节点（P3）：human-in-the-loop，AI 搞不定时的兜底。

工作方式：
- human_node 调 interrupt(payload) 把图挂起，payload 使用 LangGraph 标准
  HumanInterrupt schema，方便官方 Agent Chat UI 直接弹出坐席回复框。
  坐席输入后前端通过 command: { resume: HumanResponse } 恢复图。
- interrupt 的返回值是 HumanResponse；节点提取出坐席回复写入 messages，
  再 Command(goto=END) 结束本轮。

两个设计决策（定了就不轻易改，注释留给未来的自己）：
1. 人工回复按 AIMessage(name="human_agent") 入历史——保持 user/assistant
   交替结构；之后用户的新消息仍是 HumanMessage，专家 LLM 读到完整上下文
   （含人工答了什么），不会把坐席的话误读成用户提问
2. resume 后 goto=END 而不是交还原专家——【修正】早期版本交还原专家，
   专家会把坐席已回答的问题再答一遍（双答，实测踩过）。坐席的话就是
   本轮的答案；后续用户新消息由 router 重新判意图分流，连续性不丢。
   若将来需要在人工答复后做质检等后处理，加专用节点，
   不要把专家当后置节点用

配套接线（P3 剩余，不在本文件）：
- graph.py: builder.add_node("human", human_node)
- 触发：给各专家 make_human_handoff_tool(...)——注意转人工工具【不要】
  改 active_agent（它得始终是真实专家节点名，output_guard 打回要靠它）
- 跨进程恢复靠 P2 的 PostgresSaver，已就绪
"""

import warnings

from langchain_core.messages import AIMessage
from langgraph.graph import END
from langgraph.warnings import LangGraphDeprecatedSinceV10
from langgraph.types import Command, interrupt

# HumanInterrupt 新家在 `langchain.agents.interrupt`（LangGraph 1.x 后期版本），
# 当前环境若还没有，则回退到 `langgraph.prebuilt.interrupt`，并屏蔽迁移警告。
try:
    from langchain.agents.interrupt import (
        ActionRequest,
        HumanInterrupt,
        HumanInterruptConfig,
        HumanResponse,
    )
except ImportError:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LangGraphDeprecatedSinceV10)
        from langgraph.prebuilt.interrupt import (
            ActionRequest,
            HumanInterrupt,
            HumanInterruptConfig,
            HumanResponse,
        )

from customer_service.state import State

_CONTEXT_LINES = 6  # 挂起时带给坐席的最近对话轮数


def _format_recent(messages: list, n: int = _CONTEXT_LINES) -> str:
    """最近 n 条消息格式化成「[角色] 内容」纯文本，给人工坐席看来龙去脉。"""
    lines = []
    for m in messages[-n:]:
        role = getattr(m, "type", m.__class__.__name__)
        content = getattr(m, "content", None)
        if not content:
            # 空内容的 AI 消息多半是 tool_call 信令，展示成可读的调用说明
            tool_calls = getattr(m, "tool_calls", None) or []
            if tool_calls:
                names = "、".join(str(tc.get("name", "?")) for tc in tool_calls)
                content = f"（调用工具：{names}）"
            else:
                content = str(m)
        lines.append(f"[{role}] {content}")
    return "\n".join(lines)


def _extract_response_text(response: HumanResponse | list[HumanResponse]) -> str:
    """从 HumanResponse（或数组）里提取坐席回复文本。"""
    if isinstance(response, list):
        response = response[0] if response else None
    if not response:
        return "（坐席未回复）"

    response_type = response.get("type")
    args = response.get("args")

    if response_type == "response" and isinstance(args, str):
        return args
    if response_type in ("accept", "ignore"):
        return "（坐席已接管本回合）"
    if response_type == "edit" and isinstance(args, dict):
        # edit 返回的是 ActionRequest，取第一个字段兜底展示
        return str(next(iter(args.values()), "（坐席已编辑回复）"))
    return str(args) if args is not None else "（坐席未回复）"


def human_node(state: State) -> Command:
    """挂起图等人工坐席接管；resume 后把坐席回复写入历史，本轮直接 END 交卷。

    下一轮用户消息由 router 重新判意图分流（见文件头决策 2）。
    外层 resume 时必须传 HumanResponse（官方 UI 会自动按此格式发送）。
    """
    messages = state["messages"]
    # 最后一条用户消息要跳过 tool 信令（转人工后 messages[-1] 是 ToolMessage）
    last_user = next(
        (m for m in reversed(messages) if getattr(m, "type", "") == "human"),
        None,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LangGraphDeprecatedSinceV10)
        interrupt_request = HumanInterrupt(
            action_request=ActionRequest(
                action="human_takeover",
                args={
                    "hint": "AI 已转人工坐席，请根据上下文回复用户",
                    "latest_user_message": str(last_user.content) if last_user else "",
                    "recent_context": _format_recent(messages),
                },
            ),
            config=HumanInterruptConfig(
                allow_respond=True,
                allow_accept=False,
                allow_edit=False,
                allow_ignore=False,
            ),
            description="用户问题超出 AI 自动处理能力，需要人工坐席介入回复。",
        )

    # 挂起：返回值是 HumanResponse（官方 UI 发送 command: { resume: HumanResponse }）
    human_response = interrupt(interrupt_request)
    human_reply = _extract_response_text(human_response)

    # add_messages 归并：只追加新消息即可
    # goto=END：坐席的话就是本轮的答案（见文件头决策 2），同轮再进专家会双答
    return Command(
        goto=END,
        update={"messages": [AIMessage(content=human_reply, name="human_agent")]},
    )
