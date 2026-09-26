"""Handoff 工具工厂：专家间轮内接力（Command + graph=Command.PARENT）。

每个专家持有指向其他专家的 handoff 工具。LLM 决定转接时：
- goto 目标专家节点，graph=Command.PARENT 表示在父图层面导航（工具跑在子图里）
- update 把「转接成功」的 ToolMessage 追加进共享消息历史，并切换 active_agent

分工（见 graph.py 文件头"分层决策"）：handoff 管轮内交接——首发专家干完
自己那段、活儿跨域时接力给下家；轮级首发分流是 router 的事，别搞混。
"""

from langchain.tools import tool, ToolRuntime
from langchain_core.messages import ToolMessage
from langgraph.types import Command


def make_handoff_tool(*, agent_name: str, description: str):
    @tool(f"transfer_to_{agent_name}", description=description)
    def handoff_tool(runtime: ToolRuntime) -> Command:
        return Command(
            goto=agent_name,
            graph=Command.PARENT,
            update={
                "active_agent": agent_name,
                "messages": runtime.state["messages"]
                + [
                    ToolMessage(
                        content=f"已转接至 {agent_name}，请继续为用户服务",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            },
        )

    return handoff_tool


def make_human_handoff_tool(*, description: str):
    """转人工专用 handoff：goto human 节点，但【不改 active_agent】。

    不能复用 make_handoff_tool——它会把 active_agent 写成 "human"，而
    active_agent 必须是真实专家节点名（output_guard 打回重答靠它 goto
    原专家）。这里只追加 ToolMessage 说明，active_agent 原样保留。
    """
    @tool("transfer_to_human", description=description)
    def human_handoff_tool(runtime: ToolRuntime) -> Command:
        return Command(
            goto="human",
            graph=Command.PARENT,
            update={
                "messages": runtime.state["messages"]
                + [
                    ToolMessage(
                        content="已为您转接人工坐席，请稍候",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            },
        )

    return human_handoff_tool
