"""父图装配层：只做路由和接线，不写业务。

流程：
  START → input_guard（P5 硬闸门：规则先行，LLM 兜底；拦截直接 END + 写死模板）
        → router（首轮 LLM 意图识别，Command(goto) 分流；后续轮次凭 active_agent 直达）
        → 专家子图（内部是 create_agent 的 model⇄tools ReAct 循环）
        → END（专家间转接靠 handoff 工具的 Command(goto, graph=PARENT)，父图不需要为它们画边）

多轮记忆靠 PostgresSaver + 调用方传 thread_id：重启/换实例凭 thread_id 找回历史。
"""

from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from customer_service.agents.aftersale import aftersale_agent
from customer_service.agents.complaint import complaint_agent
from customer_service.agents.human import human_node
from customer_service.agents.order import order_agent
from customer_service.agents.presale import presale_agent
from customer_service.guardrails.input_guard import input_guard_node
from customer_service.guardrails.output_guard import output_guard_node
from customer_service.llm import get_llm
from customer_service.memory.checkpointer import get_checkpointer
from customer_service.prompts import ROUTER_PROMPT
from customer_service.state import State

EXPERTS = ("presale_expert", "aftersale_expert", "order_expert", "complaint_expert")
# P3 的 human 节点走 interrupt 不进路由表；P5 的 output_guard 是下一个扩展位。


def router_node(state: State) -> Command:
    """分流节点：老客直达当前专家，新客让 LLM 判意图。"""
    if state.get("active_agent"):
        return Command(goto=state["active_agent"])

    resp = get_llm().invoke([SystemMessage(content=ROUTER_PROMPT), *state["messages"]])
    target = resp.content.strip()
    if target not in EXPERTS:
        target = "presale_expert"  # 兜底：识别不了的一律按售前咨询接待
    return Command(goto=target, update={"active_agent": target})


def build_graph():
    builder = StateGraph(State)
    builder.add_node("input_guard", input_guard_node)
    builder.add_node("router", router_node)
    builder.add_node("presale_expert", presale_agent)
    builder.add_node("aftersale_expert", aftersale_agent)
    builder.add_node("order_expert", order_agent)
    builder.add_node("complaint_expert", complaint_agent)
    builder.add_node("human", human_node)
    builder.add_node("output_guard", output_guard_node)

    # 护栏放行是 Command(goto="router")，这里只需要 START 到护栏的边
    builder.add_edge(START, "input_guard")
    # router 用 Command(goto) 动态路由，这里不需要条件边
    # 专家的出参统一过 output_guard：通过/兜底才 END，打回 goto 原专家重答。
    # human 节点故意不过 output_guard——人工坐席的话自己负责，且 interrupt/resume
    # 流程不该被再拦一道（human_node 自己 Command(goto=END)，不经专家边）
    for name in EXPERTS:
        builder.add_edge(name, "output_guard")
    builder.add_edge("output_guard", END)

    return builder.compile(checkpointer=get_checkpointer())
