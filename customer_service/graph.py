"""父图装配层：只做路由和接线，不写业务。

流程：
  START → input_guard（P5 硬闸门：规则先行，LLM 兜底；拦截直接 END + 写死模板）
        → router（每轮 LLM 意图识别，Command(goto) 定首发专家并刷新 active_agent）
        → 专家子图（内部是 create_agent 的 model⇄tools ReAct 循环）
        → END（轮内跨域接力靠 handoff 工具的 Command(goto, graph=PARENT)，
               父图不需要为它们画边）

分层决策（两种粒度，不是冗余）：
- router 管【轮级】分流：每轮重判意图定首发，绝不凭 active_agent 粘滞直达——
  旧设计的浪费全在直达短路（售后题发给售前，白跑一次 3.4s model 调用再转走）。
  router 用小成本意图模型，首发必对，远低于错误专家白跑
- handoff 管【轮内】交接：一句话多意图时（"退货+顺便推荐新品"），首发专家
  带着完整上下文干完自己那段，再交接给下家；ToolMessage 留共享历史，无缝接力

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
    """分流节点：每轮都让 LLM 判意图，goto 首发专家并刷新 active_agent。

    不做"老客直达"——轮级意图切换靠每轮重判接住（见文件头"分层决策"）。
    router 的 prompt 短、输出一个节点名，成本远低于错误专家白跑一次。
    """
    resp = get_llm().invoke([SystemMessage(content=ROUTER_PROMPT), *state["messages"]])
    target = resp.content.strip()
    if target not in EXPERTS:
        target = "presale_expert"  # 兜底：识别不了的一律按售前咨询接待
    return Command(goto=target, update={"active_agent": target})


def _expert_exit(state: State) -> str:
    """专家完赛后的去向判定：handoff 在途 → END 收工；正常答完 → output_guard。

    不能给专家画 expert→output_guard 静态边：handoff 工具的 Command(goto) 触发
    目标专家时，静态边会【同时】触发 output_guard——护栏和目标专家并发抢跑，
    审的是没有最终答复的旧 state（最小复现钉死：guard 跑两次，首次入参无答复）。
    判定依据：handoff 生效时子图被拦停，最后一条消息是 transfer_to_* 的
    ToolMessage；正常答完则是专家的最终 AIMessage。
    """
    last = state["messages"][-1]
    if (getattr(last, "type", "") == "tool"
            and str(getattr(last, "name", "")).startswith("transfer_to_")):
        return END  # Command(goto) 已把本轮交给目标专家/human，本边不添乱
    return "output_guard"


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
    # 专家出参统一过 output_guard（条件边，见 _expert_exit 的双触发教训）：
    # 通过/兜底才 END，打回 goto 原专家重答。
    # human 节点故意不过 output_guard——人工坐席的话自己负责，且 interrupt/resume
    # 流程不该被再拦一道（human_node 自己 Command(goto=END)，不经专家边）
    for name in EXPERTS:
        builder.add_conditional_edges(
            name, _expert_exit, {"output_guard": "output_guard", END: END}
        )
    builder.add_edge("output_guard", END)

    return builder.compile(checkpointer=get_checkpointer())
