"""售前专家子图：create_agent 产物本身就是一张 CompiledStateGraph，直接当父图节点挂。"""

from langchain.agents import create_agent

from customer_service.llm import get_llm
from customer_service.prompts import PRESALE_PROMPT
from customer_service.rag import get_faq_retriever_tool, get_product_retriever_tool
from customer_service.state import State
from customer_service.tools.handoff import make_handoff_tool, make_human_handoff_tool

presale_agent = create_agent(
    model=get_llm(),
    tools=[
        get_product_retriever_tool(),
        get_faq_retriever_tool(),
        # 轮内接力：自己这段干完、诉求跨域时才交接；轮级首发分流归 router
        make_handoff_tool(agent_name="aftersale_expert", description="用户需要退货、退款、售后政策时，转接售后专家"),
        make_handoff_tool(agent_name="order_expert", description="用户要查询订单时，转接订单专家"),
        make_handoff_tool(agent_name="complaint_expert", description="用户情绪激动、言语激烈、威胁投诉/曝光时，转接投诉安抚专员"),
        make_human_handoff_tool(description="用户明确要求转人工/找真人客服时，转接人工坐席（情绪激烈先转投诉专员，别直接甩人工）"),
    ],
    system_prompt=PRESALE_PROMPT,
    state_schema=State,  # 与父图同构：messages 互通，active_agent 可写
    name="presale_expert",
)
