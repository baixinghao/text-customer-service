"""投诉 / 情绪安抚专家子图（P4）：接待高情绪浓度的用户。

人设四步走（prompts.py 的 COMPLAINT_PROMPT）：道歉 → 共情 → 方案 → 时限，
绝不与用户争辩。工具组合：工单两兄弟（create_ticket / query_ticket）+
handoff 三件套（售前/订单/售后，轮内接力）+ 转人工。
"""

from langchain.agents import create_agent

from customer_service.llm import get_llm
from customer_service.prompts import COMPLAINT_PROMPT
from customer_service.state import State
from customer_service.tools.handoff import make_handoff_tool, make_human_handoff_tool
from customer_service.tools.ticket_tools import create_ticket, query_ticket

complaint_agent = create_agent(
    model=get_llm(),
    tools=[
        create_ticket,
        query_ticket,
        # 轮内接力：自己这段干完、诉求跨域时才交接；轮级首发分流归 router
        make_handoff_tool(agent_name="presale_expert", description="用户情绪平复、转而咨询商品时，转接售前专家"),
        make_handoff_tool(agent_name="order_expert", description="用户要查订单、物流进度时，转接订单专家"),
        make_handoff_tool(agent_name="aftersale_expert", description="用户要办退款、退货、换货时，转接售后专家"),
        make_human_handoff_tool(description="用户明确要求转真人客服，或安抚无效、问题超出权限时，转接人工坐席"),
    ],
    system_prompt=COMPLAINT_PROMPT,
    state_schema=State,
    name="complaint_expert",
)
