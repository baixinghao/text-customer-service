"""订单专家子图"""

from langchain.agents import create_agent

from customer_service.llm import get_llm
from customer_service.prompts import ORDER_PROMPT
from customer_service.state import State
from customer_service.tools.handoff import make_handoff_tool, make_human_handoff_tool
from customer_service.tools.order_tools import list_orders_by_email, query_order

order_agent = create_agent(
    model=get_llm(),
    tools=[
        query_order,
        list_orders_by_email,
        make_human_handoff_tool(description="用户明确要求转人工/找真人客服，或订单问题多次未解决且用户情绪激动时，转接人工坐席"),
        make_handoff_tool(agent_name="aftersale_expert", description="用户查完订单后要退款、退货时，转接售后专家"),
        make_handoff_tool(agent_name="presale_expert", description="用户转而咨询商品时，转接售前专家"),
    ],
    system_prompt=ORDER_PROMPT,
    state_schema=State,
    name="order_expert",
)
