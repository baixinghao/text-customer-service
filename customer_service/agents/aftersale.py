"""售后专家子图"""

from langchain.agents import create_agent

from customer_service.llm import get_llm
from customer_service.prompts import AFTERSALE_PROMPT
from customer_service.rag import get_faq_retriever_tool, get_policy_retriever_tool
from customer_service.state import State
from customer_service.tools.aftersale_tools import apply_refund
from customer_service.tools.handoff import make_handoff_tool, make_human_handoff_tool
from customer_service.tools.ticket_tools import create_ticket, query_ticket

aftersale_agent = create_agent(
    model=get_llm(),
    tools=[
        get_policy_retriever_tool(),
        get_faq_retriever_tool(),
        apply_refund,
        create_ticket,
        query_ticket,
        # 轮内接力：自己这段干完、诉求跨域时才交接；轮级首发分流归 router
        make_handoff_tool(agent_name="presale_expert", description="用户转而咨询、购买商品时，转接售前专家"),
        make_handoff_tool(agent_name="order_expert", description="用户不知道订单号、要查订单时，转接订单专家"),
        make_handoff_tool(agent_name="complaint_expert", description="用户情绪激动、言语激烈、威胁投诉/曝光时，转接投诉安抚专员"),
        # 错题集：转人工必须用 make_human_handoff_tool——它不改 active_agent；
        # 用 make_handoff_tool(agent_name="human") 会把 active_agent 写成 human，
        # output_guard 打回重答会 goto 错节点
        make_human_handoff_tool(description="用户明确表明需要转人工，或同一问题重复超过2次仍未解决时，"
                                            "转接人工客服（用户愤怒/要投诉先转投诉专员，别直接甩人工）"),
    ],
    system_prompt=AFTERSALE_PROMPT,
    state_schema=State,
    name="aftersale_expert",
)
