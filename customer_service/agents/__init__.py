"""专家子图集合：一个文件一个专家，create_agent 产物直接挂父图节点。

已上线：
- presale.py    售前：商品咨询
- aftersale.py  售后：退换货政策、退款、工单
- order.py      订单：查单
- complaint.py  投诉 / 情绪安抚（P4）：安抚四步走 + 工单 + handoff
- human.py      转人工节点，human-in-the-loop interrupt（P3）
"""

from customer_service.agents.aftersale import aftersale_agent
from customer_service.agents.complaint import complaint_agent
from customer_service.agents.order import order_agent
from customer_service.agents.presale import presale_agent

# 实现完成后放开导出，并在 graph.py 的 EXPERTS 注册：
# from customer_service.agents.human import human_node

__all__ = [
    "presale_agent",
    "aftersale_agent",
    "order_agent",
    "complaint_agent",
]
