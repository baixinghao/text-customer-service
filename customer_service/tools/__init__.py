"""工具层：每个业务域一个文件，Mock 数据集中在各文件顶部常量（项目约定）。

已上线：
- handoff.py        转接工具工厂（Command 跨图跳转：专家间轮内接力 + 转人工）
- aftersale_tools.py 退款申请落库（apply_refund）
- order_tools.py    订单 Mock

已迁走：
- 商品检索 Mock（原 product_tools.py）→ rag/retriever.py 的 get_product_retriever_tool，文件已删
- 售后政策 Mock（原 REFUND_POLICY 静态文本）→ rag/retriever.py 的 get_policy_retriever_tool

骨架待填（白哥练习区）：
- ticket_tools.py   工单创建/查询（P4）
- crm_tools.py      客户标签与画像（P4 可选）
"""

from customer_service.tools.handoff import make_handoff_tool, make_human_handoff_tool

# 实现完成后放开导出：
# from customer_service.tools.ticket_tools import create_ticket, query_ticket
# from customer_service.tools.crm_tools import get_customer_profile, tag_customer

__all__ = ["make_handoff_tool", "make_human_handoff_tool"]
