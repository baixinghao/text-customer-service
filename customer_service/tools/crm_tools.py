"""客户画像 / 标签工具（骨架，实现计划 P4 可选加分项）。

作用：成熟客服系统会给用户打标签（VIP、退货意愿、新客咨询……），
让专家 agent 能"看人说话"——对 VIP 更殷勤，对高退货风险用户主动给政策。

Mock 数据集中在本文件顶部常量（项目约定），将来接真实 CRM 时整段替换。

这个文件该实现什么：
- get_customer_profile：按手机号返回画像（历史订单数、累计金额、标签列表）
- tag_customer：给用户追加标签（专家在对话中判断后主动打标）
- 配套：把画像注入专家上下文——可以做成工具让 agent 自己查，
  也可以在 router 节点查出后写进 State（规划字段 user_phone / customer_profile，见 state.py）
"""

from langchain.tools import tool

# Mock 客户库：phone -> {name, level, tags, total_orders, total_amount}
MOCK_CUSTOMERS: dict = {}


@tool
def get_customer_profile(phone: str) -> str:
    """根据手机号查询客户画像：等级、历史订单量、已有标签。"""
    raise NotImplementedError("P4 练习：查 MOCK_CUSTOMERS，组织成一段可读的画像文本返回")


@tool
def tag_customer(phone: str, tag: str) -> str:
    """给指定客户追加一个标签，如 高意向/退货风险/已安抚。"""
    raise NotImplementedError("P4 练习：写入 MOCK_CUSTOMERS[phone]['tags']，幂等去重")
