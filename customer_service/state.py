"""全局共享 State：父图与所有专家子图同构（同一套 schema）。

- messages: 会话历史，add_messages 归并，父子图天然互通（黑板模式）
- active_agent: 当前接待的专家节点名。router 首轮写入，handoff 时更新；
  后续轮次 router 直接放行，不再重复意图识别
- user_phone: 用户联系电话（P4 放开）。工单工具的入单依据，由外层写入
  （API 网关注入 / CLI --phone 参数），LLM 工具签名里不出现它
- retry_count: 输出护栏打回重答的次数（P5 放开，默认 0）。到上限放行兜底
  话术防死循环；通过一次即清零
"""

from langgraph.graph import MessagesState


class State(MessagesState):
    active_agent: str
    user_phone: str = ""    # 默认 ""：老 checkpoint 反序列化不炸（规划注释的约定）
    retry_count: int = 0    # 输出护栏的重试预算，通过后清零


# ── 规划字段（练到对应阶段再逐个放开，见 docs/IMPLEMENTATION_PLAN.md）──────────
# 放开时注意给默认值（如 str = ""），别让已存在的 checkpoint 反序列化炸掉：
#   sentiment: str       最新消息情绪标签，路由投诉专家用（P4）
