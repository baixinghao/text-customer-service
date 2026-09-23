"""会话摘要 / 长期记忆（骨架，实现计划 P2 可选加分项）。

这个文件该实现什么：
- 短期记忆（checkpointer）解决"一个会话内记得住"，但消息越攒越多会撑爆上下文；
  本模块负责在消息数超阈值时，让 LLM 把历史压成摘要，只留摘要 + 最近 N 条
- 两种落法任选（都试最好）：
  1. LangChain 1.x 的 SummarizationMiddleware，挂进 create_agent，开箱即用
  2. 自己在父图加一个 summarize 节点：判断长度 → 调 LLM 摘要 →
     用 RemoveMessage + 摘要 SystemMessage 重写 messages
- 再进一步（长期记忆）：把"用户偏好/历史结论"从摘要里提出来存库，
  新会话开场注入——这才是成熟客服系统"记得老客"的完整形态

关键 API：
  from langchain.agents.middleware import SummarizationMiddleware   # 做法 1
  from langgraph.graph.message import RemoveMessage                 # 做法 2
"""

# 做法 1 的参考 import（实现时取消注释）：
# from langchain.agents.middleware import SummarizationMiddleware
#
# def build_summarization_middleware():
#     """返回配置好的摘要中间件：触发阈值、保留条数、摘要模型都在这里定。"""
#     raise NotImplementedError("P2 加分练习")
