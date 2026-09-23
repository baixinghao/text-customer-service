"""护栏子包（骨架，实现计划 P5）：给客服系统上"紧箍咒"。

- input_guard.py   输入护栏：敏感词、辱骂、prompt 注入、越权请求，进 router 前拦截
- output_guard.py  输出护栏：AI 答复的合规审查（虚假承诺、泄露提示词、违规话术），回复用户前把关

两个都是父图普通节点，签名 (state) -> Command，接线后图变成：
  START → input_guard → router → 专家 → output_guard → END
"""
