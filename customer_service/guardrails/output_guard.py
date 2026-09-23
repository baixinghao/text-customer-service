"""输出护栏（P5 下半场）：专家答复落地前的最后一道闸——本阶段的灵魂。

复习现场三条事故全发生在出参侧："银行卡3-7个工作日"（政策外承诺）、
"投诉组主管已收到预警"（谎称系统动作）、"所有商品默认包邮"（与语料相反）。
prompt 红线全在，模型照样犯——软约束压不住的，上机械校验。

两条校验通道，一软一硬：
1. 规则/机械层（先跑，纯代码，不跟模型商量）：
   - BANNED_PHRASES 违禁词表：绝对化用语（百分百/保证到账…），命中即打回
   - 轨迹校验（对付"谎称系统动作"的刀）：答复含数字/金额/时限类事实，
     但本轮轨迹零工具调用 → 可疑。要么先检索再承诺，要么交 LLM 审查
2. LLM 审查层（兜底，只对可疑案例启用）：拿检索原文与答复对照，判
   "有没有政策外承诺"，JSON schema 结构化输出

打回重答：Command(goto=原专家) + 追加 SystemMessage 指出违规点；
retry_count（默认 0）到上限 2 次还不过 → 放行兜底话术"这个问题我需要
核实后回复您"，防死循环；通过后清零。

接线（graph.py）：专家直连 END 的边全部改道到本节点；human 节点不过
本闸——人工坐席的话自己负责，且 interrupt/resume 流程不该被再拦一道
（写在 graph.py 的注释里）。
"""

import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END
from langgraph.types import Command

from customer_service.llm import get_llm
from customer_service.state import State

# ── 违禁词表：绝对化用语，命中即打回（不走 LLM，零成本）──────────
# 实测模型会用同义变体绕字面词表（"百分百"被写成"100%"），常见变体都收进来
BANNED_PHRASES = (
    "百分百", "百分之百", "100%", "100％",
    "保证到账", "绝对没问题", "肯定能到账",
    "一定能退", "包过", "稳赚", "绝无问题",
)

# 数字/金额/时限类事实：含阿拉伯数字即视为事实陈述。
# 已知盲区：中文数字（"三天""七日"）不在内——补中文数字会误伤正常口语，
# 留给 LLM 审查层的对照语义兜
_NUMERIC_FACT_RE = re.compile(r"\d")

# 打回重答的次数上限；到上限放行兜底话术，防死循环
_MAX_BOUNCE = 2
_FALLBACK_REPLY = "这个问题我需要核实后回复您"

# LLM 审查的结构化输出（JSON schema，同 input_guard 的判官规格）
_REVIEW_SCHEMA: dict = {
    "title": "OutputReview",
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["PASS", "VIOLATION"],
            "description": "PASS=答复未超出资料；VIOLATION=有政策外承诺或无据事实",
        },
        "reason": {"type": "string", "description": "违规点，一句话"},
    },
    "required": ["verdict", "reason"],
    "additionalProperties": False,
}
_REVIEW_MAX_ATTEMPTS = 3


def _this_round(messages: list) -> list:
    """最近一条用户消息之后的部分 = 本轮轨迹（handoff 场景各自成段，按用户轮切最稳）。"""
    idx = max((i for i, m in enumerate(messages) if getattr(m, "type", "") == "human"), default=-1)
    return messages[idx + 1:]


def _final_answer(round_msgs: list):
    """本轮最终答复：倒找第一条有内容且无 tool_calls 的 AI 消息。"""
    for m in reversed(round_msgs):
        if (getattr(m, "type", "") == "ai"
                and getattr(m, "content", None)
                and not getattr(m, "tool_calls", None)):
            return m
    return None


def _business_tool_called(round_msgs: list) -> bool:
    """本轮有没有调过检索/业务工具。handoff/转接是信令不是取证，剔除。"""
    return any(
        getattr(m, "type", "") == "tool"
        and not str(getattr(m, "name", "")).startswith("transfer_to_")
        for m in round_msgs
    )


def _tool_sources(round_msgs: list) -> str:
    """本轮工具返回的原文拼接，供 LLM 审查对照。"""
    parts = [
        str(getattr(m, "content", ""))
        for m in round_msgs
        if getattr(m, "type", "") == "tool"
        and not str(getattr(m, "name", "")).startswith("transfer_to_")
    ]
    return "\n---\n".join(p for p in parts if p.strip()) or "（本轮没有任何工具检索结果）"


def _llm_review(answer_text: str, round_msgs: list) -> str | None:
    """LLM 审查层：答复 vs 工具原文对照，判政策外承诺。

    返回违规描述字符串（打回）；None = 通过。结构非法/审查层挂了按通过处理
    （宁漏勿冤：规则层已尽责，别因质检挂掉主流程）。
    """
    judge = get_llm().with_structured_output(_REVIEW_SCHEMA)
    messages = [
        SystemMessage(content="你是客服质检员，只输出符合 schema 的判定。"),
        HumanMessage(content=(
            "对照【资料原文】审查【客服答复】：是否包含资料中没有的政策外承诺、"
            "虚构的数字/金额/时限/系统动作，或与资料矛盾的表述。\n"
            f"【资料原文】\n{_tool_sources(round_msgs)}\n\n"
            f"【客服答复】\n{answer_text}"
        )),
    ]
    for _ in range(_REVIEW_MAX_ATTEMPTS):
        try:
            result = judge.invoke(messages)
        except Exception:
            continue  # 结构非法，重新生成
        if not isinstance(result, dict):
            continue
        verdict = str(result.get("verdict", "")).strip().upper()
        if verdict == "PASS":
            return None
        if verdict == "VIOLATION":
            return f"政策外承诺/无据事实（{result.get('reason', '未说明')}）"
        # 非限定词组，重新生成
    return None


def output_guard_node(state: State) -> Command:
    """出参护栏：通过 → END（retry_count 清零）；打回 → goto 原专家 + 违规说明。"""
    round_msgs = _this_round(state["messages"])
    answer = _final_answer(round_msgs)
    if answer is None:
        return Command(goto=END)

    # 防御：人工坐席的消息不该到这儿（human 节点直连 END 不过本闸），真到了也放行
    if getattr(answer, "name", "") == "human_agent":
        return Command(goto=END)

    text = str(answer.content)
    retries = state.get("retry_count") or 0

    violation: str | None = None
    banned = next((p for p in BANNED_PHRASES if p in text), None)
    if banned is not None:
        # 硬规则：绝对化用语，命中即打回，不跟模型商量
        violation = f"包含违禁绝对化用语「{banned}」"
    elif _NUMERIC_FACT_RE.search(text) and not _business_tool_called(round_msgs):
        # 轨迹校验：有数字事实但零工具调用 → 可疑，交 LLM 审查
        violation = _llm_review(text, round_msgs)

    if violation is None:
        return Command(goto=END, update={"retry_count": 0})

    if retries >= _MAX_BOUNCE:
        # 到上限：兜底话术放行，防死循环
        return Command(goto=END, update={
            "messages": [AIMessage(content=_FALLBACK_REPLY, name="output_guard")],
            "retry_count": 0,
        })
    return Command(
        goto=state.get("active_agent") or "router",
        update={
            "messages": [SystemMessage(content=(
                f"你的答复被输出护栏打回：{violation}。"
                f"请依据工具检索到的资料重新作答，不要作政策外承诺；"
                f"资料里没有的就说需要核实。"
            ))],
            "retry_count": retries + 1,
        },
    )
