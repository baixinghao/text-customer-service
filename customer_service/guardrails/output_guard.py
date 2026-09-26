"""输出护栏（P5 下半场）：专家答复落地前的最后一道闸——本阶段的灵魂。

复习现场三条事故全发生在出参侧："银行卡3-7个工作日"（政策外承诺）、
"投诉组主管已收到预警"（谎称系统动作）、"所有商品默认包邮"（与语料相反）。
prompt 红线全在，模型照样犯——软约束压不住的，上机械校验。

两条校验通道，一软一硬：
1. 规则/机械层（先跑，纯代码，不跟模型商量）：
   - BANNED_PHRASES 违禁词表：绝对化用语（百分百/保证到账…），命中即打回
   - 轨迹校验（对付"谎称系统动作"的刀）：答复含数字/金额/时限类事实，
     但本轮轨迹零工具调用 → 可疑，交 LLM 审查。触发条件看本轮，
     但证据池是全会话——跨轮引用上轮查到的资料是合法的（"推荐个键盘"
     →"多少钱"），只切本轮会把有依据的答复误判成编造（K87 事故）
2. LLM 审查层（兜底，只对可疑案例启用）：拿全会话的工具检索原文与答复
   对照，判"有没有政策外承诺"，JSON schema 结构化输出

打回重答：Command(goto=原专家)，同时做两件事——RemoveMessage 摘掉被拒稿
（不进用户对话框，也不留在上下文里误导重答）+ 追加 SystemMessage 指出违规点
（id 带 do-not-render- 前缀，UI 不渲染：质检对话是内部过程，不给用户看）。
retry_count（默认 0）到上限 2 次还不过 → 放行兜底话术"这个问题我需要
核实后回复您"，防死循环；通过后清零。

接线（graph.py）：专家直连 END 的边全部改道到本节点；human 节点不过
本闸——人工坐席的话自己负责，且 interrupt/resume 流程不该被再拦一道
（写在 graph.py 的注释里）。
"""

import re
import uuid

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END
from langgraph.graph.message import RemoveMessage
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

# 打回反馈的消息 id 前缀：与 UI 的 DO_NOT_RENDER_ID_PREFIX 约定一致，
# 带此前缀的消息不进用户对话框——质检对话是内部过程，不给用户看
_GUARD_MSG_ID_PREFIX = "do-not-render-"

# LLM 审查的证据池上限：全会话工具结果都算证据（跨轮引用合法），
# 但只带最近几条进 judge 上下文，防长会话把审查调用烧贵
_MAX_EVIDENCE_TOOLS = 5

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


def _tool_sources(messages: list) -> str:
    """全会话工具返回的原文拼接（最近 N 条），供 LLM 审查对照。

    证据池取全会话而非本轮：跨轮引用（上轮查过 K87、这轮问价格）是合法的，
    只切本轮会把有依据的答复误判成编造。
    """
    parts = [
        str(getattr(m, "content", ""))
        for m in messages
        if getattr(m, "type", "") == "tool"
        and not str(getattr(m, "name", "")).startswith("transfer_to_")
    ]
    parts = [p for p in parts if p.strip()]
    return "\n---\n".join(parts[-_MAX_EVIDENCE_TOOLS:]) or "（本会话暂无任何工具检索结果）"


def _llm_review(answer_text: str, conv_messages: list) -> str | None:
    """LLM 审查层：答复 vs 全会话工具原文对照，判政策外承诺。

    返回违规描述字符串（打回）；None = 通过。结构非法/审查层挂了按通过处理
    （宁漏勿冤：规则层已尽责，别因质检挂掉主流程）。
    """
    judge = get_llm().with_structured_output(_REVIEW_SCHEMA)
    judge_messages = [
        SystemMessage(content="你是客服质检员，只输出符合 schema 的判定。"),
        HumanMessage(content=(
            "对照【资料原文】审查【客服答复】：是否包含资料中没有的政策外承诺、"
            "虚构的数字/金额/时限/系统动作，或与资料矛盾的表述。\n"
            "【资料原文】是本会话历次工具检索的真实结果（可能来自之前的对话轮次，"
            "跨轮引用不算违规）；若资料原文为空，答复中任何具体数字/金额/时限"
            "一律判 VIOLATION，理由直接写「全会话无任何检索依据」。\n"
            f"【资料原文】\n{_tool_sources(conv_messages)}\n\n"
            f"【客服答复】\n{answer_text}"
        )),
    ]
    for _ in range(_REVIEW_MAX_ATTEMPTS):
        try:
            result = judge.invoke(judge_messages)
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
    """出参护栏：通过 → END（retry_count 清零）；打回 → 摘稿 + 隐身反馈 + goto 原专家。"""
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
        # 轨迹校验：本轮零工具调用却报数字 → 可疑，交 LLM 审查。
        # 触发条件看本轮，证据池看全会话（见 _tool_sources）
        violation = _llm_review(text, state["messages"])

    if violation is None:
        return Command(goto=END, update={"retry_count": 0, "final_reply": text})

    # 打回先摘稿：被拒的答复不进用户对话框，也不留在上下文里误导重答
    removals = [RemoveMessage(id=answer.id)] if answer.id else []

    if retries >= _MAX_BOUNCE:
        # 到上限：摘掉违规稿，放行兜底话术，防死循环
        return Command(goto=END, update={
            "messages": [*removals, AIMessage(content=_FALLBACK_REPLY, name="output_guard")],
            "retry_count": 0,
            "final_reply": _FALLBACK_REPLY,
        })
    return Command(
        goto=state.get("active_agent") or "router",
        update={
            "messages": [*removals, SystemMessage(
                content=(
                    f"你的答复被输出护栏打回：{violation}。"
                    "注意：本会话的工具检索记录中找不到该答复的依据。"
                    "涉及价格/参数/政策/时效等事实，必须先调用相应工具取证再作答；"
                    "工具资料里没有的信息，如实告知用户需要核实，禁止凭记忆或推测给出。"
                ),
                id=f"{_GUARD_MSG_ID_PREFIX}{uuid.uuid4()}",
            )],
            "retry_count": retries + 1,
        },
    )
