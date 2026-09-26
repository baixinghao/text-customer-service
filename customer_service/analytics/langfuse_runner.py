"""Langfuse Dataset 跑批器：一个数据集，一次执行，多维打分。

设计原则（2026-09-26 与丽姐定稿）：
    跑 Agent 是评测里唯一的大成本，评分只是零头——所以**同一批题只跑一遍**，
    机械校验、LLM 裁判、ragas 指标挂在同一次执行上多维产出：
      turn_pass_rate    机械校验（路由 + "="数字雷 + 拒答红线，判生死）
      relevance/accuracy  LLM 裁判（软性质量，只出分不判死）
      faithfulness / answer_relevancy / answer_correctness /
      context_recall / context_precision   ragas 指标（仅 metadata.category=rag-* 的题）
    编造率/误杀率两条红线由机械拒答校验出数（库外题须拒答，库内题拒答即误杀）。

数据集契约（cs-agent-eval，题在 Langfuse UI 维护）：
    input           {"phone": "...", "turns": [{"input": "..."}], "in_kb": true|false(可选)}
    expected_output {"turns": [{"expect_agent": ..., "expect_points": [...]}],
                     "ground_truth": "...(可选，rag 题有）"}
    metadata        {"name": ..., "category": route|handoff|mine|guard|anaphora|rag-in-kb|rag-out-of-kb}

运行（真实调 LLM，确认额度再跑）：
    python -m customer_service.analytics.langfuse_runner
    python -m customer_service.analytics.langfuse_runner --only kb- --run-name debug-1
"""

from __future__ import annotations

import argparse
import datetime
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langfuse import Evaluation

from customer_service import build_graph
from customer_service.analytics.dataset_sync import DATASET_NAME
from customer_service.analytics.evaluator import _judge_scores
from customer_service.analytics.langfuse_client import get_langfuse
from customer_service.tracing import eval_trace

# 拒答标记族：命中即认为"承认没查到"。宁窄勿宽——"抱歉"这类客套话不算
REFUSAL_MARKERS = ["没查到", "没有查到", "没有找到", "没有相关", "无法确认",
                   "暂未收录", "不确定", "转人工", "建工单"]

_current_run_name = ""  # run() 开跑前赋值，给 trace 打 run 标签用

# 跑批统计（进程内收集，跑完打印；平台上的分数由 run_experiment 自己回写）
_stats: dict = {"turns": 0, "passed": 0, "rel": 0, "acc": 0, "judged": 0,
                "failures": [], "ragas": {},
                "in_kb_total": 0, "in_kb_refused": 0,
                "oob_total": 0, "oob_fab": 0}


# ── task：多轮驱动图，把每轮结果结构化地带出来 ──────────────────
def run_case(*, item, **kwargs) -> dict:
    """每条用例独立 thread_id 跑完整多轮脚本，返回每轮的答复/路由/证据。

    整条用例包在一个 eval_trace span 里：多轮进同一条 trace，
    并打上 offline-eval / run / category 标签，Tracing 视图按标签过滤分类。
    """
    graph = build_graph()
    case_name = (item.metadata or {}).get("name", item.id)
    category = str((item.metadata or {}).get("category", ""))
    thread_id = f"eval-{case_name}-{uuid.uuid4().hex[:6]}"
    config = {"configurable": {"thread_id": thread_id}}
    state: dict = {"messages": [], "active_agent": None,
                   "user_phone": (item.input or {}).get("phone", "")}

    turns_out: list[dict] = []
    tags = ["offline-eval", f"run:{_current_run_name}", f"category:{category}"]
    with eval_trace(thread_id, tags):
        for turn_in in (item.input or {}).get("turns", []):
            state["messages"] = state["messages"] + [HumanMessage(turn_in["input"])]
            out = graph.invoke(state, config=config)
            state = out
            interrupted = bool(out.get("__interrupt__"))
            contexts = _this_round_contexts(out["messages"])
            turns_out.append({
                "input": turn_in["input"],
                "reply": _last_ai_text(out["messages"]),
                "active_agent": out.get("active_agent"),
                "last_msg_name": getattr(out["messages"][-1], "name", "") or "",
                "contexts": contexts,
                "tool_sources": "\n---\n".join(contexts)[:2000],
                "interrupted": interrupted,
            })
            if interrupted:
                break  # 转人工挂起：E2-3 才支持 resume 续跑，当前记失败并终止本用例
    return {"case_name": case_name, "turns": turns_out}


# ── 评分器 1：机械校验（零成本，判生死）─────────────────────────
def mechanical_evaluator(*, input, output, expected_output, metadata, **kwargs) -> Evaluation:
    expected_turns = (expected_output or {}).get("turns", [])
    problems: list[str] = []
    total = len(expected_turns)
    passed = 0

    for i, exp in enumerate(expected_turns):
        actual = output["turns"][i] if i < len(output["turns"]) else None
        if actual is None:
            problems.append(f"第{i + 1}轮未执行（前轮转人工挂起）")
            continue
        turn_problems = _check_mechanical(exp, actual)
        if turn_problems:
            problems.extend(f"第{i + 1}轮：{p}" for p in turn_problems)
        else:
            passed += 1

    # 拒答红线（仅 rag 题带 in_kb 标记）：库外不拒=编造嫌疑，库内拒答=误杀
    in_kb = (input or {}).get("in_kb")
    if in_kb is not None and output["turns"]:
        refused = any(m in output["turns"][-1]["reply"] for m in REFUSAL_MARKERS)
        if in_kb:
            _stats["in_kb_total"] += 1
            if refused:
                _stats["in_kb_refused"] += 1
                passed = max(passed - 1, 0)
                problems.append("误杀：库内题被拒答")
        else:
            _stats["oob_total"] += 1
            if not refused:
                _stats["oob_fab"] += 1
                passed = max(passed - 1, 0)
                problems.append("编造嫌疑：库外题未拒答")

    rate = passed / max(total, 1)
    _stats["turns"] += total
    _stats["passed"] += passed
    if problems:
        _stats["failures"].append(f"{output['case_name']}：" + "；".join(problems))
    return Evaluation(
        name="turn_pass_rate", value=rate,
        comment="；".join(problems) if problems else "",
    )


def _variants(spec: str) -> list[str]:
    """机械要点的可接受形态："=a|b|c" 任一中即可；外层括号只是分组符，先剥掉。

    修过的坑：不剥括号时 "=(7天|七天)" 会拆出 "(7天"、"七天）"，永远匹配不上。
    """
    spec = spec.strip()
    if spec.startswith("(") and spec.endswith(")"):
        spec = spec[1:-1]
    return spec.split("|")


def _check_mechanical(exp: dict, actual: dict) -> list[str]:
    problems: list[str] = []
    if actual["interrupted"]:
        problems.append("意外挂起（转人工interrupt）")

    expect_agent = exp.get("expect_agent")
    if expect_agent == "input_guard":
        if actual["last_msg_name"] != "input_guard":
            problems.append(f"期望 input_guard 拦截，实际走到 {actual['active_agent']}")
    elif expect_agent and actual["active_agent"] != expect_agent:
        problems.append(f"期望 agent={expect_agent}，实际={actual['active_agent']}")

    normalized = actual["reply"].replace(" ", "")
    for p in exp.get("expect_points", []):
        if p.startswith("=") and not any(
                variant in normalized for variant in _variants(p[1:])):
            problems.append(f"机械要点缺失：{p}")
    return problems


# ── 评分器 2：LLM 裁判（软性质量，只出分不判死）───────────────────
def judge_evaluator(*, input, output, expected_output, metadata, **kwargs) -> list[Evaluation]:
    expected_turns = (expected_output or {}).get("turns", [])
    ground_truth = (expected_output or {}).get("ground_truth", "")
    rel_sum = acc_sum = n = 0
    comments: list[str] = []

    for i, actual in enumerate(output["turns"]):
        exp = expected_turns[i] if i < len(expected_turns) else {}
        soft_points = [p for p in exp.get("expect_points", [])
                       if not p.startswith("=")]
        if ground_truth:
            soft_points.append(f"标准答案：{ground_truth}")
        scores = _judge_scores(actual["reply"], actual["input"],
                               soft_points, actual["tool_sources"])
        if not scores:
            continue
        rel_sum += scores.get("relevance", 0)
        acc_sum += scores.get("accuracy", 0)
        n += 1
        if scores.get("relevance", 5) + scores.get("accuracy", 5) <= 4 \
                and scores.get("comment"):
            comments.append(f"第{i + 1}轮：{scores['comment']}")

    _stats["rel"] += rel_sum
    _stats["acc"] += acc_sum
    _stats["judged"] += n
    if n == 0:
        return []
    return [
        Evaluation(name="relevance", value=rel_sum / n,
                   comment="；".join(comments) if comments else ""),
        Evaluation(name="accuracy", value=acc_sum / n),
    ]


# ── 评分器 3：ragas（仅 rag 题；裁判模型走 DashScope 兼容端点）──────
def _ragas_clients():
    from openai import OpenAI
    from ragas.embeddings import OpenAIEmbeddings
    from ragas.llms import llm_factory

    load_dotenv(Path(__file__).parent.parent / ".env")
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DASHSCOPE_BASE_URL",
                           "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    )
    llm = llm_factory(os.getenv("MODEL_NAME", "qwen-plus"),
                      provider="openai", client=client)
    emb = OpenAIEmbeddings(client=client,
                           model=os.getenv("EMBEDDING_MODEL", "text-embedding-v4"))
    return llm, emb


async def ragas_evaluator(*, input, output, expected_output, metadata, **kwargs) -> list[Evaluation]:
    """faithfulness 忠于检索原文 / relevancy 防跑题 / correctness 对标准答案 /
    context_recall+precision 检索质量（有 ground_truth 才能算）。
    库外题只打 relevancy：无证据无参照，别的分打了也是噪声。"""
    if not str((metadata or {}).get("category", "")).startswith("rag"):
        return []
    from ragas.metrics.collections import (
        AnswerCorrectness, AnswerRelevancy, ContextPrecisionWithReference,
        ContextRecall, Faithfulness,
    )

    llm, emb = _ragas_clients()
    turn = output["turns"][0]
    question, reply, contexts = turn["input"], turn["reply"], turn["contexts"]
    ground_truth = (expected_output or {}).get("ground_truth", "")
    in_kb = (input or {}).get("in_kb", True)

    evals: list[Evaluation] = []

    async def _add(metric, name: str, **kw):
        result = await metric.ascore(**kw)
        value = float(result.value)
        evals.append(Evaluation(name=name, value=value))
        _stats["ragas"].setdefault(name, []).append(value)

    await _add(AnswerRelevancy(llm=llm, embeddings=emb), "answer_relevancy",
               user_input=question, response=reply)
    if in_kb and contexts:
        await _add(Faithfulness(llm=llm), "faithfulness",
                   user_input=question, response=reply, retrieved_contexts=contexts)
    if in_kb and ground_truth:
        await _add(AnswerCorrectness(llm=llm, embeddings=emb), "answer_correctness",
                   user_input=question, response=reply, reference=ground_truth)
        if contexts:
            await _add(ContextRecall(llm=llm), "context_recall",
                       user_input=question, retrieved_contexts=contexts,
                       reference=ground_truth)
            await _add(ContextPrecisionWithReference(llm=llm), "context_precision",
                       user_input=question, reference=ground_truth,
                       retrieved_contexts=contexts)
    return evals


# ── 工具函数 ─────────────────────────────────────────────────
def _last_ai_text(messages: list) -> str:
    for m in reversed(messages):
        if getattr(m, "type", "") == "ai" and getattr(m, "content", None):
            return str(m.content)
    return ""


def _this_round_contexts(messages: list) -> list[str]:
    """本轮（最近一条用户消息之后）工具返回的原文列表，ragas 的 contexts 用。"""
    idx = max((i for i, m in enumerate(messages) if getattr(m, "type", "") == "human"),
              default=-1)
    return [
        str(getattr(m, "content", ""))
        for m in messages[idx + 1:]
        if getattr(m, "type", "") == "tool"
        and not str(getattr(m, "name", "")).startswith("transfer_to_")
        and str(getattr(m, "content", "")).strip()
    ]


def run(only: str | None = None, run_name: str | None = None,
        dataset_name: str = DATASET_NAME, category: str | None = None,
        golden: bool = False) -> None:
    global _current_run_name
    client = get_langfuse()
    items = client.get_dataset(dataset_name).items
    if only:
        items = [it for it in items if only in (it.metadata or {}).get("name", "")]
    if category:
        items = [it for it in items
                 if str((it.metadata or {}).get("category", "")).startswith(category)]
    if golden:
        # 黄金集：metadata.golden=true 的用例（评审签字的核心集，门禁跑它）
        items = [it for it in items if (it.metadata or {}).get("golden")]
    if not items:
        print("没有匹配的用例"); return

    run_name = run_name or f"eval-{datetime.date.today():%Y%m%d}-{uuid.uuid4().hex[:4]}"
    _current_run_name = run_name
    print(f"数据集 {dataset_name}：{len(items)} 条用例，run={run_name}")

    client.run_experiment(
        name=run_name,
        data=items,
        task=run_case,
        evaluators=[mechanical_evaluator, judge_evaluator, ragas_evaluator],
        max_concurrency=4,  # DashScope 限流友好；线程级并发，thread_id 各自独立
        metadata={"dataset": dataset_name},
    )
    client.flush()

    total, passed = _stats["turns"], _stats["passed"]
    print("\n===== 评估报告 =====")
    print(f"轮次通过率：{passed}/{total}（{passed / max(total, 1):.0%}）")
    if _stats["judged"]:
        print(f"裁判平均分（{_stats['judged']} 轮）：相关性 "
              f"{_stats['rel'] / _stats['judged']:.1f} / 准确性 "
              f"{_stats['acc'] / _stats['judged']:.1f}")
    for name, vals in _stats["ragas"].items():
        print(f"ragas·{name}: {sum(vals) / len(vals):.2f}（{len(vals)} 条）")
    if _stats["oob_total"]:
        fab = _stats["oob_fab"] / _stats["oob_total"]
        print(f"编造率：{_stats['oob_fab']}/{_stats['oob_total']}（{fab:.0%}）"
              f"红线 <5%  {'✅' if fab < 0.05 else '❌ 破线'}")
    if _stats["in_kb_total"]:
        kill = _stats["in_kb_refused"] / _stats["in_kb_total"]
        print(f"误杀率：{_stats['in_kb_refused']}/{_stats['in_kb_total']}（{kill:.0%}）"
              f"红线 <10%  {'✅' if kill < 0.10 else '❌ 破线'}")
    if _stats["failures"]:
        print(f"\n失败 {len(_stats['failures'])} 条用例：")
        for f in _stats["failures"]:
            print(f"  ✗ {f}")
    else:
        print("全部通过 ✓")
    print(f"\nLangfuse 对比视图：Datasets → {dataset_name} → Runs → {run_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Langfuse Dataset 跑批质检（一次执行多维打分）")
    parser.add_argument("--only", default=None, help="只跑名字含该关键词的用例")
    parser.add_argument("--run-name", default=None, help="Dataset Run 名（默认按日期生成）")
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--category", default=None,
                        help="按分类前缀过滤，如 rag / route / guard")
    parser.add_argument("--golden", action="store_true",
                        help="只跑黄金集（metadata.golden=true 的评审签字用例）")
    args = parser.parse_args()
    run(only=args.only, run_name=args.run_name, dataset_name=args.dataset,
        category=args.category, golden=args.golden)
