# 反幻觉设计（RAG 无命中时的硬约束）

> 适用场景：`config` 用的是 flash 级小模型（deepseek-v41-flash / qwen-flash / gemini-flash 一类），
> 表现是**检索不到就编造、无视 prompt 里的红线**。
>
> 核心判断先说清楚：**这不是提示词问题，是接线问题。**
> 当前链路上"资料不包含就说不知道"只是一句 prompt 文字，没有任何代码在拦；
> 而 flash 级模型在小样本下对"格式正确但内容无关"的资料几乎没有免疫力——
> 它看到的是上下文里有像模像样的条目，于是就拼一个答案出来。
>
> 本文按「代码硬约束优先，prompt 只做锦上添花」排序。P0 三项做完，编造率就有量级下降。

---

## 一、根因定位（对应当前代码）

| # | 位置 | 现状 | 为什么会导致编造 |
|---|---|---|---|
| 1 | `rag/retriever.py:122` `_build_fusion_retriever` | 向量检索 `similarity_top_k=5`，**无相似度下限** | 知识库只有 5 条数据时，任何查询都会返回 5 条"最像的"。没有下限 = 永远有资料 = 模型从不触发"没查到" |
| 2 | `rag/retriever.py:189` `reranker.postprocess_nodes` | 只用 rerank 重排取 top_n，**分数看完就丢** | rerank score 是唯一可信的相关性证据，却被丢掉，没参与任何判断 |
| 3 | `rag/retriever.py:159` `_format_nodes` | 空结果返回一句普通文本 `"知识库中没有检索到相关信息。"` | ①无机器可判的硬标记；②这句话本身在用户消息里也可能出现，模型分不清"工具说没有"还是"用户说了这句话" |
| 4 | `prompts.py:16,26,36,53` | 四条专家 prompt 各挂一句 `🔴规则红线:...禁止编造事实` | 纯文本约束，flash 模型在工具结果"看起来有料"时会直接违背；而且四份重复，维护会漂移 |
| 5 | `graph.py:54-55` | 专家节点**直连 END** | 答复没有任何出口审查。`guardrails/output_guard.py` 还是 `NotImplementedError` 骨架 |
| 6 | `guardrails/input_guard.py:23` | 未接线、未实现 | 注入类攻击（"忽略之前的指令"、伪造 `【来源：...】`）畅通无阻 |
| 7 | `llm.py:18` | `temperature=0.3` | 事实型问答 + 小模型，0.3 已经足够让它在无证据时"顺着编" |

> 一句话：**检索层没有"没找到"这个状态，输出层没有"没证据"这个检查。**
> 模型被放在了唯一一个它天然会做错的决策点上。

---

## 二、P0 三项（做完立刻见效，不动骨架结构）

### P0-1 检索层加硬门槛：让"没找到"成为一等状态

在 `rag/retriever.py` 里给 rerank 结果加阈值过滤 + 相对差距兜底：

```python
# rag/retriever.py 顶部常量
RERANK_MIN_SCORE = 0.30   # 绝对下限，待实测校准（见 P0-1 校准法）
RERANK_MIN_GAP = 0.15     # 相对兜底：最高分没比阈值高出这么多，也视为无命中

def _filter_by_score(nodes: list[NodeWithScore]) -> list[NodeWithScore]:
    """rerank 后的相关性硬门槛。宁缺毋滥：不确定就当没查到。"""
    if not nodes:
        return []
    top = nodes[0].score or 0.0
    if top < RERANK_MIN_SCORE + RERANK_MIN_GAP:
        return []
    return [n for n in nodes if (n.score or 0.0) >= RERANK_MIN_SCORE]
```

然后每个工具里把两行改成三行：

```python
    def search_product(keyword: str) -> str:
        nodes = retriever.retrieve(keyword)
        nodes = reranker.postprocess_nodes(nodes, query_str=keyword)
        return _format_nodes(_filter_by_score(nodes))   # ← 只多这一层
```

**阈值校准法（10 分钟，别拍脑袋）**：临时加一行 `print([(n.score, n.text[:20]) for n in nodes])`，
分别跑 5 个**确实存在**的问题（如"退款多久到账"）和 5 个**知识库外**的问题（如"你们支持比特币付款吗"），
看两组分数分布的间隔，阈值取在间隔中间。Qwen3-Reranker 的 score 在 0–1 区间，
参考起点 0.3 左右，但**以你实测的分布为准**[^1][^2]。

> 注意：`top_k=5` 而知识库只有 5 条时，top_k 形同虚设。
> 知识库小的时候，**门槛比 top_k 重要一个数量级。**

### P0-2 工具返回值改结构化硬标记：让"无证据"不可被忽略

替换 `_format_nodes` 的空分支：

```python
NO_EVIDENCE = "[NO_EVIDENCE]"   # 机器可判的硬标记，prompt 与 output_guard 都认这个

def _format_nodes(nodes: list[NodeWithScore]) -> str:
    if not nodes:
        return (
            f"{NO_EVIDENCE} 检索无结果（分数未达相关性门槛）。\n"
            "纪律：禁止基于常识或推测回答本问题；"
            "应如实告知用户你没有查到相关资料，并给出下一步（转人工 / 建工单 / 换个说法再查）。"
        )
    ...
```

同时在**有结果**时也加来源标签，为 P0-3 的引用校验铺路（现在 `_format_nodes:168,172` 已有
`【来源：{source}】`，保留并确保每段必有）：

```python
    parts.append(f"【来源：{source}】\n{text}")
```

关键点：**`[NO_EVIDENCE]` 是全项目唯一的"查无此项"信号**，
input_guard / output_guard / 评测脚本都只认这一个字符串，避免多处文案漂移。

### P0-3 接线 output_guard：出口做证据校验，而不是让模型自律

`guardrails/output_guard.py` 现在是空骨架，先用**纯规则版**（零成本零延迟）就能挡住大部分编造：

```python
# guardrails/output_guard.py
import re

from langchain_core.messages import AIMessage
from langgraph.graph import END
from langgraph.types import Command

from customer_service.rag.retriever import NO_EVIDENCE

BANNED_PHRASES = [
    "百分百", "100%", "绝对没问题", "保证到账", "一定能", "必定",
    "我保证", "随时可以退", "无条件退",
]
# 事实型断言特征：编造时最常见的是凭空出现具体数字/时限/金额
FACT_PATTERNS = [
    r"\d+\s*(个)?工作日", r"\d+\s*小时", r"\d+\s*天", r"\d+\s*元", r"\d+\s*%",
]


def _last_tool_text(messages) -> str:
    """本轮所有 ToolMessage 的拼接，作为唯一证据来源。"""
    return "\n".join(
        m.content for m in messages
        if getattr(m, "type", "") == "tool" and isinstance(m.content, str)
    )


def output_guard_node(state) -> Command:
    """出口审查：①绝对化承诺 ②无证据事实断言 ③提示词泄露。"""
    reply = state["messages"][-1]
    text = reply.content if isinstance(reply.content, str) else ""
    evidence = _last_tool_text(state["messages"])

    # ① 绝对化承诺 → 改写为低风险表述
    for bad in BANNED_PHRASES:
        if bad in text:
            text = text.replace(bad, "以页面/政策实际显示为准")

    # ② 无证据时，禁止出现具体数字断言 → 打回重答（带硬指令）
    no_evidence = NO_EVIDENCE in evidence or not evidence.strip()
    if no_evidence and any(re.search(p, text) for p in FACT_PATTERNS):
        retry = state.get("retry_count", 0)
        if retry < 2:
            return Command(
                goto=state["active_agent"],
                update={
                    "retry_count": retry + 1,
                    "messages": [AIMessage(
                        "[系统纠偏] 本轮检索无结果，你的答复里出现了资料中没有的具体数字。"
                        "请重答：如实告知未查到，不要给出任何时限/金额/比例，"
                        "并给出下一步（转人工或建工单）。"
                    )],
                },
            )
        # 超过重试上限：兜底固定话术，绝不放行编造内容
        text = "抱歉，这项信息我这边暂时没有查到确切依据，为避免给您错误信息，我为您转接人工同事核实。"

    # ③ 提示词泄露
    if any(k in text for k in ("system prompt", "系统提示词", "You are a")) or "expert" in text:
        text = "抱歉，这个我无法提供，我们回到您的问题上好吗？"

    return Command(goto=END, update={"messages": [AIMessage(text)]})


# state.py 同步放开规划字段（给默认值，老 checkpoint 不炸）
#   retry_count: int = 0
```

`graph.py` 改接线（原 `add_edge(name, END)` 那两行）：

```python
    for name in EXPERTS:
        builder.add_edge(name, "output_guard")     # 专家不再直连 END
    builder.add_edge("output_guard", END)
```

> 打回重答 `goto=state["active_agent"]` 依赖 `active_agent` 已被 router/handoff 正确写入。
> 重试上限 2 次（`state.py:21` 的规划字段 `retry_count` 正好为此准备），到顶走固定话术。
> **绝不允许"重试到超时"或"超限放行原文"** —— 这是死循环与事故的两个来源。

---

## 三、P1 两项（把 prompt 从"唯一防线"降级为"辅助"）

### P1-1 prompt 改写：把红线从"劝告"变成"决策树"

四条专家 prompt 各挂一句红线的写法有两个毛病：**重复**（改一处漏三处）和**抽象**
（"禁止编造"没有告诉模型在无证据时*做什么*）。建议在 `prompts.py` 里抽公共常量：

```python
# prompts.py
EVIDENCE_CLAUSE = """
【最高优先级 · 资料纪律】
1. 你的每一句事实性陈述都必须能在本轮工具返回的资料里找到出处。
2. 工具返回含 [NO_EVIDENCE] 标记 → 本轮禁止给出任何具体参数、价格、时限、金额、比例；
   只允许回答："这项我没有查到确切资料"，并给出下一步（转人工 / 建工单 / 换个关键词再查）。
3. 资料里有相关内容但不完整 → 只讲资料里有的部分，明确说明"其余信息我需要为您核实"。
4. 不确定时，说"不确定"永远优于给出一个看起来合理的答案。
   猜错的代价由用户承担，追问的代价只是一句话。
"""
```

然后各专家 `PROMPT = HEADER + EVIDENCE_CLAUSE`，删掉各自那行 `🔴规则红线:...`。

**顺序有讲究**：长 system prompt 里，把资料纪律放在**头部**比尾部有效，
同时它位置稳定 → 对 KV cache 友好（同一专家多轮对话时前缀不变，命中缓存）。

### P1-2 模型参数：`temperature` 降到 0

`llm.py:18` 的 `temperature=0.3` 建议改成 `0.0`（事实型客服不需要发散）。
如果后面做自洽性检查（P2），再单独用一个 `temperature=0.7` 的实例做多次采样，别混用。

---

## 四、P2 两项（量化兜底，看你要做到哪一步）

### P2-1 引用强校验（Citation Check）

prompt 里要求答复带 `【来源：refund_policy.md】`，`output_guard` 校验：
答复中出现的每个来源标签，必须是本轮 ToolMessage 里真实出现过的。
出现凭空来源 → 打回。好处是**对用户透明**（用户能看到依据），
代价是 prompt 要重写、且要容忍模型漏标（漏标不能算错，**标错才算错**）。

### P2-2 logprobs 不确定性闸门

DashScope OpenAI 兼容接口支持 `logprobs`（需确认你选的 flash 型号是否支持），
拿生成 token 的平均对数概率做置信度：

```python
resp = get_llm().bind(logprobs=True, top_logprobs=3).invoke(msgs)
# 平均 logprob < -1.0 或存在 logprob < -3 的 token → 视为低置信
# 低置信 + 无证据 → 直接转人工，不输出
```

这是**软信号**，只做"加一道闸门"，不能替代 P0。阈值必须用你自己的评测集调，别抄别人的数。

---

## 五、评测：把"不编造"变成可回归的指标

没有评测集，上面的阈值和 prompt 都是玄学，改一次 prompt 可能悄悄退化。

建 `customer_service/test/eval_dataset.jsonl`，**50 条起手，其中 20 条是知识库外的**：

```jsonl
{"q": "退款多久到账", "in_kb": true,  "must_contain": ["工作日"], "must_not_contain": []}
{"q": "你们收比特币吗", "in_kb": false, "must_contain": [], "must_not_contain": ["支持", "可以", "元", "工作日"]}
```

跑批脚本（不调真模型，遵守项目约定：真调用前找白哥确认）至少算两个数：

| 指标 | 定义 | 目标 |
|---|---|---|
| **编造率** | 知识库外问题中出现 must_not_contain 词的比例 | < 5% |
| **误杀率** | 知识库内问题被拒答/转人工的比例 | < 10% |

改 prompt / 改阈值 / 换模型后**必跑**。这两条线任何一条破了就当回归处理。

---

## 六、实施顺序（建议）

```
P0-1 检索门槛 + 阈值校准     ← 最高性价比，半天，改 3 处
P0-2 NO_EVIDENCE 硬标记      ← 与 P0-1 同一批改
P0-3 output_guard 规则版接线 ← 顺手完成 P5 骨架的一半
P1-1 prompt 抽公共纪律条款   ← 半小时
P1-2 temperature → 0         ← 一行
P2-1 引用校验 / P2-2 logprobs ← 有余力再做
P5   input_guard（注入防御）  ← 与本文第 7 条根因对应
```

做完 P0 三项，用第五节的两条线量一次，把数字记下来——
**后面每一次"感觉变好了"都要有数字背书。**

---

[^1]: Qwen3-Reranker 相关性分数为 0–1 区间，阈值设定需按场景实测：
      [Qwen3-Reranker-8B 入门指南：理解 rerank score 含义与阈值设定逻辑](https://blog.csdn.net/weixin_42627459/article/details/159370087)
[^2]: 官方接口文档（OpenAI 兼容重排序）：
      [阿里云百炼 OpenAI 兼容重排序](https://platform.qianwenai.com/docs/api-reference/rerank/openai-rerank)
