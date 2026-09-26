# 智能文本客服 · 评测计划（丽姐接手专用）

> 定位：与 `IMPLEMENTATION_PLAN.md`（白哥的开发主线）平行的**评测主线**。
> 开发侧 P0–P5、P7 已完成，P1（RAG）与 P6（离线质检）代码已写完但验收未跑。
> 本计划按「观测 → 功能质检 → RAG 专项 → 回归门禁」四步走，
> 每步有明确验收标准，跑完勾 checkbox。
>
> 背景对表（给测试同行看的翻译）：
> | 评测概念 | 传统测试对应 |
> |---|---|
> | OTel trace / Langfuse | 全链路日志 + 调用链监控（APM） |
> | Agent eval（预设用例 + 断言） | 功能测试用例 + 自动化断言 |
> | LLM 裁判打分 | 探索性测试的自动化替身（有锚点 rubric） |
> | Ragas 指标 | RAG 模块的专项质量指标（类似接口的准确率/召回率） |
> | 编造率/误杀率红线 | 质量门禁 / 回归基线 |

## 〇、现状盘点（接手时已确认）

| 资产 | 状态 | 评测侧怎么用 |
|---|---|---|
| `analytics/evaluator.py` | ✅ 已实现：17 条用例（路由/handoff/数字埋雷/护栏/指代），机械校验 + LLM 裁判双维度打分 | E2 的基线，直接跑 |
| `docker-compose.observability.yml` | ✅ Langfuse v3 + ClickHouse + MinIO + OTel Collector(:4317/:4318) 已容器化 | E1 起栈即用 |
| 应用侧 OTel 接线 | ❌ 未做（compose 注释里留了环境变量） | E1 要补的唯一代码 |
| LangSmith | ❌ 被 `customer_service/__init__.py` 硬关（本机连不上 api.smith） | **不走 LangSmith，追踪全走 OTel→Langfuse** |
| `docs/ANTIHALLUCINATION.md` | ✅ 反幻觉设计 + 编造率/误杀率两条线已定义 | E3 的门禁指标直接沿用 |
| Ragas | ❌ 未引入 | E3 新增 `eval` extra |

## E1 观测接线：OTel + Langfuse ⭐第一步

- 目标：每一次对话在 Langfuse 里能看到完整调用链（router LLM → 专家 LLM →
  每个工具调用），含 token 消耗、延迟、报错。
- 前置：核心层 `docker compose up -d` 已起，然后
  `docker compose -f docker-compose.observability.yml up -d`（首次要等 ClickHouse 迁移）。
  控制台 http://localhost:3000 ，账号 `lijie@tcs.local`（密码见 compose 文件）。
- 动手（2026-09-23 已完成接线，此为复盘）：
  - `uv sync --extra obs`（openinference-instrumentation-langchain + OTLP HTTP exporter）
  - `customer_service/tracing.py`：TracerProvider + OTLP HTTP exporter +
    `LangChainInstrumentor().instrument()`，在 `__init__.py` import 图之前初始化；
    CLI 入口可用 `bind_session(thread_id)` 打 session 维度，langgraph dev 下靠
    LangGraph 自动注入的 thread_id metadata 归 session
  - 应用侧只需 `OTEL_EXPORTER_OTLP_ENDPOINT` 一个变量（认证在 Collector，无需 key）
  - 两个实测踩坑：① tracing 初始化早于 llm/db 的 dotenv 加载，必须自己 load
    `customer_service/.env`；② 改观测代码后必须**重启 langgraph dev**（初始化在
    进程启动时一次完成，热重载不覆盖），且 Windows 下要 `PYTHONUTF8=1` 起服务，
    否则 langgraph_api 读 OpenAPI 文件 GBK 解码崩；③ **Langfuse 项目错配**：
    UI 手工建过同名项目（id 不同、密钥不同），.env 里的 LANGFUSE_* 指向错项目时
    数据集会写进"另一个 text-customer-service"——排查法：PG 的 langfuse 库
    `select id,name from projects` 对 id，别对名字；防御：脚本连接时打印
    host+key 前缀横幅，并用 compose INIT 的那对 key 作为唯一口径
  - 注意：`__init__.py` 硬关 LangSmith 那两行**不动**——OTel 是独立通道，不冲突
- 练的知识点：OTel span/trace 模型 / OpenInference 对 LangChain 的自动埋点 /
  Collector 转发（应用 → :4318 → Langfuse 原生 OTel 端点）/ trace 与会话维度
- 验收标准（走 Web UI http://localhost:5173，URL 里的 threadId 即 thread_id）：
  - [x] UI 里聊一轮含工具调用的对话（如"查一下订单 1"），Langfuse 里
        能看到完整 trace 树：LLM 调用、工具调用、各段延迟与 token 数
  - [x] 多轮对话的 trace 能按 thread_id（session）聚合查看
  - [x] 故意触发一次护栏拦截，能在 trace 里看到 input_guard 的路径
  - [x] 触发一次转人工（interrupt → 坐席 resume），trace 里能看到挂起与恢复两段

## E2 Agent 功能质检（数据集托管在 Langfuse 平台）

> 2026-09-24 定稿方向：**用例不落代码/jsonl，统一放 Langfuse Dataset**——
> UI 可增删改、有版本、每次跑批生成可对比的 Dataset Run、线上 trace 可回灌。
> 代码侧只留两个脚本：sync（一次性迁移）+ runner（执行与评分）。

- 动手（接线已完成，此为复盘）：
  1. `analytics/dataset_sync.py`：EVAL_CASES 17 条已迁入数据集 `cs-agent-eval`
     （按 metadata.name 判重，幂等，不覆盖 UI 里人工改过的用例）
  2. `analytics/langfuse_runner.py`：执行引擎用 langfuse v4 的 `run_experiment`
     （task=多轮驱动图 / 两个评分器=机械校验 turn_pass_rate 判生死 + LLM 裁判
     relevance/accuracy 只出分），分数自动回写平台
  3. 跑基线（真实调 LLM，自己的 key）：
     `PYTHONIOENCODING=utf-8 .venv/Scripts/python -m customer_service.analytics.langfuse_runner --run-name baseline-v1`
  4. 后续补用例（转人工 resume、多轮记忆、边界输入）直接在 Langfuse UI 加，
     数据契约：`input.turns[].input` + `expected_output.turns[].expect_agent|expect_points`
  5. 测试侧打标（2026-09-26 补）：每条用例的 trace 带 `offline-eval` / `run:<名>` /
     `category:<分类>` 标签（`tracing.eval_trace`），Tracing 视图按标签把评测流量
     和生产流量分开；Datasets→Runs 即跑测历史。两个实测坑：openinference 的
     `using_attributes(tags=)` 不落 Langfuse，要在 span 上直接打 `langfuse.tags`；
     BatchSpanProcessor 攒批发送，脚本退出前不 flush 会丢尾巴
  6. 子集执行：`--category rag` 按分类前缀跑专项；`--golden` 只跑评审签字
     （metadata.golden=true）的黄金集——门禁用
- 练的知识点：LLM-as-judge 锚点设计 / 机械断言 vs 软性评分分工 /
  评测资产的平台化管理（git vs 平台的取舍）/ 跑批并发与 API 限流的平衡
- 验收标准：
  - [x] baseline-v1 跑出报告（16/22=73%，裁判相关性 4.9 / 准确性 4.3；5 条失败已归因：
        断言括号 bug 已修、测试数据失效已改、表述变体已放宽）
  - [x] Langfuse「Datasets → cs-agent-eval → Runs」能看到 baseline-v1 及逐条分数
  - [ ] 用例集 38 条已达标；转人工续跑用例待补（runner 的 interrupt 续跑逻辑见 E2-3）

## E3 Ragas 专项（RAG 质量）

- 目标：给检索问答质量定量——答的忠不忠于检索结果、检索本身准不准
- 状态（2026-09-26 定稿，采纳丽姐意见）：**不搞独立评测脚本**——RAG 用例并进
  `cs-agent-eval`（单轮 = turns 只有一条，带 ground_truth），ragas 指标作为第三个
  评分器挂进 `langfuse_runner.py` 的同一次跑批：**一次执行，多维打分**——
  跑 Agent 是唯一大成本，不为评分把同一批题跑两遍（LLM 要钱）
- 结构（已就位）：
  - 数据集 38 条 = 功能 17 + RAG 21（16 库内带 ground_truth + 5 库外测拒答），
    `ragas_eval.py` 已删除，逻辑并入 runner
  - ragas 评分器按 `metadata.category=rag-*` 门控只打 RAG 题：
    faithfulness / answer_relevancy / answer_correctness / context_recall /
    context_precision（后两个有 ground_truth 才能算，咱恰好都有）
  - 拒答红线并入机械校验：库外题须出现拒答标记，库内题拒答即误杀
  - 踩坑记录：ragas 0.4.3 硬 import `langchain-community<0.4` 的 vertexai 模块，
    版本已在 pyproject 钉死；ragas 裁判走 `llm_factory` + DashScope 兼容端点
- 动手（丽姐执行）：
  1. 全量跑批：`PYTHONIOENCODING=utf-8 .venv/Scripts/python -m customer_service.analytics.langfuse_runner --run-name baseline-v2`
  2. 做一次检索参数对照实验（如 rerank 阈值 0.30 vs 0.40），用数字说明哪个更好
- 练的知识点：一次执行多维打分的成本结构 / ragas 指标适用前提 /
  拒答判定的机械口径 / 检索参数 A/B 对照
- 验收标准：
  - [ ] baseline-v2 报告：功能通过率 + 裁判均分 + 五个 ragas 指标 + 两条红线结论
  - [ ] 编造率 < 5%、误杀率 < 10%
  - [ ] 一次检索参数对照实验，结论写报告

## E4 回归门禁（前三步站稳后再做）

- 目标：评测从"一次性体检"变成"持续守门"
- 动手：
  - 约定：改 prompt / 改检索参数 / 换模型，**必跑 E2 + E3**，任一红线破了当回归处理
    （ANTIHALLUCINATION.md 第五节原话）
  - Langfuse 上定期抽查真实对话 trace，人工标注好/坏，回灌进 E2/E3 数据集
    （线上样本是最值钱的测试数据）
  - 可选：跑批脚本挂定时任务，报告输出到固定目录
- 验收标准：
  - [ ] 有一次真实的"改动 → 跑评测 → 拿数字做决策"完整记录

## 顺序建议与排雷

```
E1 观测接线（半天，纯接线零风险，先让一切可见）
 → E2-1 跑基线（1 小时，先有数）
  → E3 造数据集 + 基线（1-2 天，最重的活在造数据）
   → E2 扩用例 / E4 门禁（滚动做）
```

- 真调 LLM 的脚本都自己跑（现在烧的是自己的百炼额度，注意量大时先看一眼价格）
- Windows 照旧：`PYTHONIOENCODING=utf-8` + `.venv/Scripts/python`
- ragas 指标别贪多，先把 faithfulness 和编造率这两个和反幻觉直接挂钩的跑稳
- Langfuse 是 v3，网上很多 v2 的教程（`@observe` 装饰器那套）SDK 用法不一样，
  本项目走的是 OTel 通道，查资料认准 "Langfuse v3 + OpenTelemetry"
