# 项目约定（长期实验项目，随项目演进同步更新本文件）

## 技术栈红线
- LangChain / LangGraph **只用 1.x 新语法**，禁止抄 0.x 的 `AgentExecutor`、`initialize_agent`、旧 `Runtime` 等老 API
- Agent 一律 `from langchain.agents import create_agent`
- 工具一律 `from langchain.tools import tool, ToolRuntime`
- 图内跳转优先 `Command(goto=..., update=...)`，让边只做固定路由，逻辑收进节点函数
- **存储一律走 PostgreSQL**：向量库 / 文本库（docstore）/ 索引库（index_store）用
  llama-index PG 三件套（需 pgvector 扩展），会话 checkpointer 用 `PostgresSaver`；
  禁止本地文件落盘 / SQLite，连接信息只从 `.env` 的 `PG_*` 读

## 环境
- uv 管理依赖；虚拟环境在 `.venv`
- Windows Git Bash 下跑 Python 必须 `PYTHONIOENCODING=utf-8`，解释器用 `.venv/Scripts/python`
- 涉及真实 LLM 调用的脚本，未经白哥同意不得替他运行（烧他的 API 额度）；只做 `py_compile` 和不调模型的编译级冒烟测试

## 架构
- 父图只做路由和装配，业务全在 `agents/` 各专家子图里
- 父子图共享同一套 `State`（见 `state.py`），消息历史互通，靠 `active_agent` 记录当前接待者
- Mock 数据全部集中在 `tools/` 各文件顶部的常量里，将来接真库 / RAG 时整段替换
- `rag/`、`memory/`、`guardrails/`、`analytics/`、`api.py` 是练习骨架包：只有签名和注释，
  函数体一律 `raise NotImplementedError("Px 练习：...")`，是刻意留的空位，
  由白哥按 `docs/IMPLEMENTATION_PLAN.md` 逐阶段填，不要替他实现
- 骨架包引用了可选依赖（llama-index、fastapi 等），可选依赖一律走
  `pyproject.toml` 的 `[project.optional-dependencies]`，不污染基础环境；
  没装对应 extra 时 import 骨架包报 ModuleNotFoundError 属预期

## Web UI
- `ui/` 是官方 Agent Chat UI（Vite 版，已裁掉自带 TS 示例 agents），非练习区，可以直接改
- 对接方式：`ui/apps/web/.env.local` 的 `VITE_API_URL` + `VITE_ASSISTANT_ID`
  指向根目录 `langgraph.json` 注册的图（`customer_service`）
- 起服务：`.venv/Scripts/langgraph dev --no-browser --port 80`（**必须 80**：Langfuse
  评测 webhook 只放行 80/443，挂在 langgraph dev 的自定义路由上）+ `cd ui && pnpm dev`（:5173）
- Windows 坑：`langgraph dev` 需要 `colorama`（已在 dev 组）；无全局 pnpm 时用 `corepack pnpm`
