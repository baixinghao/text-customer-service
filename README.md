# text-customer-service

智能文本客服 · 长期实验项目（LangChain / LangGraph 1.x）

## 架构

父图只路由，专家全是子图（`create_agent` 产物，直接挂为父图节点）。
首轮 router 做 LLM 意图识别 → `Command(goto)` 分流；专家间用 handoff 工具
`Command(goto, graph=Command.PARENT)` 一跳直达，不回主图中转。
父子图共享同一套 `State`：`messages` 黑板互通，`active_agent` 记录当前接待者，
后续轮次直达专家，不重复分流。

```
main.py                        # CLI 入口（InMemorySaver + thread_id 多轮记忆）
api.py                         # 【骨架】FastAPI 对外服务（P7）
docs/IMPLEMENTATION_PLAN.md    # 分阶段实现计划（练习主线，先读这个）
customer_service/
├── graph.py                   # 父图：router 分流 + 装配接线
├── state.py                   # 全局 State（MessagesState + active_agent）
├── llm.py                     # 唯一 LLM 入口（.env 可换模型）
├── prompts.py                 # 全部系统提示词
├── agents/                    # 专家子图，一个专家一个文件
│   ├── presale.py             # 售前：商品咨询
│   ├── aftersale.py           # 售后：退换货政策、退款
│   ├── order.py               # 订单：查单、查物流
│   ├── complaint.py           # 【骨架 P4】投诉/情绪安抚专家
│   └── human.py               # 【骨架 P3】转人工 interrupt 节点
├── tools/
│   ├── handoff.py             # 转接工具工厂（Command 跨图跳转）
│   ├── product_tools.py       # 【RAG 替换点】商品检索
│   ├── aftersale_tools.py     # 【RAG 替换点】售后政策问答
│   ├── order_tools.py         # 订单 Mock 数据
│   ├── ticket_tools.py        # 【骨架 P4】工单创建/查询
│   └── crm_tools.py           # 【骨架 P4 可选】客户标签/画像
├── rag/                       # 【骨架 P1】LlamaIndex 知识库（存 PG/pgvector）
│   ├── ingest.py              # 语料 → 向量索引 → PG
│   ├── retriever.py           # 索引 → LangChain 工具（换芯不换线）
│   └── knowledge/             # 语料：faq / refund_policy / products
├── memory/                    # 【骨架 P2】PostgresSaver 持久化 + 会话摘要
├── guardrails/                # 【骨架 P5】输入/输出护栏
└── analytics/                 # 【骨架 P6】离线质检评估
```

标注【骨架】的文件只有签名和注释（函数体 `raise NotImplementedError`），
是刻意留的练习空位，填法见 `docs/IMPLEMENTATION_PLAN.md`。

## 跑起来

```bash
uv venv && uv sync              # 基础环境
docker compose up -d            # 起 PG(ParadeDB) + Redis，等 docker compose ps 显示 healthy
cp .env.example customer_service/.env                 # 填入 DASHSCOPE_API_KEY
PYTHONIOENCODING=utf-8 .venv/Scripts/python db/apply_schema.py   # 建业务表（幂等）
PYTHONIOENCODING=utf-8 .venv/Scripts/python main.py   # Git Bash
```

注意：代码实际加载的是 **`customer_service/.env`**（不是根目录 `.env`）；
连接信息统一走 `DATABASE_URL` 一个变量，端口 5433 是 compose 的宿主机映射端口。
组件容器的说明见 `docker-compose.yml`，端到端共享流程见 `docs/SHARING.md`。

Mock 账号：手机号 `13800001234`（订单 DD1001 键盘 / DD1002 鼠标）、`13900005678`（DD1003 显示器）。

## Web UI（Agent Chat UI）

前端是官方 [Agent Chat UI](https://docs.langchain.com/oss/python/langgraph/ui)（Vite 版），
只留了 `apps/web`，通过环境变量对接本地 `langgraph dev` 服务：

```bash
# 终端 1：起 LangGraph 本地服务（加载根目录 langgraph.json 里的 customer_service 图）
.venv/Scripts/langgraph dev --no-browser

# 终端 2：起前端（首次先 cd ui && pnpm install）
cd ui && pnpm dev               # http://localhost:5173
```

连接配置在 `ui/apps/web/.env.local`（`VITE_API_URL` / `VITE_ASSISTANT_ID`），
改端口或图名只动这两个值。也可直接用在线版 https://agentchat.vercel.app
填入 `http://localhost:2024` + 图名 `customer_service`。

注意：Windows 下 `langgraph dev` 依赖 `colorama`（已进 dev 依赖组）；
机器上没有全局 pnpm 时，用 `corepack pnpm` 代替。

可选依赖按阶段装：`uv sync --extra rag` / `--extra memory` / `--extra api`，
未安装就 import 对应骨架包会报 ModuleNotFoundError，属预期。

## 共享 / 协作

代码走 GitHub 私有仓库，**PG + Redis 走 `docker-compose.yml`**，Python 与前端运行时留在本地
（保留热重载与 `langgraph dev` Studio）——完整理由、朋友的四步上手、验收命令和常见坑见
**`docs/SHARING.md`**。

推仓库前务必先读该文第 1 节：`customer_service/.env` 里有真实 Key，`.gitignore` 已覆盖
`**/.env`，但仍需先轮换 Key 并用 `git status --short` 逐个确认暂存内容。

## 路线图（详见 docs/IMPLEMENTATION_PLAN.md）

- [ ] P1 RAG：LlamaIndex 替换 `search_product` 与售后政策静态文本（可选 LightRAG 对照）
- [x] P2 持久化：PostgresSaver 替换 InMemorySaver；可选会话摘要
- [x] P3 转人工：human-in-the-loop（interrupt + resume）
- [x] P4 工单 + 投诉/情绪专家（工单走 PG 真表 tickets + ticket_events；crm 画像未做）
- [x] P5 输入/输出护栏（注入拦截、违规承诺审查）
- [ ] P6 离线质检 + LangSmith 追踪 / `langgraph dev` Studio 看图
- [x] P7 FastAPI 对外服务
- [ ] P8 Redis：会话缓存 / 限流 / 任务队列（容器已由 compose 备好，代码未接线）
- [ ] 远期：deepagents 重构专家、router 置信度兜底、真实订单库
