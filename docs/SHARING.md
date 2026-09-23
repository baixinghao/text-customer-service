# 共享指南：把项目 + PG + Redis 交给朋友

> 面向两个人：**要推仓库的维护者**（第 0、1、5 节）和**拿到仓库的朋友**（第 2、3、4 节）。

## 0. 方案：代码走 GitHub，组件走 Docker

不是"二选一"，而是**分层**——代码和组件各自用最合适的交付方式：

| 层 | 交付方式 | 理由 |
|---|---|---|
| 项目代码 | GitHub 私有仓库 | 这是练习型项目，`rag/`、`analytics/` 里还有骨架等着填，朋友要读代码、改代码、跑 `langgraph dev` 看图 |
| PostgreSQL | `docker-compose.yml` | 有状态、版本敏感。**必须用 ParadeDB 镜像**，见下 |
| Redis | `docker-compose.yml` | 同上，且为 P8 提前备好 |
| Python / 前端运行时 | **不进容器**，本地 uv + pnpm | 保留热重载；`langgraph dev` 的 Studio 调试体验在容器里会打折 |

**为什么不选全 Docker**：全 docker 的唯一优势是"零环境配置"，但代价直接打在这个项目的核心用法上——改一行代码要重 build 镜像、Vite HMR 与 `uvicorn --reload` 在 Windows bind mount 下变慢、镜像体积几个 G。PG / Redis 这种"装起来烦、版本还敏感"的东西才值得容器化，代码不值得。

## 1. 🔴 推仓库前必做：密钥清理（唯一会出事故的地方）

`.env` 里有真实凭据：DeepSeek Key、DashScope Key、LangSmith Key、Tavily Key，以及 `DATABASE_URL` 里的明文密码。仓库还没初始化 git，所以**这一步没做就 `git init && git push` 会一次性全送出去**。

1. **轮换 Key**：上面几把 Key 已经在本机明文躺了很久，即使不入库也建议去各家控制台重新签发一遍，旧的全部吊销。
2. **`.gitignore` 已修好**（本次改动）：原来只有 `.env`，只挡根目录，**挡不住 `customer_service/.env` 和 `customer_service/.env.bak-*`**；现在用 `**/.env` / `**/.env.*` 覆盖所有子目录，并保留 `.env.example`。
3. **删掉含密钥的备份文件**：`customer_service/.env.bak-20260912`（历史 Key，没有留的必要）。
4. **推之前先验一次**：

   ```bash
   git init
   git add -A
   git status --short          # ← 逐个看，确认没有任何 .env / node_modules / .venv
   git ls-files | grep -i env  # 期望只输出 .env.example 和 ui/.env.example
   ```

   确认干净后再 commit + push。**先看 `git status` 再 commit，不要 `git add -A && git commit` 一气呵成。**
5. **建私有仓库**，把朋友加成 collaborator（Settings → Collaborators）。

## 2. 朋友上手：四步

### 前置
- Python **3.12+**、[uv](https://docs.astral.sh/uv/)、Node 20+（`corepack enable` 即可拿到 pnpm）、Docker Desktop

### ① 拿代码 + 建 Python 环境
```bash
git clone <私有仓库地址> && cd text-customer-service
uv sync                    # 建 .venv 并装基础依赖（含 dev 组）
```

### ② 起组件（PG + Redis）
```bash
docker compose up -d
docker compose ps          # 等 postgres 和 redis 都变成 healthy
```
首次启动会自动执行 `db/init.sql`，在库里装好 `vector`(pgvector) 与 `pg_search`(BM25) 扩展。

### ③ 建业务表 + 填 Key
```bash
# 建业务表：refund_applications / tickets / ticket_events（幂等，可重复跑）
PYTHONIOENCODING=utf-8 .venv/Scripts/python db/apply_schema.py

# 填 Key（Windows PowerShell 用 Copy-Item）
cp .env.example customer_service/.env
```
编辑 `customer_service/.env`，**只需填 `DASHSCOPE_API_KEY`**（用自己的百炼 Key，有免费额度）。
`DATABASE_URL` 已经和 compose 的默认值对齐，不用改。

> ⚠️ 项目代码实际加载的是 `customer_service/.env`，**不是根目录 `.env`**。

### ④ 起服务
```bash
# 终端 1：LangGraph 本地服务 → http://localhost:2024
.venv/Scripts/langgraph dev --no-browser

# 终端 2：前端 → http://localhost:5173
cd ui && corepack pnpm install && corepack pnpm dev
```

浏览器打开 <http://localhost:5173> 即可对话。

## 3. 验收：确认三件套真的通了

```bash
# ① PG 连得上、扩展齐、表已建
docker exec -it tcs-postgres psql -U postgres -d pg4mysql -c "SELECT extname, extversion FROM pg_extension ORDER BY 1;"
docker exec -it tcs-postgres psql -U postgres -d pg4mysql -c "\dt"

# ② Redis 活着
docker exec -it tcs-redis redis-cli ping        # 期望 PONG

# ③ 端到端：在 UI 里问「帮我查一下订单 DD1001」
#    期望路由到 order 专家并返回订单信息（Mock 数据）
```

Mock 账号：手机号 `13800001234`（订单 DD1001 键盘 / DD1002 鼠标）、`13900005678`（DD1003 显示器）。

## 4. 日常操作 & 常见坑

```bash
docker compose stop                # 暂停，保留数据
docker compose up -d               # 再起
docker compose down                # 删容器，保留数据卷
docker compose down -v             # ⚠️ 连数据卷一起删，PG 和 Redis 全清空
docker compose logs -f postgres    # 看 PG 日志
```

| 症状 | 原因 / 处理 |
|---|---|
| `pg_isready` 一直不 healthy | 宿主机 5433 被占用了。改 `.env` 的 `PG_PORT`，并同步改 `DATABASE_URL` 的端口 |
| Python 报 `DATABASE_URL 未配置` | 没建 `customer_service/.env`，或建到了根目录 |
| 模型调用 401 | `customer_service/.env` 里 `DASHSCOPE_API_KEY` 没填 |
| import `rag/` 报 `ModuleNotFoundError` | 属预期——可选依赖按阶段装：`uv sync --extra rag` / `--extra memory` / `--extra api` |
| BM25 相关导入失败 | 用了官方 `postgres` 镜像而不是 `paradedb/paradedb`，缺 `pg_search` |
| 想重建空库 | `docker compose down -v && docker compose up -d`，然后重跑 `db/apply_schema.py` |

## 5. 维护者备注

- **不确定的地方（未验证）**：`paradedb/paradedb:0.25.9` 这个 tag 是按本机 `pg_search 0.25.9` 锁的，但我这边沙箱连不上 Docker Hub 无法确认 tag 是否存在。起容器前先 `docker pull paradedb/paradedb:0.25.9`；若报 tag 不存在，就改用 `latest` 并在 `docker-compose.yml` 注释里记下实际版本。本机原实例是 ParadeDB（`pg_search 0.25.9` / `pgvector 0.8.4` / `postgis` / `pg_trgm` / `pg_cron` / `pg_ivm`），镜像预装清单见[官方文档](https://www.paradedb.com/docs/operate/deploy/third-party-extensions)。
- **`pg_cron` 用不了**：ParadeDB 镜像只在默认 `postgres` 库预配置 `pg_cron`，业务库 `pg4mysql` 启用会报错。项目目前没用到，`db/init.sql` 里刻意没启用。
- **库名 `pg4mysql` 不是笔误**：早期三个实例合并成一个时留下的名字，改它要同步改 `DATABASE_URL`、compose 变量和所有已有数据，不值得动。
- **Redis 当前是"备而不用"**：代码里还没有任何 `redis` 引用，`REDIS_URL` 已在 `.env.example` 里预留。P8 真正接线时再写代码，compose 侧不用再改。
- **`customer_service/.env` 与容器初始化没有关系**：容器的 `POSTGRES_*` 变量来自 compose（默认值或根目录 `.env`），朋友那边只要 `DATABASE_URL` 对得上就行。这意味着**密钥和 DSN 是分离的**：`.env.example` 里的 `DATABASE_URL` 是给朋友用的本地默认值（`localhost:5433`），你本机要连自己的实例时按需覆盖。
- **一个已知隐患（未改代码）**：`customer_service/llm.py` 用的是裸 `load_dotenv()`（读当前工作目录），而其他模块都定向读 `customer_service/.env`。目前能工作，是因为 `customer_service/__init__.py → graph → … → db.py` 的导入链先把 `.env` 灌进了 `os.environ`。但导入顺序一变就可能悄悄失效（`DASHSCOPE_API_KEY` 变 None）。要根治就把 `llm.py` 也改成 `load_dotenv(Path(__file__).parent / ".env")`。
- **本次新增文件**：`docker-compose.yml`、`db/init.sql`、`docs/SHARING.md`；改动 `.gitignore`、`.env.example`、`README.md`。
