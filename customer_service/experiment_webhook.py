"""Langfuse Remote Experiment 的 webhook 入口（挂进 langgraph dev 的自定义路由）。

定位：只收触发、异步执行，真正的跑批逻辑全在 analytics/langfuse_runner.py
（run_case / 两个 evaluator / run_experiment 回写），本文件一行不重复实现。

接线（langgraph.json）：
    "http": { "app": "./customer_service/experiment_webhook.py:app" }
langgraph dev 会把它自己的 /threads /runs 等路由挂到本 app 上，前端不受影响。

Langfuse 侧契约（官方文档）：
- Dataset → Run experiment → via Webhook → Configure 里填本端点 URL
- 触发时 POST JSON：{datasetId, datasetName, config}（字段名做防御性兼容）
- 本端点必须快速返回 2xx，跑批放后台异步——否则会判定超时
- ⚠️ Langfuse 只放行 80/443 端口的 URL（SSRF 防护），所以 langgraph dev
  要用 --port 80 起；且 Langfuse 在容器里，回宿主机只能走 host.docker.internal

冒烟（dry_run，不烧 LLM 额度）：
    curl -X POST http://localhost/webhook/langfuse-experiment \
         -H "Content-Type: application/json" \
         -d '{"datasetName":"cs-agent-eval","config":{"dry_run":true}}'
"""

from __future__ import annotations

import datetime
import os
import subprocess
import sys
import threading
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="customer-service-webhooks")  # 无 lifespan：不建驻留图，省内存

# 可选令牌：在 Langfuse 的 Default config 里配 {"token": "..."}，环境变量也配上才校验
_WEBHOOK_TOKEN = os.getenv("LANGFUSE_WEBHOOK_TOKEN", "")

_PROJECT_ROOT = Path(__file__).parent.parent
_RUN_LOG_DIR = _PROJECT_ROOT / "logs" / "eval"


def _spawn_experiment(dataset_name: str, run_name: str, only: str | None,
                      log_path: Path) -> None:
    """拉独立子进程跑批。本函数全是阻塞调用，只能在【裸线程】里跑。

    层层都有讲究：
    - 不能直接进 async 端点：blockbuster 检测器逮 os.mkdir/Popen（实测踩过）
    - 不能用 anyio 线程池（BackgroundTasks）：contextvar 跟着进线程，照样被逮
    - threading.Thread 裸线程不传播 contextvar，检测器管不着；
      子进程则彻底跳出 ASGI 进程，dev server 热重载不杀跑批
    """
    cmd = [sys.executable, "-m", "customer_service.analytics.langfuse_runner",
           "--dataset", dataset_name, "--run-name", run_name]
    if only:
        cmd += ["--only", only]
    _RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    # DEVNULL stdin 防子进程等输入；CREATE_NEW_PROCESS_GROUP 让 dev server
    # Ctrl+C 的信号不牵连跑批（Windows）
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    with open(log_path, "a", encoding="utf-8") as log_file:
        subprocess.Popen(cmd, cwd=_PROJECT_ROOT, env=env,
                         stdin=subprocess.DEVNULL,
                         stdout=log_file, stderr=subprocess.STDOUT,
                         creationflags=creationflags)


@app.post("/webhook/langfuse-experiment")
async def langfuse_experiment(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"status": "rejected", "reason": "invalid JSON"}, 400)

    config = payload.get("config") or {}
    if _WEBHOOK_TOKEN and config.get("token") != _WEBHOOK_TOKEN:
        return JSONResponse({"status": "rejected", "reason": "bad token"}, 403)

    # 字段名防御性兼容：文档写 datasetName/datasetId，变体也收
    dataset_name = (payload.get("datasetName") or payload.get("dataset_name")
                    or payload.get("name") or "cs-agent-eval")
    run_name = config.get("run_name") or (
        f"eval-ui-{datetime.datetime.now():%Y%m%d-%H%M%S}")

    if config.get("dry_run"):
        # 冒烟档：只回执不跑批，验证 Langfuse → 本端点链路通不通
        print(f"[webhook] dry_run 收到触发：dataset={dataset_name} "
              f"payload={payload}")
        return JSONResponse({"status": "accepted", "dry_run": True,
                             "dataset": dataset_name, "run_name": run_name})

    # 拼路径不碰文件系统（非阻塞）；mkdir + Popen 全在裸线程里做
    log_path = _RUN_LOG_DIR / f"{run_name}.log"
    threading.Thread(
        target=_spawn_experiment,
        args=(dataset_name, run_name, config.get("only"), log_path),
        daemon=True,
    ).start()
    print(f"[webhook] 已受理：dataset={dataset_name} run={run_name} "
          f"日志={log_path}")
    return JSONResponse({"status": "accepted", "dataset": dataset_name,
                         "run_name": run_name, "log": str(log_path)})
