import os

# 硬关 LangSmith 追踪：本机连 api.smith.langchain.com 超时，后台批量上传
# 失败会在 stderr 刷 "Failed to multipart ingest runs" 报错（仅噪音，不影响运行）。
# 注意必须是直接赋值而非 setdefault——终端会话里若已 export 过 true 会赢。
# 想重新打开追踪：删掉这两行。
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ.pop("LANGCHAIN_API_KEY", None)  # 顺手摘掉 key，双保险

from customer_service.tracing import init_tracing

init_tracing()  # OTel→Langfuse 自动埋点；未设 OTEL_EXPORTER_OTLP_ENDPOINT 时静默跳过

from customer_service.graph import build_graph

__all__ = ["build_graph"]
