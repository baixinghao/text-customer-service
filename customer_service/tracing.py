"""OTel 观测接线（E1）：OpenInference 自动埋点 LangChain/LangGraph。

链路：应用 --OTLP HTTP--> 本地 OTel Collector(:4318) --转发--> Langfuse(:3000)。
应用侧不需要 Langfuse Key——Basic Auth 写在 Collector 的 exporter 里
（observability/otel-collector-config.yaml），应用只管往本地发 span。

开关：`OTEL_EXPORTER_OTLP_ENDPOINT` 存在即启用，否则静默跳过；
没装 obs extra（`uv sync --extra obs`）同样跳过——观测是增强项，不是运行前提。
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv

_initialized = False


def init_tracing() -> bool:
    """初始化 OTel + OpenInference 自动埋点，返回是否真正启用。

    必须在 import 图之前调用（见 customer_service/__init__.py）。初始化一次后，
    所有 LLM / 工具 / 链调用自动上报 span，业务代码零改动。
    """
    global _initialized
    if _initialized:
        return True
    # 本模块先于 llm/db 被 import（__init__.py 里在图之前初始化），
    # customer_service/.env 此刻还没人加载，必须自己 load——与 db.py 同款写法；
    # override=False：终端里 export 的真实环境变量优先
    load_dotenv(Path(__file__).parent / ".env")
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip().rstrip("/")
    if not endpoint:
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from openinference.instrumentation.langchain import LangChainInstrumentor
    except ImportError:
        return False

    provider = TracerProvider(
        resource=Resource.create({"service.name": "text-customer-service"})
    )
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces"))
    )
    trace.set_tracer_provider(provider)
    LangChainInstrumentor().instrument(tracer_provider=provider)
    _initialized = True
    print(f"[tracing] OTel → {endpoint} 已启用（Collector 转发 Langfuse）")
    return True


@contextmanager
def bind_session(thread_id: str, user_id: str = "") -> Iterator[None]:
    """把一次 invoke 归到 Langfuse 的 session 维度（同一会话的多轮聚合查看）。

    用法：
        with bind_session(thread_id):
            graph.invoke(state, config)
    """
    if not _initialized:
        yield
        return
    from openinference.instrumentation import using_attributes

    with using_attributes(session_id=thread_id, user_id=user_id):
        yield


@contextmanager
def eval_trace(session_id: str, tags: list[str]) -> Iterator[None]:
    """评测执行的 trace 标签：Langfuse Tracing 视图按 eval/run/分类过滤靠它。

    关键细节：OpenInference 的 using_attributes(tags=...) 走 Collector 不会落成
    Langfuse trace tags（实测），必须在 span 上直接打 `langfuse.tags` /
    `langfuse.session.id` 属性。一条用例包一个 span，多轮全部挂进同一条 trace。
    """
    if not _initialized:
        yield
        return
    from opentelemetry import trace

    tracer = trace.get_tracer("cs.eval")
    with tracer.start_as_current_span("eval-case") as span:
        span.set_attribute("langfuse.session.id", session_id)
        span.set_attribute("langfuse.tags", tags)
        yield
