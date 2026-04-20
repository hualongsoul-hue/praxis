"""分布式追踪。

基于 OpenTelemetry SDK，支持 Agent → 子代理 → 工具的完整调用链路追踪。
每个 Span 携带：组件名、操作类型、耗时、状态码。
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
from opentelemetry.trace import Span

from praxis.config.schemas import TelemetryConfig

tracer: trace.Tracer | None = None


def configure_tracing(config: TelemetryConfig) -> None:
    """根据配置初始化分布式追踪。"""
    global tracer

    provider = TracerProvider()

    if config.tracing_enabled:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer("praxis")


def get_tracer() -> trace.Tracer:
    global tracer
    if tracer is None:
        configure_tracing(TelemetryConfig(tracing_export="console"))
    return tracer  # type: ignore[return-value]


def start_span(
    name: str,
    parent: Span | None = None,
    component: str | None = None,
    operation: str | None = None,
) -> Span:
    """创建追踪 Span。

    Args:
        name: Span 名称。
        parent: 父 Span，用于建立调用链。
        component: 组件标识。
        operation: 操作类型。

    Returns:
        新创建的 Span。调用方负责通过 ``span.end()`` 结束。
    """
    current_tracer = get_tracer()

    context = None
    if parent is not None:
        context = trace.set_span_in_context(parent)

    span = current_tracer.start_span(name, context=context)

    if component:
        span.set_attribute("praxis.component", component)
    if operation:
        span.set_attribute("praxis.operation", operation)

    return span
