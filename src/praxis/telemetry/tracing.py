"""分布式追踪。

基于 OpenTelemetry SDK，支持 Agent → 子代理 → 工具的完整调用链路追踪。
每个 Span 携带：组件名、操作类型、耗时、状态码。
"""

from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
from opentelemetry.trace import Span

from praxis.config.schemas import TelemetryConfig
from praxis.telemetry.logger import get_logger

log = get_logger("telemetry.tracing")

tracer: trace.Tracer | None = None


def _otlp_processor(endpoint: str | None) -> Any:
    """构造 OTLP BatchSpanProcessor；导出器未安装时返回 None 并降级。"""
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
    except Exception:
        log.warning(
            "未安装 OTLP 导出器（opentelemetry-exporter-otlp），追踪降级为 console。"
            "生产环境请安装该依赖。",
        )
        return None
    exporter = OTLPSpanExporter(endpoint=endpoint) if endpoint else OTLPSpanExporter()
    return BatchSpanProcessor(exporter)


def configure_tracing(config: TelemetryConfig) -> None:
    """根据配置初始化分布式追踪。

    tracing_export: console（默认）/ otlp（生产，需 otlp 导出器）/ none（关闭）。
    """
    global tracer

    provider = TracerProvider()

    if config.tracing_enabled and config.tracing_export != "none":
        processor = None
        if config.tracing_export == "otlp":
            processor = _otlp_processor(config.otlp_endpoint)
        if processor is None:  # console 或 otlp 降级
            processor = SimpleSpanProcessor(ConsoleSpanExporter())
        provider.add_span_processor(processor)

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
