"""复用宿主 Provider 的上下文级 OpenTelemetry 适配。"""

from contextvars import ContextVar, Token
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
from opentelemetry.trace import Span, Tracer

from praxis.config.schemas import TelemetryConfig
from praxis.telemetry.logger import get_logger

log = get_logger("telemetry.tracing")
current_tracer: ContextVar[Tracer | None] = ContextVar("praxis_tracer", default=None)


class TracingLifecycle:
    """Own and close a locally created tracing provider exactly once."""

    def __init__(
        self,
        tracer: Tracer,
        provider: TracerProvider | None,
        owns_provider: bool,
        context_token: Token[Tracer | None],
    ) -> None:
        self.tracer = tracer
        self.provider = provider
        self.owns_provider = owns_provider
        self.context_token = context_token
        self.closed = False

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            if self.provider is not None and self.owns_provider:
                self.provider.force_flush()
                self.provider.shutdown()
        finally:
            current_tracer.reset(self.context_token)


def configure_tracing_lifecycle(config: TelemetryConfig) -> TracingLifecycle:
    """Configure context-local tracing and return its explicit lifecycle."""
    if not config.tracing_enabled or config.tracing_export == "none":
        tracer = trace.get_tracer("praxis")
        context_token = current_tracer.set(tracer)
        return TracingLifecycle(tracer, None, False, context_token)

    provider = TracerProvider()
    if config.tracing_export == "otlp":
        provider.add_span_processor(otlp_processor(config.otlp_endpoint))
    else:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    tracer = provider.get_tracer("praxis")
    context_token = current_tracer.set(tracer)
    return TracingLifecycle(tracer, provider, True, context_token)


def otlp_processor(endpoint: str | None) -> Any:
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    except ModuleNotFoundError as exc:
        raise RuntimeError("OTLP 导出不可用，请安装 praxis[otlp]") from exc
    exporter = OTLPSpanExporter(endpoint=endpoint) if endpoint else OTLPSpanExporter()
    return BatchSpanProcessor(exporter)


def configure_tracing(config: TelemetryConfig) -> TracingLifecycle:
    """Create an explicitly owned tracing lifecycle for the current context."""
    return configure_tracing_lifecycle(config)


def get_tracer() -> Tracer:
    return current_tracer.get() or trace.get_tracer("praxis")


def start_span(
    name: str,
    parent: Span | None = None,
    component: str | None = None,
    operation: str | None = None,
) -> Span:
    context = trace.set_span_in_context(parent) if parent is not None else None
    span = get_tracer().start_span(name, context=context)
    if component:
        span.set_attribute("praxis.component", component)
    if operation:
        span.set_attribute("praxis.operation", operation)
    return span
