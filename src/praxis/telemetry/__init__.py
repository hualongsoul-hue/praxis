"""实例化遥测组件；仅 CLI 可选择配置进程级日志和导出器。"""

from praxis.config.schemas import TelemetryConfig
from praxis.telemetry.audit import AuditService, NullAuditSink
from praxis.telemetry.logger import StructuredLogger, configure_logging, get_logger
from praxis.telemetry.metrics import (
    MetricsCollector,
    MetricsExporter,
    configure_metrics,
    emit_metric,
    export_prometheus,
)
from praxis.telemetry.tracing import (
    TracingLifecycle,
    configure_tracing,
    configure_tracing_lifecycle,
    start_span,
)


def configure_cli_telemetry(config: TelemetryConfig) -> TracingLifecycle:
    """Configure CLI telemetry and return the tracing resource that must be closed."""
    configure_logging(config)
    configure_metrics(config)
    return configure_tracing_lifecycle(config)


__all__ = [
    "AuditService",
    "MetricsCollector",
    "MetricsExporter",
    "NullAuditSink",
    "StructuredLogger",
    "configure_cli_telemetry",
    "configure_logging",
    "configure_metrics",
    "configure_tracing",
    "emit_metric",
    "export_prometheus",
    "get_logger",
    "start_span",
]
