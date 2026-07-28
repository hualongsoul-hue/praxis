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
from praxis.telemetry.tracing import configure_tracing, start_span


def configure_cli_telemetry(config: TelemetryConfig) -> None:
    """由 CLI 显式配置进程级日志/指标/追踪；SDK Runtime 不调用。"""
    configure_logging(config)
    configure_metrics(config)
    configure_tracing(config)


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
