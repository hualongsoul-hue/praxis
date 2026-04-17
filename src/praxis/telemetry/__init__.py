"""praxis.telemetry — 遥测系统（S2）：结构化日志、指标采集、分布式追踪、审计日志。"""

from praxis.telemetry.audit import configure_audit, query_audit, record_audit
from praxis.telemetry.logger import StructuredLogger, configure_logging, get_logger
from praxis.telemetry.metrics import (
    MetricsCollector,
    configure_metrics,
    emit_metric,
    export_prometheus,
)
from praxis.telemetry.tracing import configure_tracing, start_span

__all__ = [
    "MetricsCollector",
    "StructuredLogger",
    "configure_audit",
    "configure_logging",
    "configure_metrics",
    "configure_tracing",
    "emit_metric",
    "export_prometheus",
    "get_logger",
    "query_audit",
    "record_audit",
    "start_span",
]
