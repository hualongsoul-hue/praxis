"""praxis.telemetry — 遥测系统（S2）：结构化日志、指标采集、分布式追踪、审计日志。"""

from typing import Any

from praxis.telemetry.audit import configure_audit, query_audit, record_audit
from praxis.telemetry.logger import StructuredLogger, configure_logging, get_logger
from praxis.telemetry.metrics import (
    MetricsCollector,
    configure_metrics,
    emit_metric,
    export_prometheus,
)
from praxis.telemetry.tracing import configure_tracing, start_span


def configure_telemetry(config: Any, store: Any = None) -> None:
    """按配置统一初始化日志/指标/追踪/审计（进程级）。

    应在应用启动时调用一次。create_agent_session 在传入 telemetry_config
    时也会调用本函数。

    Args:
        config: 遥测配置。
        store: 审计持久化存储；None 时不改变审计存储，仅应用其余配置。
    """
    configure_logging(config)
    configure_metrics(config)
    configure_tracing(config)
    if store is not None:
        configure_audit(store, enabled=config.audit_enabled)


__all__ = [
    "MetricsCollector",
    "StructuredLogger",
    "configure_audit",
    "configure_logging",
    "configure_metrics",
    "configure_telemetry",
    "configure_tracing",
    "emit_metric",
    "export_prometheus",
    "get_logger",
    "query_audit",
    "record_audit",
    "start_span",
]
