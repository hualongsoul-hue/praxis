"""结构化日志框架。

每条日志包含：时间戳、级别、子系统名、会话 ID、轮次号、消息、结构化字段。
支持 JSON（机器可读）和 Text（人类可读）两种输出格式。
日志级别可按子系统独立配置。
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from praxis.config.subsystems import TelemetryConfig

STANDARD_ATTRS = frozenset({
    "args", "created", "exc_info", "exc_text", "filename", "funcName",
    "levelname", "levelno", "lineno", "message", "module", "msecs", "msg",
    "name", "pathname", "process", "processName", "relativeCreated",
    "stack_info", "thread", "threadName", "taskName",
    "subsystem", "session_id", "turn",
})


class JsonFormatter(logging.Formatter):
    """JSON 格式日志格式化器。"""

    def format(self, record: logging.LogRecord) -> str:
        data: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "subsystem": getattr(record, "subsystem", ""),
            "message": record.getMessage(),
        }
        for ctx_key in ("session_id", "turn"):
            val = getattr(record, ctx_key, None)
            if val is not None:
                data[ctx_key] = val
        for key, val in record.__dict__.items():
            if key not in STANDARD_ATTRS and not key.startswith("_"):
                data[key] = val
        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            data["exception"] = record.exc_text
        return json.dumps(data, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    """人类可读格式日志格式化器。"""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s [%(levelname)-8s] %(subsystem)-14s │ %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "subsystem"):
            record.subsystem = ""  # type: ignore[attr-defined]
        return super().format(record)


class StructuredLogger:
    """结构化日志器，支持绑定持久上下文字段。

    通过 ``bind()`` 方法创建携带额外上下文（如 session_id、turn）的派生实例。
    """

    def __init__(self, name: str, logger: logging.Logger) -> None:
        self._name = name
        self._logger = logger
        self._context: dict[str, Any] = {"subsystem": name}

    def bind(self, **kwargs: Any) -> "StructuredLogger":
        """创建绑定额外上下文字段的新 Logger 实例。"""
        new = StructuredLogger(self._name, self._logger)
        new._context = {**self._context, **kwargs}
        return new

    def debug(self, msg: str, **kwargs: Any) -> None:
        self._log(logging.DEBUG, msg, **kwargs)

    def info(self, msg: str, **kwargs: Any) -> None:
        self._log(logging.INFO, msg, **kwargs)

    def warning(self, msg: str, **kwargs: Any) -> None:
        self._log(logging.WARNING, msg, **kwargs)

    def error(self, msg: str, **kwargs: Any) -> None:
        self._log(logging.ERROR, msg, **kwargs)

    def critical(self, msg: str, **kwargs: Any) -> None:
        self._log(logging.CRITICAL, msg, **kwargs)

    def _log(self, level: int, msg: str, **kwargs: Any) -> None:
        if not self._logger.isEnabledFor(level):
            return
        extra = {**self._context, **kwargs}
        self._logger.log(level, msg, extra=extra)


initialized = False


def configure_logging(config: TelemetryConfig) -> None:
    """根据配置初始化日志系统。可重复调用以更新配置。"""
    global initialized

    root = logging.getLogger("praxis")
    root.handlers.clear()
    root.propagate = False

    handler = logging.StreamHandler(sys.stderr)
    if config.log_format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(TextFormatter())
    root.addHandler(handler)
    root.setLevel(getattr(logging, config.log_level.upper(), logging.INFO))

    for subsystem, level_str in config.log_levels.items():
        sub_logger = logging.getLogger(f"praxis.{subsystem}")
        sub_logger.setLevel(
            getattr(logging, level_str.upper(), logging.INFO)
        )

    initialized = True


def get_logger(name: str) -> StructuredLogger:
    """获取指定子系统的结构化日志器。

    首次调用时自动用默认配置初始化日志系统。
    """
    global initialized
    if not initialized:
        configure_logging(TelemetryConfig())

    logger = logging.getLogger(f"praxis.{name}")
    return StructuredLogger(name, logger)
