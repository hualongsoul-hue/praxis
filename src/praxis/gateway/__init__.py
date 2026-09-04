"""praxis.gateway — 模型网关（S4）：基于 LiteLLM 的统一 LLM 接入。

门面使用按需导入，Protocol 调用辅助与非模型组件不会提前初始化 LiteLLM。
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from praxis.gateway.callbacks import TelemetryCallback
    from praxis.gateway.chat import chat, chat_stream
    from praxis.gateway.metering import (
        check_budget,
        completion_cost,
        get_max_tokens,
        get_token_count,
        record_usage,
    )
    from praxis.gateway.resilience import map_litellm_exception
    from praxis.gateway.router import GatewayRouter
    from praxis.gateway.tasks import judge, summarize

GATEWAY_EXPORTS = {
    "GatewayRouter": ("praxis.gateway.router", "GatewayRouter"),
    "TelemetryCallback": ("praxis.gateway.callbacks", "TelemetryCallback"),
    "chat": ("praxis.gateway.chat", "chat"),
    "chat_stream": ("praxis.gateway.chat", "chat_stream"),
    "check_budget": ("praxis.gateway.metering", "check_budget"),
    "completion_cost": ("praxis.gateway.metering", "completion_cost"),
    "get_max_tokens": ("praxis.gateway.metering", "get_max_tokens"),
    "get_token_count": ("praxis.gateway.metering", "get_token_count"),
    "judge": ("praxis.gateway.tasks", "judge"),
    "map_litellm_exception": (
        "praxis.gateway.resilience",
        "map_litellm_exception",
    ),
    "record_usage": ("praxis.gateway.metering", "record_usage"),
    "summarize": ("praxis.gateway.tasks", "summarize"),
}


def __getattr__(name: str) -> object:
    """Resolve one public gateway adapter or helper on demand."""
    target = GATEWAY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    return getattr(import_module(module_name), attribute_name)

__all__ = [
    "GatewayRouter",
    "TelemetryCallback",
    "chat",
    "chat_stream",
    "check_budget",
    "completion_cost",
    "get_max_tokens",
    "get_token_count",
    "judge",
    "map_litellm_exception",
    "record_usage",
    "summarize",
]
