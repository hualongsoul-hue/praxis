"""praxis.gateway — 模型网关（S4）：基于 LiteLLM 的统一 LLM 接入。"""

from praxis.gateway.callbacks import TelemetryCallback, register_callbacks
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
    "register_callbacks",
    "summarize",
]
