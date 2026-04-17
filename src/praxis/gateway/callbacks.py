"""可观测性回调。

通过 LiteLLM callback 机制桥接 S2 遥测系统，
自动发射 llm_latency / llm_tokens_* / llm_cost / llm_error 指标。
"""

from typing import Any

import litellm
from litellm.integrations.custom_logger import CustomLogger

from praxis.telemetry.metrics import emit_metric


class TelemetryCallback(CustomLogger):
    """LiteLLM 自定义回调，将调用指标桥接到 S2。"""

    def log_success_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """成功调用时发射指标。"""
        model = kwargs.get("model", "unknown")
        tags = {"model": model}

        if hasattr(response_obj, "usage") and response_obj.usage:
            usage = response_obj.usage
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            emit_metric("llm_tokens_input", float(prompt_tokens), tags, "counter")
            emit_metric("llm_tokens_output", float(completion_tokens), tags, "counter")

        try:
            cost = litellm.completion_cost(completion_response=response_obj)
            emit_metric("llm_cost", cost, tags, "counter")
        except Exception:
            pass

        if start_time and end_time:
            latency_ms = (end_time - start_time).total_seconds() * 1000
            emit_metric("llm_latency", latency_ms, tags, "histogram")

        emit_metric("llm_requests_total", 1.0, {**tags, "status": "success"}, "counter")

    def log_failure_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """失败调用时发射错误指标。"""
        model = kwargs.get("model", "unknown")
        error_type = type(kwargs.get("exception", Exception())).__name__
        tags = {"model": model, "status": "error", "error_type": error_type}
        emit_metric("llm_requests_total", 1.0, tags, "counter")
        emit_metric("llm_errors_total", 1.0, {"model": model, "error_type": error_type}, "counter")


def register_callbacks() -> None:
    """注册 Praxis 遥测回调到 LiteLLM。"""
    callback = TelemetryCallback()
    litellm.callbacks.append(callback)  # type: ignore[arg-type]
