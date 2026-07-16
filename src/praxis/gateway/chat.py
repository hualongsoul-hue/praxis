"""类型安全的非流式与流式模型调用。"""

import asyncio
import warnings
from collections.abc import AsyncIterator
from typing import Any, cast

from pydantic.warnings import PydanticDeprecatedSince211

from praxis.config.schemas import ModelDeployment
from praxis.gateway.metering import (
    completion_cost,
    estimate_input_cost,
    estimate_output_cost,
    get_token_count,
    record_usage,
)
from praxis.gateway.resilience import map_litellm_exception
from praxis.gateway.router import GatewayRouter, UsageReservation
from praxis.models.responses import (
    FunctionCallDelta,
    ModelResponse,
    ModelResponseChunk,
    ToolCallDelta,
    Usage,
)
from praxis.models.tools import FunctionCall, ToolCall


def extract_reasoning(obj: Any) -> str | None:
    """提取不同兼容端点使用的推理字段。"""
    for attr in ("reasoning_content", "thinking"):
        value = getattr(obj, attr, None)
        if isinstance(value, str) and value:
            return value
    return None


def extract_refusal(obj: Any) -> str | None:
    value = getattr(obj, "refusal", None)
    return value if isinstance(value, str) and value else None


def build_usage(usage_data: Any) -> Usage:
    def detail(name: str, field: str) -> int:
        details = getattr(usage_data, name, None)
        value = getattr(details, field, None) if details is not None else None
        return int(value) if isinstance(value, int) else 0

    return Usage(
        prompt_tokens=int(getattr(usage_data, "prompt_tokens", 0) or 0),
        completion_tokens=int(getattr(usage_data, "completion_tokens", 0) or 0),
        total_tokens=int(getattr(usage_data, "total_tokens", 0) or 0),
        reasoning_tokens=detail("completion_tokens_details", "reasoning_tokens"),
        cached_prompt_tokens=detail("prompt_tokens_details", "cached_tokens"),
    )


def convert_response(raw: Any) -> ModelResponse:
    choice = raw.choices[0]
    message = choice.message
    tool_calls = None
    if message.tool_calls:
        tool_calls = [
            ToolCall(
                id=item.id,
                type="function",
                function=FunctionCall(
                    name=item.function.name,
                    arguments=item.function.arguments,
                ),
            )
            for item in message.tool_calls
        ]
    return ModelResponse(
        id=raw.id,
        content=message.content,
        reasoning_content=extract_reasoning(message),
        refusal=extract_refusal(message),
        tool_calls=tool_calls,
        usage=build_usage(raw.usage),
        model=raw.model,
        finish_reason=choice.finish_reason,
        created=raw.created,
        system_fingerprint=getattr(raw, "system_fingerprint", None),
    )


def convert_stream_chunk(raw: Any) -> ModelResponseChunk:
    choice = raw.choices[0] if raw.choices else None
    delta = choice.delta if choice else None
    delta_tool_calls: list[ToolCallDelta] | None = None
    if delta and delta.tool_calls:
        delta_tool_calls = [
            ToolCallDelta(
                index=item.index,
                id=getattr(item, "id", None),
                type=getattr(item, "type", None),
                function=(
                    FunctionCallDelta(
                        name=item.function.name,
                        arguments=item.function.arguments,
                    )
                    if item.function
                    else None
                ),
            )
            for item in delta.tool_calls
        ]
    usage = build_usage(raw.usage) if getattr(raw, "usage", None) else None
    return ModelResponseChunk(
        id=raw.id,
        delta_content=delta.content if delta else None,
        delta_reasoning_content=extract_reasoning(delta) if delta else None,
        delta_refusal=extract_refusal(delta) if delta else None,
        delta_tool_calls=delta_tool_calls,
        usage=usage,
        model=getattr(raw, "model", None),
        finish_reason=choice.finish_reason if choice else None,
        system_fingerprint=getattr(raw, "system_fingerprint", None),
    )


def prepare_reservation(
    gateway: GatewayRouter,
    deployment: ModelDeployment,
    messages: list[dict[str, Any]],
    kwargs: dict[str, Any],
) -> UsageReservation:
    max_budget = gateway.config.max_budget
    max_total_tokens = gateway.config.max_total_tokens
    constrained = (
        isinstance(max_budget, (int, float))
        or isinstance(max_total_tokens, int)
    )
    if not constrained:
        return gateway.reserve_usage(estimated_tokens=0, estimated_cost=None)

    prompt_tokens = get_token_count(messages, deployment.model)
    output_tokens = int(
        kwargs.get("max_tokens", deployment.default_max_output_tokens)
        or deployment.default_max_output_tokens
    )
    output_tokens = max(output_tokens, 0)
    estimated_cost: float | None = None
    if isinstance(max_budget, (int, float)):
        input_cost = (
            prompt_tokens * deployment.input_cost_per_token
            if deployment.input_cost_per_token is not None
            else estimate_input_cost(prompt_tokens, deployment.model)
        )
        output_cost = (
            output_tokens * deployment.output_cost_per_token
            if deployment.output_cost_per_token is not None
            else estimate_output_cost(output_tokens, deployment.model)
        )
        if input_cost is not None and output_cost is not None:
            estimated_cost = input_cost + output_cost
    return gateway.reserve_usage(
        estimated_tokens=prompt_tokens + output_tokens,
        estimated_cost=estimated_cost,
    )


def actual_cost(
    deployment: ModelDeployment,
    usage: Usage,
    raw: Any | None = None,
) -> float | None:
    input_rate = deployment.input_cost_per_token
    output_rate = deployment.output_cost_per_token
    if isinstance(input_rate, (int, float)) and isinstance(output_rate, (int, float)):
        return (
            usage.prompt_tokens * input_rate
            + usage.completion_tokens * output_rate
        )
    try:
        if raw is not None:
            return completion_cost(raw)
        import litellm

        prompt_cost, output_cost = litellm.cost_per_token(
            model=deployment.model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        )
        return float(prompt_cost) + float(output_cost)
    except Exception:
        return None


def settlement_cost(settlement: object, fallback: float | None) -> float:
    if isinstance(settlement, tuple):
        values = cast(tuple[object, ...], settlement)
        if len(values) != 2:
            return fallback or 0.0
        value = values[1]
        if isinstance(value, (int, float)):
            return float(value)
    return fallback or 0.0


async def chat(
    gateway: GatewayRouter,
    messages: list[dict[str, Any]],
    model: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> ModelResponse:
    model_name = model or gateway.config.default_model
    deployment = gateway.resolve_deployment(model_name)
    reservation = prepare_reservation(gateway, deployment, messages, kwargs)
    call_kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        **kwargs,
    }
    if tools:
        call_kwargs["tools"] = tools
        # Praxis owns MCP discovery/execution. Bypass LiteLLM's proxy-only MCP
        # bridge, which otherwise imports web-server dependencies for SDK calls.
        call_kwargs["_skip_mcp_handler"] = True

    try:
        async with gateway.request_slot():
            raw: Any = cast(
                Any,
                await gateway.router.acompletion(  # pyright: ignore[reportUnknownMemberType]
                    **call_kwargs,
                ),
            )
    except Exception as exc:
        gateway.release_usage(reservation)
        raise map_litellm_exception(exc) from exc

    try:
        response = convert_response(raw)
    except Exception:
        gateway.settle_usage(reservation, actual_tokens=None, actual_cost=None)
        raise
    calculated_cost = actual_cost(deployment, response.usage, raw)
    settlement = gateway.settle_usage(
        reservation,
        actual_tokens=response.usage.total_tokens,
        actual_cost=calculated_cost,
    )
    cost = settlement_cost(settlement, calculated_cost)
    record_usage(
        response.model,
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
        cost,
    )
    return response


async def chat_stream(
    gateway: GatewayRouter,
    messages: list[dict[str, Any]],
    model: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> AsyncIterator[ModelResponseChunk]:
    model_name = model or gateway.config.default_model
    deployment = gateway.resolve_deployment(model_name)
    reservation = prepare_reservation(gateway, deployment, messages, kwargs)
    call_kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        **kwargs,
    }
    if tools:
        call_kwargs["tools"] = tools
        call_kwargs["_skip_mcp_handler"] = True

    final_usage: Usage | None = None
    final_model = model_name
    started = False
    settled = False
    try:
        async with gateway.request_slot():
            stream: Any = cast(
                Any,
                await gateway.router.acompletion(  # pyright: ignore[reportUnknownMemberType]
                    **call_kwargs,
                ),
            )
            started = True
            with warnings.catch_warnings():
                # LiteLLM 1.92 introspects Pydantic model instances while
                # deciding whether a usage chunk is empty. Pydantic 2.11+
                # emits this specific compatibility warning from that path.
                warnings.filterwarnings(
                    "ignore",
                    message=(
                        "Accessing the 'model_(?:computed_)?fields' attribute on the instance "
                        "is deprecated.*"
                    ),
                    category=PydanticDeprecatedSince211,
                )
                async for raw_chunk in stream:
                    chunk = convert_stream_chunk(raw_chunk)
                    if chunk.usage is not None:
                        final_usage = chunk.usage
                    if chunk.model:
                        final_model = chunk.model
                    yield chunk
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        if not started:
            gateway.release_usage(reservation)
            settled = True
        raise map_litellm_exception(exc) from exc
    finally:
        if not settled:
            if final_usage is None:
                gateway.settle_usage(
                    reservation,
                    actual_tokens=None,
                    actual_cost=None,
                )
            else:
                calculated_cost = actual_cost(deployment, final_usage)
                settlement = gateway.settle_usage(
                    reservation,
                    actual_tokens=final_usage.total_tokens,
                    actual_cost=calculated_cost,
                )
                cost = settlement_cost(settlement, calculated_cost)
                record_usage(
                    final_model,
                    final_usage.prompt_tokens,
                    final_usage.completion_tokens,
                    cost,
                )
