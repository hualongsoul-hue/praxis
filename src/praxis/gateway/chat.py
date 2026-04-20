"""chat / chat_stream 接口。

提供同步完整调用和异步流式调用两种模式，
自动将 LiteLLM 响应转换为 Praxis 内部数据模型。
"""

from collections.abc import AsyncIterator
from typing import Any

from praxis.gateway.metering import check_budget, completion_cost, get_token_count, record_usage
from praxis.gateway.resilience import map_litellm_exception
from praxis.gateway.router import GatewayRouter
from praxis.models.responses import (
    FunctionCallDelta,
    ModelResponse,
    ModelResponseChunk,
    ToolCallDelta,
    Usage,
)
from praxis.models.tools import FunctionCall, ToolCall


def convert_response(raw: Any) -> ModelResponse:
    """将 LiteLLM 原始响应转换为 Praxis ModelResponse。"""
    choice = raw.choices[0]
    message = choice.message

    tool_calls: list[ToolCall] | None = None
    if message.tool_calls:
        tool_calls = [
            ToolCall(
                id=tc.id,
                type="function",
                function=FunctionCall(
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                ),
            )
            for tc in message.tool_calls
        ]

    usage_data = raw.usage
    usage = Usage(
        prompt_tokens=usage_data.prompt_tokens,
        completion_tokens=usage_data.completion_tokens,
        total_tokens=usage_data.total_tokens,
    )

    return ModelResponse(
        id=raw.id,
        content=message.content,
        tool_calls=tool_calls,
        usage=usage,
        model=raw.model,
        finish_reason=choice.finish_reason,
        created=raw.created,
    )


def convert_stream_chunk(raw: Any) -> ModelResponseChunk:
    """将 LiteLLM 流式响应块转换为 Praxis ModelResponseChunk。"""
    choice = raw.choices[0] if raw.choices else None
    delta = choice.delta if choice else None

    delta_content: str | None = None
    delta_tool_calls: list[ToolCallDelta] | None = None

    if delta:
        delta_content = delta.content
        if delta.tool_calls:
            delta_tool_calls = [
                ToolCallDelta(
                    index=tc.index,
                    id=getattr(tc, "id", None),
                    type=getattr(tc, "type", None),
                    function=FunctionCallDelta(
                        name=tc.function.name if tc.function else None,
                        arguments=tc.function.arguments if tc.function else None,
                    ) if tc.function else None,
                )
                for tc in delta.tool_calls
            ]

    usage: Usage | None = None
    if hasattr(raw, "usage") and raw.usage:
        usage = Usage(
            prompt_tokens=raw.usage.prompt_tokens or 0,
            completion_tokens=raw.usage.completion_tokens or 0,
            total_tokens=raw.usage.total_tokens or 0,
        )

    return ModelResponseChunk(
        id=raw.id,
        delta_content=delta_content,
        delta_tool_calls=delta_tool_calls,
        usage=usage,
        model=getattr(raw, "model", None),
        finish_reason=choice.finish_reason if choice else None,
    )


async def chat(
    gateway: GatewayRouter,
    messages: list[dict[str, Any]],
    model: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> ModelResponse:
    """同步完整调用 LLM。

    Args:
        gateway: 网关路由器实例。
        messages: OpenAI 格式消息列表。
        model: 模型别名，None 时使用默认模型。
        tools: 可选工具 Schema 列表。
        **kwargs: 额外推理参数（temperature、max_tokens 等）透传。

    Returns:
        完整的 ModelResponse。
    """
    model_name = model or gateway.config.default_model

    # 预算检查（仅在 max_budget 配置时才计算 Token，避免热路径开销）
    if gateway.config.max_budget is not None:
        estimated = get_token_count(messages, model_name)
        check_budget(gateway, estimated, model_name)

    call_kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        **kwargs,
    }
    if tools:
        call_kwargs["tools"] = tools

    try:
        raw = await gateway.router.acompletion(**call_kwargs)
    except Exception as exc:
        raise map_litellm_exception(exc) from exc

    response = convert_response(raw)

    try:
        cost = completion_cost(raw)
    except Exception:
        cost = 0.0
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
    """异步流式调用 LLM。

    逐块 yield ModelResponseChunk，支持流式传输中工具调用增量解析。
    流中断时已产出的块保留。

    Args:
        gateway: 网关路由器实例。
        messages: OpenAI 格式消息列表。
        model: 模型别名，None 时使用默认模型。
        tools: 可选工具 Schema 列表。
        **kwargs: 额外推理参数透传。

    Yields:
        ModelResponseChunk 流式响应块。
    """
    model_name = model or gateway.config.default_model

    # 预算检查（仅在 max_budget 配置时才计算 Token）
    if gateway.config.max_budget is not None:
        estimated = get_token_count(messages, model_name)
        check_budget(gateway, estimated, model_name)

    call_kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "stream": True,
        **kwargs,
    }
    if tools:
        call_kwargs["tools"] = tools

    try:
        stream = await gateway.router.acompletion(**call_kwargs)
    except Exception as exc:
        raise map_litellm_exception(exc) from exc

    try:
        async for raw_chunk in stream:
            yield convert_stream_chunk(raw_chunk)
    except Exception as exc:
        raise map_litellm_exception(exc) from exc
