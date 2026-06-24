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


def extract_reasoning(obj: Any) -> str | None:
    """从 LiteLLM message/delta 对象提取思考内容。

    LiteLLM 将推理/思考内容统一映射到 ``reasoning_content``；
    部分实现可能使用 ``thinking``，此处作为兼容兜底。
    """
    for attr in ("reasoning_content", "thinking"):
        value = getattr(obj, attr, None)
        if isinstance(value, str) and value:
            return value
    return None


def extract_refusal(obj: Any) -> str | None:
    """从 LiteLLM message/delta 提取 refusal（OpenAI 安全拒答）。"""
    value = getattr(obj, "refusal", None)
    if isinstance(value, str) and value:
        return value
    return None


def build_usage(usage_data: Any) -> Usage:
    """从 LiteLLM usage 结构构造 ``Usage``，兼容 *_tokens_details 嵌套字段。"""
    def _detail(name: str, field: str) -> int:
        details = getattr(usage_data, name, None)
        if details is None:
            return 0
        val = getattr(details, field, None)
        return int(val) if isinstance(val, int) else 0

    return Usage(
        prompt_tokens=getattr(usage_data, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage_data, "completion_tokens", 0) or 0,
        total_tokens=getattr(usage_data, "total_tokens", 0) or 0,
        reasoning_tokens=_detail("completion_tokens_details", "reasoning_tokens"),
        cached_prompt_tokens=_detail("prompt_tokens_details", "cached_tokens"),
    )


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
    """将 LiteLLM 流式响应块转换为 Praxis ModelResponseChunk。"""
    choice = raw.choices[0] if raw.choices else None
    delta = choice.delta if choice else None

    delta_content: str | None = None
    delta_reasoning: str | None = None
    delta_refusal: str | None = None
    delta_tool_calls: list[ToolCallDelta] | None = None

    if delta:
        delta_content = delta.content
        delta_reasoning = extract_reasoning(delta)
        delta_refusal = extract_refusal(delta)
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
        usage = build_usage(raw.usage)

    return ModelResponseChunk(
        id=raw.id,
        delta_content=delta_content,
        delta_reasoning_content=delta_reasoning,
        delta_refusal=delta_refusal,
        delta_tool_calls=delta_tool_calls,
        usage=usage,
        model=getattr(raw, "model", None),
        finish_reason=choice.finish_reason if choice else None,
        system_fingerprint=getattr(raw, "system_fingerprint", None),
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
    # 累计实际花费，供后续调用的预算检查使用
    gateway.add_spend(cost)
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
