"""Protocol-facing model calls used by framework components."""

from collections.abc import AsyncIterator
from typing import Any

from opentelemetry.trace import StatusCode

from praxis.lifecycle import close_async_stream
from praxis.models.responses import ModelResponse, ModelResponseChunk
from praxis.protocols import ModelGateway
from praxis.telemetry.tracing import operation_span, start_span


async def complete(
    gateway: ModelGateway,
    messages: list[dict[str, Any]],
    model: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> ModelResponse:
    """Invoke the public gateway completion contract."""
    with operation_span("praxis.model.complete") as span:
        response = await gateway.complete(messages, model=model, tools=tools, **kwargs)
        span.set_attribute("gen_ai.usage.input_tokens", response.usage.prompt_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", response.usage.completion_tokens)
        return response


async def stream(
    gateway: ModelGateway,
    messages: list[dict[str, Any]],
    model: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> AsyncIterator[ModelResponseChunk]:
    """Invoke the public gateway streaming contract."""
    # Never attach a tracing context across yield: the consumer owns that context.
    span = start_span("praxis.model.stream")
    source: AsyncIterator[ModelResponseChunk] | None = None
    try:
        source = gateway.stream(messages, model=model, tools=tools, **kwargs)
        async for chunk in source:
            if chunk.usage is not None:
                span.set_attribute("gen_ai.usage.input_tokens", chunk.usage.prompt_tokens)
                span.set_attribute("gen_ai.usage.output_tokens", chunk.usage.completion_tokens)
            yield chunk
    except GeneratorExit:
        span.set_attribute("praxis.stream.interrupted", True)
        raise
    except BaseException as error:
        span.set_attribute("error.type", type(error).__name__)
        span.set_status(StatusCode.ERROR)
        raise
    finally:
        try:
            await close_async_stream(source)
        finally:
            span.end()
