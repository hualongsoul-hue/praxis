"""Protocol-facing model calls used by framework components."""

from collections.abc import AsyncIterator
from typing import Any

from praxis.models.responses import ModelResponse, ModelResponseChunk
from praxis.protocols import ModelGateway


async def complete(
    gateway: ModelGateway,
    messages: list[dict[str, Any]],
    model: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> ModelResponse:
    """Invoke the public gateway completion contract."""
    return await gateway.complete(messages, model=model, tools=tools, **kwargs)


def stream(
    gateway: ModelGateway,
    messages: list[dict[str, Any]],
    model: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> AsyncIterator[ModelResponseChunk]:
    """Invoke the public gateway streaming contract."""
    return gateway.stream(messages, model=model, tools=tools, **kwargs)
