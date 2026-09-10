"""Public Runtime smoke workflow; run with the installed wheel's isolated Python."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Any

from praxis import PraxisRuntime, load_config
from praxis.config import PraxisConfig
from praxis.config.schemas import GatewayConfig, ModelCapabilities
from praxis.models.orchestrator import EventType, TextDeltaPayload
from praxis.models.responses import ModelResponse, ModelResponseChunk, Usage
from praxis.models.runtime import HealthStatus


class OfflineGateway:
    """Model boundary substitute; all Runtime components remain real."""

    def __init__(self) -> None:
        self.config = GatewayConfig()
        self.closed = False

    async def complete(
        self, messages: list[dict[str, Any]], **kwargs: Any,
    ) -> ModelResponse:
        return ModelResponse(
            id="wheel-smoke", content="offline response", model="default", created=0,
            usage=Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        )

    async def stream(
        self, messages: list[dict[str, Any]], **kwargs: Any,
    ) -> AsyncIterator[ModelResponseChunk]:
        yield ModelResponseChunk(id="wheel-smoke", delta_content="offline ")
        yield ModelResponseChunk(
            id="wheel-smoke", delta_content="response", finish_reason="stop",
            usage=Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        )

    def capabilities(self, model_name: str | None = None) -> ModelCapabilities:
        return ModelCapabilities()

    async def health(self) -> bool:
        return not self.closed

    async def close(self) -> None:
        self.closed = True


async def smoke(config: PraxisConfig) -> tuple[str, str, bool]:
    gateway = OfflineGateway()
    async with PraxisRuntime(config, gateway=gateway, own_gateway=True) as runtime:
        async with runtime.session() as session:
            response = await session.run("Exercise the installed SDK")
            assert response.content == "offline response"
            chunks: list[str] = []
            async with aclosing(session.run_stream("Exercise the installed stream")) as stream:
                async for event in stream:
                    if event.event_type is EventType.CONTENT_DELTA:
                        assert isinstance(event.data, TextDeltaPayload)
                        chunks.append(event.data.text)
            assert "".join(chunks) == "offline response"
            health = await runtime.health()
            assert health.components["model"].status is HealthStatus.READY
            assert health.components["storage"].status is HealthStatus.READY
        assert not runtime.sessions
    assert gateway.closed
    assert not runtime.started
    return response.content, "".join(chunks), gateway.closed


if __name__ == "__main__":
    asyncio.run(smoke(load_config("wheel-smoke.yaml")))
