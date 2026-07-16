"""Real-socket contracts for Runtime, Session, GatewayRouter, and LiteLLM."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from praxis.config.schemas import (
    GatewayConfig,
    InputConfig,
    MemoryConfig,
    ModelCapabilities,
    ModelDeployment,
    PersistenceConfig,
    SessionConfig,
)
from praxis.config.settings import PraxisConfig
from praxis.exceptions import AuthenticationError, GatewayError, GatewayTimeoutError
from praxis.gateway.router import GatewayRouter
from praxis.models.inputs import AudioInput, FileInput, ImageInput, UserInput, VideoInput
from praxis.runtime import PraxisRuntime

if TYPE_CHECKING:
    from tests.fixtures.openai_compatible_server import OpenAICompatibleServer

pytest_plugins = ("tests.fixtures.openai_compatible_server",)
pytestmark = pytest.mark.e2e


def make_contract_config(tmp_path: Path, api_base: str) -> PraxisConfig:
    """Build an isolated Runtime configuration for a local contract provider."""
    return PraxisConfig(
        gateway=GatewayConfig(
            deployments=[
                ModelDeployment(
                    model_name="contract",
                    model="openai/contract-model",
                    api_base=api_base,
                    capabilities=ModelCapabilities(
                        image=True,
                        audio=True,
                        video=True,
                        file=True,
                    ),
                )
            ],
            default_model="contract",
            num_retries=0,
        ),
        inputs=InputConfig(
            max_attachment_bytes=1024,
            max_total_bytes=4096,
        ),
        persistence=PersistenceConfig(
            backend="filesystem",
            filesystem_path=str(tmp_path / "store"),
        ),
        session=SessionConfig(auto_checkpoint=False),
        memory=MemoryConfig(background_enabled=False, dream_enabled=False),
    )


def all_content_input() -> UserInput:
    """Return resolver-valid bytes and MIME types for every attachment modality."""
    return UserInput(
        text="contract",
        parts=(
            ImageInput.from_bytes(
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR",
                media_type="image/png",
                filename="pixel.png",
            ),
            AudioInput.from_bytes(
                b"RIFF$\x00\x00\x00WAVEfmt ",
                media_type="audio/wav",
                filename="tone.wav",
            ),
            VideoInput.from_bytes(
                b"\x00\x00\x00\x18ftypmp42mp42isom",
                media_type="video/mp4",
                filename="clip.mp4",
            ),
            FileInput.from_bytes(
                b"%PDF-1.4\n%%EOF\n",
                media_type="application/pdf",
                filename="contract.pdf",
            ),
        ),
    )


def make_gateway(config: PraxisConfig) -> GatewayRouter:
    """Create the real GatewayRouter with a fake, non-secret local credential."""
    return GatewayRouter(
        config.gateway,
        environ={"PRAXIS_MODEL_API_KEY": "fake-contract-credential"},
    )


def normalize_stream_transport(request: dict[str, Any]) -> dict[str, Any]:
    """Remove only the transport switches that distinguish streaming calls."""
    normalized = dict(request)
    normalized.pop("stream", None)
    normalized.pop("stream_options", None)
    return normalized


async def test_runtime_sends_all_content_parts_over_real_http(
    tmp_path: Path,
    openai_compatible_server: OpenAICompatibleServer,
) -> None:
    config = make_contract_config(tmp_path, openai_compatible_server.base_url)
    gateway = make_gateway(config)

    async with PraxisRuntime(config, gateway=gateway) as runtime:
        async with runtime.session() as session:
            response = await session.run(all_content_input())

    request = openai_compatible_server.requests[0]
    user_message = next(message for message in request["messages"] if message["role"] == "user")
    content = user_message["content"]
    assert [part["type"] for part in content] == [
        "text",
        "image_url",
        "input_audio",
        "video_url",
        "file",
    ]
    assert response.content == "contract-ok"


async def test_runtime_streams_sse_over_real_http(
    tmp_path: Path,
    openai_compatible_server: OpenAICompatibleServer,
) -> None:
    config = make_contract_config(tmp_path, openai_compatible_server.base_url)
    async with PraxisRuntime(config, gateway=make_gateway(config)) as runtime:
        async with runtime.session() as session:
            events = [event async for event in session.run_stream("stream contract")]

    content = "".join(
        str(event.data["text"])
        for event in events
        if event.event_type == "content_delta"
    )
    assert content == "contract-ok"
    assert openai_compatible_server.requests[0]["stream"] is True
    assert events[-1].event_type == "termination"


async def test_stream_and_regular_requests_differ_only_in_transport_options(
    tmp_path: Path,
    openai_compatible_server: OpenAICompatibleServer,
) -> None:
    config = make_contract_config(tmp_path, openai_compatible_server.base_url)
    async with PraxisRuntime(config, gateway=make_gateway(config)) as runtime:
        async with runtime.session() as regular_session:
            regular_response = await regular_session.run(all_content_input())
        async with runtime.session() as streaming_session:
            streaming_events = [
                event async for event in streaming_session.run_stream(all_content_input())
            ]

    regular_request, streaming_request = openai_compatible_server.requests
    assert normalize_stream_transport(streaming_request) == normalize_stream_transport(
        regular_request
    )
    assert regular_response.content == "contract-ok"
    assert "".join(
        str(event.data["text"])
        for event in streaming_events
        if event.event_type == "content_delta"
    ) == "contract-ok"


async def test_gateway_receives_forced_tool_call_over_real_http(
    tmp_path: Path,
    openai_compatible_server: OpenAICompatibleServer,
) -> None:
    config = make_contract_config(tmp_path, openai_compatible_server.base_url)
    gateway = make_gateway(config)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "contract_tool",
                "description": "Return a deterministic contract value",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                },
            },
        }
    ]
    try:
        response = await gateway.complete(
            [{"role": "user", "content": "call the contract tool"}],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "contract_tool"}},
        )
    finally:
        await gateway.close()

    assert response.tool_calls is not None
    assert response.tool_calls[0].function.name == "contract_tool"
    assert response.tool_calls[0].function.arguments == '{"value":"forced"}'
    assert openai_compatible_server.requests[0]["tool_choice"] == {
        "type": "function",
        "function": {"name": "contract_tool"},
    }


async def test_runtime_stream_cancellation_releases_real_http_handler(
    tmp_path: Path,
    openai_compatible_server: OpenAICompatibleServer,
) -> None:
    openai_compatible_server.block_stream_after_first_frame()
    config = make_contract_config(tmp_path, openai_compatible_server.base_url)
    async with PraxisRuntime(config, gateway=make_gateway(config)) as runtime:
        async with runtime.session() as session:
            events = []
            content_received = asyncio.Event()

            async def consume_stream() -> None:
                async for event in session.run_stream("cancel contract"):
                    events.append(event)
                    if event.event_type == "content_delta":
                        content_received.set()

            pending = asyncio.create_task(consume_stream())
            try:
                received = await asyncio.to_thread(
                    openai_compatible_server.stream_frame_sent.wait,
                    5,
                )
                assert received is True
                await asyncio.wait_for(content_received.wait(), timeout=5)
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
            finally:
                openai_compatible_server.release_response()
                if not pending.done():
                    pending.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await pending

    assert len(openai_compatible_server.requests) == 1
    assert [event.event_type for event in events[:3]] == [
        "turn_start",
        "llm_request",
        "content_delta",
    ]


async def test_runtime_maps_real_http_timeout(
    tmp_path: Path,
    openai_compatible_server: OpenAICompatibleServer,
) -> None:
    openai_compatible_server.block_response()
    base_config = make_contract_config(tmp_path, openai_compatible_server.base_url)
    config = base_config.model_copy(
        update={"gateway": base_config.gateway.model_copy(update={"timeout": 0.05})}
    )
    async with PraxisRuntime(config, gateway=make_gateway(config)) as runtime:
        async with runtime.session() as session:
            pending = asyncio.create_task(session.run("timeout contract"))
            try:
                received = await asyncio.to_thread(
                    openai_compatible_server.request_received.wait,
                    5,
                )
                assert received is True
                with pytest.raises(GatewayTimeoutError) as captured:
                    await pending
            finally:
                openai_compatible_server.release_response()
                if not pending.done():
                    pending.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await pending

    assert captured.value.details["original_type"] == "Timeout"


async def test_runtime_maps_malformed_real_http_response(
    tmp_path: Path,
    openai_compatible_server: OpenAICompatibleServer,
) -> None:
    openai_compatible_server.choose_response("malformed")
    config = make_contract_config(tmp_path, openai_compatible_server.base_url)
    async with PraxisRuntime(config, gateway=make_gateway(config)) as runtime:
        async with runtime.session() as session:
            with pytest.raises(GatewayError) as captured:
                await session.run("malformed contract")

    assert captured.value.details["original_type"]
    assert len(openai_compatible_server.requests) == 1


async def test_runtime_maps_real_http_unauthorized_response(
    tmp_path: Path,
    openai_compatible_server: OpenAICompatibleServer,
) -> None:
    openai_compatible_server.choose_response("unauthorized")
    config = make_contract_config(tmp_path, openai_compatible_server.base_url)
    async with PraxisRuntime(config, gateway=make_gateway(config)) as runtime:
        async with runtime.session() as session:
            with pytest.raises(AuthenticationError) as captured:
                await session.run("unauthorized contract")

    assert captured.value.details["original_type"] == "AuthenticationError"
    assert len(openai_compatible_server.requests) == 1
