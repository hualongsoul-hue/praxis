"""指定 OpenAI 兼容端点的显式 live-model 验收测试。"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from contextlib import aclosing

import pytest

from praxis.config import GatewayConfig, ModelDeployment
from praxis.exceptions import (
    AuthenticationError,
    GatewayError,
    GatewayTimeoutError,
    ModelNotFoundError,
)
from praxis.gateway.chat import chat, chat_stream
from praxis.gateway.router import GatewayRouter

MODEL = "openai/glm-5.1-openai"
API_BASE = "http://172.24.23.192:3000/v1"

pytestmark = pytest.mark.live_model


@pytest.fixture
async def live_gateway() -> AsyncIterator[GatewayRouter]:
    api_key = os.environ.get("PRAXIS_MODEL_API_KEY")
    if not api_key:
        pytest.skip("需要 PRAXIS_MODEL_API_KEY 才能执行 live model 测试")

    gateway = GatewayRouter(
        GatewayConfig(
            deployments=[
                ModelDeployment(
                    model_name="default",
                    model=MODEL,
                    api_base=API_BASE,
                    api_key_env="PRAXIS_MODEL_API_KEY",
                    default_max_output_tokens=128,
                ),
            ],
            default_model="default",
            timeout=30.0,
            num_retries=0,
            max_concurrent_requests=1,
            max_total_tokens=20_000,
        ),
        environ={"PRAXIS_MODEL_API_KEY": api_key},
    )
    try:
        yield gateway
    finally:
        await gateway.close()


async def test_live_regular_response(live_gateway: GatewayRouter) -> None:
    response = None
    error_category: str | None = None
    try:
        response = await chat(
            live_gateway,
            [{"role": "user", "content": "只回复单词 pong"}],
            max_tokens=256,
        )
    except GatewayError as exc:
        error_category = type(exc).__name__

    if error_category is not None:
        pytest.fail(f"regular response returned {error_category}")
    assert response is not None
    assert response.content
    assert "pong" in response.content.lower()
    assert response.usage.total_tokens > 0


async def test_live_streaming_response(live_gateway: GatewayRouter) -> None:
    chunks = []
    error_category: str | None = None
    try:
        chunks = [
            chunk
            async for chunk in chat_stream(
                live_gateway,
                [{"role": "user", "content": "用一句简短中文问候"}],
                max_tokens=256,
            )
        ]
    except GatewayError as exc:
        error_category = type(exc).__name__

    if error_category is not None:
        pytest.fail(f"streaming response returned {error_category}")
    assert chunks
    assert any(
        chunk.delta_content or chunk.delta_reasoning_content
        for chunk in chunks
    )
    assert any(chunk.finish_reason for chunk in chunks)
    assert all(chunk is not None for chunk in chunks)


async def test_live_forced_tool_call(live_gateway: GatewayRouter) -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "原样返回文本",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                    "additionalProperties": False,
                },
            },
        },
    ]
    response = None
    error_category: str | None = None
    try:
        response = await chat(
            live_gateway,
            [{"role": "user", "content": "调用 echo，text 参数使用 hello"}],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "echo"}},
            max_tokens=256,
        )
    except GatewayError as exc:
        error_category = type(exc).__name__

    if error_category is not None:
        pytest.fail(f"forced tool call returned {error_category}")
    assert response is not None
    assert response.tool_calls
    call = response.tool_calls[0]
    assert call.function.name == "echo"
    assert json.loads(call.function.arguments)["text"] == "hello"


async def test_live_stream_can_be_cancelled(live_gateway: GatewayRouter) -> None:
    first_chunk_received = asyncio.Event()
    hold_stream = asyncio.Event()

    async def consume() -> None:
        async with aclosing(
            chat_stream(
                live_gateway,
                [{"role": "user", "content": "写一篇较长的分布式系统说明"}],
                max_tokens=512,
            )
        ) as stream:
            async for chunk in stream:
                assert chunk is not None
                first_chunk_received.set()
                await hold_stream.wait()

    task = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(first_chunk_received.wait(), timeout=30.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        if not task.done():
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    response = None
    error_category: str | None = None
    try:
        response = await asyncio.wait_for(
            chat(
                live_gateway,
                [{"role": "user", "content": "只回复单词 healthy"}],
                timeout=30.0,
                max_tokens=32,
            ),
            timeout=35.0,
        )
    except GatewayError as exc:
        error_category = type(exc).__name__
    if error_category is not None:
        pytest.fail(f"post-cancellation text request returned {error_category}")
    assert response is not None
    assert response.content
    assert "healthy" in response.content.lower()


async def test_live_timeout_is_mapped(live_gateway: GatewayRouter) -> None:
    observed_error: GatewayError | None = None
    try:
        await chat(
            live_gateway,
            [{"role": "user", "content": "回复任意内容"}],
            timeout=0.000001,
            max_tokens=16,
        )
    except GatewayError as exc:
        observed_error = exc

    assert type(observed_error) is GatewayTimeoutError
    assert observed_error.details["original_type"] == "Timeout"


async def test_live_not_found_is_mapped() -> None:
    api_key = os.environ.get("PRAXIS_MODEL_API_KEY")
    if not api_key:
        pytest.skip("需要 PRAXIS_MODEL_API_KEY 才能执行 live model 测试")

    gateway = GatewayRouter(
        GatewayConfig(
            deployments=[
                ModelDeployment(
                    model_name="default",
                    model=MODEL,
                    api_base=f"{API_BASE}/unreachable-path",
                ),
            ],
            num_retries=0,
            timeout=5.0,
        ),
        environ={"PRAXIS_MODEL_API_KEY": api_key},
    )
    try:
        observed_error: GatewayError | None = None
        try:
            await chat(
                gateway,
                [{"role": "user", "content": "ping"}],
                max_tokens=8,
            )
        except GatewayError as exc:
            observed_error = exc
    finally:
        await gateway.close()

    assert type(observed_error) is ModelNotFoundError
    assert observed_error.details["original_type"] == "NotFoundError"


async def test_live_authentication_error_is_mapped() -> None:
    if not os.environ.get("PRAXIS_MODEL_API_KEY"):
        pytest.skip("需要 PRAXIS_MODEL_API_KEY 才能执行 live model 测试")
    gateway = GatewayRouter(
        GatewayConfig(
            deployments=[
                ModelDeployment(
                    model_name="default",
                    model=MODEL,
                    api_base=API_BASE,
                ),
            ],
            num_retries=0,
            timeout=30.0,
        ),
        environ={"PRAXIS_MODEL_API_KEY": "intentionally-invalid-live-credential"},
    )
    try:
        observed_error: GatewayError | None = None
        try:
            await chat(
                gateway,
                [{"role": "user", "content": "ping"}],
                timeout=30.0,
                max_tokens=8,
            )
        except GatewayError as exc:
            observed_error = exc
    finally:
        await gateway.close()

    assert type(observed_error) is AuthenticationError
    assert observed_error.details["original_type"] == "AuthenticationError"
