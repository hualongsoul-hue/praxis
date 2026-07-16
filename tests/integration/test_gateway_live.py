"""指定 OpenAI 兼容端点的显式 live-model 验收测试。"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator

import pytest

from praxis.config import GatewayConfig, ModelDeployment
from praxis.exceptions import GatewayError, GatewayTimeoutError
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
            max_concurrent_requests=2,
            max_total_tokens=20_000,
        ),
        environ={"PRAXIS_MODEL_API_KEY": api_key},
    )
    try:
        yield gateway
    finally:
        await gateway.close()


async def test_live_regular_response(live_gateway: GatewayRouter) -> None:
    response = await chat(
        live_gateway,
        [{"role": "user", "content": "只回复单词 pong"}],
        max_tokens=256,
    )

    assert response.content
    assert "pong" in response.content.lower()
    assert response.usage.total_tokens > 0


async def test_live_streaming_response(live_gateway: GatewayRouter) -> None:
    chunks = [
        chunk
        async for chunk in chat_stream(
            live_gateway,
            [{"role": "user", "content": "用一句简短中文问候"}],
            max_tokens=256,
        )
    ]

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
    response = await chat(
        live_gateway,
        [{"role": "user", "content": "调用 echo，text 参数使用 hello"}],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "echo"}},
        max_tokens=256,
    )

    assert response.tool_calls
    call = response.tool_calls[0]
    assert call.function.name == "echo"
    assert json.loads(call.function.arguments)["text"] == "hello"


async def test_live_stream_can_be_cancelled(live_gateway: GatewayRouter) -> None:
    started = asyncio.Event()

    async def consume() -> None:
        started.set()
        async for _chunk in chat_stream(
            live_gateway,
            [{"role": "user", "content": "写一篇较长的分布式系统说明"}],
            max_tokens=512,
        ):
            pass

    task = asyncio.create_task(consume())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert await live_gateway.health()


async def test_live_timeout_is_mapped(live_gateway: GatewayRouter) -> None:
    with pytest.raises((GatewayTimeoutError, GatewayError)):
        await chat(
            live_gateway,
            [{"role": "user", "content": "回复任意内容"}],
            timeout=0.000001,
            max_tokens=16,
        )


async def test_live_endpoint_error_is_mapped() -> None:
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
        with pytest.raises(GatewayError):
            await chat(
                gateway,
                [{"role": "user", "content": "ping"}],
                max_tokens=8,
            )
    finally:
        await gateway.close()
