"""S4 模型网关验证测试。"""

import asyncio
import json
import warnings
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic.warnings import PydanticDeprecatedSince211

from praxis.config.schemas import GatewayConfig, ModelCapabilities, ModelDeployment
from praxis.exceptions import (
    AuthenticationError,
    BudgetExceededError,
    GatewayError,
    GatewayTimeoutError,
    ModelValidationError,
    RateLimitError,
)
from praxis.gateway.callbacks import TelemetryCallback
from praxis.gateway.chat import (
    build_usage,
    chat,
    chat_stream,
    convert_response,
    convert_stream_chunk,
)
from praxis.gateway.metering import estimate_input_cost, get_max_tokens, get_token_count
from praxis.gateway.resilience import EXCEPTION_MAP, map_litellm_exception
from praxis.gateway.router import GatewayRouter
from praxis.gateway.tasks import judge, summarize
from praxis.models.gateway import JudgeResult
from praxis.models.responses import ModelResponse, ModelResponseChunk

SAMPLE_DEPLOYMENTS = [
    ModelDeployment(model_name="default", model="openai/gpt-4o"),
    ModelDeployment(model_name="default", model="anthropic/claude-sonnet-4-20250514"),
    ModelDeployment(model_name="fast", model="openai/gpt-4o-mini"),
]


def make_config(**overrides: object) -> GatewayConfig:
    defaults = {"deployments": SAMPLE_DEPLOYMENTS}
    defaults.update(overrides)
    return GatewayConfig(**defaults)


def test_model_capability_probe_uses_typed_deployment() -> None:
    gateway = GatewayRouter(
        GatewayConfig(
            deployments=[
                ModelDeployment(
                    capabilities=ModelCapabilities(
                        image=True,
                        audio=False,
                        video=False,
                        file=True,
                    )
                )
            ],
        )
    )
    expected = ModelCapabilities(image=True, file=True)
    assert gateway.capabilities() == expected
    assert gateway.capabilities("default") == expected


@pytest.fixture(autouse=True)
def model_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRAXIS_MODEL_API_KEY", "unit-test-model-key")


def make_raw_response(
    content: str = "Hello!",
    model: str = "gpt-4o",
    tool_calls: list | None = None,
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
    reasoning_content: str | None = None,
    refusal: str | None = None,
    reasoning_tokens: int = 0,
    cached_tokens: int = 0,
    system_fingerprint: str | None = None,
) -> SimpleNamespace:
    """构建模拟 LiteLLM 原始响应。"""
    message = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        reasoning_content=reasoning_content,
        refusal=refusal,
    )
    choice = SimpleNamespace(message=message, finish_reason="stop")
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning_tokens),
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
    )
    return SimpleNamespace(
        id="chatcmpl-test-123",
        choices=[choice],
        usage=usage,
        model=model,
        created=1700000000,
        system_fingerprint=system_fingerprint,
    )


def make_raw_stream_chunk(
    content: str | None = "Hi",
    finish_reason: str | None = None,
    reasoning_content: str | None = None,
    refusal: str | None = None,
    system_fingerprint: str | None = None,
) -> SimpleNamespace:
    """构建模拟 LiteLLM 流式响应块。"""
    delta = SimpleNamespace(
        content=content,
        tool_calls=None,
        reasoning_content=reasoning_content,
        refusal=refusal,
    )
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(
        id="chatcmpl-stream-123",
        choices=[choice],
        usage=None,
        model="gpt-4o",
        system_fingerprint=system_fingerprint,
    )


class TestGatewayRouter:
    """Task 4.1: LiteLLM Router 集成验证。"""

    def test_init_with_valid_config(self) -> None:
        config = make_config()
        gw = GatewayRouter(config)
        assert gw.router is not None
        assert gw.config is config

    def test_empty_deployments_rejected(self) -> None:
        with pytest.raises(ModelValidationError):
            GatewayConfig(deployments=[])

    def test_get_model_names(self) -> None:
        gw = GatewayRouter(make_config())
        names = gw.get_model_names()
        assert "default" in names
        assert "fast" in names
        assert len(names) == 2

    def test_get_model_list(self) -> None:
        gw = GatewayRouter(make_config())
        ml = gw.get_model_list()
        assert len(ml) == 3

    def test_register_model(self) -> None:
        gw = GatewayRouter(make_config())
        new_deploy = ModelDeployment(model_name="local", model="ollama/llama3")
        gw.register_model(new_deploy)
        assert "local" in gw.get_model_names()
        assert len(gw.get_model_list()) == 4

    def test_unregister_model(self) -> None:
        gw = GatewayRouter(make_config())
        removed = gw.unregister_model("fast")
        assert removed == 1
        assert "fast" not in gw.get_model_names()

    def test_missing_credential_and_unknown_alias_fail_explicitly(self) -> None:
        with pytest.raises(AuthenticationError, match="PRAXIS_MODEL_API_KEY"):
            GatewayRouter(make_config(), environ={})

        gw = GatewayRouter(make_config())
        with pytest.raises(GatewayError) as error:
            gw.resolve_deployment("missing")
        assert error.value.details == {"model_name": "missing"}

    def test_unregister_last_model_is_atomic(self) -> None:
        config = GatewayConfig(deployments=[ModelDeployment()])
        gw = GatewayRouter(config)

        with pytest.raises(GatewayError, match="最后一个"):
            gw.unregister_model("default")

        assert gw.get_model_names() == ["default"]
        assert gw.unregister_model("missing") == 0

    def test_usage_reservation_settlement_and_release_are_idempotent(self) -> None:
        gw = GatewayRouter(make_config(max_total_tokens=20, max_budget=1.0))
        gw.add_usage(cost=-1.0, tokens=-1)
        assert gw.total_spend == 0.0
        assert gw.total_tokens == 0

        with pytest.raises(ValueError, match="estimated_tokens"):
            gw.reserve_usage(estimated_tokens=-1, estimated_cost=0.0)
        with pytest.raises(BudgetExceededError, match="成本"):
            gw.reserve_usage(estimated_tokens=1, estimated_cost=2.0)

        reservation = gw.reserve_usage(estimated_tokens=3, estimated_cost=0.25)
        assert gw.settle_usage(
            reservation,
            actual_tokens=None,
            actual_cost=None,
        ) == (3, 0.25)
        assert gw.settle_usage(
            reservation,
            actual_tokens=10,
            actual_cost=0.5,
        ) == (0, 0.0)

        released = gw.reserve_usage(estimated_tokens=2, estimated_cost=0.1)
        gw.release_usage(released)
        gw.release_usage(released)
        assert gw.total_tokens == 3
        assert gw.total_spend == 0.25

    async def test_health_and_close_are_non_networking(self) -> None:
        gw = GatewayRouter(make_config())
        assert await gw.health() is True
        await gw.close()


class TestChat:
    """Task 4.2: chat / chat_stream 接口验证。"""

    def test_convert_response_basic(self) -> None:
        raw = make_raw_response(content="Hello world")
        resp = convert_response(raw)
        assert isinstance(resp, ModelResponse)
        assert resp.content == "Hello world"
        assert resp.usage.prompt_tokens == 10
        assert resp.usage.completion_tokens == 20
        assert resp.model == "gpt-4o"
        assert resp.finish_reason == "stop"

    def test_convert_response_with_tool_calls(self) -> None:
        tc = SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(name="read_file", arguments='{"path": "/tmp"}'),
        )
        raw = make_raw_response(content=None, tool_calls=[tc])
        resp = convert_response(raw)
        assert resp.tool_calls is not None
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].function.name == "read_file"

    def test_convert_stream_chunk(self) -> None:
        raw = make_raw_stream_chunk(content="partial")
        chunk = convert_stream_chunk(raw)
        assert isinstance(chunk, ModelResponseChunk)
        assert chunk.delta_content == "partial"

    def test_convert_response_with_reasoning(self) -> None:
        raw = make_raw_response(content="答案", reasoning_content="推理过程…")
        resp = convert_response(raw)
        assert resp.content == "答案"
        assert resp.reasoning_content == "推理过程…"

    def test_convert_stream_chunk_with_reasoning(self) -> None:
        raw = make_raw_stream_chunk(content=None, reasoning_content="思考中")
        chunk = convert_stream_chunk(raw)
        assert chunk.delta_content is None
        assert chunk.delta_reasoning_content == "思考中"

    def test_convert_response_without_reasoning(self) -> None:
        raw = make_raw_response(content="no-think")
        resp = convert_response(raw)
        assert resp.reasoning_content is None

    def test_convert_response_with_refusal(self) -> None:
        raw = make_raw_response(content=None, refusal="不能回答该请求")
        resp = convert_response(raw)
        assert resp.refusal == "不能回答该请求"
        assert resp.content is None

    def test_convert_response_with_token_details(self) -> None:
        raw = make_raw_response(
            content="ok",
            prompt_tokens=100,
            completion_tokens=80,
            reasoning_tokens=40,
            cached_tokens=60,
        )
        resp = convert_response(raw)
        assert resp.usage.reasoning_tokens == 40
        assert resp.usage.cached_prompt_tokens == 60

    def test_convert_response_with_system_fingerprint(self) -> None:
        raw = make_raw_response(content="ok", system_fingerprint="fp_abc123")
        resp = convert_response(raw)
        assert resp.system_fingerprint == "fp_abc123"

    def test_convert_stream_chunk_with_refusal(self) -> None:
        raw = make_raw_stream_chunk(content=None, refusal="部分拒答")
        chunk = convert_stream_chunk(raw)
        assert chunk.delta_refusal == "部分拒答"

    def test_build_usage_missing_details(self) -> None:
        """缺少 *_tokens_details 的老版本响应需优雅退化为 0。"""
        usage_data = SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        usage = build_usage(usage_data)
        assert usage.reasoning_tokens == 0
        assert usage.cached_prompt_tokens == 0

    async def test_chat_success(self) -> None:
        gw = GatewayRouter(make_config())
        raw = make_raw_response(content="answer")
        gw.router.acompletion = AsyncMock(return_value=raw)

        resp = await chat(gw, [{"role": "user", "content": "hi"}])
        assert resp.content == "answer"
        gw.router.acompletion.assert_called_once()

    async def test_tool_calls_bypass_litellm_mcp_proxy_bridge(self) -> None:
        gw = GatewayRouter(make_config())
        gw.router.acompletion = AsyncMock(return_value=make_raw_response())
        tools = [{"type": "function", "function": {"name": "echo"}}]

        await chat(gw, [{"role": "user", "content": "hi"}], tools=tools)

        kwargs = gw.router.acompletion.await_args.kwargs
        assert kwargs["tools"] == tools
        assert kwargs["_skip_mcp_handler"] is True

    async def test_chat_records_actual_tokens(self) -> None:
        gw = GatewayRouter(make_config())
        gw.router.acompletion = AsyncMock(return_value=make_raw_response(
            prompt_tokens=11,
            completion_tokens=7,
        ))
        with patch("praxis.gateway.chat.completion_cost", return_value=0.25):
            await chat(gw, [{"role": "user", "content": "hi"}])
        assert gw.total_tokens == 18
        assert gw.total_spend == 0.25

    async def test_token_limit_is_independent_from_cost_budget(self) -> None:
        gw = GatewayRouter(make_config(max_total_tokens=10, max_budget=None))
        gw.router.acompletion = AsyncMock(return_value=make_raw_response())
        with (
            patch("praxis.gateway.chat.get_token_count", return_value=11),
            pytest.raises(BudgetExceededError, match="Token"),
        ):
            await chat(gw, [{"role": "user", "content": "too large"}])
        gw.router.acompletion.assert_not_called()

    async def test_unknown_model_price_fails_closed_when_budget_enabled(self) -> None:
        gw = GatewayRouter(make_config(max_budget=1.0))
        gw.router.acompletion = AsyncMock(return_value=make_raw_response())
        with (
            patch("praxis.gateway.chat.get_token_count", return_value=2),
            patch("praxis.gateway.chat.estimate_input_cost", return_value=None),
            pytest.raises(BudgetExceededError, match="价格"),
        ):
            await chat(gw, [{"role": "user", "content": "hi"}])
        gw.router.acompletion.assert_not_called()

    async def test_gateway_enforces_shared_concurrency_limit(self) -> None:
        gw = GatewayRouter(make_config(max_concurrent_requests=1))
        active = 0
        maximum = 0

        async def completion(**completion_options: object):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(0.05)
            active -= 1
            return make_raw_response()

        gw.router.acompletion = AsyncMock(side_effect=completion)
        await asyncio.gather(
            chat(gw, [{"role": "user", "content": "one"}]),
            chat(gw, [{"role": "user", "content": "two"}]),
        )
        assert maximum == 1

    async def test_concurrent_budget_reservations_fail_closed(self) -> None:
        deployment = ModelDeployment(
            model_name="default",
            model="openai/glm-5.1-openai",
            default_max_output_tokens=4,
        )
        gw = GatewayRouter(GatewayConfig(
            deployments=[deployment],
            max_total_tokens=5,
            max_concurrent_requests=1,
        ))
        started = asyncio.Event()
        release = asyncio.Event()

        async def completion(**completion_options: object) -> SimpleNamespace:
            started.set()
            await release.wait()
            return make_raw_response(prompt_tokens=1, completion_tokens=4)

        gw.router.acompletion = AsyncMock(side_effect=completion)
        with patch("praxis.gateway.chat.get_token_count", return_value=1):
            first = asyncio.create_task(chat(
                gw,
                [{"role": "user", "content": "one"}],
                max_tokens=4,
            ))
            await started.wait()
            with pytest.raises(BudgetExceededError, match="Token"):
                await chat(
                    gw,
                    [{"role": "user", "content": "two"}],
                    max_tokens=4,
                )
            release.set()
            await first

        assert gw.total_tokens == 5
        assert gw.router.acompletion.await_count == 1

    async def test_chat_maps_exception(self) -> None:
        import litellm

        gw = GatewayRouter(make_config())
        gw.router.acompletion = AsyncMock(
            side_effect=litellm.AuthenticationError(
                message="Invalid key", llm_provider="openai", model="gpt-4o"
            )
        )
        with pytest.raises(AuthenticationError):
            await chat(gw, [{"role": "user", "content": "hi"}])

    async def test_chat_stream_yields_chunks(self) -> None:
        gw = GatewayRouter(make_config())

        async def mock_stream():
            yield make_raw_stream_chunk(content="Hello")
            yield make_raw_stream_chunk(content=" world")
            yield make_raw_stream_chunk(content=None, finish_reason="stop")

        gw.router.acompletion = AsyncMock(return_value=mock_stream())

        chunks: list[ModelResponseChunk] = []
        async for chunk in chat_stream(gw, [{"role": "user", "content": "hi"}]):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0].delta_content == "Hello"
        assert chunks[1].delta_content == " world"
        assert chunks[2].finish_reason == "stop"

    async def test_stream_suppresses_known_litellm_pydantic_warning(self) -> None:
        gw = GatewayRouter(make_config())

        async def warning_stream():
            for attribute in ("model_computed_fields", "model_fields"):
                warnings.warn(
                    f"Accessing the '{attribute}' attribute on the instance is deprecated. "
                    "Instead, you should access this attribute from the model class.",
                    PydanticDeprecatedSince211,
                    stacklevel=2,
                )
            yield make_raw_stream_chunk(content="ok")

        gw.router.acompletion = AsyncMock(return_value=warning_stream())

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            chunks = [
                chunk
                async for chunk in chat_stream(
                    gw,
                    [{"role": "user", "content": "hi"}],
                )
            ]

        assert [chunk.delta_content for chunk in chunks] == ["ok"]

    async def test_stream_does_not_retry_after_partial_output(self) -> None:
        gw = GatewayRouter(make_config(num_retries=3))

        async def broken_stream():
            yield make_raw_stream_chunk(content="partial")
            raise RuntimeError("connection lost")

        gw.router.acompletion = AsyncMock(return_value=broken_stream())
        chunks: list[ModelResponseChunk] = []
        with pytest.raises(GatewayError):
            async for chunk in chat_stream(gw, [{"role": "user", "content": "hi"}]):
                chunks.append(chunk)
        assert [chunk.delta_content for chunk in chunks] == ["partial"]
        gw.router.acompletion.assert_awaited_once()

    async def test_stream_cancellation_settles_budget_and_releases_slot(self) -> None:
        deployment = ModelDeployment(
            model_name="default",
            model="openai/glm-5.1-openai",
            default_max_output_tokens=9,
        )
        gw = GatewayRouter(GatewayConfig(
            deployments=[deployment],
            max_total_tokens=100,
            max_concurrent_requests=1,
        ))
        emitted = asyncio.Event()

        async def pending_stream():
            emitted.set()
            yield make_raw_stream_chunk(content="partial")
            await asyncio.Event().wait()

        gw.router.acompletion = AsyncMock(return_value=pending_stream())

        async def consume() -> None:
            async for chunk in chat_stream(  # noqa: B007 - public discard name
                gw,
                [{"role": "user", "content": "cancel"}],
                max_tokens=9,
            ):
                pass

        with patch("praxis.gateway.chat.get_token_count", return_value=1):
            task = asyncio.create_task(consume())
            await emitted.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert gw.total_tokens == 10
        async with asyncio.timeout(0.2):
            async with gw.request_slot():
                pass


class TestTasks:
    """Task 4.3: summarize / judge 便捷接口验证。"""

    async def test_summarize(self) -> None:
        gw = GatewayRouter(make_config())
        raw = make_raw_response(content="这是摘要内容")
        gw.router.acompletion = AsyncMock(return_value=raw)

        result = await summarize(gw, "一段很长的文本...", instruction="请摘要")
        assert result == "这是摘要内容"

    async def test_judge_valid_json(self) -> None:
        gw = GatewayRouter(make_config())
        judge_json = json.dumps({
            "verdict": True,
            "confidence": 0.95,
            "reasoning": "内容符合标准",
        })
        raw = make_raw_response(content=judge_json)
        gw.router.acompletion = AsyncMock(return_value=raw)

        result = await judge(gw, "是否正确", "待评估内容")
        assert isinstance(result, JudgeResult)
        assert result.verdict is True
        assert result.confidence == 0.95
        assert result.reasoning == "内容符合标准"

    async def test_judge_invalid_json_fallback(self) -> None:
        gw = GatewayRouter(make_config())
        raw = make_raw_response(content="这不是 JSON")
        gw.router.acompletion = AsyncMock(return_value=raw)

        result = await judge(gw, "是否正确", "待评估内容")
        assert result.verdict is False
        assert result.confidence == 0.0
        assert "无法解析" in result.reasoning


class TestMetering:
    """Task 4.4: Token 计量验证。"""

    @patch("litellm.token_counter", return_value=42)
    def test_get_token_count(self, mock_tc: MagicMock) -> None:
        count = get_token_count([{"role": "user", "content": "hello"}], model="gpt-4o")
        assert count == 42
        mock_tc.assert_called_once()

    @patch("litellm.get_max_tokens", return_value=128000)
    def test_get_max_tokens(self, mock_mt: MagicMock) -> None:
        max_t = get_max_tokens("gpt-4o")
        assert max_t == 128000

    @patch("litellm.cost_per_token", return_value=(0.0, 0.0))
    def test_zero_price_is_unknown_without_explicit_deployment_price(
        self,
        mock_cost: MagicMock,
    ) -> None:
        assert estimate_input_cost(100, "custom/model") is None


class TestResilience:
    """Task 4.5: 异常标准化验证。"""

    def test_exception_map_covers_key_types(self) -> None:
        assert len(EXCEPTION_MAP) >= 7

    def test_map_authentication_error(self) -> None:
        import litellm

        exc = litellm.AuthenticationError(
            message="bad key", llm_provider="openai", model="gpt-4o"
        )
        mapped = map_litellm_exception(exc)
        assert isinstance(mapped, AuthenticationError)
        assert "bad key" in mapped.message

    def test_map_rate_limit_error(self) -> None:
        import litellm

        exc = litellm.RateLimitError(
            message="rate limited", llm_provider="openai", model="gpt-4o"
        )
        mapped = map_litellm_exception(exc)
        assert isinstance(mapped, RateLimitError)

    def test_map_timeout_error(self) -> None:
        import litellm

        exc = litellm.Timeout(
            message="timeout", llm_provider="openai", model="gpt-4o"
        )
        mapped = map_litellm_exception(exc)
        assert isinstance(mapped, GatewayTimeoutError)

    def test_map_unknown_fallback(self) -> None:
        exc = RuntimeError("unknown error")
        mapped = map_litellm_exception(exc)
        assert isinstance(mapped, GatewayError)
        assert mapped.details["original_type"] == "RuntimeError"


class TestCallbacks:
    """Task 4.5: 可观测性回调验证。"""

    def test_log_success_event(self) -> None:
        cb = TelemetryCallback()
        from datetime import datetime, timedelta
        start = datetime.now()
        end = start + timedelta(milliseconds=150)
        kwargs = {"model": "gpt-4o"}
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=50)
        response_obj = SimpleNamespace(usage=usage)

        cb.log_success_event(kwargs, response_obj, start, end)

    def test_log_failure_event(self) -> None:
        cb = TelemetryCallback()
        kwargs = {"model": "gpt-4o", "exception": ValueError("test")}
        cb.log_failure_event(kwargs, None, None, None)
