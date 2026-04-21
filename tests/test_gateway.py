"""S4 模型网关验证测试。"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from praxis.config.schemas import GatewayConfig
from praxis.exceptions import (
    AuthenticationError,
    BudgetExceededError,
    GatewayError,
    GatewayTimeoutError,
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
from praxis.gateway.metering import get_max_tokens, get_token_count
from praxis.gateway.resilience import EXCEPTION_MAP, map_litellm_exception
from praxis.gateway.router import GatewayRouter
from praxis.gateway.tasks import judge, summarize
from praxis.models.gateway import JudgeResult
from praxis.models.responses import ModelResponse, ModelResponseChunk


SAMPLE_MODEL_LIST = [
    {
        "model_name": "default",
        "litellm_params": {"model": "openai/gpt-4o", "api_key": "test-key-1"},
    },
    {
        "model_name": "default",
        "litellm_params": {"model": "anthropic/claude-sonnet-4-20250514", "api_key": "test-key-2"},
    },
    {
        "model_name": "fast",
        "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "test-key-1"},
    },
]


def make_config(**overrides: object) -> GatewayConfig:
    defaults = {"model_list": SAMPLE_MODEL_LIST}
    defaults.update(overrides)
    return GatewayConfig(**defaults)


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

    @patch("praxis.gateway.router.register_callbacks")
    def test_init_with_valid_config(self, mock_cb: MagicMock) -> None:
        config = make_config()
        gw = GatewayRouter(config)
        assert gw.router is not None
        assert gw.config is config
        mock_cb.assert_called_once()

    @patch("praxis.gateway.router.register_callbacks")
    def test_empty_model_list_raises(self, mock_cb: MagicMock) -> None:
        config = GatewayConfig(model_list=[])
        with pytest.raises(GatewayError, match="model_list 为空"):
            GatewayRouter(config)

    @patch("praxis.gateway.router.register_callbacks")
    def test_get_model_names(self, mock_cb: MagicMock) -> None:
        gw = GatewayRouter(make_config())
        names = gw.get_model_names()
        assert "default" in names
        assert "fast" in names
        assert len(names) == 2

    @patch("praxis.gateway.router.register_callbacks")
    def test_get_model_list(self, mock_cb: MagicMock) -> None:
        gw = GatewayRouter(make_config())
        ml = gw.get_model_list()
        assert len(ml) == 3

    @patch("praxis.gateway.router.register_callbacks")
    def test_register_model(self, mock_cb: MagicMock) -> None:
        gw = GatewayRouter(make_config())
        new_deploy = {
            "model_name": "local",
            "litellm_params": {"model": "ollama/llama3"},
        }
        gw.register_model(new_deploy)
        assert "local" in gw.get_model_names()
        assert len(gw.get_model_list()) == 4

    @patch("praxis.gateway.router.register_callbacks")
    def test_register_model_no_name_raises(self, mock_cb: MagicMock) -> None:
        gw = GatewayRouter(make_config())
        with pytest.raises(GatewayError, match="model_name"):
            gw.register_model({"litellm_params": {"model": "test"}})

    @patch("praxis.gateway.router.register_callbacks")
    def test_unregister_model(self, mock_cb: MagicMock) -> None:
        gw = GatewayRouter(make_config())
        removed = gw.unregister_model("fast")
        assert removed == 1
        assert "fast" not in gw.get_model_names()


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

    @patch("praxis.gateway.router.register_callbacks")
    async def test_chat_success(self, mock_cb: MagicMock) -> None:
        gw = GatewayRouter(make_config())
        raw = make_raw_response(content="answer")
        gw._router.acompletion = AsyncMock(return_value=raw)

        resp = await chat(gw, [{"role": "user", "content": "hi"}])
        assert resp.content == "answer"
        gw._router.acompletion.assert_called_once()

    @patch("praxis.gateway.router.register_callbacks")
    async def test_chat_maps_exception(self, mock_cb: MagicMock) -> None:
        import litellm

        gw = GatewayRouter(make_config())
        gw._router.acompletion = AsyncMock(
            side_effect=litellm.AuthenticationError(
                message="Invalid key", llm_provider="openai", model="gpt-4o"
            )
        )
        with pytest.raises(AuthenticationError):
            await chat(gw, [{"role": "user", "content": "hi"}])

    @patch("praxis.gateway.router.register_callbacks")
    async def test_chat_stream_yields_chunks(self, mock_cb: MagicMock) -> None:
        gw = GatewayRouter(make_config())

        async def mock_stream():
            yield make_raw_stream_chunk(content="Hello")
            yield make_raw_stream_chunk(content=" world")
            yield make_raw_stream_chunk(content=None, finish_reason="stop")

        gw._router.acompletion = AsyncMock(return_value=mock_stream())

        chunks: list[ModelResponseChunk] = []
        async for chunk in chat_stream(gw, [{"role": "user", "content": "hi"}]):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0].delta_content == "Hello"
        assert chunks[1].delta_content == " world"
        assert chunks[2].finish_reason == "stop"


class TestTasks:
    """Task 4.3: summarize / judge 便捷接口验证。"""

    @patch("praxis.gateway.router.register_callbacks")
    async def test_summarize(self, mock_cb: MagicMock) -> None:
        gw = GatewayRouter(make_config())
        raw = make_raw_response(content="这是摘要内容")
        gw._router.acompletion = AsyncMock(return_value=raw)

        result = await summarize(gw, "一段很长的文本...", instruction="请摘要")
        assert result == "这是摘要内容"

    @patch("praxis.gateway.router.register_callbacks")
    async def test_judge_valid_json(self, mock_cb: MagicMock) -> None:
        gw = GatewayRouter(make_config())
        judge_json = json.dumps({
            "verdict": True,
            "confidence": 0.95,
            "reasoning": "内容符合标准",
        })
        raw = make_raw_response(content=judge_json)
        gw._router.acompletion = AsyncMock(return_value=raw)

        result = await judge(gw, "是否正确", "待评估内容")
        assert isinstance(result, JudgeResult)
        assert result.verdict is True
        assert result.confidence == 0.95
        assert result.reasoning == "内容符合标准"

    @patch("praxis.gateway.router.register_callbacks")
    async def test_judge_invalid_json_fallback(self, mock_cb: MagicMock) -> None:
        gw = GatewayRouter(make_config())
        raw = make_raw_response(content="这不是 JSON")
        gw._router.acompletion = AsyncMock(return_value=raw)

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
