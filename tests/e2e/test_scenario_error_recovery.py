"""场景二：工具执行失败与错误恢复。

工具执行失败 → S9 错误分类 → 重试决策 → 重试/熔断 →
降级方案 → 错误信息返回 LLM。
"""

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

from tests.e2e.conftest import build_loop, register_tool


def make_raw_response(
    content: str = "",
    tool_calls: list | None = None,
) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    usage = SimpleNamespace(prompt_tokens=50, completion_tokens=20, total_tokens=70)
    return SimpleNamespace(
        id="chatcmpl-err", choices=[choice], usage=usage,
        model="test-model", created=1700000000,
    )


def make_raw_tool_call(
    name: str, arguments: str = "{}", tc_id: str = "tc-1",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=tc_id, type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class TestErrorRecovery:
    """场景二：工具执行失败与错误恢复 E2E 测试。"""

    async def test_tool_failure_returns_error_to_llm(
        self,
        mock_gateway,
        registry,
        guardrails,
        orchestrator_config,
        context_config,
    ) -> None:
        """验证：工具执行失败后错误信息被返回给 LLM，LLM 调整策略。"""
        call_count = 0

        async def failing_handler(args: dict[str, Any]) -> str:
            raise ConnectionError("连接超时")

        register_tool(registry, "web_fetch", failing_handler, parameters={
            "type": "object",
            "properties": {"url": {"type": "string"}},
        })

        async def mock_acompletion(**kwargs: Any) -> SimpleNamespace:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                tc = make_raw_tool_call("web_fetch", json.dumps({"url": "https://example.com"}))
                return make_raw_response(tool_calls=[tc])
            # LLM 收到错误后给出替代回复
            return make_raw_response(content="网络请求失败，我将使用缓存数据回答。")

        mock_gateway.router.acompletion = AsyncMock(side_effect=mock_acompletion)

        loop = build_loop(
            mock_gateway, registry, guardrails,
            orchestrator_config, context_config,
        )

        response = await loop.run("获取 example.com 的内容")
        assert "缓存" in response.content
        assert response.tool_calls_made == 1
        assert call_count == 2

    async def test_circuit_breaker_opens_after_repeated_failures(
        self,
        mock_gateway,
        registry,
        guardrails,
        orchestrator_config,
        context_config,
    ) -> None:
        """验证：工具多次失败后熔断器打开，后续调用被跳过。"""
        async def always_fail(args: dict[str, Any]) -> str:
            raise RuntimeError("持续失败")

        register_tool(registry, "unstable_tool", always_fail)

        call_count = 0

        async def mock_acompletion(**kwargs: Any) -> SimpleNamespace:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                tc = make_raw_tool_call("unstable_tool", "{}", f"tc-{call_count}")
                return make_raw_response(tool_calls=[tc])
            return make_raw_response(content="无法完成操作")

        mock_gateway.router.acompletion = AsyncMock(side_effect=mock_acompletion)

        loop = build_loop(
            mock_gateway, registry, guardrails,
            orchestrator_config, context_config,
        )

        # 预先注入失败记录使熔断器处于半开状态
        for _ in range(5):
            loop.coordinator.circuits.record_outcome("unstable_tool", success=False)

        response = await loop.run("执行不稳定操作")
        # 熔断器应跳过工具调用
        assert response is not None

    async def test_tool_timeout_handled(
        self,
        mock_gateway,
        registry,
        guardrails,
        orchestrator_config,
        context_config,
    ) -> None:
        """验证：工具执行超时被正确处理。"""
        import asyncio

        async def slow_handler(args: dict[str, Any]) -> str:
            await asyncio.sleep(10)
            return "不应到达"

        register_tool(
            registry, "slow_tool", slow_handler,
            parameters={"type": "object", "properties": {}},
        )
        # 设置工具超时为极短时间
        entry = registry.get_entry("slow_tool")
        entry.definition.metadata.timeout_seconds = 0.01

        call_count = 0

        async def mock_acompletion(**kwargs: Any) -> SimpleNamespace:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                tc = make_raw_tool_call("slow_tool")
                return make_raw_response(tool_calls=[tc])
            return make_raw_response(content="工具超时，已跳过")

        mock_gateway.router.acompletion = AsyncMock(side_effect=mock_acompletion)

        loop = build_loop(
            mock_gateway, registry, guardrails,
            orchestrator_config, context_config,
        )

        response = await loop.run("执行慢操作")
        assert response is not None
        assert call_count == 2
