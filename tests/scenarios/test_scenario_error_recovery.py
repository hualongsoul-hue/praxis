"""场景二：工具执行失败与错误恢复。

工具执行失败 → S9 错误分类 → 重试决策 → 重试/熔断 →
降级方案 → 错误信息返回 LLM。
"""

from typing import Any
from unittest.mock import AsyncMock

from tests.scenarios.conftest import (
    build_loop,
    make_model_response,
    make_tool_call,
    register_tool,
    resolved_text_input,
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

        async def mock_completion(*args: Any, **kwargs: Any):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_model_response(
                    tool_calls=[make_tool_call(
                        "web_fetch",
                        '{"url": "https://example.com"}',
                    )],
                    finish_reason="tool_calls",
                )
            # LLM 收到错误后给出替代回复
            return make_model_response(content="网络请求失败，我将使用缓存数据回答。")

        mock_gateway.complete = AsyncMock(side_effect=mock_completion)

        loop = build_loop(
            mock_gateway, registry, guardrails,
            orchestrator_config, context_config,
        )

        response = await loop.run(resolved_text_input("获取 example.com 的内容"))
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
        execution_count = 0

        async def always_fail(args: dict[str, Any]) -> str:
            nonlocal execution_count
            execution_count += 1
            raise RuntimeError("持续失败")

        register_tool(registry, "unstable_tool", always_fail)

        call_count = 0

        async def mock_completion(*args: Any, **kwargs: Any):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return make_model_response(
                    tool_calls=[make_tool_call(
                        "unstable_tool", "{}", f"tc-{call_count}"
                    )],
                    finish_reason="tool_calls",
                )
            return make_model_response(content="无法完成操作")

        mock_gateway.complete = AsyncMock(side_effect=mock_completion)

        loop = build_loop(
            mock_gateway, registry, guardrails,
            orchestrator_config, context_config,
        )

        # 预先注入失败记录使熔断器处于半开状态
        for failure_index in range(5):  # noqa: B007 - public discard name
            loop.coordinator.circuits.record_outcome("unstable_tool", success=False)

        response = await loop.run(resolved_text_input("执行不稳定操作"))

        assert response.content == "无法完成操作"
        assert execution_count == 0
        assert call_count == 3
        assert any(
            event.event_type == "tool_call_end" and event.data.skipped
            for event in response.events
        )

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
        entry.definition = entry.definition.model_copy(update={
            "metadata": entry.definition.metadata.model_copy(
                update={"timeout_seconds": 0.01}
            )
        })

        call_count = 0

        async def mock_completion(*args: Any, **kwargs: Any):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_model_response(
                    tool_calls=[make_tool_call("slow_tool")],
                    finish_reason="tool_calls",
                )
            return make_model_response(content="工具超时，已跳过")

        mock_gateway.complete = AsyncMock(side_effect=mock_completion)

        loop = build_loop(
            mock_gateway, registry, guardrails,
            orchestrator_config, context_config,
        )

        response = await loop.run(resolved_text_input("执行慢操作"))

        assert response.content == "工具超时，已跳过"
        assert call_count == 2
        timeout_error_types = [
            event.data.error_type
            for event in response.events
            if event.event_type == "tool_call_end"
        ]
        assert any(
            error_type.endswith("ToolTimeoutError")
            for error_type in timeout_error_types
        ), timeout_error_types
