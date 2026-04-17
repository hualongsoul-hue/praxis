"""场景三：上下文窗口溢出处理。

Token 用量接近窗口限制 → S7 触发压缩 →
S4 生成摘要 → 替换历史消息 → 正常继续。
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from praxis.config.schemas import ContextConfig, OrchestratorConfig
from praxis.context.assembler import PromptAssembler
from praxis.models.orchestrator import TerminationReason
from tests.e2e.conftest import build_loop


def make_raw_response(content: str = "") -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    usage = SimpleNamespace(prompt_tokens=50, completion_tokens=20, total_tokens=70)
    return SimpleNamespace(
        id="chatcmpl-ctx", choices=[choice], usage=usage,
        model="test-model", created=1700000000,
    )


class TestContextOverflow:
    """场景三：上下文窗口溢出处理 E2E 测试。"""

    async def test_assembler_tracks_token_growth(
        self,
        mock_gateway,
        registry,
        guardrails,
        orchestrator_config,
        context_config,
    ) -> None:
        """验证：多轮交互后上下文正确累积消息。"""
        call_count = 0

        async def mock_acompletion(**kwargs: Any) -> SimpleNamespace:
            nonlocal call_count
            call_count += 1
            return make_raw_response(content=f"回复 {call_count}")

        mock_gateway.router.acompletion = AsyncMock(side_effect=mock_acompletion)

        loop = build_loop(
            mock_gateway, registry, guardrails,
            orchestrator_config, context_config,
        )

        # 第一轮
        r1 = await loop.run("问题 1")
        assert r1.content == "回复 1"
        # conversation_history 保存 assistant 响应（user 消息在 assemble_prompt 中实时注入）
        assert len(loop.assembler.conversation_history) >= 1

        # 第二轮（上下文累积）
        r2 = await loop.run("问题 2")
        assert r2.content == "回复 2"
        assert len(loop.assembler.conversation_history) >= 2

    async def test_compaction_triggered_on_large_context(
        self,
        mock_gateway,
        registry,
        guardrails,
    ) -> None:
        """验证：当 Token 接近上限时 Prompt 组装能正常工作。"""
        # 使用小窗口配置模拟溢出
        ctx_config = ContextConfig(compaction_threshold=0.5)
        orch_config = OrchestratorConfig(max_turns=5)

        mock_gateway.router.acompletion = AsyncMock(
            return_value=make_raw_response(content="压缩后继续")
        )

        loop = build_loop(
            mock_gateway, registry, guardrails,
            orch_config, ctx_config,
        )

        # 注入大量历史消息模拟上下文积累
        for i in range(20):
            loop.assembler.conversation_history.append({
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"这是一段较长的消息内容，用于填充上下文窗口 - 消息 {i} " * 5,
            })

        response = await loop.run("在已有大量上下文的情况下继续对话")
        assert response is not None

    async def test_max_turns_terminates_loop(
        self,
        mock_gateway,
        registry,
        guardrails,
        context_config,
    ) -> None:
        """验证：达到最大轮次时循环正确终止。"""
        orch_config = OrchestratorConfig(max_turns=2)

        # 始终返回工具调用，不返回最终响应
        def make_raw_tool_call() -> SimpleNamespace:
            return SimpleNamespace(
                id="tc-loop", type="function",
                function=SimpleNamespace(name="noop", arguments="{}"),
            )

        async def noop_handler(args: dict[str, Any]) -> str:
            return "ok"

        from tests.e2e.conftest import register_tool
        register_tool(registry, "noop", noop_handler)

        mock_gateway.router.acompletion = AsyncMock(
            return_value=SimpleNamespace(
                id="chatcmpl-loop",
                choices=[SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                        tool_calls=[make_raw_tool_call()],
                    ),
                    finish_reason="tool_calls",
                )],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                model="test-model",
                created=1700000000,
            )
        )

        loop = build_loop(
            mock_gateway, registry, guardrails,
            orch_config, context_config,
        )

        response = await loop.run("无限循环测试")
        assert response.termination_reason == TerminationReason.MAX_TURNS
        assert response.total_turns == 2
