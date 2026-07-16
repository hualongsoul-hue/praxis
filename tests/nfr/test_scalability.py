"""非功能需求验证——可扩展性。

PRD § 9.4:
- 100+ 工具注册与查找
- 1000+ 轮次对话
- 10000+ 记忆条目
- 大规模规则集护栏
"""

import time
from typing import Any
from unittest.mock import patch

import pytest

from praxis.config.schemas import ContextConfig
from praxis.context.assembler import PromptAssembler
from praxis.guardrails.engine import GuardrailEngine
from praxis.guardrails.permissions import PermissionManager
from praxis.guardrails.rules import GuardrailRule, RuleEngine, RuleTarget
from praxis.models.context import TurnContext
from praxis.models.guardrails import VerdictType
from praxis.models.memory import WorkingMemory, WorkingMemoryMessage
from praxis.models.tools import ToolDefinition, ToolMetadata
from praxis.recovery.circuit_breaker import CircuitBreakerRegistry
from praxis.tools.registry import ToolRegistry


@pytest.fixture(autouse=True)
def mock_litellm():
    with patch("praxis.context.assembler.get_max_tokens", return_value=128000), \
         patch("praxis.context.assembler.get_token_count", return_value=100):
        yield


class TestToolScalability:
    """100+ 工具注册与查找可扩展性。"""

    def test_register_100_tools(self) -> None:
        """验证：注册 100+ 工具后系统正常运行。"""
        registry = ToolRegistry()

        async def handler(args: dict[str, Any]) -> str:
            return "ok"

        for i in range(150):
            registry.register(
                ToolDefinition(
                    name=f"tool_{i:03d}",
                    description=f"工具 {i}",
                    parameters={"type": "object", "properties": {}},
                    metadata=ToolMetadata(category="file_ops"),
                ),
                handler,
            )

        assert len(registry.list_tools()) == 150
        assert registry.has_tool("tool_000")
        assert registry.has_tool("tool_149")

    def test_tool_lookup_performance_at_scale(self) -> None:
        """验证：100+ 工具时查找性能 <1ms。"""
        registry = ToolRegistry()

        async def handler(args: dict[str, Any]) -> str:
            return "ok"

        for i in range(200):
            registry.register(
                ToolDefinition(
                    name=f"tool_{i:03d}",
                    description=f"工具 {i}",
                    parameters={"type": "object", "properties": {}},
                ),
                handler,
            )

        # 查找性能
        start = time.perf_counter()
        for lookup_iteration in range(1000):  # noqa: B007 - public discard name
            registry.get_entry("tool_100")
        elapsed = (time.perf_counter() - start) / 1000 * 1000  # ms

        assert elapsed < 1, f"工具查找延迟 {elapsed:.3f}ms 超过 1ms"

    def test_schema_export_at_scale(self) -> None:
        """验证：100+ 工具的 Schema 批量导出。"""
        registry = ToolRegistry()

        async def handler(args: dict[str, Any]) -> str:
            return "ok"

        for i in range(100):
            registry.register(
                ToolDefinition(
                    name=f"tool_{i:03d}",
                    description=f"工具 {i} 的详细描述",
                    parameters={
                        "type": "object",
                        "properties": {
                            "arg1": {"type": "string", "description": f"参数 {i}"},
                        },
                    },
                ),
                handler,
            )

        start = time.perf_counter()
        schemas = registry.get_tool_schemas()
        elapsed = (time.perf_counter() - start) * 1000

        assert len(schemas) == 100
        assert elapsed < 50, f"Schema 导出 {elapsed:.2f}ms 超过 50ms"


class TestConversationScalability:
    """1000+ 轮次对话可扩展性。"""

    def test_large_conversation_history(self) -> None:
        """验证：1000+ 消息的对话历史管理。"""
        assembler = PromptAssembler(ContextConfig(), model="test-model")

        for i in range(1000):
            assembler.conversation_history.append({
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"消息 {i}",
            })

        assert len(assembler.conversation_history) == 1000
        assert assembler.total_turns == 500  # 500 个 assistant 消息

    def test_prompt_assembly_with_large_history(self) -> None:
        """验证：大量历史消息下 Prompt 组装仍正常。"""
        assembler = PromptAssembler(ContextConfig(), model="test-model")

        for i in range(500):
            assembler.conversation_history.append({
                "role": "assistant",
                "content": f"回复 {i}: 这是一段内容。",
            })

        turn = TurnContext(user_message="新消息")
        prompt = assembler.assemble_prompt(turn)

        assert len(prompt.messages) > 500
        assert prompt.messages[-1]["content"] == "新消息"


class TestMemoryScalability:
    """10000+ 记忆条目可扩展性。"""

    def test_working_memory_10000_messages(self) -> None:
        """验证：工作记忆支持 10000+ 消息追加。"""
        wm = WorkingMemory(session_id="scale-test", max_messages=20000)

        start = time.perf_counter()
        for i in range(10000):
            wm.append(WorkingMemoryMessage(
                role="user" if i % 2 == 0 else "assistant",
                content=f"消息 {i}",
            ))
        elapsed = (time.perf_counter() - start) * 1000

        assert len(wm.messages) == 10000
        assert elapsed < 5000, f"10000 条消息追加耗时 {elapsed:.0f}ms 超过 5s"

    def test_working_memory_export_at_scale(self) -> None:
        """验证：大规模工作记忆导出性能。"""
        wm = WorkingMemory(session_id="scale-test", max_messages=10000)
        for i in range(5000):
            wm.append(WorkingMemoryMessage(role="user", content=f"msg {i}"))

        start = time.perf_counter()
        messages = wm.get_recent(limit=100)
        elapsed = (time.perf_counter() - start) * 1000

        assert len(messages) == 100
        assert elapsed < 100, f"记忆导出 {elapsed:.2f}ms 超过 100ms"


class TestGuardrailScalability:
    """大规模规则集护栏可扩展性。"""

    async def test_100_rules_evaluation(self) -> None:
        """验证：100 条规则下护栏裁决性能。"""
        rule_engine = RuleEngine()
        for i in range(100):
            rule_engine.register_rule(GuardrailRule(
                name=f"rule_{i:03d}",
                description=f"规则 {i}",
                target=RuleTarget.INPUT,
                patterns=[f"forbidden_pattern_{i}_\\d+"],
                verdict=VerdictType.BLOCK,
            ))

        engine = GuardrailEngine(rule_engine, PermissionManager())

        start = time.perf_counter()
        for guardrail_iteration in range(100):  # noqa: B007 - public discard name
            await engine.check_input("正常消息，不匹配任何规则模式")
        elapsed = (time.perf_counter() - start) / 100 * 1000

        assert elapsed < 10, f"100 规则护栏裁决 {elapsed:.2f}ms 超过 10ms"

    async def test_rule_matching_at_scale(self) -> None:
        """验证：大量规则中精确匹配目标规则。"""
        rule_engine = RuleEngine()
        for i in range(100):
            rule_engine.register_rule(GuardrailRule(
                name=f"rule_{i:03d}",
                description=f"规则 {i}",
                target=RuleTarget.INPUT,
                patterns=[f"exact_match_token_{i:03d}"],
                verdict=VerdictType.BLOCK,
            ))

        engine = GuardrailEngine(rule_engine, PermissionManager())

        # 匹配第 50 条规则（使用精确 token 避免子串匹配干扰）
        verdict = await engine.check_input("消息包含 exact_match_token_050")
        assert verdict.verdict == VerdictType.BLOCK
        assert "rule_050" in verdict.rule_name


class TestCircuitBreakerScalability:
    """大规模熔断器可扩展性。"""

    def test_100_tool_circuit_breakers(self) -> None:
        """验证：100+ 工具各自独立的熔断器。"""
        registry = CircuitBreakerRegistry(failure_threshold=3)

        # 为 100 个工具创建熔断器
        for i in range(100):
            registry.get(f"tool_{i:03d}")

        assert len(registry.breakers) == 100

        # 单个工具熔断不影响其他
        for failure_index in range(4):  # noqa: B007 - public discard name
            registry.record_outcome("tool_050", success=False)

        from praxis.models.recovery import CircuitState
        assert registry.check_circuit("tool_050") == CircuitState.OPEN
        assert registry.check_circuit("tool_049") == CircuitState.CLOSED
        assert registry.check_circuit("tool_051") == CircuitState.CLOSED
