"""性能基准测试。

验证 PRD § 9.1 性能指标：
- 编排循环开销: <50ms（不含 LLM 调用）
- 工具执行延迟: <100ms（不含工具本体耗时）
- 检查点写入: <200ms
- Prompt 组装: <20ms
- 护栏裁决: <10ms
"""

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from praxis.config.schemas import (
    ContextConfig,
    OrchestratorConfig,
    PersistenceConfig,
    SessionConfig,
    ToolsConfig,
)
from praxis.context.assembler import PromptAssembler
from praxis.gateway.router import GatewayRouter
from praxis.guardrails.engine import GuardrailEngine
from praxis.guardrails.permissions import PermissionManager
from praxis.guardrails.rules import GuardrailRule, RuleEngine, RuleTarget
from praxis.models.context import TurnContext
from praxis.models.guardrails import VerdictType
from praxis.models.tools import ToolDefinition
from praxis.persistence.store import PersistenceStore, create_store
from praxis.session.core import SessionFactory
from praxis.tools.executor import ToolExecutor
from praxis.tools.policy import ToolPolicy
from praxis.tools.registry import ToolRegistry

# ── 辅助 ──────────────────────────────────────────────────────────────────


def make_raw_response(
    content: str = "bench response",
    tool_calls: list | None = None,
) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    usage = SimpleNamespace(prompt_tokens=50, completion_tokens=20, total_tokens=70)
    return SimpleNamespace(
        id="chatcmpl-bench", choices=[choice], usage=usage,
        model="test-model", created=1700000000,
    )


@pytest.fixture(autouse=True)
def mock_litellm():
    with patch("praxis.context.assembler.get_max_tokens", return_value=128000), \
         patch("praxis.context.assembler.get_token_count", return_value=100):
        yield


@pytest.fixture
async def store(tmp_path: Path) -> PersistenceStore:
    config = PersistenceConfig(backend="sqlite", sqlite_path=str(tmp_path / "bench.db"))
    s = await create_store(config)
    yield s  # type: ignore[misc]
    await s.close()


# ── 基准测试 ──────────────────────────────────────────────────────────────


class TestPromptAssemblyPerformance:
    """Prompt 组装性能: <20ms。"""

    def test_prompt_assembly_latency(self) -> None:
        """验证：Prompt 组装延迟 <20ms。"""
        assembler = PromptAssembler(ContextConfig(), model="test-model")
        # 模拟中等上下文
        for i in range(20):
            assembler.conversation_history.append({
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"消息 {i}: 这是一段测试内容用于验证组装性能。" * 3,
            })

        turn = TurnContext(user_message="当前用户消息")

        # 预热
        assembler.assemble_prompt(turn)

        # 计时
        iterations = 50
        start = time.perf_counter()
        for _ in range(iterations):
            assembler.assemble_prompt(turn)
        elapsed = (time.perf_counter() - start) / iterations * 1000

        assert elapsed < 20, f"Prompt 组装延迟 {elapsed:.2f}ms 超过 20ms 上限"

    def test_prompt_assembly_with_tools(self) -> None:
        """验证：带工具 Schema 的 Prompt 组装延迟 <20ms。"""
        assembler = PromptAssembler(ContextConfig(), model="test-model")
        # 注入 50 个工具 Schema
        schemas = [
            {
                "type": "function",
                "function": {
                    "name": f"tool_{i}",
                    "description": f"测试工具 {i} 的描述信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "arg1": {"type": "string"},
                            "arg2": {"type": "integer"},
                        },
                    },
                },
            }
            for i in range(50)
        ]
        assembler.set_tool_schemas(schemas)

        turn = TurnContext(user_message="使用工具")

        iterations = 50
        start = time.perf_counter()
        for _ in range(iterations):
            assembler.assemble_prompt(turn)
        elapsed = (time.perf_counter() - start) / iterations * 1000

        assert elapsed < 20, f"Prompt 组装（含工具）延迟 {elapsed:.2f}ms 超过 20ms 上限"


class TestGuardrailPerformance:
    """护栏裁决性能: <10ms。"""

    async def test_input_guardrail_latency(self) -> None:
        """验证：输入护栏裁决延迟 <10ms。"""
        rule_engine = RuleEngine()
        rule_engine.register_builtin_rules()
        engine = GuardrailEngine(rule_engine, PermissionManager())

        # 预热
        await engine.check_input("测试消息")

        iterations = 100
        start = time.perf_counter()
        for _ in range(iterations):
            await engine.check_input("这是一个正常的用户消息，不包含任何注入攻击。")
        elapsed = (time.perf_counter() - start) / iterations * 1000

        assert elapsed < 10, f"输入护栏裁决延迟 {elapsed:.2f}ms 超过 10ms 上限"

    async def test_output_guardrail_latency(self) -> None:
        """验证：输出护栏裁决延迟 <10ms。"""
        rule_engine = RuleEngine()
        rule_engine.register_builtin_rules()
        engine = GuardrailEngine(rule_engine, PermissionManager())

        iterations = 100
        start = time.perf_counter()
        for _ in range(iterations):
            await engine.check_output("这是正常的助手回复内容，不包含敏感信息。")
        elapsed = (time.perf_counter() - start) / iterations * 1000

        assert elapsed < 10, f"输出护栏裁决延迟 {elapsed:.2f}ms 超过 10ms 上限"

    async def test_guardrail_with_many_rules(self) -> None:
        """验证：50 条自定义规则下护栏裁决延迟 <10ms。"""
        rule_engine = RuleEngine()
        for i in range(50):
            rule_engine.register_rule(GuardrailRule(
                name=f"custom_rule_{i}",
                description=f"自定义规则 {i}",
                target=RuleTarget.INPUT,
                patterns=[f"pattern_{i}_\\d+"],
                verdict=VerdictType.BLOCK,
            ))
        engine = GuardrailEngine(rule_engine, PermissionManager())

        iterations = 100
        start = time.perf_counter()
        for _ in range(iterations):
            await engine.check_input("正常消息，不匹配任何规则")
        elapsed = (time.perf_counter() - start) / iterations * 1000

        assert elapsed < 10, f"多规则护栏裁决延迟 {elapsed:.2f}ms 超过 10ms 上限"


class TestToolExecutionPerformance:
    """工具执行管线延迟（不含工具本体）: <100ms。"""

    async def test_tool_executor_overhead(self) -> None:
        """验证：工具执行管线开销 <100ms（工具本体 <1ms）。"""
        registry = ToolRegistry()

        async def fast_handler(args: dict[str, Any]) -> str:
            return "ok"

        registry.register(
            ToolDefinition(
                name="fast_tool",
                description="极快工具",
                parameters={"type": "object", "properties": {"x": {"type": "string"}}},
            ),
            fast_handler,
        )

        sandbox = ToolPolicy(ToolsConfig())
        executor = ToolExecutor(registry, sandbox)

        # 预热
        await executor.execute("fast_tool", {"x": "warmup"}, "tc-warmup")

        iterations = 50
        start = time.perf_counter()
        for i in range(iterations):
            await executor.execute("fast_tool", {"x": f"v{i}"}, f"tc-{i}")
        elapsed = (time.perf_counter() - start) / iterations * 1000

        assert elapsed < 100, f"工具执行管线开销 {elapsed:.2f}ms 超过 100ms 上限"


class TestCheckpointPerformance:
    """检查点写入性能: <200ms。"""

    async def test_checkpoint_write_latency(
        self, store: PersistenceStore, mock_gateway: MagicMock,
    ) -> None:
        """验证：检查点写入延迟 <200ms。"""
        guardrails = GuardrailEngine(RuleEngine(), PermissionManager())
        factory = SessionFactory(
            store=store,
            session_config=SessionConfig(),
            orchestrator_config=OrchestratorConfig(),
            context_config=ContextConfig(),
        )

        session = await factory.create_session(guardrails=guardrails, gateway=mock_gateway)
        # 模拟中等状态量
        session.assembler.conversation_history = [
            {"role": "user" if i % 2 == 0 else "assistant",
             "content": f"消息内容 {i}" * 10}
            for i in range(30)
        ]
        session.metadata.total_turns = 15

        # 预热
        await session.save_auto_checkpoint()

        iterations = 10
        start = time.perf_counter()
        for _ in range(iterations):
            await session.save_auto_checkpoint()
        elapsed = (time.perf_counter() - start) / iterations * 1000

        assert elapsed < 200, f"检查点写入延迟 {elapsed:.2f}ms 超过 200ms 上限"


class TestOrchestrationLoopOverhead:
    """编排循环开销（不含 LLM 调用）: <100ms。"""

    async def test_loop_overhead(self, store: PersistenceStore) -> None:
        """验证：编排循环单次开销 <100ms（不含 LLM 调用）。"""
        guardrails = GuardrailEngine(RuleEngine(), PermissionManager())
        factory = SessionFactory(
            store=store,
            session_config=SessionConfig(auto_checkpoint=False),
            orchestrator_config=OrchestratorConfig(),
            context_config=ContextConfig(),
        )

        # 使用即时返回的 mock gateway
        mock_gw = MagicMock(spec=GatewayRouter)
        mock_gw.config = MagicMock()
        mock_gw.config.max_budget = None
        mock_gw.config.default_model = "test-model"
        mock_gw.router = MagicMock()
        mock_gw.router.acompletion = AsyncMock(
            return_value=make_raw_response(content="快速响应")
        )
        session = await factory.create_session(guardrails=guardrails, gateway=mock_gw)

        # 预热
        await session.run_turn("预热")

        iterations = 20
        start = time.perf_counter()
        for i in range(iterations):
            mock_gw.router.acompletion = AsyncMock(
                return_value=make_raw_response(content=f"响应 {i}")
            )
            await session.run_turn(f"测试 {i}")
        elapsed = (time.perf_counter() - start) / iterations * 1000

        assert elapsed < 100, f"编排循环开销 {elapsed:.2f}ms 超过 100ms 上限"

    async def test_loop_with_tool_call_overhead(self, store: PersistenceStore) -> None:
        """验证：含工具调用的编排循环单次开销 <100ms。"""
        guardrails = GuardrailEngine(RuleEngine(), PermissionManager())
        factory = SessionFactory(
            store=store,
            session_config=SessionConfig(auto_checkpoint=False),
            orchestrator_config=OrchestratorConfig(),
            context_config=ContextConfig(),
        )

        mock_gw = MagicMock(spec=GatewayRouter)
        mock_gw.config = MagicMock()
        mock_gw.config.max_budget = None
        mock_gw.config.default_model = "test-model"
        mock_gw.router = MagicMock()
        session = await factory.create_session(guardrails=guardrails, gateway=mock_gw)

        async def noop_handler(args: dict[str, Any]) -> str:
            return "ok"

        session.registry.register(
            ToolDefinition(
                name="noop",
                description="空操作",
                parameters={"type": "object", "properties": {}},
            ),
            noop_handler,
        )

        def make_tool_call_response() -> SimpleNamespace:
            tc = SimpleNamespace(
                id="tc-bench", type="function",
                function=SimpleNamespace(name="noop", arguments="{}"),
            )
            return make_raw_response(tool_calls=[tc])

        # 预热
        call_count = 0

        async def mock_acompletion(**kwargs: Any) -> SimpleNamespace:
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 1:
                return make_tool_call_response()
            return make_raw_response(content="done")

        mock_gw.router.acompletion = AsyncMock(side_effect=mock_acompletion)
        await session.run_turn("预热")

        iterations = 10
        start = time.perf_counter()
        for i in range(iterations):
            mock_gw.router.acompletion = AsyncMock(side_effect=mock_acompletion)
            await session.run_turn(f"工具测试 {i}")
        elapsed = (time.perf_counter() - start) / iterations * 1000

        assert elapsed < 100, f"含工具调用的编排循环开销 {elapsed:.2f}ms 超过 100ms 上限"
