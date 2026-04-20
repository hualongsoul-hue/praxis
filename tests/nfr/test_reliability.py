"""非功能需求验证——可靠性。

PRD § 9.2:
- 单步可靠性: ≥99.5%（解析/工具/护栏不误崩溃）
- 恢复成功率: 100%（检查点恢复后状态完整）
- 熔断器正确三态转换
- 重试策略指数退避
"""

import json
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
)
from praxis.gateway.router import GatewayRouter
from praxis.guardrails.engine import GuardrailEngine
from praxis.guardrails.permissions import PermissionManager
from praxis.guardrails.rules import RuleEngine
from praxis.models.recovery import CircuitState
from praxis.models.tools import ToolDefinition
from praxis.orchestrator.parser import OutputParser
from praxis.persistence.store import PersistenceStore, create_store
from praxis.recovery.circuit_breaker import CircuitBreaker, CircuitBreakerRegistry
from praxis.recovery.retry import RetryPolicy
from praxis.session.checkpoint import CheckpointManager
from praxis.session.core import SessionFactory
from praxis.session.resume import SessionResumer
from praxis.tools.executor import ToolExecutor
from praxis.models.guardrails import VerdictType
from praxis.tools.registry import ToolRegistry
from praxis.tools.sandbox import Sandbox


def make_raw_response(content: str = "ok") -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return SimpleNamespace(
        id="chatcmpl-rel", choices=[choice], usage=usage,
        model="test-model", created=1700000000,
    )


@pytest.fixture(autouse=True)
def mock_litellm():
    with patch("praxis.context.assembler.get_max_tokens", return_value=128000), \
         patch("praxis.context.assembler.get_token_count", return_value=100):
        yield


@pytest.fixture
async def store(tmp_path: Path) -> PersistenceStore:
    config = PersistenceConfig(backend="sqlite", sqlite_path=str(tmp_path / "rel.db"))
    s = await create_store(config)
    yield s  # type: ignore[misc]
    await s.close()


class TestCircuitBreakerReliability:
    """熔断器三态模型可靠性。"""

    def test_closed_to_open_on_threshold(self) -> None:
        """验证：连续失败达到阈值后从 CLOSED 转为 OPEN。"""
        cb = CircuitBreaker("test_tool", failure_threshold=3, cooldown_seconds=0.1)
        assert cb.check() == CircuitState.CLOSED

        cb.record_failure()
        cb.record_failure()
        assert cb.check() == CircuitState.CLOSED  # 2 次不到阈值

        cb.record_failure()
        assert cb.check() == CircuitState.OPEN  # 3 次到达阈值

    def test_open_to_half_open_after_cooldown(self) -> None:
        """验证：OPEN 状态超过 cooldown 后转为 HALF_OPEN。"""
        cb = CircuitBreaker("test_tool", failure_threshold=1, cooldown_seconds=0.01)
        cb.record_failure()
        assert cb.check() == CircuitState.OPEN

        time.sleep(0.02)
        assert cb.check() == CircuitState.HALF_OPEN

    def test_half_open_success_resets_to_closed(self) -> None:
        """验证：HALF_OPEN 状态下成功执行重置为 CLOSED。"""
        cb = CircuitBreaker("test_tool", failure_threshold=1, cooldown_seconds=0.01)
        cb.record_failure()
        time.sleep(0.02)
        cb.check()  # 触发 HALF_OPEN 转换

        cb.record_success()
        assert cb.check() == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_half_open_failure_returns_to_open(self) -> None:
        """验证：HALF_OPEN 状态下失败重新转为 OPEN。"""
        cb = CircuitBreaker("test_tool", failure_threshold=1, cooldown_seconds=0.01)
        cb.record_failure()
        time.sleep(0.02)
        cb.check()  # HALF_OPEN

        cb.record_failure()
        assert cb.check() == CircuitState.OPEN

    def test_registry_per_tool_isolation(self) -> None:
        """验证：注册表中每个工具的熔断器独立。"""
        reg = CircuitBreakerRegistry(failure_threshold=2)

        reg.record_outcome("tool_a", success=False)
        reg.record_outcome("tool_a", success=False)
        reg.record_outcome("tool_b", success=False)

        assert reg.check_circuit("tool_a") == CircuitState.OPEN
        assert reg.check_circuit("tool_b") == CircuitState.CLOSED


class TestRetryPolicyReliability:
    """重试策略可靠性。"""

    def test_exponential_backoff(self) -> None:
        """验证：重试延迟呈指数增长。"""
        policy = RetryPolicy(max_retries=5, initial_delay=1.0, jitter_factor=0.0)

        d0 = policy.get_retry_decision("t", 0)
        d1 = policy.get_retry_decision("t", 1)
        d2 = policy.get_retry_decision("t", 2)

        assert d0.should_retry is True
        assert d1.should_retry is True
        assert d2.should_retry is True
        # 指数增长（无抖动时精确）
        assert d1.wait_seconds > d0.wait_seconds
        assert d2.wait_seconds > d1.wait_seconds

    def test_max_retries_respected(self) -> None:
        """验证：超过最大重试次数后停止重试。"""
        policy = RetryPolicy(max_retries=2)

        assert policy.get_retry_decision("t", 0).should_retry is True
        assert policy.get_retry_decision("t", 1).should_retry is True
        assert policy.get_retry_decision("t", 2).should_retry is False

    def test_max_delay_cap(self) -> None:
        """验证：重试延迟不超过上限。"""
        policy = RetryPolicy(
            max_retries=20, initial_delay=1.0, max_delay=5.0, jitter_factor=0.0,
        )

        d10 = policy.get_retry_decision("t", 10)
        assert d10.wait_seconds <= 5.0


class TestCheckpointRecovery:
    """检查点恢复可靠性: 100%。"""

    async def test_full_state_recovery(
        self, store: PersistenceStore, mock_gateway: MagicMock,
    ) -> None:
        """验证：恢复后所有关键状态完整。"""
        guardrails = GuardrailEngine(RuleEngine(), PermissionManager())
        factory = SessionFactory(
            store=store,
            session_config=SessionConfig(),
            orchestrator_config=OrchestratorConfig(),
            context_config=ContextConfig(),
        )

        session = await factory.create_session(guardrails=guardrails, gateway=mock_gateway)
        sid = session.session_id

        # 构建非平凡状态
        session.assembler.conversation_history = [
            {"role": "assistant", "content": f"回复 {i}"} for i in range(10)
        ]
        session.assembler.file_refs = ["a.py", "b.py", "c.py"]
        session.metadata.total_turns = 10
        session.metadata.total_tokens = 5000

        await session.save_auto_checkpoint()

        # 恢复
        cp_mgr = CheckpointManager(store)
        resumer = SessionResumer(factory, cp_mgr)
        restored = await resumer.resume_session(sid, guardrails, gateway=mock_gateway)

        assert restored is not None
        try:
            assert restored.session_id == sid
            assert restored.metadata.total_turns == 10
            assert restored.metadata.total_tokens == 5000
            assert len(restored.assembler.conversation_history) == 10
            assert restored.assembler.file_refs == ["a.py", "b.py", "c.py"]
        finally:
            await session.terminate()
            await restored.terminate()

    async def test_repeated_save_restore_idempotent(
        self, store: PersistenceStore, mock_gateway: MagicMock,
    ) -> None:
        """验证：重复保存/恢复的结果一致。"""
        guardrails = GuardrailEngine(RuleEngine(), PermissionManager())
        factory = SessionFactory(
            store=store,
            session_config=SessionConfig(),
            orchestrator_config=OrchestratorConfig(),
            context_config=ContextConfig(),
        )

        session = await factory.create_session(guardrails=guardrails, gateway=mock_gateway)
        sid = session.session_id
        session.assembler.conversation_history = [
            {"role": "assistant", "content": "幂等性测试"},
        ]

        # 保存 3 次
        for _ in range(3):
            await session.save_auto_checkpoint()

        cp_mgr = CheckpointManager(store)
        resumer = SessionResumer(factory, cp_mgr)
        restored = await resumer.resume_session(sid, guardrails, gateway=mock_gateway)

        assert restored is not None
        assert len(restored.assembler.conversation_history) == 1
        await session.terminate()
        await restored.terminate()


class TestSingleStepReliability:
    """单步可靠性: ≥99.5%（解析/工具/护栏不误崩溃）。"""

    async def test_parser_handles_edge_cases(self) -> None:
        """验证：OutputParser 对各种边界输入不崩溃。"""
        parser = OutputParser()

        edge_cases = [
            make_raw_response(content=""),
            make_raw_response(content=None),
            make_raw_response(content="正常内容"),
            make_raw_response(content="   \n\n\t   "),
            make_raw_response(content="a" * 10000),
        ]

        for raw in edge_cases:
            from praxis.gateway.chat import convert_response
            model_response = convert_response(raw)
            result = parser.parse(model_response)
            assert result is not None

    async def test_tool_executor_handles_exception(self) -> None:
        """验证：工具执行异常不导致系统崩溃。"""
        registry = ToolRegistry()

        async def crasher(args: dict[str, Any]) -> str:
            raise ValueError("boom")

        registry.register(
            ToolDefinition(
                name="crash_tool",
                description="会崩溃的工具",
                parameters={"type": "object", "properties": {}},
            ),
            crasher,
        )

        from praxis.config.schemas import ToolsConfig
        sandbox = Sandbox(ToolsConfig())
        executor = ToolExecutor(registry, sandbox)

        result = await executor.execute("crash_tool", {}, "tc-crash")
        assert result.success is False
        assert result.error is not None

    async def test_guardrail_handles_unicode(self) -> None:
        """验证：护栏引擎正确处理 Unicode 输入。"""
        rule_engine = RuleEngine()
        rule_engine.register_builtin_rules()
        engine = GuardrailEngine(rule_engine, PermissionManager())

        unicode_inputs = [
            "你好世界 🌍",
            "日本語テスト",
            "العربية",
            "Ñoño con eñe",
            "Ελληνικά",
            "emojis: 🎉🔥💻🤖",
        ]
        for msg in unicode_inputs:
            verdict = await engine.check_input(msg)
            assert verdict.verdict == VerdictType.PASS
