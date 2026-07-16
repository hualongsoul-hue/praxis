"""场景四：会话恢复（跨窗口续接）。

用户恢复中断的会话 → S12 加载检查点 → S3 读取快照 →
恢复 S6/S7 状态 → 重建无状态组件 → 继续执行。
"""

from pathlib import Path

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
from praxis.models.session import ContinuationPhase, SessionStatus
from praxis.persistence.store import PersistenceStore, create_store
from praxis.session.checkpoint import CheckpointManager
from praxis.session.continuation import ContinuationManager
from praxis.session.core import SessionFactory
from praxis.session.resume import SessionResumer


@pytest.fixture
async def e2e_store(tmp_path: Path) -> PersistenceStore:
    config = PersistenceConfig(backend="sqlite", sqlite_path=str(tmp_path / "resume.db"))
    s = await create_store(config)
    yield s  # type: ignore[misc]
    await s.close()


@pytest.fixture
def e2e_factory(e2e_store: PersistenceStore) -> SessionFactory:
    return SessionFactory(
        store=e2e_store,
        session_config=SessionConfig(),
        orchestrator_config=OrchestratorConfig(),
        context_config=ContextConfig(),
    )


@pytest.fixture
def e2e_guardrails() -> GuardrailEngine:
    return GuardrailEngine(RuleEngine(), PermissionManager())


class TestSessionResume:
    """场景四：会话恢复 E2E 测试。"""

    async def test_full_checkpoint_save_and_resume(
        self,
        e2e_store: PersistenceStore,
        e2e_factory: SessionFactory,
        e2e_guardrails: GuardrailEngine,
        mock_gateway: GatewayRouter,
    ) -> None:
        """验证：完整的检查点保存 → 恢复 → 续接流程。"""
        # 创建原始会话
        session = await e2e_factory.create_session(guardrails=e2e_guardrails, gateway=mock_gateway)
        session_id = session.session_id
        assert session.status == SessionStatus.ACTIVE

        # 模拟几轮对话后的状态
        session.assembler.conversation_history = [
            {"role": "user", "content": "请分析代码"},
            {"role": "assistant", "content": "我来分析..."},
            {"role": "user", "content": "继续"},
            {"role": "assistant", "content": "分析结果如下..."},
        ]
        session.assembler.file_refs = ["src/main.py", "src/utils.py"]
        session.metadata.total_turns = 4
        session.metadata.total_tokens = 2000

        # 保存检查点
        cp_id = await session.save_auto_checkpoint()
        assert cp_id is not None

        # 恢复会话
        cp_mgr = CheckpointManager(e2e_store)
        resumer = SessionResumer(e2e_factory, cp_mgr)
        restored = await resumer.resume_session(session_id, e2e_guardrails, gateway=mock_gateway)

        # 验证恢复完整性
        assert restored is not None
        assert restored.session_id == session_id
        assert restored.metadata.total_turns == 4
        assert restored.metadata.total_tokens == 2000
        assert len(restored.assembler.conversation_history) == 4
        assert restored.assembler.file_refs == ["src/main.py", "src/utils.py"]
        assert restored.metadata.continuation_phase == ContinuationPhase.WARMUP

    async def test_multiple_checkpoints_and_restore_latest(
        self,
        e2e_store: PersistenceStore,
        e2e_factory: SessionFactory,
        e2e_guardrails: GuardrailEngine,
        mock_gateway: GatewayRouter,
    ) -> None:
        """验证：多次检查点保存后恢复最新状态。"""
        session = await e2e_factory.create_session(guardrails=e2e_guardrails, gateway=mock_gateway)
        session_id = session.session_id

        # 第一次检查点
        session.metadata.total_turns = 2
        session.assembler.conversation_history = [
            {"role": "user", "content": "v1"},
            {"role": "assistant", "content": "v1-reply"},
        ]
        await session.save_auto_checkpoint()

        # 第二次检查点（更新状态）
        session.metadata.total_turns = 5
        session.assembler.conversation_history.extend([
            {"role": "user", "content": "v2"},
            {"role": "assistant", "content": "v2-reply"},
            {"role": "user", "content": "v3"},
        ])
        await session.save_auto_checkpoint()

        # 恢复应该得到最新状态
        cp_mgr = CheckpointManager(e2e_store)
        resumer = SessionResumer(e2e_factory, cp_mgr)
        restored = await resumer.resume_session(session_id, e2e_guardrails, gateway=mock_gateway)

        assert restored is not None
        assert restored.metadata.total_turns == 5
        assert len(restored.assembler.conversation_history) == 5

    async def test_continuation_manager_warmup_phase(self) -> None:
        """验证：续接管理器在恢复后提供热身指令。"""
        mgr = ContinuationManager()
        prompt = mgr.get_system_prompt(ContinuationPhase.WARMUP)
        assert "恢复" in prompt
        assert "热身" in prompt

    async def test_resume_nonexistent_session(
        self,
        e2e_store: PersistenceStore,
        e2e_factory: SessionFactory,
        e2e_guardrails: GuardrailEngine,
        mock_gateway: GatewayRouter,
    ) -> None:
        """验证：恢复不存在的会话返回 None。"""
        cp_mgr = CheckpointManager(e2e_store)
        resumer = SessionResumer(e2e_factory, cp_mgr)
        result = await resumer.resume_session("nonexistent-id", e2e_guardrails, gateway=mock_gateway)
        assert result is None
