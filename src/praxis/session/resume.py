"""会话恢复。

resume_session 从 S3 加载检查点，
恢复 S6 记忆状态、S7 上下文状态，重建无状态组件，验证完整性。
"""

from typing import Any

from praxis.gateway.router import GatewayRouter
from praxis.session.checkpoint import CheckpointManager
from praxis.session.core import Session, SessionFactory
from praxis.memory.core import CognitiveMemory
from praxis.models.orchestrator import LoopState
from praxis.models.session import (
    ContinuationPhase,
    SessionMetadata,
    SessionSnapshot,
    SessionStatus,
)
from praxis.guardrails.engine import GuardrailEngine
from praxis.skills.manager import SkillManager
from praxis.telemetry.logger import get_logger
from praxis.tools.registry import ToolRegistry
from praxis.verification.registry import VerifierRegistry

log = get_logger("session.resume")


class SessionResumer:
    """会话恢复器。

    从检查点恢复完整会话状态。
    """

    def __init__(
        self,
        factory: SessionFactory,
        checkpoint_mgr: CheckpointManager,
    ) -> None:
        self.factory = factory
        self.checkpoint_mgr = checkpoint_mgr

    async def resume_session(
        self,
        session_id: str,
        guardrails: GuardrailEngine,
        gateway: GatewayRouter,
        registry: ToolRegistry | None = None,
        model: str = "default",
        checkpoint_id: str | None = None,
        memory: CognitiveMemory | None = None,
        skill_manager: SkillManager | None = None,
        verifier_registry: VerifierRegistry | None = None,
    ) -> Session | None:
        """从检查点恢复会话。

        Args:
            session_id: 会话 ID。
            guardrails: 护栏引擎。
            registry: 工具注册表（可选）。
            model: LLM 模型名。
            checkpoint_id: 指定检查点 ID，None 时加载最新。
            memory: S6 记忆管线（可选）。
            skill_manager: S14 技能管理器（可选）。
            verifier_registry: S10 验证器注册表（可选）。

        Returns:
            恢复后的 Session，检查点不存在时返回 None。
        """
        if checkpoint_id:
            checkpoint = await self.checkpoint_mgr.load_checkpoint(
                session_id, checkpoint_id
            )
        else:
            checkpoint = await self.checkpoint_mgr.load_latest(session_id)

        if checkpoint is None:
            log.warning("检查点未找到", session_id=session_id)
            return None

        snapshot = await self.checkpoint_mgr.extract_snapshot(checkpoint)

        # 创建新会话（重建无状态组件）
        session = await self.factory.create_session(
            guardrails=guardrails,
            gateway=gateway,
            registry=registry,
            model=model,
            memory=memory,
            skill_manager=skill_manager,
            verifier_registry=verifier_registry,
        )

        # 恢复有状态组件
        self.restore_metadata(session, snapshot.metadata)
        self.restore_context_state(session, snapshot.context_state)
        self.restore_loop_state(session, snapshot.loop_state)
        await self.restore_memory_state(session, snapshot.memory_state)

        session.metadata.continuation_phase = ContinuationPhase.WARMUP

        log.info(
            "会话已恢复",
            session_id=session_id,
            checkpoint_id=checkpoint.checkpoint_id,
            turn=snapshot.metadata.total_turns,
        )
        return session

    @staticmethod
    def restore_metadata(session: Session, saved: SessionMetadata) -> None:
        """恢复会话元数据。"""
        session.metadata.session_id = saved.session_id
        session.metadata.created_at = saved.created_at
        session.metadata.total_turns = saved.total_turns
        session.metadata.total_tokens = saved.total_tokens
        session.metadata.checkpoint_count = saved.checkpoint_count
        session.metadata.last_checkpoint_id = saved.last_checkpoint_id
        session.metadata.status = SessionStatus.PAUSED

    @staticmethod
    def restore_context_state(session: Session, state: dict[str, Any]) -> None:
        """恢复 S7 上下文状态。"""
        session.assembler.conversation_history = state.get("messages", [])
        session.assembler.file_refs = state.get("file_refs", [])
        session.assembler.compaction_count = state.get("compaction_count", 0)
        session.assembler.tool_schemas = state.get("tool_schemas", [])

    @staticmethod
    def restore_loop_state(session: Session, state: dict[str, Any]) -> None:
        """恢复 S11 循环状态。"""
        if state:
            session.loop.state = LoopState.model_validate(state)

    @staticmethod
    async def restore_memory_state(session: Session, state: dict[str, Any]) -> None:
        """恢复 S6 记忆状态。"""
        if session.memory is not None and state:
            await session.memory.import_state(state)

    @staticmethod
    def validate_integrity(session: Session) -> list[str]:
        """验证恢复状态完整性。

        Returns:
            问题列表，空表示完整。
        """
        issues: list[str] = []
        if not session.metadata.session_id:
            issues.append("session_id 为空")
        if session.metadata.status not in (SessionStatus.PAUSED, SessionStatus.ACTIVE):
            issues.append(f"状态异常: {session.metadata.status}")
        return issues
