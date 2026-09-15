"""会话恢复。

resume_session 从 S3 加载检查点，
恢复 S6 记忆状态、S7 上下文状态，重建无状态组件，验证完整性。
"""

from typing import Any, cast

from praxis.config.schemas import ToolsConfig
from praxis.context.jit_retrieval import JITRetriever
from praxis.guardrails.engine import GuardrailEngine
from praxis.memory.core import CognitiveMemory
from praxis.models.orchestrator import LoopState
from praxis.models.session import (
    ContinuationPhase,
    SessionMetadata,
    SessionStatus,
)
from praxis.models.tools import ToolExecutionRecord, ToolExecutionState
from praxis.protocols import ModelGateway
from praxis.session.checkpoint import CheckpointManager
from praxis.session.core import Session, SessionFactory
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
        gateway: ModelGateway,
        registry: ToolRegistry | None = None,
        model: str = "default",
        checkpoint_id: str | None = None,
        memory: CognitiveMemory | None = None,
        skill_manager: SkillManager | None = None,
        verifier_registry: VerifierRegistry | None = None,
        tools_config: ToolsConfig | None = None,
        include_builtins: bool = True,
        jit_retriever: JITRetriever | None = None,
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
            tools_config: 宿主的工具安全策略；恢复时显式传递，默认保持关闭敏感能力。
            include_builtins: 是否注册内置工具（仅在未提供 registry 时生效）。
            jit_retriever: 宿主的内容加载器（可选）。

        输入策略使用 factory.input_config，模型能力重新由 gateway.capabilities 解析。
        安全策略与输入策略由可信宿主提供，不从检查点反序列化授权配置。

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
            tools_config=tools_config,
            include_builtins=include_builtins,
            jit_retriever=jit_retriever,
            session_id=snapshot.metadata.session_id,
        )

        # 恢复有状态组件
        self.restore_metadata(session, snapshot.metadata)
        self.restore_context_state(session, snapshot.context_state)
        self.restore_loop_state(session, snapshot.loop_state)
        self.restore_strategy_state(session, snapshot.strategy_state)
        self.restore_recovery_state(session, snapshot.recovery_state)
        self.restore_tool_execution_ledger(session, snapshot.tool_execution_ledger)
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
    def restore_strategy_state(session: Session, state: dict[str, Any]) -> None:
        """Restore request planning state."""
        if state:
            session.loop.strategy.import_state(state)

    @staticmethod
    def restore_recovery_state(session: Session, state: dict[str, Any]) -> None:
        """Restore circuit-breaker and retry state."""
        circuits = state.get("circuits")
        if isinstance(circuits, dict):
            session.loop.coordinator.circuits.import_state(
                cast(dict[str, Any], circuits),
            )
        retry = state.get("retry")
        if isinstance(retry, dict):
            session.loop.coordinator.retry_policy.import_state(
                cast(dict[str, object], retry),
            )

    @staticmethod
    def restore_tool_execution_ledger(
        session: Session,
        ledger: dict[str, ToolExecutionRecord],
    ) -> None:
        """Restore the ledger and convert interrupted writes to UNCERTAIN."""
        session.tool_execution_ledger.clear()
        for call_id, value in ledger.items():
            record = value
            if record.state is ToolExecutionState.STARTED and not (
                record.readonly or record.idempotent
            ):
                record = record.model_copy(update={"state": ToolExecutionState.UNCERTAIN})
            session.tool_execution_ledger[call_id] = record

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
