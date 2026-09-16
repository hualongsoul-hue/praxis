"""自动检查点。

每次 S11 循环终止后自动保存，
检查点包含会话元数据 + S6/S7/S11 状态快照，
通过 S3 写入。
"""

import time
from typing import Any, cast

from praxis.exceptions import CheckpointCorruptionError
from praxis.models.persistence import Checkpoint
from praxis.models.session import (
    CheckpointInfo,
    SessionMetadata,
    SessionSnapshot,
)
from praxis.persistence.store import PersistenceStore
from praxis.telemetry.logger import get_logger
from praxis.telemetry.metrics import emit_metric

log = get_logger("session.checkpoint")

CHECKPOINT_NAMESPACE = "checkpoints"


class CheckpointManager:
    """检查点管理器。

    负责保存、加载和列出检查点。
    """

    def __init__(self, store: PersistenceStore) -> None:
        self.store = store

    async def save_checkpoint(
        self,
        metadata: SessionMetadata,
        context_state: dict[str, Any],
        memory_state: dict[str, Any],
        loop_state: dict[str, Any],
        strategy_state: dict[str, Any] | None = None,
        recovery_state: dict[str, Any] | None = None,
        approval_state: dict[str, Any] | None = None,
        skill_state: dict[str, Any] | None = None,
        tool_execution_ledger: dict[str, Any] | None = None,
        file_refs: list[str] | None = None,
        description: str = "",
    ) -> str:
        """保存检查点。

        Args:
            metadata: 会话元数据。
            context_state: S7 上下文状态快照。
            memory_state: S6 记忆状态快照。
            loop_state: S11 循环状态快照。
            file_refs: 文件引用列表。
            description: 检查点描述。

        Returns:
            检查点 ID。
        """
        start = time.perf_counter()

        snapshot = SessionSnapshot(
            metadata=metadata.model_copy(),
            context_state=context_state,
            memory_state=memory_state,
            loop_state=loop_state,
            strategy_state=strategy_state or {},
            recovery_state=recovery_state or {},
            approval_state=approval_state or {},
            skill_state=skill_state or {},
            tool_execution_ledger=tool_execution_ledger or {},
            file_refs=file_refs or [],
        )

        checkpoint = Checkpoint(
            session_id=metadata.session_id,
            state=snapshot.model_dump(mode="json"),
            description=description or f"Turn {metadata.total_turns}",
        )

        key = f"{metadata.session_id}:{checkpoint.checkpoint_id}"
        await self.store.save(
            CHECKPOINT_NAMESPACE,
            key,
            checkpoint.model_dump(mode="json"),
        )

        # 更新最新检查点引用
        await self.store.save(
            CHECKPOINT_NAMESPACE,
            f"{metadata.session_id}:latest",
            checkpoint.checkpoint_id,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000
        emit_metric("checkpoint_save_ms", elapsed_ms, {}, "histogram")
        log.info(
            "检查点已保存",
            checkpoint_id=checkpoint.checkpoint_id,
            session_id=metadata.session_id,
            elapsed_ms=round(elapsed_ms, 1),
        )

        metadata.checkpoint_count += 1
        metadata.last_checkpoint_id = checkpoint.checkpoint_id
        return checkpoint.checkpoint_id

    async def load_checkpoint(
        self,
        session_id: str,
        checkpoint_id: str,
    ) -> Checkpoint | None:
        """加载指定检查点。

        Args:
            session_id: 会话 ID。
            checkpoint_id: 检查点 ID。

        Returns:
            检查点对象，不存在时返回 None。
        """
        key = f"{session_id}:{checkpoint_id}"
        exists, data = await self.store.load_with_presence(CHECKPOINT_NAMESPACE, key)
        if not exists:
            return None
        return Checkpoint.from_storage(cast(object, data))

    async def load_latest(self, session_id: str) -> Checkpoint | None:
        """加载最新检查点。"""
        exists, latest_id = await self.store.load_with_presence(
            CHECKPOINT_NAMESPACE,
            f"{session_id}:latest",
        )
        if not exists:
            return None
        if not isinstance(latest_id, str) or not latest_id:
            raise CheckpointCorruptionError("最新检查点引用损坏")
        checkpoint = await self.load_checkpoint(session_id, latest_id)
        if checkpoint is None:
            raise CheckpointCorruptionError("最新检查点正文缺失")
        return checkpoint

    async def list_checkpoints(self, session_id: str) -> list[CheckpointInfo]:
        """列出会话的所有检查点。

        Args:
            session_id: 会话 ID。

        Returns:
            检查点摘要列表。
        """
        keys = await self.store.list_keys(
            CHECKPOINT_NAMESPACE,
            prefix=f"{session_id}:",
        )

        infos: list[CheckpointInfo] = []
        for key in keys:
            if key.endswith(":latest"):
                continue
            data = await self.store.load(CHECKPOINT_NAMESPACE, key)
            if data is None or not isinstance(data, dict):
                continue
            cp = Checkpoint.from_storage(cast(object, data))
            state = cp.state or {}
            meta = state.get("metadata", {})
            infos.append(CheckpointInfo(
                checkpoint_id=cp.checkpoint_id,
                session_id=cp.session_id,
                created_at=cp.created_at,
                turn_number=meta.get("total_turns", 0),
                description=cp.description or "",
            ))

        infos.sort(key=lambda x: x.created_at)
        return infos

    async def delete_checkpoint(
        self,
        session_id: str,
        checkpoint_id: str,
    ) -> None:
        """删除检查点。"""
        latest_key = f"{session_id}:latest"
        latest_exists, latest_id = await self.store.load_with_presence(
            CHECKPOINT_NAMESPACE,
            latest_key,
        )
        if latest_exists:
            if not isinstance(latest_id, str) or not latest_id:
                raise CheckpointCorruptionError("最新检查点引用损坏")
            if latest_id == checkpoint_id:
                await self.store.delete(CHECKPOINT_NAMESPACE, latest_key)
        key = f"{session_id}:{checkpoint_id}"
        await self.store.delete(CHECKPOINT_NAMESPACE, key)

    async def extract_snapshot(self, checkpoint: Checkpoint) -> SessionSnapshot:
        """从检查点提取会话快照。"""
        return SessionSnapshot.model_validate(checkpoint.state)
