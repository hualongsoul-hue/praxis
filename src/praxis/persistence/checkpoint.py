"""检查点管理。

支持完整 Agent 状态快照的保存/加载/列出，
基于 PersistenceStore 实现，按会话 ID 组织。
"""

from typing import Any

from praxis.models.persistence import Checkpoint
from praxis.persistence.store import PersistenceStore

CHECKPOINT_NAMESPACE = "checkpoints"


class CheckpointManager:
    """检查点管理器。"""

    def __init__(self, store: PersistenceStore) -> None:
        self.store = store

    async def save_checkpoint(
        self,
        session_id: str,
        state: dict[str, Any],
        description: str | None = None,
    ) -> str:
        """保存检查点。

        Args:
            session_id: 会话 ID。
            state: 完整 Agent 状态快照。
            description: 可选描述信息。

        Returns:
            生成的检查点 ID。
        """
        cp = Checkpoint(
            session_id=session_id,
            state=state,
            description=description,
        )
        key = f"{session_id}:{cp.checkpoint_id}"
        await self.store.save(
            CHECKPOINT_NAMESPACE,
            key,
            cp.model_dump(mode="json"),
        )
        return cp.checkpoint_id

    async def load_checkpoint(self, checkpoint_id: str) -> dict[str, Any] | None:
        """加载指定检查点的状态快照。

        Args:
            checkpoint_id: 检查点 ID（需包含 session_id 前缀）。

        Returns:
            状态快照字典，不存在时返回 None。
        """
        data = await self.store.load(CHECKPOINT_NAMESPACE, checkpoint_id)
        if data is None:
            return None
        cp = Checkpoint.from_storage(data)
        return cp.state

    async def list_checkpoints(self, session_id: str) -> list[Checkpoint]:
        """列出指定会话的所有检查点，按时间排序。

        Args:
            session_id: 会话 ID。

        Returns:
            检查点列表，按 created_at 升序排列。
        """
        keys = await self.store.list_keys(
            CHECKPOINT_NAMESPACE, prefix=f"{session_id}:"
        )
        checkpoints: list[Checkpoint] = []
        for key in keys:
            data = await self.store.load(CHECKPOINT_NAMESPACE, key)
            if data is not None:
                checkpoints.append(Checkpoint.from_storage(data))
        return sorted(checkpoints, key=lambda c: c.created_at)
