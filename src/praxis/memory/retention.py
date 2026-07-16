"""记忆保留与版本管理。

时间衰减、动态遗忘、版本链、不可变审计。
"""

import math
from datetime import UTC, datetime
from uuid import uuid4

from praxis.memory.store import ScopedMemoryStore
from praxis.models.memory import (
    MemoryEntry,
    MemoryScope,
    MemoryStatus,
    MemoryVersion,
)
from praxis.persistence.store import PersistenceStore
from praxis.telemetry.logger import get_logger
from praxis.telemetry.metrics import emit_metric

log = get_logger("memory.retention")

VERSION_NAMESPACE = "memory_versions"


class RetentionManager:
    """记忆保留与版本管理器。"""

    def __init__(
        self,
        scoped_store: ScopedMemoryStore,
        version_store: PersistenceStore,
        decay_half_life_days: float = 30.0,
        inactivity_threshold_days: float = 90.0,
        min_access_count: int = 1,
    ) -> None:
        self.scoped_store = scoped_store
        self.version_store = version_store
        self.decay_half_life_days = decay_half_life_days
        self.inactivity_threshold_days = inactivity_threshold_days
        self.min_access_count = min_access_count

    def compute_decay(self, entry: MemoryEntry) -> float:
        """指数衰减：factor = 0.5 ^ (age_days / half_life)。"""
        now = datetime.now(UTC)
        age_days = (now - entry.updated_at).total_seconds() / 86400.0
        return math.pow(0.5, age_days / self.decay_half_life_days)

    def compute_relevance(self, entry: MemoryEntry) -> float:
        """综合 confidence × 衰减 × 访问活跃度。"""
        decay = self.compute_decay(entry)
        if entry.access_count > 0:
            access_factor = min(entry.access_count / 10.0, 1.0)
        else:
            access_factor = 0.1
        return entry.confidence * decay * access_factor

    async def mark_inactive(self, entry: MemoryEntry, reason: str = "") -> None:
        """将记忆标记为 INACTIVE（动态遗忘）。不物理删除。"""
        await self.save_version(entry, reason or "标记为 INACTIVE（动态遗忘）")
        entry.status = MemoryStatus.INACTIVE
        entry.updated_at = datetime.now(UTC)
        await self.scoped_store.update(entry)

        log.info("记忆标记为 INACTIVE", memory_id=entry.memory_id, reason=reason)
        emit_metric(
            "memory_retention_inactive",
            1.0,
            {"memory_type": entry.memory_type.value},
            "counter",
        )

    async def supersede(
        self,
        old_entry: MemoryEntry,
        new_entry: MemoryEntry,
        reason: str = "",
    ) -> str:
        """用新记忆取代旧记忆（版本链）。"""
        await self.save_version(old_entry, reason or "被新版本取代")

        # 若调用方复用了旧条目的 memory_id，分配新 ID 防止覆盖 SUPERSEDED 记录
        if new_entry.memory_id == old_entry.memory_id:
            new_entry.memory_id = uuid4().hex

        old_entry.status = MemoryStatus.SUPERSEDED
        old_entry.superseded_by = new_entry.memory_id
        old_entry.updated_at = datetime.now(UTC)
        await self.scoped_store.update(old_entry)

        new_entry.version = old_entry.version + 1
        await self.scoped_store.save(new_entry)

        log.info(
            "记忆版本更新",
            old_id=old_entry.memory_id,
            new_id=new_entry.memory_id,
            version=new_entry.version,
        )
        emit_metric(
            "memory_retention_superseded",
            1.0,
            {"memory_type": old_entry.memory_type.value},
            "counter",
        )
        return new_entry.memory_id

    async def save_version(self, entry: MemoryEntry, reason: str = "") -> None:
        """保存记忆版本快照到审计日志。"""
        version = MemoryVersion(
            memory_id=entry.memory_id,
            version=entry.version,
            content=entry.content,
            status=entry.status,
            change_reason=reason,
        )
        key = f"{entry.memory_id}:v{entry.version}"
        await self.version_store.save(
            VERSION_NAMESPACE,
            key,
            version.model_dump(mode="json"),
        )

    async def get_version_history(self, memory_id: str) -> list[MemoryVersion]:
        """获取记忆的完整版本历史。"""
        prefix = f"{memory_id}:v"
        keys = await self.version_store.list_keys(VERSION_NAMESPACE, prefix)
        versions: list[MemoryVersion] = []
        for key in keys:
            data = await self.version_store.load(VERSION_NAMESPACE, key)
            if data is not None:
                versions.append(MemoryVersion.model_validate(data))
        versions.sort(key=lambda v: v.version)
        return versions

    async def run_decay_sweep(self, scope: MemoryScope) -> int:
        """执行衰减扫描，将低相关性记忆标记为 INACTIVE。"""
        entries = await self.scoped_store.list_scope(scope)
        count = 0
        now = datetime.now(UTC)

        for entry in entries:
            last_access = entry.last_accessed_at or entry.created_at
            inactive_days = (now - last_access).total_seconds() / 86400.0

            if (
                inactive_days > self.inactivity_threshold_days
                and entry.access_count <= self.min_access_count
            ):
                relevance = self.compute_relevance(entry)
                if relevance < 0.1:
                    await self.mark_inactive(
                        entry,
                        f"衰减扫描: relevance={relevance:.4f}, inactive_days={inactive_days:.1f}",
                    )
                    count += 1

        emit_metric(
            "memory_decay_sweep",
            float(count),
            {"scope": scope.to_string()},
            "gauge",
        )
        return count

    async def enforce_capacity(self, scope: MemoryScope, max_memories: int) -> int:
        """容量上限：活跃记忆超过 max_memories 时，淘汰相关性最低的多余条目。

        Args:
            scope: 作用域。
            max_memories: 该作用域允许的最大活跃记忆数；<=0 表示不限制。

        Returns:
            被淘汰（标记 INACTIVE）的条目数。
        """
        if max_memories <= 0:
            return 0
        active = [
            e for e in await self.scoped_store.list_scope(scope)
            if e.status == MemoryStatus.ACTIVE
        ]
        if len(active) <= max_memories:
            return 0
        active.sort(key=self.compute_relevance)  # 升序：相关性最低在前
        excess = active[: len(active) - max_memories]
        for entry in excess:
            await self.mark_inactive(
                entry, f"容量上限 {max_memories}: 淘汰低相关性记忆"
            )
        emit_metric(
            "memory_capacity_evict",
            float(len(excess)),
            {"scope": scope.to_string()},
            "counter",
        )
        return len(excess)
