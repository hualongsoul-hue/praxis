"""记忆保留与版本管理。

时间衰减（可配置衰减曲线）、动态遗忘（低相关性标记 INACTIVE）、
版本历史（关键事实更新保留版本链）、不可变审计。
"""

import math
from datetime import datetime, timezone

from praxis.models.memory import (
    MemoryEntry,
    MemoryScope,
    MemoryStatus,
    MemoryVersion,
)
from praxis.memory.scope import ScopedMemoryStore
from praxis.persistence.store import PersistenceStore
from praxis.telemetry.logger import get_logger
from praxis.telemetry.metrics import emit_metric

log = get_logger("memory.retention")

VERSION_NAMESPACE = "memory_versions"


class RetentionManager:
    """记忆保留与版本管理器。

    管理时间衰减、动态遗忘、版本历史和不可变审计。
    """

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
        """计算记忆的时间衰减因子。

        使用指数衰减：factor = 0.5 ^ (age_days / half_life)

        Args:
            entry: 记忆条目。

        Returns:
            衰减因子 [0.0, 1.0]。
        """
        now = datetime.now(timezone.utc)
        age_days = (now - entry.updated_at).total_seconds() / 86400.0
        return math.pow(0.5, age_days / self.decay_half_life_days)

    def compute_relevance(self, entry: MemoryEntry) -> float:
        """计算记忆的综合相关性评分。

        综合 confidence × 时间衰减 × 访问活跃度。

        Args:
            entry: 记忆条目。

        Returns:
            相关性评分 [0.0, 1.0]。
        """
        decay = self.compute_decay(entry)
        access_factor = min(entry.access_count / 10.0, 1.0) if entry.access_count > 0 else 0.1
        return entry.confidence * decay * access_factor

    async def mark_inactive(self, entry: MemoryEntry, reason: str = "") -> None:
        """将记忆标记为 INACTIVE（动态遗忘）。

        不物理删除，仅标记状态。

        Args:
            entry: 记忆条目。
            reason: 标记原因。
        """
        await self.save_version(entry, reason or "标记为 INACTIVE（动态遗忘）")
        entry.status = MemoryStatus.INACTIVE
        entry.updated_at = datetime.now(timezone.utc)
        await self.scoped_store.update(entry)

        log.info(
            "记忆标记为 INACTIVE",
            memory_id=entry.memory_id,
            reason=reason,
        )
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
        """用新记忆取代旧记忆（版本链）。

        旧记忆标记 SUPERSEDED，新记忆保存为 ACTIVE。

        Args:
            old_entry: 旧记忆条目。
            new_entry: 新记忆条目。
            reason: 取代原因。

        Returns:
            新记忆的 ID。
        """
        await self.save_version(old_entry, reason or "被新版本取代")
        old_entry.status = MemoryStatus.SUPERSEDED
        old_entry.superseded_by = new_entry.memory_id
        old_entry.updated_at = datetime.now(timezone.utc)
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
        """保存记忆版本快照到审计日志。

        Args:
            entry: 当前记忆条目。
            reason: 变更原因。
        """
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
        """获取记忆的完整版本历史。

        Args:
            memory_id: 记忆 ID。

        Returns:
            版本记录列表（按版本号排序）。
        """
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
        """执行衰减扫描，将低相关性记忆标记为 INACTIVE。

        Args:
            scope: 扫描的作用域。

        Returns:
            标记为 INACTIVE 的记忆数量。
        """
        entries = await self.scoped_store.list_scope(scope)
        count = 0
        now = datetime.now(timezone.utc)

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
                        f"衰减扫描: relevance={relevance:.4f}, "
                        f"inactive_days={inactive_days:.1f}",
                    )
                    count += 1

        emit_metric(
            "memory_decay_sweep",
            float(count),
            {"scope": scope.to_string()},
            "gauge",
        )
        return count
