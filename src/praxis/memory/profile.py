"""语义档案管理（F6.1 Profile 模式）。

每 (scope, schema_name) 对应唯一档案文档，字段字典就地合并更新。
档案独立于集合模式的 SemanticMemory，不参与 search_memory 的语义检索。
"""

from datetime import datetime, timezone
from typing import Any

from praxis.memory.store import ProfileStore
from praxis.models.memory import MemoryScope, SemanticProfile
from praxis.telemetry.logger import get_logger
from praxis.telemetry.metrics import emit_metric

log = get_logger("memory.profile")


class ProfileManager:
    """档案管理器：读取/合并更新语义档案。"""

    def __init__(self, profile_store: ProfileStore) -> None:
        self.profile_store = profile_store

    async def get(
        self,
        scope: MemoryScope,
        schema_name: str,
    ) -> dict[str, Any] | None:
        """读取档案字段字典，未找到返回 None。"""
        profile = await self.profile_store.load(scope, schema_name)
        if profile is None:
            return None
        return dict(profile.fields)

    async def update(
        self,
        scope: MemoryScope,
        schema_name: str,
        fields: dict[str, Any],
    ) -> SemanticProfile:
        """就地合并更新档案字段。不存在则创建。"""
        profile = await self.profile_store.load(scope, schema_name)
        now = datetime.now(timezone.utc)

        if profile is None:
            profile = SemanticProfile(
                scope=scope,
                schema_name=schema_name,
                fields=dict(fields),
            )
        else:
            merged = dict(profile.fields)
            merged.update(fields)
            profile.fields = merged
            profile.version += 1
            profile.updated_at = now

        await self.profile_store.save(profile)
        emit_metric(
            "memory_profile_updated",
            1.0,
            {"schema": schema_name},
            "counter",
        )
        log.info(
            "档案更新",
            scope=scope.to_string(),
            schema=schema_name,
            version=profile.version,
        )
        return profile

    async def list_profiles(self, scope: MemoryScope) -> list[SemanticProfile]:
        """列出指定作用域下的所有档案。"""
        return await self.profile_store.list_scope(scope)

    async def delete(self, scope: MemoryScope, schema_name: str) -> None:
        """删除指定档案。"""
        await self.profile_store.delete(scope, schema_name)
