"""作用域隔离的记忆存储与反序列化工厂。

每条记忆按 scope 隔离，支持单作用域和复合作用域查询。
反序列化时按 memory_type 选择对应子类，保留 EpisodicMemory/ProceduralMemory 的结构化字段。
"""

from typing import Any

from praxis.models.memory import (
    EpisodicMemory,
    MemoryEntry,
    MemoryScope,
    MemoryStatus,
    MemoryType,
    ProceduralMemory,
    SemanticMemory,
    SemanticProfile,
)
from praxis.persistence.store import PersistenceStore

MEMORY_NAMESPACE = "memory"
PROFILE_NAMESPACE = "memory_profile"


def entry_from_dict(data: dict[str, Any]) -> MemoryEntry:
    """按 memory_type 反序列化为正确子类，保留结构化字段。"""
    mem_type = data.get("memory_type")
    if mem_type == MemoryType.EPISODIC.value:
        return EpisodicMemory.model_validate(data)
    if mem_type == MemoryType.PROCEDURAL.value:
        return ProceduralMemory.model_validate(data)
    if mem_type == MemoryType.SEMANTIC.value:
        return SemanticMemory.model_validate(data)
    return MemoryEntry.model_validate(data)


class ScopedMemoryStore:
    """作用域隔离的记忆存储。

    每条记忆按 scope 隔离存储，支持单作用域和复合作用域查询。
    """

    def __init__(self, store: PersistenceStore) -> None:
        self.store = store

    def scope_key(self, scope: MemoryScope, memory_id: str) -> str:
        """生成带作用域前缀的存储键。"""
        return f"{scope.to_string()}:{memory_id}"

    async def save(self, entry: MemoryEntry) -> str:
        """保存记忆条目到对应作用域。"""
        key = self.scope_key(entry.scope, entry.memory_id)
        await self.store.save(MEMORY_NAMESPACE, key, entry.model_dump(mode="json"))
        return entry.memory_id

    async def load(self, scope: MemoryScope, memory_id: str) -> MemoryEntry | None:
        """从指定作用域加载记忆条目。"""
        key = self.scope_key(scope, memory_id)
        data = await self.store.load(MEMORY_NAMESPACE, key)
        if data is None:
            return None
        return entry_from_dict(data)

    async def delete(self, scope: MemoryScope, memory_id: str) -> None:
        """从指定作用域删除记忆条目。"""
        key = self.scope_key(scope, memory_id)
        await self.store.delete(MEMORY_NAMESPACE, key)

    async def list_scope(
        self,
        scope: MemoryScope,
        memory_type: MemoryType | None = None,
        status: MemoryStatus = MemoryStatus.ACTIVE,
    ) -> list[MemoryEntry]:
        """列出指定作用域中的所有记忆。"""
        prefix = scope.to_string() + ":"
        keys = await self.store.list_keys(MEMORY_NAMESPACE, prefix)
        entries: list[MemoryEntry] = []
        for key in keys:
            data = await self.store.load(MEMORY_NAMESPACE, key)
            if data is None:
                continue
            entry = entry_from_dict(data)
            if entry.status != status:
                continue
            if memory_type is not None and entry.memory_type != memory_type:
                continue
            entries.append(entry)
        return entries

    async def query(
        self,
        scopes: list[MemoryScope],
        memory_type: MemoryType | None = None,
        tags: list[str] | None = None,
        status: MemoryStatus = MemoryStatus.ACTIVE,
    ) -> list[MemoryEntry]:
        """复合作用域查询，作用域之间严格隔离。"""
        results: list[MemoryEntry] = []
        for scope in scopes:
            entries = await self.list_scope(scope, memory_type, status)
            if tags:
                entries = [e for e in entries if all(t in e.tags for t in tags)]
            results.extend(entries)
        return results

    async def find_by_id(self, memory_id: str) -> MemoryEntry | None:
        """跨所有作用域按 ID 查找（开销较大，仅用于 update/delete 兜底）。"""
        keys = await self.store.list_keys(MEMORY_NAMESPACE, "")
        for key in keys:
            if not key.endswith(f":{memory_id}"):
                continue
            data = await self.store.load(MEMORY_NAMESPACE, key)
            if data is not None:
                return entry_from_dict(data)
        return None

    async def update(self, entry: MemoryEntry) -> None:
        """更新记忆条目（就地替换）。"""
        key = self.scope_key(entry.scope, entry.memory_id)
        await self.store.save(MEMORY_NAMESPACE, key, entry.model_dump(mode="json"))

    async def clear_scope(self, scope: MemoryScope) -> int:
        """清除指定作用域的所有记忆。返回清除数量。"""
        prefix = scope.to_string() + ":"
        keys = await self.store.list_keys(MEMORY_NAMESPACE, prefix)
        for key in keys:
            await self.store.delete(MEMORY_NAMESPACE, key)
        return len(keys)


class ProfileStore:
    """语义档案存储（F6.1 Profile 模式）。

    每 (scope, schema_name) 对应唯一档案文档。
    """

    def __init__(self, store: PersistenceStore) -> None:
        self.store = store

    def profile_key(self, scope: MemoryScope, schema_name: str) -> str:
        return f"{scope.to_string()}:{schema_name}"

    async def load(self, scope: MemoryScope, schema_name: str) -> SemanticProfile | None:
        data = await self.store.load(PROFILE_NAMESPACE, self.profile_key(scope, schema_name))
        if data is None:
            return None
        return SemanticProfile.model_validate(data)

    async def save(self, profile: SemanticProfile) -> None:
        key = self.profile_key(profile.scope, profile.schema_name)
        await self.store.save(PROFILE_NAMESPACE, key, profile.model_dump(mode="json"))

    async def delete(self, scope: MemoryScope, schema_name: str) -> None:
        await self.store.delete(PROFILE_NAMESPACE, self.profile_key(scope, schema_name))

    async def list_scope(self, scope: MemoryScope) -> list[SemanticProfile]:
        prefix = scope.to_string() + ":"
        keys = await self.store.list_keys(PROFILE_NAMESPACE, prefix)
        profiles: list[SemanticProfile] = []
        for key in keys:
            data = await self.store.load(PROFILE_NAMESPACE, key)
            if data is not None:
                profiles.append(SemanticProfile.model_validate(data))
        return profiles
