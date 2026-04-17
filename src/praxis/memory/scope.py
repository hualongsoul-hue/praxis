"""多作用域记忆隔离。

session/<id>、project/<name>、user/<id>、global 四级作用域，
复合查询支持，跨作用域隔离防泄露。
"""

from typing import Any

from praxis.models.memory import (
    MemoryEntry,
    MemoryScope,
    MemoryStatus,
    MemoryType,
    ScopeType,
)
from praxis.persistence.store import PersistenceStore

MEMORY_NAMESPACE = "memory"


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
        """保存记忆条目到对应作用域。

        Returns:
            记忆 ID。
        """
        key = self.scope_key(entry.scope, entry.memory_id)
        await self.store.save(MEMORY_NAMESPACE, key, entry.model_dump(mode="json"))
        return entry.memory_id

    async def load(self, scope: MemoryScope, memory_id: str) -> MemoryEntry | None:
        """从指定作用域加载记忆条目。"""
        key = self.scope_key(scope, memory_id)
        data = await self.store.load(MEMORY_NAMESPACE, key)
        if data is None:
            return None
        return MemoryEntry.model_validate(data)

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
        """列出指定作用域中的所有记忆。

        Args:
            scope: 目标作用域。
            memory_type: 可选类型过滤。
            status: 状态过滤，默认只返回 ACTIVE。

        Returns:
            匹配的记忆条目列表。
        """
        prefix = scope.to_string() + ":"
        keys = await self.store.list_keys(MEMORY_NAMESPACE, prefix)
        entries: list[MemoryEntry] = []
        for key in keys:
            data = await self.store.load(MEMORY_NAMESPACE, key)
            if data is None:
                continue
            entry = MemoryEntry.model_validate(data)
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
        """复合作用域查询。

        支持跨多个作用域检索，按类型/标签/状态过滤。
        不同作用域之间严格隔离，仅返回明确指定的作用域内容。

        Args:
            scopes: 目标作用域列表。
            memory_type: 可选类型过滤。
            tags: 可选标签过滤（需全部包含）。
            status: 状态过滤。

        Returns:
            匹配的记忆条目列表。
        """
        results: list[MemoryEntry] = []
        for scope in scopes:
            entries = await self.list_scope(scope, memory_type, status)
            if tags:
                entries = [
                    e for e in entries
                    if all(tag in e.tags for tag in tags)
                ]
            results.extend(entries)
        return results

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
