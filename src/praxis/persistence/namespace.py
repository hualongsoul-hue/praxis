"""命名空间隔离管理。

不同组件的数据通过命名空间隔离，
同一组件的不同会话通过会话 ID 隔离。
"""

from praxis.persistence.store import PersistenceStore


class NamespaceManager:
    """命名空间管理器。"""

    def __init__(self, store: PersistenceStore) -> None:
        self._store = store

    async def clear_namespace(self, namespace: str) -> int:
        """清理指定命名空间的所有数据。

        Args:
            namespace: 命名空间名称。

        Returns:
            清理的条目数。
        """
        return await self._store.clear_namespace(namespace)

    async def namespace_exists(self, namespace: str) -> bool:
        """检查命名空间是否存在（即是否有数据）。"""
        keys = await self._store.list_keys(namespace)
        return len(keys) > 0

    async def namespace_key_count(self, namespace: str) -> int:
        """返回命名空间中的数据条目数。"""
        keys = await self._store.list_keys(namespace)
        return len(keys)
