"""工作记忆 Scratchpad。

Agent 主动维护的结构化笔记，持久化到 S3 文件系统。
支持 progress.json / todos.json / features.json。
"""

from typing import Any

from praxis.persistence.store import PersistenceStore

SCRATCHPAD_NAMESPACE = "scratchpad"


class Scratchpad:
    """Scratchpad 存储管理。

    提供 read/write 接口，数据持久化到 S3。
    上下文重置后可通过读取 Scratchpad 恢复工作状态。
    """

    KNOWN_KEYS: set[str] = {"progress.json", "todos.json", "features.json"}

    def __init__(self, store: PersistenceStore, session_id: str) -> None:
        self.store = store
        self.session_id = session_id

    def storage_key(self, key: str) -> str:
        """生成带会话前缀的存储键。"""
        return f"{self.session_id}:{key}"

    async def read(self, key: str) -> Any | None:
        """读取 Scratchpad 内容。

        Args:
            key: 草稿键名（如 progress.json）。

        Returns:
            草稿内容（JSON 可序列化对象），或 None。
        """
        return await self.store.load(SCRATCHPAD_NAMESPACE, self.storage_key(key))

    async def write(self, key: str, content: Any) -> None:
        """写入 Scratchpad 内容。

        Args:
            key: 草稿键名。
            content: 草稿内容（JSON 可序列化对象）。
        """
        await self.store.save(SCRATCHPAD_NAMESPACE, self.storage_key(key), content)

    async def delete(self, key: str) -> None:
        """删除 Scratchpad 条目。"""
        await self.store.delete(SCRATCHPAD_NAMESPACE, self.storage_key(key))

    async def list_keys(self) -> list[str]:
        """列出当前会话的所有 Scratchpad 键名。"""
        prefix = f"{self.session_id}:"
        raw_keys = await self.store.list_keys(SCRATCHPAD_NAMESPACE, prefix)
        return [k.removeprefix(prefix) for k in raw_keys]

    async def export_state(self) -> dict[str, Any]:
        """导出所有 Scratchpad 状态，用于检查点。"""
        keys = await self.list_keys()
        state: dict[str, Any] = {}
        for key in keys:
            value = await self.read(key)
            if value is not None:
                state[key] = value
        return state

    async def import_state(self, state: dict[str, Any]) -> None:
        """从检查点恢复 Scratchpad 状态。"""
        for key, value in state.items():
            await self.write(key, value)

    async def clear(self) -> int:
        """清空当前会话所有 Scratchpad。返回清除数量。"""
        keys = await self.list_keys()
        for key in keys:
            await self.delete(key)
        return len(keys)
