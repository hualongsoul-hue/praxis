"""工作记忆 Scratchpad。

Agent 主动维护的结构化笔记，持久化到 S3 键值存储。
仅允许 KNOWN_KEYS 白名单中的键（progress/todos/features）。
"""

from typing import Any

from praxis.persistence.store import PersistenceStore

SCRATCHPAD_NAMESPACE = "scratchpad"


class Scratchpad:
    """Scratchpad 存储管理。"""

    KNOWN_KEYS: set[str] = {"progress.json", "todos.json", "features.json"}

    def __init__(self, store: PersistenceStore, session_id: str) -> None:
        self.store = store
        self.session_id = session_id

    def storage_key(self, key: str) -> str:
        """生成带会话前缀的存储键。"""
        return f"{self.session_id}:{key}"

    def check_key(self, key: str) -> None:
        """校验 key 是否在白名单中。"""
        if key not in self.KNOWN_KEYS:
            raise ValueError(
                f"不支持的 scratchpad 键: {key!r}，允许值为 {sorted(self.KNOWN_KEYS)}"
            )

    async def read(self, key: str) -> Any | None:
        """读取 Scratchpad 内容。key 必须在白名单中。"""
        self.check_key(key)
        return await self.store.load(SCRATCHPAD_NAMESPACE, self.storage_key(key))

    async def write(self, key: str, content: Any) -> None:
        """写入 Scratchpad 内容。key 必须在白名单中。"""
        self.check_key(key)
        await self.store.save(SCRATCHPAD_NAMESPACE, self.storage_key(key), content)

    async def delete(self, key: str) -> None:
        """删除 Scratchpad 条目。key 必须在白名单中。"""
        self.check_key(key)
        await self.store.delete(SCRATCHPAD_NAMESPACE, self.storage_key(key))

    async def list_keys(self) -> list[str]:
        """列出当前会话已写入的 Scratchpad 键名。"""
        prefix = f"{self.session_id}:"
        raw_keys = await self.store.list_keys(SCRATCHPAD_NAMESPACE, prefix)
        return [k.removeprefix(prefix) for k in raw_keys]

    async def export_state(self) -> dict[str, Any]:
        """导出所有 Scratchpad 状态，用于检查点。"""
        keys = await self.list_keys()
        state: dict[str, Any] = {}
        for key in keys:
            if key not in self.KNOWN_KEYS:
                continue
            value = await self.store.load(SCRATCHPAD_NAMESPACE, self.storage_key(key))
            if value is not None:
                state[key] = value
        return state

    async def import_state(self, state: dict[str, Any]) -> None:
        """从检查点恢复 Scratchpad 状态。忽略白名单外的 key。"""
        for key, value in state.items():
            if key not in self.KNOWN_KEYS:
                continue
            await self.store.save(SCRATCHPAD_NAMESPACE, self.storage_key(key), value)

    async def clear(self) -> int:
        """清空当前会话所有 Scratchpad。返回清除数量。"""
        keys = await self.list_keys()
        count = 0
        for key in keys:
            if key not in self.KNOWN_KEYS:
                continue
            await self.store.delete(SCRATCHPAD_NAMESPACE, self.storage_key(key))
            count += 1
        return count
