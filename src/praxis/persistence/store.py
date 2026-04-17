"""统一存储抽象层。

屏蔽底层存储差异，提供 save/load/delete/list_keys 统一接口。
数据自动 JSON 序列化/反序列化，支持 Pydantic 模型。
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from praxis.config.schemas import PersistenceConfig
from praxis.exceptions import PersistenceError
from praxis.persistence.backends.filesystem import FilesystemBackend
from praxis.persistence.backends.redis import RedisBackend
from praxis.persistence.backends.sqlite import SqliteBackend


def json_default(obj: Any) -> Any:
    """JSON 序列化扩展：支持 Pydantic 模型、datetime、Path、bytes。"""
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def serialize(data: Any) -> bytes:
    """将数据序列化为 JSON bytes。"""
    return json.dumps(data, ensure_ascii=False, default=json_default).encode("utf-8")


def deserialize(raw: bytes) -> Any:
    """将 JSON bytes 反序列化为数据。"""
    return json.loads(raw.decode("utf-8"))


@runtime_checkable
class StorageBackend(Protocol):
    """存储后端协议。所有后端（SQLite/Redis/Filesystem）实现此接口。"""

    async def save(self, namespace: str, key: str, data: bytes) -> None: ...

    async def load(self, namespace: str, key: str) -> bytes | None: ...

    async def delete(self, namespace: str, key: str) -> None: ...

    async def list_keys(
        self, namespace: str, prefix: str | None = None
    ) -> list[str]: ...

    async def clear_namespace(self, namespace: str) -> int: ...

    async def close(self) -> None: ...


class PersistenceStore:
    """统一持久化存储。

    在 StorageBackend 上叠加 JSON 序列化/反序列化，对上层组件屏蔽存储细节。
    """

    def __init__(self, backend: StorageBackend) -> None:
        self._backend = backend

    async def save(self, namespace: str, key: str, data: Any) -> None:
        """保存数据到指定命名空间。"""
        raw = serialize(data)
        await self._backend.save(namespace, key, raw)

    async def load(self, namespace: str, key: str) -> Any | None:
        """从指定命名空间加载数据，不存在时返回 None。"""
        raw = await self._backend.load(namespace, key)
        if raw is None:
            return None
        return deserialize(raw)

    async def delete(self, namespace: str, key: str) -> None:
        """删除指定命名空间中的数据。"""
        await self._backend.delete(namespace, key)

    async def list_keys(
        self, namespace: str, prefix: str | None = None
    ) -> list[str]:
        """列出指定命名空间中的所有键，可按前缀过滤。"""
        return await self._backend.list_keys(namespace, prefix)

    async def clear_namespace(self, namespace: str) -> int:
        """清理指定命名空间的所有数据。返回清理的条目数。"""
        return await self._backend.clear_namespace(namespace)

    async def close(self) -> None:
        """关闭存储后端连接。"""
        await self._backend.close()


async def create_store(config: PersistenceConfig) -> PersistenceStore:
    """根据配置创建持久化存储实例。

    Args:
        config: 持久化引擎配置，``backend`` 字段决定使用的后端。

    Returns:
        初始化完毕的 PersistenceStore。
    """
    if config.backend == "sqlite":
        backend = await SqliteBackend.create(config.sqlite_path)
    elif config.backend == "filesystem":
        backend = FilesystemBackend(config.filesystem_path)
    elif config.backend == "redis":
        backend = await RedisBackend.create(config.redis_url)
    else:
        raise PersistenceError(f"不支持的存储后端: {config.backend}")
    return PersistenceStore(backend)
