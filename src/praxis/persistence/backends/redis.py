"""Redis 存储后端（可选）。

分布式部署场景使用。需要安装 ``praxis[redis]`` 可选依赖。
键格式：``praxis:{namespace}:{key}``。
"""

from collections.abc import AsyncIterator, Awaitable
from typing import cast

import redis.asyncio as aioredis

from praxis.exceptions import PersistenceError

KEY_PREFIX = "praxis"


class RedisBackend:
    """Redis 异步存储后端。"""

    def __init__(self, client: aioredis.Redis) -> None:
        self._client = client

    @classmethod
    async def create(cls, url: str | None) -> "RedisBackend":
        """创建并连接 Redis 后端。"""
        if not url:
            raise PersistenceError("Redis 后端需要配置 redis_url")
        client = aioredis.from_url(url, decode_responses=False)
        await cast(
            Awaitable[bool],
            client.ping(),  # pyright: ignore[reportUnknownMemberType]
        )
        return cls(client)

    def _full_key(self, namespace: str, key: str) -> str:
        return f"{KEY_PREFIX}:{namespace}:{key}"

    async def save(self, namespace: str, key: str, data: bytes) -> None:
        await self._client.set(self._full_key(namespace, key), data)

    async def save_if_absent(self, namespace: str, key: str, data: bytes) -> bool:
        created = await self._client.set(self._full_key(namespace, key), data, nx=True)
        return bool(created)

    async def load(self, namespace: str, key: str) -> bytes | None:
        result = await self._client.get(self._full_key(namespace, key))
        return bytes(result) if result else None

    async def delete(self, namespace: str, key: str) -> None:
        await self._client.delete(self._full_key(namespace, key))

    async def list_keys(
        self, namespace: str, prefix: str | None = None
    ) -> list[str]:
        if prefix:
            pattern = f"{KEY_PREFIX}:{namespace}:{prefix}*"
        else:
            pattern = f"{KEY_PREFIX}:{namespace}:*"
        strip_prefix = f"{KEY_PREFIX}:{namespace}:"
        keys: list[str] = []
        iterator = cast(
            AsyncIterator[bytes | str],
            self._client.scan_iter(match=pattern),  # pyright: ignore[reportUnknownMemberType]
        )
        async for raw_key in iterator:
            decoded = raw_key.decode("utf-8") if isinstance(raw_key, bytes) else raw_key
            keys.append(decoded.removeprefix(strip_prefix))
        return sorted(keys)

    async def clear_namespace(self, namespace: str) -> int:
        pattern = f"{KEY_PREFIX}:{namespace}:*"
        count = 0
        batch: list[bytes | str] = []
        iterator = cast(
            AsyncIterator[bytes | str],
            self._client.scan_iter(match=pattern),  # pyright: ignore[reportUnknownMemberType]
        )
        async for raw_key in iterator:
            batch.append(raw_key)
            if len(batch) >= 500:
                count += int(await self._client.delete(*batch))
                batch.clear()
        if batch:
            count += int(await self._client.delete(*batch))
        return count

    async def close(self) -> None:
        await self._client.aclose()
