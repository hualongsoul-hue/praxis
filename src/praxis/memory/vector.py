"""向量嵌入存储与语义搜索。

记忆内容向量化存储，基于余弦相似度进行语义检索。
嵌入通过 Hugging Face Text Embeddings Inference (TEI) 服务获取。
"""

import math
from collections.abc import Awaitable, Callable

import httpx

from praxis.memory.store import ScopedMemoryStore
from praxis.models.memory import MemoryEntry, MemoryScope, MemoryStatus, MemoryType
from praxis.telemetry.logger import get_logger

log = get_logger("memory.vector")

EmbeddingFunc = Callable[[str], Awaitable[list[float]]]


DEFAULT_EMBEDDING_API_BASE = "http://localhost:8080"


async def tei_embed(
    text: str,
    api_base: str = DEFAULT_EMBEDDING_API_BASE,
    api_key: str = "",
    timeout: float = 30.0,
    client: httpx.AsyncClient | None = None,
) -> list[float]:
    """通过 TEI 服务获取文本嵌入向量。

    传入 ``client`` 时复用该连接池；否则临时创建并关闭一个客户端。
    """
    url = f"{api_base.rstrip('/')}/embed"
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {"inputs": [text], "normalize": True, "truncate": True}

    if client is not None:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        embeddings = response.json()
    else:
        async with httpx.AsyncClient(timeout=timeout) as tmp_client:
            response = await tmp_client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            embeddings = response.json()

    if not isinstance(embeddings, list) or len(embeddings) == 0:
        raise RuntimeError(f"TEI 返回异常结果: {str(embeddings)[:200]}")
    return embeddings[0]


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """计算两个向量的余弦相似度。"""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class VectorStore:
    """内存向量存储，支持语义搜索。

    与 ScopedMemoryStore 协同：ScopedMemoryStore 负责持久化，
    VectorStore 维护内存中的嵌入索引用于快速语义检索。
    """

    def __init__(
        self,
        scoped_store: ScopedMemoryStore,
        embed_func: EmbeddingFunc | None = None,
        api_base: str = DEFAULT_EMBEDDING_API_BASE,
        api_key: str = "",
        timeout: float = 30.0,
    ) -> None:
        self.scoped_store = scoped_store
        self.api_base = api_base
        self.api_key = api_key
        self.timeout = timeout
        self.embed_func: EmbeddingFunc = embed_func or self.default_embed
        self.index: dict[str, tuple[MemoryEntry, list[float]]] = {}
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """惰性创建并复用共享 httpx 客户端（连接池）。"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def default_embed(self, text: str) -> list[float]:
        return await tei_embed(
            text,
            api_base=self.api_base,
            api_key=self.api_key,
            timeout=self.timeout,
            client=self._get_client(),
        )

    async def aclose(self) -> None:
        """关闭共享 httpx 客户端。"""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    async def add(self, entry: MemoryEntry) -> None:
        """添加记忆并建立嵌入索引。"""
        if entry.embedding is not None:
            vector = entry.embedding
        else:
            vector = await self.embed_func(entry.content)
            entry.embedding = vector
        self.index[entry.memory_id] = (entry, vector)
        await self.scoped_store.save(entry)

    async def remove(self, memory_id: str) -> None:
        """从索引中移除记忆。"""
        self.index.pop(memory_id, None)

    async def search(
        self,
        query: str,
        scopes: list[MemoryScope] | None = None,
        memory_type: MemoryType | None = None,
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[MemoryEntry, float]]:
        """语义搜索。返回 (记忆条目, 相似度) 列表，按相似度降序。

        索引为空时跳过 embedding 调用，避免无意义的 LLM 开销。
        """
        if not self.index:
            return []

        query_vector = await self.embed_func(query)

        scope_strings: set[str] | None = None
        if scopes:
            scope_strings = {s.to_string() for s in scopes}

        scored: list[tuple[MemoryEntry, float]] = []
        for entry, vector in self.index.values():
            if entry.status != MemoryStatus.ACTIVE:
                continue
            if scope_strings and entry.scope.to_string() not in scope_strings:
                continue
            if memory_type and entry.memory_type != memory_type:
                continue
            score = cosine_similarity(query_vector, vector)
            if score >= min_score:
                scored.append((entry, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    async def rebuild_index(self, scopes: list[MemoryScope]) -> int:
        """从持久化存储重建嵌入索引。"""
        count = 0
        for scope in scopes:
            entries = await self.scoped_store.list_scope(scope)
            for entry in entries:
                if entry.embedding is not None:
                    self.index[entry.memory_id] = (entry, entry.embedding)
                else:
                    vector = await self.embed_func(entry.content)
                    entry.embedding = vector
                    self.index[entry.memory_id] = (entry, vector)
                    await self.scoped_store.update(entry)
                count += 1
        return count

    def clear(self) -> None:
        """清空内存索引。"""
        self.index.clear()
