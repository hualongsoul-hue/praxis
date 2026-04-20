"""向量嵌入存储与语义搜索。

记忆内容向量化存储，基于余弦相似度进行语义检索。
嵌入通过 LiteLLM embedding API（S4）获取。
"""

import math
from collections.abc import Awaitable, Callable

import litellm

from praxis.memory.store import ScopedMemoryStore
from praxis.models.memory import MemoryEntry, MemoryScope, MemoryStatus, MemoryType

EmbeddingFunc = Callable[[str], Awaitable[list[float]]]


async def litellm_embed(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """通过 LiteLLM 获取文本嵌入向量。"""
    response = await litellm.aembedding(model=model, input=[text])
    return response.data[0]["embedding"]


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
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        self.scoped_store = scoped_store
        self.embedding_model = embedding_model
        self.embed_func: EmbeddingFunc = embed_func or self.default_embed
        self.index: dict[str, tuple[MemoryEntry, list[float]]] = {}

    async def default_embed(self, text: str) -> list[float]:
        return await litellm_embed(text, model=self.embedding_model)

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
