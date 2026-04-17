"""向量嵌入存储与语义搜索。

记忆内容向量化存储，基于余弦相似度进行语义检索。
嵌入通过 LiteLLM embedding API（S4）获取。
"""

import math
from typing import Any, Callable, Awaitable

import litellm

from praxis.models.memory import MemoryEntry, MemoryScope, MemoryStatus, MemoryType
from praxis.memory.scope import ScopedMemoryStore


EmbeddingFunc = Callable[[str], Awaitable[list[float]]]


async def litellm_embed(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """通过 LiteLLM 获取文本嵌入向量。

    Args:
        text: 待嵌入文本。
        model: 嵌入模型名称。

    Returns:
        浮点向量。
    """
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
    ) -> None:
        self.scoped_store = scoped_store
        self.embed_func = embed_func or litellm_embed
        self.index: dict[str, tuple[MemoryEntry, list[float]]] = {}

    async def add(self, entry: MemoryEntry) -> None:
        """添加记忆并建立嵌入索引。

        Args:
            entry: 记忆条目（已持久化或即将持久化）。
        """
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
        """语义搜索。

        Args:
            query: 查询文本。
            scopes: 可选作用域过滤。
            memory_type: 可选类型过滤。
            top_k: 返回最相关的前 K 条。
            min_score: 最低相似度阈值。

        Returns:
            (记忆条目, 相似度分数) 列表，按相似度降序。
        """
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

    async def rebuild_index(
        self,
        scopes: list[MemoryScope],
    ) -> int:
        """从持久化存储重建嵌入索引。

        Args:
            scopes: 需要重建索引的作用域列表。

        Returns:
            已索引的条目数量。
        """
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
