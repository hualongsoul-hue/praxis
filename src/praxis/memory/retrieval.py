"""元数据过滤 + 重排序 + 渐进式检索。

三层检索架构：轻量索引 → 摘要 → 完整内容。
支持按作用域/类型/标签/时间范围过滤，语义候选集二次评分重排。
"""

from datetime import datetime, timezone
from typing import Any

from praxis.models.memory import (
    MemoryEntry,
    MemoryIndexEntry,
    MemoryScope,
    MemorySearchResult,
    MemoryStatus,
    MemoryType,
)
from praxis.memory.scope import ScopedMemoryStore
from praxis.memory.vector_store import VectorStore


class MemoryRetriever:
    """渐进式记忆检索器。

    整合向量搜索、元数据过滤和重排序。
    """

    def __init__(
        self,
        scoped_store: ScopedMemoryStore,
        vector_store: VectorStore,
    ) -> None:
        self.scoped_store = scoped_store
        self.vector_store = vector_store

    async def get_memory_index(
        self,
        scopes: list[MemoryScope] | None = None,
        memory_type: MemoryType | None = None,
    ) -> list[MemoryIndexEntry]:
        """获取轻量索引（第一层，~150 字符/条），始终加载到系统提示。

        Args:
            scopes: 作用域过滤。
            memory_type: 类型过滤。

        Returns:
            轻量索引条目列表。
        """
        if scopes is None:
            scopes = [MemoryScope.from_string("global")]

        entries = await self.scoped_store.query(scopes, memory_type=memory_type)
        index_entries: list[MemoryIndexEntry] = []
        for entry in entries:
            summary = entry.summary or entry.content[:150]
            index_entries.append(MemoryIndexEntry(
                memory_id=entry.memory_id,
                memory_type=entry.memory_type,
                scope=entry.scope.to_string(),
                summary=summary,
                tags=entry.tags,
                confidence=entry.confidence,
                updated_at=entry.updated_at,
            ))
        return index_entries

    async def search_memory(
        self,
        query: str,
        scopes: list[MemoryScope] | None = None,
        memory_type: MemoryType | None = None,
        tags: list[str] | None = None,
        time_after: datetime | None = None,
        time_before: datetime | None = None,
        top_k: int = 10,
    ) -> list[MemorySearchResult]:
        """语义搜索 + 元数据过滤 + 重排序。

        Args:
            query: 搜索查询文本。
            scopes: 作用域过滤。
            memory_type: 类型过滤。
            tags: 标签过滤（需全部包含）。
            time_after: 时间范围下限。
            time_before: 时间范围上限。
            top_k: 返回结果数量。

        Returns:
            按相关性排序的搜索结果。
        """
        candidates = await self.vector_store.search(
            query=query,
            scopes=scopes,
            memory_type=memory_type,
            top_k=top_k * 3,
        )

        filtered: list[tuple[MemoryEntry, float]] = []
        for entry, score in candidates:
            if tags and not all(tag in entry.tags for tag in tags):
                continue
            if time_after and entry.created_at < time_after:
                continue
            if time_before and entry.created_at > time_before:
                continue
            filtered.append((entry, score))

        reranked = self.rerank(filtered)

        results: list[MemorySearchResult] = []
        for entry, score in reranked[:top_k]:
            results.append(MemorySearchResult(
                entry=entry,
                relevance_score=score,
                source=entry.scope.to_string(),
            ))
        return results

    async def load_memory_detail(self, scope: MemoryScope, memory_id: str) -> MemoryEntry | None:
        """加载完整记忆内容（第三层）。

        Args:
            scope: 记忆所在作用域。
            memory_id: 记忆 ID。

        Returns:
            完整记忆条目，或 None。
        """
        entry = await self.scoped_store.load(scope, memory_id)
        if entry is not None:
            entry.access_count += 1
            entry.last_accessed_at = datetime.now(timezone.utc)
            await self.scoped_store.update(entry)
        return entry

    @staticmethod
    def rerank(
        candidates: list[tuple[MemoryEntry, float]],
    ) -> list[tuple[MemoryEntry, float]]:
        """二次评分重排序。

        综合语义相似度、新鲜度、访问频率和置信度。

        Args:
            candidates: (记忆条目, 语义相似度) 列表。

        Returns:
            重排序后的列表。
        """
        now = datetime.now(timezone.utc)
        scored: list[tuple[MemoryEntry, float]] = []

        for entry, semantic_score in candidates:
            age_hours = max(
                (now - entry.updated_at).total_seconds() / 3600.0,
                0.01,
            )
            freshness = 1.0 / (1.0 + age_hours / 168.0)

            popularity = min(entry.access_count / 10.0, 1.0)

            final_score = (
                0.6 * semantic_score
                + 0.2 * freshness
                + 0.1 * popularity
                + 0.1 * entry.confidence
            )
            scored.append((entry, final_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
