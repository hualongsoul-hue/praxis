"""元数据过滤 + 综合重排 + 渐进式检索。

三层检索架构：轻量索引 → 摘要 → 完整内容。
"""

from datetime import datetime, timezone

from praxis.memory.store import ScopedMemoryStore
from praxis.memory.vector import VectorStore
from praxis.models.memory import (
    MemoryEntry,
    MemoryIndexEntry,
    MemoryScope,
    MemorySearchResult,
    MemoryType,
)
from praxis.telemetry.metrics import emit_metric


class MemoryRetriever:
    """渐进式记忆检索器。

    整合向量搜索、元数据过滤和综合重排。
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
        """第一层：轻量索引（~150 字符/条），始终加载到系统提示。"""
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
        """第二层：语义搜索 + 元数据过滤 + 综合重排。"""
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

        emit_metric(
            "memory_search",
            1.0,
            {"hit": "1" if results else "0"},
            "counter",
        )
        emit_metric(
            "memory_search_hits",
            float(len(results)),
            {},
            "histogram",
        )
        return results

    async def load_memory_detail(
        self,
        scope: MemoryScope,
        memory_id: str,
    ) -> MemoryEntry | None:
        """第三层：加载完整记忆内容，同时累计 access_count。"""
        entry = await self.scoped_store.load(scope, memory_id)
        if entry is not None:
            entry.access_count += 1
            entry.last_accessed_at = datetime.now(timezone.utc)
            await self.scoped_store.update(entry)
            emit_metric("memory_detail_loaded", 1.0, {}, "counter")
        return entry

    @staticmethod
    def rerank(
        candidates: list[tuple[MemoryEntry, float]],
    ) -> list[tuple[MemoryEntry, float]]:
        """综合重排：final = 0.6·semantic + 0.2·freshness + 0.1·popularity + 0.1·confidence。"""
        now = datetime.now(timezone.utc)
        scored: list[tuple[MemoryEntry, float]] = []

        for entry, semantic_score in candidates:
            age_hours = max((now - entry.updated_at).total_seconds() / 3600.0, 0.01)
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
