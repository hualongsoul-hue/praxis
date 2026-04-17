"""双路径处理管线。

热路径：Agent 通过 save_memory/search_memory 主动调用。
后台路径：append_message 内部触发异步任务自动提取+整合。
"""

from datetime import datetime, timezone
from typing import Any

from praxis.gateway.router import GatewayRouter
from praxis.memory.consolidation import ConsolidationResult, MemoryConsolidator
from praxis.memory.extraction import MemoryExtractor
from praxis.memory.lifecycle import LifecycleManager
from praxis.memory.retrieval import MemoryRetriever
from praxis.memory.scope import ScopedMemoryStore
from praxis.memory.vector_store import VectorStore
from praxis.models.memory import (
    MemoryEntry,
    MemoryIndexEntry,
    MemoryScope,
    MemorySearchResult,
    MemoryType,
    WorkingMemory,
    WorkingMemoryMessage,
)
from praxis.telemetry.logger import get_logger
from praxis.telemetry.metrics import emit_metric

log = get_logger("memory.pipeline")


class MemoryPipeline:
    """记忆管线——统一热路径和后台路径。

    管理工作记忆、长期记忆存取，以及后台提取/整合的协调。
    """

    def __init__(
        self,
        scoped_store: ScopedMemoryStore,
        vector_store: VectorStore,
        retriever: MemoryRetriever,
        extractor: MemoryExtractor,
        consolidator: MemoryConsolidator,
        lifecycle: LifecycleManager,
        session_id: str,
        default_scope: MemoryScope | None = None,
    ) -> None:
        self.scoped_store = scoped_store
        self.vector_store = vector_store
        self.retriever = retriever
        self.extractor = extractor
        self.consolidator = consolidator
        self.lifecycle = lifecycle
        self.session_id = session_id
        self.default_scope = default_scope or MemoryScope.from_string(
            f"session/{session_id}"
        )
        self.working_memory = WorkingMemory(session_id=session_id)
        self.pending_messages: list[dict[str, str]] = []
        self.last_processed_cursor: int = 0

    def append_message(self, message: WorkingMemoryMessage) -> None:
        """追加消息到工作记忆，并记录待处理消息。

        此方法是同步的，不阻塞调用方。
        后台任务通过 process_pending 消费待处理消息。

        Args:
            message: 消息对象。
        """
        self.working_memory.append(message)
        self.pending_messages.append({
            "role": message.role,
            "content": message.content,
        })
        emit_metric("memory_message_appended", 1.0, {}, "counter")

    def get_message_history(self, limit: int | None = None) -> list[WorkingMemoryMessage]:
        """获取消息历史。

        Args:
            limit: 可选数量限制。

        Returns:
            消息列表。
        """
        return self.working_memory.get_recent(limit)

    async def save_memory(
        self,
        content: str,
        scope: MemoryScope | None = None,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        tags: list[str] | None = None,
    ) -> str:
        """热路径：Agent 主动保存记忆。

        即时写入持久化存储和向量索引。

        Args:
            content: 记忆内容。
            scope: 作用域，默认使用 default_scope。
            memory_type: 认知类型。
            tags: 标签。

        Returns:
            记忆 ID。
        """
        target_scope = scope or self.default_scope
        entry = MemoryEntry(
            memory_type=memory_type,
            scope=target_scope,
            content=content,
            summary=content[:150],
            tags=tags or [],
            metadata={"source": "hot_path"},
        )
        await self.vector_store.add(entry)
        emit_metric("memory_save_hot", 1.0, {"type": memory_type.value}, "counter")
        log.info("热路径保存记忆", memory_id=entry.memory_id, type=memory_type.value)
        return entry.memory_id

    async def search_memory(
        self,
        query: str,
        scopes: list[MemoryScope] | None = None,
        memory_type: MemoryType | None = None,
        tags: list[str] | None = None,
        top_k: int = 10,
    ) -> list[MemorySearchResult]:
        """热路径：Agent 主动搜索记忆。

        Args:
            query: 搜索查询。
            scopes: 作用域过滤。
            memory_type: 类型过滤。
            tags: 标签过滤。
            top_k: 返回数量。

        Returns:
            搜索结果列表。
        """
        return await self.retriever.search_memory(
            query=query,
            scopes=scopes,
            memory_type=memory_type,
            tags=tags,
            top_k=top_k,
        )

    async def update_memory(self, memory_id: str, content: str) -> None:
        """更新记忆内容。

        Args:
            memory_id: 记忆 ID。
            content: 新内容。
        """
        entry = await self.scoped_store.load(self.default_scope, memory_id)
        if entry is None:
            log.warning("更新记忆失败: 未找到", memory_id=memory_id)
            return

        new_entry = MemoryEntry(
            memory_type=entry.memory_type,
            scope=entry.scope,
            content=content,
            summary=content[:150],
            tags=entry.tags,
            confidence=entry.confidence,
            metadata={**entry.metadata, "source": "manual_update"},
        )
        await self.lifecycle.supersede(entry, new_entry, "手动更新")
        await self.vector_store.remove(entry.memory_id)
        await self.vector_store.add(new_entry)

    async def delete_memory(self, memory_id: str) -> None:
        """删除记忆（标记为 INACTIVE）。

        Args:
            memory_id: 记忆 ID。
        """
        entry = await self.scoped_store.load(self.default_scope, memory_id)
        if entry is None:
            return
        await self.lifecycle.mark_inactive(entry, "手动删除")
        await self.vector_store.remove(entry.memory_id)

    async def get_memory_index(
        self,
        scopes: list[MemoryScope] | None = None,
    ) -> list[MemoryIndexEntry]:
        """获取轻量索引。

        Args:
            scopes: 作用域过滤。

        Returns:
            索引条目列表。
        """
        return await self.retriever.get_memory_index(scopes=scopes)

    async def load_memory_detail(
        self,
        scope: MemoryScope,
        memory_id: str,
    ) -> MemoryEntry | None:
        """加载完整记忆内容。"""
        return await self.retriever.load_memory_detail(scope, memory_id)

    async def process_pending(self, scope: MemoryScope | None = None) -> int:
        """后台路径：处理待提取的消息。

        从 pending_messages 中取出未处理的消息，
        调用 Extractor 提取 → Consolidator 整合。

        Args:
            scope: 提取的记忆归属作用域。

        Returns:
            本次提取的记忆条目数。
        """
        if not self.pending_messages:
            return 0

        target_scope = scope or self.default_scope
        messages_to_process = list(self.pending_messages)
        self.pending_messages.clear()

        entries = await self.extractor.extract(messages_to_process, target_scope)
        if not entries:
            self.last_processed_cursor = len(self.working_memory.messages)
            return 0

        await self.consolidator.consolidate_batch(entries)
        self.last_processed_cursor = len(self.working_memory.messages)

        emit_metric(
            "memory_background_extracted",
            float(len(entries)),
            {},
            "counter",
        )
        return len(entries)

    def export_state(self) -> dict[str, Any]:
        """导出管线状态，用于检查点。"""
        return {
            "session_id": self.session_id,
            "last_processed_cursor": self.last_processed_cursor,
            "working_memory": self.working_memory.model_dump(mode="json"),
            "pending_count": len(self.pending_messages),
        }

    async def import_state(self, snapshot: dict[str, Any]) -> None:
        """从检查点恢复管线状态。"""
        self.last_processed_cursor = snapshot.get("last_processed_cursor", 0)
        wm_data = snapshot.get("working_memory")
        if wm_data:
            self.working_memory = WorkingMemory.model_validate(wm_data)

    async def clear_session(self) -> None:
        """清理会话——清空工作记忆和待处理队列。"""
        self.working_memory.clear()
        self.pending_messages.clear()
        self.last_processed_cursor = 0
