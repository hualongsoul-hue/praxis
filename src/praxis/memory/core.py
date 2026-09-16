"""CognitiveMemory——S6 认知记忆系统统一门面。

基于认知科学的四类记忆模型（语义/情景/程序/工作），集成所有记忆子组件，
对外暴露 PRD F6 定义的完整接口契约。

生命周期的 start()/stop() 由 S12（SessionFactory）管理，Agent 不直接接触。
"""

import asyncio
import os
from datetime import UTC, datetime
from typing import Any, cast

from praxis.config.schemas import MemoryConfig
from praxis.exceptions import CognitiveMemoryError
from praxis.lifecycle import AsyncResourceOwner
from praxis.memory.consolidator import MemoryConsolidator
from praxis.memory.dream import DreamConsolidator, DreamReport, DreamScheduler
from praxis.memory.extractor import MemoryExtractor
from praxis.memory.profile import ProfileManager
from praxis.memory.project_loader import ProjectMemoryLoader
from praxis.memory.retention import RetentionManager
from praxis.memory.retriever import MemoryRetriever
from praxis.memory.scratchpad import Scratchpad
from praxis.memory.store import ProfileStore, ScopedMemoryStore
from praxis.memory.vector import VectorStore
from praxis.memory.worker import BackgroundWorker
from praxis.models.memory import (
    MemoryEntry,
    MemoryIndexEntry,
    MemoryScope,
    MemorySearchResult,
    MemoryType,
    ScopeType,
    SemanticMemory,
    SemanticProfile,
    WorkingMemory,
    WorkingMemoryMessage,
)
from praxis.persistence.store import PersistenceStore
from praxis.protocols import EmbeddingProvider, ModelGateway
from praxis.telemetry.logger import get_logger
from praxis.telemetry.metrics import emit_metric

log = get_logger("memory.core")

COGNITIVE_META_NAMESPACE = "cognitive_memory_meta"


class CognitiveMemory:
    """认知记忆系统统一门面（S6）。

    管理工作记忆、长期记忆、档案、Scratchpad、Dream 调度和后台自治 Worker。
    """

    def __init__(
        self,
        store: PersistenceStore,
        gateway: ModelGateway,
        session_id: str,
        config: MemoryConfig | None = None,
        default_scope: MemoryScope | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.store = store
        self.gateway = gateway
        self.session_id = session_id
        self.config = config or MemoryConfig()
        self.default_scope = default_scope or MemoryScope(
            scope_type=ScopeType.SESSION, scope_id=session_id,
        )
        default_model = self.gateway.config.default_model

        # 持久化层
        self.scoped_store = ScopedMemoryStore(store)
        self.profile_store = ProfileStore(store)

        # 向量索引
        self.owns_vector_store = vector_store is None
        self.vector_store = vector_store or VectorStore(
            self.scoped_store,
            embed_func=embedding_provider.embed if embedding_provider is not None else None,
            api_base=self.config.embedding_api_base,
            api_key=(
                os.getenv(self.config.embedding_api_key_env, "")
                if self.config.embedding_api_key_env
                else ""
            ),
            model=self.config.embedding_model,
            timeout=self.config.embedding_timeout,
            dimensions=self.config.embedding_dimensions,
            provider_id=(
                self.config.embedding_model
                or ("injected-provider" if embedding_provider is not None else "local-lexical-v1")
            ),
            max_memories=self.config.max_memories,
        )

        # 生命周期管理
        self.retention = RetentionManager(
            self.scoped_store,
            version_store=store,
            decay_half_life_days=self.config.decay_half_life_days,
            inactivity_threshold_days=self.config.inactivity_threshold_days,
        )

        # 检索
        self.retriever = MemoryRetriever(self.scoped_store, self.vector_store)

        # 提取 + 整合
        self.extractor = MemoryExtractor(
            gateway=self.gateway,
            model=default_model,
            prompts=self.config.extraction_prompts,
        )
        self.consolidator = MemoryConsolidator(
            gateway=self.gateway,
            vector_store=self.vector_store,
            retention=self.retention,
            model=default_model,
            similarity_threshold=self.config.consolidation_similarity_threshold,
        )

        # 档案
        self.profile_manager = ProfileManager(self.profile_store)

        # Scratchpad
        self.scratchpad = Scratchpad(store, session_id)

        # 项目预加载器
        self.project_loader = ProjectMemoryLoader(self.scoped_store, self.vector_store)

        # Dream
        self.dream_consolidator = DreamConsolidator(
            gateway=self.gateway,
            scoped_store=self.scoped_store,
            retention=self.retention,
            meta_store=store,
            model=default_model,
            min_hours_since_last=self.config.dream_min_hours,
            min_sessions=self.config.dream_min_sessions,
            decay_enabled=self.config.decay_enabled,
            max_memories=self.config.max_memories,
        )
        self.dream_scopes: list[MemoryScope] = [
            MemoryScope.from_string(s) for s in self.config.dream_scopes
        ]
        self.dream_scheduler = DreamScheduler(
            consolidator=self.dream_consolidator,
            scopes=self.dream_scopes,
            check_interval_seconds=self.config.dream_check_interval_seconds,
            session_count_getter=lambda: self.dream_session_count,
        )

        # 工作记忆 + 后台 Worker
        self.working_memory = WorkingMemory(session_id=session_id)
        self.worker = BackgroundWorker(
            process_fn=self.process_batch,
            scope=self.default_scope,
            batch_threshold=self.config.background_batch_threshold,
            interval_seconds=self.config.background_interval_seconds,
        )

        # 持久化的 dream 会话计数（clear_session 时 +1）
        self.dream_session_count: int = 0
        self.project_preloaded: bool = False

    # ──────────────────────────────────────────────────────────────────
    # 生命周期（S12 调用）
    # ──────────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """启动后台 Worker 与 Dream 调度器；首次启动时预加载 praxis.md。"""
        await self.load_meta()
        await self.vector_store.start()
        await self.preload_project_memory()
        if self.config.background_enabled:
            self.worker.start()
        if self.config.dream_enabled and self.dream_scopes:
            self.dream_scheduler.start()
        log.info("CognitiveMemory 已启动", session_id=self.session_id)

    async def stop(self) -> None:
        """停止后台 Worker 与 Dream 调度器。"""
        owner = AsyncResourceOwner()
        # Reverse registration order gives tasks -> metadata -> index shutdown.
        if self.owns_vector_store:
            owner.register("memory-index", self.vector_store.aclose)
        owner.register("memory-metadata", self.save_meta)
        owner.register("memory-dream", self.dream_scheduler.stop)
        owner.register("memory-worker", self.worker.stop)
        failures = await owner.close()
        for failure in failures:
            if isinstance(failure, asyncio.CancelledError):
                raise failure
        if failures:
            raise CognitiveMemoryError(
                "记忆关闭期间发生错误",
                details={"failure_types": [type(error).__name__ for error in failures]},
            ) from None
        log.info("CognitiveMemory 已停止", session_id=self.session_id)

    # ──────────────────────────────────────────────────────────────────
    # 工作记忆
    # ──────────────────────────────────────────────────────────────────

    def append_message(self, message: WorkingMemoryMessage) -> None:
        """同步追加消息；通知后台 Worker，不阻塞。"""
        self.working_memory.append(message)
        if self.config.background_enabled:
            self.worker.notify(message)
        emit_metric("memory_message_appended", 1.0, {}, "counter")

    def get_message_history(self, limit: int | None = None) -> list[WorkingMemoryMessage]:
        """获取工作记忆消息历史。"""
        return self.working_memory.get_recent(limit)

    # ──────────────────────────────────────────────────────────────────
    # 热路径：Agent 主动写入/检索
    # ──────────────────────────────────────────────────────────────────

    async def save_memory(
        self,
        content: str,
        scope: MemoryScope | None = None,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """热路径写入：走整合链路。"""
        target_scope = scope or self.default_scope
        entry = SemanticMemory(
            memory_type=memory_type,
            scope=target_scope,
            content=content,
            summary=content[:150],
            tags=tags or [],
            metadata={**(metadata or {}), "source": "hot_path"},
        )
        result = await self.consolidator.consolidate(entry)
        emit_metric(
            "memory_save_hot",
            1.0,
            {"type": memory_type.value, "action": result.action.value},
            "counter",
        )
        log.info(
            "热路径写入",
            memory_id=entry.memory_id,
            type=memory_type.value,
            action=result.action.value,
        )
        return entry.memory_id

    async def search_memory(
        self,
        query: str,
        scopes: list[MemoryScope] | None = None,
        memory_type: MemoryType | None = None,
        tags: list[str] | None = None,
        top_k: int = 10,
    ) -> list[MemorySearchResult]:
        """语义 + 元数据 + 重排。"""
        return await self.retriever.search_memory(
            query=query,
            scopes=scopes,
            memory_type=memory_type,
            tags=tags,
            top_k=top_k,
        )

    async def update_memory(self, memory_id: str, content: str) -> None:
        """更新指定记忆（跨作用域按 ID 查找）。"""
        entry = await self.scoped_store.find_by_id(memory_id)
        if entry is None:
            log.warning("更新记忆失败: 未找到", memory_id=memory_id)
            return

        updated = entry.model_copy(update={
            "content": content,
            "summary": content[:150],
            "metadata": {**entry.metadata, "source": "manual_update"},
            "embedding": None,
        })
        await self.retention.supersede(entry, updated, "手动更新")
        await self.vector_store.remove(entry.memory_id)
        await self.vector_store.add(updated)

    async def delete_memory(self, memory_id: str) -> None:
        """删除指定记忆（标记为 INACTIVE）。"""
        entry = await self.scoped_store.find_by_id(memory_id)
        if entry is None:
            return
        await self.retention.mark_inactive(entry, "手动删除")
        await self.vector_store.remove(entry.memory_id)

    # ──────────────────────────────────────────────────────────────────
    # 渐进式检索
    # ──────────────────────────────────────────────────────────────────

    async def get_memory_index(
        self,
        scopes: list[MemoryScope] | None = None,
    ) -> list[MemoryIndexEntry]:
        """轻量索引（第一层）。"""
        return await self.retriever.get_memory_index(scopes=scopes)

    async def load_memory_detail(
        self,
        scope: MemoryScope,
        memory_id: str,
    ) -> MemoryEntry | None:
        """完整内容（第三层）。"""
        return await self.retriever.load_memory_detail(scope, memory_id)

    # ──────────────────────────────────────────────────────────────────
    # 档案（Profile 模式）
    # ──────────────────────────────────────────────────────────────────

    async def get_profile(
        self,
        scope: MemoryScope,
        schema_name: str,
    ) -> dict[str, Any] | None:
        return await self.profile_manager.get(scope, schema_name)

    async def update_profile(
        self,
        scope: MemoryScope,
        schema_name: str,
        fields: dict[str, Any],
    ) -> SemanticProfile:
        return await self.profile_manager.update(scope, schema_name, fields)

    # ──────────────────────────────────────────────────────────────────
    # Scratchpad
    # ──────────────────────────────────────────────────────────────────

    async def read_scratchpad(self, key: str) -> Any | None:
        return await self.scratchpad.read(key)

    async def write_scratchpad(self, key: str, content: Any) -> None:
        await self.scratchpad.write(key, content)

    # ──────────────────────────────────────────────────────────────────
    # Dream
    # ──────────────────────────────────────────────────────────────────

    async def run_dream(
        self,
        scopes: list[MemoryScope] | None = None,
    ) -> DreamReport:
        """手动触发梦境整理。"""
        target_scopes = scopes or self.dream_scopes
        return await self.dream_consolidator.run_dream(target_scopes)

    # ──────────────────────────────────────────────────────────────────
    # 会话级清理 / 检查点
    # ──────────────────────────────────────────────────────────────────

    async def clear_session(self) -> None:
        """会话级清理：清空工作记忆、pending、默认作用域；dream 会话计数 +1。"""
        self.working_memory.clear()
        self.worker.clear_pending()
        await self.scratchpad.clear()
        if self.default_scope.scope_type == ScopeType.SESSION:
            await self.scoped_store.clear_scope(self.default_scope)
        self.dream_session_count += 1
        await self.save_meta()
        log.info(
            "会话级清理完成",
            session_id=self.session_id,
            dream_session_count=self.dream_session_count,
        )

    def export_state(self) -> dict[str, Any]:
        """导出记忆系统状态，用于检查点。"""
        return {
            "session_id": self.session_id,
            "working_memory": self.working_memory.model_dump(mode="json"),
            "last_processed_message_id": self.worker.last_processed_message_id,
            "pending_count": len(self.worker.pending),
            "pending": [m.model_dump(mode="json") for m in self.worker.pending],
            "dream_session_count": self.dream_session_count,
            "project_preloaded": self.project_preloaded,
        }

    async def import_state(self, snapshot: dict[str, Any]) -> None:
        """从检查点恢复。若 Worker 已启动，继续按游标消费。"""
        wm_data = snapshot.get("working_memory")
        if wm_data:
            self.working_memory = WorkingMemory.model_validate(wm_data)
        self.worker.last_processed_message_id = snapshot.get(
            "last_processed_message_id"
        )
        # 恢复尚未消费的待提取消息（旧检查点可能仅有 pending_count，缺省为空）
        pending_data = snapshot.get("pending")
        if not self.config.background_enabled:
            self.worker.pending.clear()
        elif isinstance(pending_data, list):
            self.worker.pending = [
                WorkingMemoryMessage.model_validate(item)
                for item in cast(list[object], pending_data)
            ]
        self.dream_session_count = int(snapshot.get("dream_session_count", 0) or 0)
        self.project_preloaded = bool(snapshot.get("project_preloaded", False))

    # ──────────────────────────────────────────────────────────────────
    # 内部：后台处理入口 / 元数据 / 项目预加载
    # ──────────────────────────────────────────────────────────────────

    async def process_batch(
        self,
        messages: list[WorkingMemoryMessage],
        scope: MemoryScope,
    ) -> int:
        """后台 Worker 的处理入口：提取 → 整合。"""
        if not messages:
            return 0

        conversation = [
            {"role": m.role, "content": m.content}
            for m in messages
        ]
        extracted = await self.extractor.extract(conversation, scope)
        if not extracted:
            return 0
        await self.consolidator.consolidate_batch(extracted)
        return len(extracted)

    async def load_meta(self) -> None:
        """从持久化存储恢复系统元数据（dream 计数等）。"""
        data = await self.store.load(COGNITIVE_META_NAMESPACE, self.session_id)
        if isinstance(data, dict):
            payload = cast(dict[str, object], data)
            count = payload.get("dream_session_count", 0)
            self.dream_session_count = int(count) if isinstance(count, (int, str)) else 0
            self.project_preloaded = bool(payload.get("project_preloaded", False))

    async def save_meta(self) -> None:
        """持久化系统元数据。"""
        await self.store.save(
            COGNITIVE_META_NAMESPACE,
            self.session_id,
            {
                "dream_session_count": self.dream_session_count,
                "project_preloaded": self.project_preloaded,
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )

    async def preload_project_memory(self) -> None:
        """首次启动时加载 praxis.md 到 project 作用域。"""
        if not self.config.load_project_praxis_md:
            return
        if self.project_preloaded:
            return
        if not self.config.project_root:
            return
        project_name = self.config.project_name or "default"
        await self.project_loader.load(
            project_root=self.config.project_root,
            default_project_name=project_name,
        )
        # 无论是否发现文件，标记为已尝试以避免重复扫描
        self.project_preloaded = True
        await self.save_meta()

    # ──────────────────────────────────────────────────────────────────
    # 维护操作
    # ──────────────────────────────────────────────────────────────────

    async def run_decay_sweep(self, scope: MemoryScope) -> int:
        """触发衰减扫描，标记低相关性记忆为 INACTIVE。"""
        return await self.retention.run_decay_sweep(scope)

    async def rebuild_vector_index(self, scopes: list[MemoryScope]) -> int:
        """从持久化存储重建嵌入索引。"""
        return await self.vector_store.rebuild_index(scopes)
