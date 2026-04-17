"""S6 记忆系统——智能管线验证测试。

测试 Extraction、Consolidation、Pipeline、Background、Dream 模块。
LLM 调用全部 mock。
"""

import asyncio
import json
from datetime import datetime, timezone, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from praxis.config.schemas import MemoryConfig, PersistenceConfig
from praxis.memory.background import BackgroundProcessor
from praxis.memory.consolidation import ConsolidationResult, MemoryConsolidator
from praxis.memory.dream import DreamConsolidator, DreamReport
from praxis.memory.extraction import MemoryExtractor
from praxis.memory.retention import RetentionManager
from praxis.memory.pipeline import MemoryPipeline
from praxis.memory.retrieval import MemoryRetriever
from praxis.memory.scope import ScopedMemoryStore
from praxis.memory.vector_store import VectorStore
from praxis.models.memory import (
    ConsolidationAction,
    MemoryEntry,
    MemoryScope,
    MemoryStatus,
    MemoryType,
    ScopeType,
    WorkingMemoryMessage,
)
from praxis.persistence.store import PersistenceStore, create_store


@pytest.fixture
async def store(tmp_path) -> PersistenceStore:
    config = PersistenceConfig(backend="sqlite", sqlite_path=str(tmp_path / "test.db"))
    s = await create_store(config)
    yield s
    await s.close()


@pytest.fixture
def scoped(store: PersistenceStore) -> ScopedMemoryStore:
    return ScopedMemoryStore(store)


async def mock_embed(text: str) -> list[float]:
    """确定性 mock 嵌入函数。"""
    if "PostgreSQL" in text or "数据库" in text or "DB" in text:
        return [0.9, 0.1, 0.0]
    if "Python" in text or "编程" in text:
        return [0.1, 0.9, 0.0]
    if "Redis" in text or "缓存" in text:
        return [0.8, 0.2, 0.0]
    return [0.4, 0.4, 0.2]


def make_mock_gateway() -> MagicMock:
    """创建 mock GatewayRouter。"""
    gw = MagicMock()
    gw.config = MagicMock()
    gw.config.default_model = "test-model"
    gw.router = MagicMock()
    return gw


# ── Task 8.1: 记忆提取 ─────────────────────────────────────────────────────


class TestExtraction:
    """Task 8.1: 模型辅助记忆提取验证。"""

    async def test_extract_semantic(self) -> None:
        gw = make_mock_gateway()
        extraction_response = json.dumps([
            {"content": "用户偏好 Python 开发", "tags": ["preference"], "confidence": 0.9},
            {"content": "项目使用 PostgreSQL", "tags": ["tech"], "confidence": 0.85},
        ])

        mock_response = MagicMock()
        mock_response.content = extraction_response
        mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        mock_response.id = "test-id"
        mock_response.model = "test-model"
        mock_response.created = 1234567890

        mock_raw = MagicMock()
        mock_raw.choices = [MagicMock()]
        mock_raw.choices[0].message = MagicMock(
            content=extraction_response, tool_calls=None,
        )
        mock_raw.choices[0].finish_reason = "stop"
        mock_raw.usage = MagicMock(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        mock_raw.id = "test-id"
        mock_raw.model = "test-model"
        mock_raw.created = 1234567890

        with patch("praxis.memory.extraction.chat", return_value=mock_response) as mock_chat:
            mock_chat.return_value = mock_response
            extractor = MemoryExtractor(gw)
            scope = MemoryScope(scope_type=ScopeType.USER, scope_id="u1")
            conversation = [
                {"role": "user", "content": "我喜欢用 Python 开发"},
                {"role": "assistant", "content": "好的，我记住了"},
            ]
            entries = await extractor.extract(
                conversation, scope, memory_types=[MemoryType.SEMANTIC],
            )
            assert len(entries) == 2
            assert entries[0].content == "用户偏好 Python 开发"
            assert entries[0].memory_type == MemoryType.SEMANTIC

    async def test_extract_empty_conversation(self) -> None:
        gw = make_mock_gateway()
        mock_response = MagicMock()
        mock_response.content = "[]"

        with patch("praxis.memory.extraction.chat", return_value=mock_response):
            extractor = MemoryExtractor(gw)
            scope = MemoryScope(scope_type=ScopeType.USER, scope_id="u1")
            entries = await extractor.extract(
                [{"role": "user", "content": "你好"}],
                scope,
                memory_types=[MemoryType.SEMANTIC],
            )
            assert len(entries) == 0

    async def test_extract_malformed_response(self) -> None:
        gw = make_mock_gateway()
        mock_response = MagicMock()
        mock_response.content = "not valid json"

        with patch("praxis.memory.extraction.chat", return_value=mock_response):
            extractor = MemoryExtractor(gw)
            scope = MemoryScope(scope_type=ScopeType.GLOBAL)
            entries = await extractor.extract(
                [{"role": "user", "content": "test"}],
                scope,
                memory_types=[MemoryType.SEMANTIC],
            )
            assert len(entries) == 0

    def test_format_conversation(self) -> None:
        result = MemoryExtractor.format_conversation([
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！"},
        ])
        assert "[user]: 你好" in result
        assert "[assistant]: 你好！" in result

    def test_build_entry_empty_content(self) -> None:
        scope = MemoryScope(scope_type=ScopeType.GLOBAL)
        result = MemoryExtractor.build_entry(
            {"content": "", "tags": []}, scope, MemoryType.SEMANTIC,
        )
        assert result is None

    def test_build_entry_valid(self) -> None:
        scope = MemoryScope(scope_type=ScopeType.GLOBAL)
        result = MemoryExtractor.build_entry(
            {"content": "测试", "tags": ["t1"], "confidence": 0.9},
            scope, MemoryType.SEMANTIC,
        )
        assert result is not None
        assert result.content == "测试"
        assert result.confidence == 0.9


# ── Task 8.2: 记忆整合 ─────────────────────────────────────────────────────


class TestConsolidation:
    """Task 8.2: 记忆整合验证。"""

    async def test_add_no_similar(self, store: PersistenceStore) -> None:
        scoped = ScopedMemoryStore(store)
        vs = VectorStore(scoped, embed_func=mock_embed)
        lifecycle = RetentionManager(scoped, store)
        gw = make_mock_gateway()

        consolidator = MemoryConsolidator(gw, vs, lifecycle)
        scope = MemoryScope(scope_type=ScopeType.GLOBAL)
        entry = MemoryEntry(
            memory_type=MemoryType.SEMANTIC, scope=scope,
            content="全新的知识",
        )
        result = await consolidator.consolidate(entry)
        assert result.action == ConsolidationAction.ADD

    async def test_noop_duplicate(self, store: PersistenceStore) -> None:
        scoped = ScopedMemoryStore(store)
        vs = VectorStore(scoped, embed_func=mock_embed)
        lifecycle = RetentionManager(scoped, store)
        gw = make_mock_gateway()

        scope = MemoryScope(scope_type=ScopeType.GLOBAL)
        existing = MemoryEntry(
            memory_type=MemoryType.SEMANTIC, scope=scope,
            content="项目使用 PostgreSQL 数据库",
        )
        await vs.add(existing)

        noop_response = MagicMock()
        noop_response.content = json.dumps({
            "action": "noop",
            "reasoning": "冗余信息",
            "merged_content": "",
        })

        with patch("praxis.memory.consolidation.chat", return_value=noop_response):
            consolidator = MemoryConsolidator(gw, vs, lifecycle, similarity_threshold=0.3)
            new_entry = MemoryEntry(
                memory_type=MemoryType.SEMANTIC, scope=scope,
                content="DB 是 PostgreSQL",
            )
            result = await consolidator.consolidate(new_entry)
            assert result.action == ConsolidationAction.NOOP

    async def test_update_existing(self, store: PersistenceStore) -> None:
        scoped = ScopedMemoryStore(store)
        vs = VectorStore(scoped, embed_func=mock_embed)
        lifecycle = RetentionManager(scoped, store)
        gw = make_mock_gateway()

        scope = MemoryScope(scope_type=ScopeType.GLOBAL)
        existing = MemoryEntry(
            memory_type=MemoryType.SEMANTIC, scope=scope,
            content="项目使用 PostgreSQL 数据库",
        )
        await vs.add(existing)

        update_response = MagicMock()
        update_response.content = json.dumps({
            "action": "update",
            "reasoning": "补充版本信息",
            "merged_content": "项目使用 PostgreSQL 15 数据库，部署在 AWS RDS",
        })

        with patch("praxis.memory.consolidation.chat", return_value=update_response):
            consolidator = MemoryConsolidator(gw, vs, lifecycle, similarity_threshold=0.3)
            new_entry = MemoryEntry(
                memory_type=MemoryType.SEMANTIC, scope=scope,
                content="PostgreSQL 部署在 AWS RDS 数据库服务",
            )
            result = await consolidator.consolidate(new_entry)
            assert result.action == ConsolidationAction.UPDATE
            assert "AWS RDS" in result.merged_content

    def test_parse_decision_fallback(self) -> None:
        result = MemoryConsolidator.parse_decision("invalid json")
        assert result.action == ConsolidationAction.ADD


# ── Task 8.3: 双路径处理 ───────────────────────────────────────────────────


class TestPipeline:
    """Task 8.3: 双路径处理验证。"""

    async def test_hot_path_save_memory(self, store: PersistenceStore) -> None:
        scoped = ScopedMemoryStore(store)
        vs = VectorStore(scoped, embed_func=mock_embed)
        retriever = MemoryRetriever(scoped, vs)
        gw = make_mock_gateway()
        extractor = MemoryExtractor(gw)
        lifecycle = RetentionManager(scoped, store)
        consolidator = MemoryConsolidator(gw, vs, lifecycle)

        pipeline = MemoryPipeline(
            scoped, vs, retriever, extractor, consolidator, lifecycle,
            session_id="sess-1",
        )
        mid = await pipeline.save_memory("Python 是首选语言")
        assert mid is not None

        results = await pipeline.search_memory("Python")
        assert len(results) > 0

    async def test_append_message_and_pending(self, store: PersistenceStore) -> None:
        scoped = ScopedMemoryStore(store)
        vs = VectorStore(scoped, embed_func=mock_embed)
        retriever = MemoryRetriever(scoped, vs)
        gw = make_mock_gateway()
        extractor = MemoryExtractor(gw)
        lifecycle = RetentionManager(scoped, store)
        consolidator = MemoryConsolidator(gw, vs, lifecycle)

        pipeline = MemoryPipeline(
            scoped, vs, retriever, extractor, consolidator, lifecycle,
            session_id="sess-1",
        )
        pipeline.append_message(WorkingMemoryMessage(role="user", content="你好"))
        pipeline.append_message(WorkingMemoryMessage(role="assistant", content="你好！"))

        assert len(pipeline.pending_messages) == 2
        assert len(pipeline.get_message_history()) == 2

    async def test_process_pending(self, store: PersistenceStore) -> None:
        scoped = ScopedMemoryStore(store)
        vs = VectorStore(scoped, embed_func=mock_embed)
        retriever = MemoryRetriever(scoped, vs)
        gw = make_mock_gateway()
        extractor = MemoryExtractor(gw)
        lifecycle = RetentionManager(scoped, store)
        consolidator = MemoryConsolidator(gw, vs, lifecycle)

        pipeline = MemoryPipeline(
            scoped, vs, retriever, extractor, consolidator, lifecycle,
            session_id="sess-1",
        )
        pipeline.append_message(WorkingMemoryMessage(role="user", content="使用 Python"))

        extract_response = MagicMock()
        extract_response.content = json.dumps([
            {"content": "用户使用 Python", "tags": ["lang"], "confidence": 0.9},
        ])

        with patch("praxis.memory.extraction.chat", return_value=extract_response):
            count = await pipeline.process_pending()
            assert count == 3
            assert len(pipeline.pending_messages) == 0

    async def test_export_import_state(self, store: PersistenceStore) -> None:
        scoped = ScopedMemoryStore(store)
        vs = VectorStore(scoped, embed_func=mock_embed)
        retriever = MemoryRetriever(scoped, vs)
        gw = make_mock_gateway()
        extractor = MemoryExtractor(gw)
        lifecycle = RetentionManager(scoped, store)
        consolidator = MemoryConsolidator(gw, vs, lifecycle)

        pipeline = MemoryPipeline(
            scoped, vs, retriever, extractor, consolidator, lifecycle,
            session_id="sess-1",
        )
        pipeline.append_message(WorkingMemoryMessage(role="user", content="test"))
        pipeline.last_processed_cursor = 5

        state = pipeline.export_state()
        assert state["last_processed_cursor"] == 5
        assert state["session_id"] == "sess-1"

        pipeline2 = MemoryPipeline(
            scoped, vs, retriever, extractor, consolidator, lifecycle,
            session_id="sess-2",
        )
        await pipeline2.import_state(state)
        assert pipeline2.last_processed_cursor == 5

    async def test_clear_session(self, store: PersistenceStore) -> None:
        scoped = ScopedMemoryStore(store)
        vs = VectorStore(scoped, embed_func=mock_embed)
        retriever = MemoryRetriever(scoped, vs)
        gw = make_mock_gateway()
        extractor = MemoryExtractor(gw)
        lifecycle = RetentionManager(scoped, store)
        consolidator = MemoryConsolidator(gw, vs, lifecycle)

        pipeline = MemoryPipeline(
            scoped, vs, retriever, extractor, consolidator, lifecycle,
            session_id="sess-1",
        )
        pipeline.append_message(WorkingMemoryMessage(role="user", content="test"))
        await pipeline.clear_session()
        assert len(pipeline.working_memory.messages) == 0
        assert len(pipeline.pending_messages) == 0


# ── Task 8.4: 后台异步自治任务 ─────────────────────────────────────────────


class TestBackground:
    """Task 8.4: 后台处理器验证。"""

    async def test_start_stop(self, store: PersistenceStore) -> None:
        scoped = ScopedMemoryStore(store)
        vs = VectorStore(scoped, embed_func=mock_embed)
        retriever = MemoryRetriever(scoped, vs)
        gw = make_mock_gateway()
        extractor = MemoryExtractor(gw)
        lifecycle = RetentionManager(scoped, store)
        consolidator = MemoryConsolidator(gw, vs, lifecycle)

        pipeline = MemoryPipeline(
            scoped, vs, retriever, extractor, consolidator, lifecycle,
            session_id="sess-1",
        )
        config = MemoryConfig(background_batch_threshold=2, background_interval_seconds=0.1)
        bg = BackgroundProcessor(pipeline, config)

        bg.start()
        assert bg.running is True
        await asyncio.sleep(0.05)

        await bg.stop()
        assert bg.running is False

    async def test_notify_triggers_processing(self, store: PersistenceStore) -> None:
        scoped = ScopedMemoryStore(store)
        vs = VectorStore(scoped, embed_func=mock_embed)
        retriever = MemoryRetriever(scoped, vs)
        gw = make_mock_gateway()
        extractor = MemoryExtractor(gw)
        lifecycle = RetentionManager(scoped, store)
        consolidator = MemoryConsolidator(gw, vs, lifecycle)

        pipeline = MemoryPipeline(
            scoped, vs, retriever, extractor, consolidator, lifecycle,
            session_id="sess-1",
        )
        config = MemoryConfig(background_batch_threshold=2, background_interval_seconds=0.1)
        bg = BackgroundProcessor(pipeline, config)
        bg.start()

        pipeline.append_message(WorkingMemoryMessage(role="user", content="msg1"))
        pipeline.append_message(WorkingMemoryMessage(role="user", content="msg2"))
        bg.notify()

        extract_response = MagicMock()
        extract_response.content = json.dumps([
            {"content": "测试记忆", "tags": [], "confidence": 0.8},
        ])

        with patch("praxis.memory.extraction.chat", return_value=extract_response):
            await asyncio.sleep(0.3)

        await bg.stop()

    async def test_export_import_state(self, store: PersistenceStore) -> None:
        scoped = ScopedMemoryStore(store)
        vs = VectorStore(scoped, embed_func=mock_embed)
        retriever = MemoryRetriever(scoped, vs)
        gw = make_mock_gateway()
        extractor = MemoryExtractor(gw)
        lifecycle = RetentionManager(scoped, store)
        consolidator = MemoryConsolidator(gw, vs, lifecycle)

        pipeline = MemoryPipeline(
            scoped, vs, retriever, extractor, consolidator, lifecycle,
            session_id="sess-1",
        )
        config = MemoryConfig()
        bg = BackgroundProcessor(pipeline, config)

        pipeline.last_processed_cursor = 10
        state = bg.export_state()
        assert state["pipeline"]["last_processed_cursor"] == 10


# ── Task 8.5: 梦境整理 ─────────────────────────────────────────────────────


class TestDream:
    """Task 8.5: 梦境整理验证。"""

    async def test_should_run_first_time(self, store: PersistenceStore) -> None:
        scoped = ScopedMemoryStore(store)
        lifecycle = RetentionManager(scoped, store)
        gw = make_mock_gateway()

        dc = DreamConsolidator(
            gw, scoped, lifecycle, store,
            min_sessions=3,
        )
        assert await dc.should_run(3) is True
        assert await dc.should_run(2) is False

    async def test_run_dream_empty(self, store: PersistenceStore) -> None:
        scoped = ScopedMemoryStore(store)
        lifecycle = RetentionManager(scoped, store)
        gw = make_mock_gateway()

        dc = DreamConsolidator(gw, scoped, lifecycle, store)
        scope = MemoryScope(scope_type=ScopeType.GLOBAL)
        report = await dc.run_dream([scope])
        assert report.total_reviewed == 0
        assert report.summary == "无记忆条目需要整理"

    async def test_run_dream_with_entries(self, store: PersistenceStore) -> None:
        scoped = ScopedMemoryStore(store)
        lifecycle = RetentionManager(scoped, store)
        gw = make_mock_gateway()

        scope = MemoryScope(scope_type=ScopeType.GLOBAL)
        e1 = MemoryEntry(
            memory_type=MemoryType.SEMANTIC, scope=scope,
            content="昨天部署了新版本",
        )
        e2 = MemoryEntry(
            memory_type=MemoryType.SEMANTIC, scope=scope,
            content="用户喜欢 Python",
        )
        await scoped.save(e1)
        await scoped.save(e2)

        dream_response = MagicMock()
        dream_response.content = json.dumps({
            "anchored": [
                {"memory_id": e1.memory_id, "updated_content": "2026-04-16 部署了新版本"},
            ],
            "conflicts": [],
            "stale": [],
            "merge_suggestions": [],
            "summary": "完成 2 条记忆整理",
        })

        with patch("praxis.memory.dream.chat", return_value=dream_response):
            dc = DreamConsolidator(gw, scoped, lifecycle, store)
            report = await dc.run_dream([scope])
            assert report.total_reviewed == 2
            assert report.anchored_count == 1
            assert report.summary == "完成 2 条记忆整理"

        updated = await scoped.load(scope, e1.memory_id)
        assert updated is not None
        assert "2026-04-16" in updated.content

    async def test_run_dream_stale_marking(self, store: PersistenceStore) -> None:
        scoped = ScopedMemoryStore(store)
        lifecycle = RetentionManager(scoped, store)
        gw = make_mock_gateway()

        scope = MemoryScope(scope_type=ScopeType.GLOBAL)
        entry = MemoryEntry(
            memory_type=MemoryType.SEMANTIC, scope=scope,
            content="旧任务已完成",
        )
        await scoped.save(entry)

        dream_response = MagicMock()
        dream_response.content = json.dumps({
            "anchored": [],
            "conflicts": [],
            "stale": [entry.memory_id],
            "merge_suggestions": [],
            "summary": "清理 1 条陈旧记忆",
        })

        with patch("praxis.memory.dream.chat", return_value=dream_response):
            dc = DreamConsolidator(gw, scoped, lifecycle, store)
            report = await dc.run_dream([scope])
            assert report.stale_marked == 1

        loaded = await scoped.load(scope, entry.memory_id)
        assert loaded is not None
        assert loaded.status == MemoryStatus.INACTIVE

    def test_parse_dream_response_invalid(self) -> None:
        result = DreamConsolidator.parse_dream_response("not json")
        assert result == {}

    def test_format_entries(self) -> None:
        entry = MemoryEntry(
            memory_type=MemoryType.SEMANTIC,
            scope=MemoryScope(scope_type=ScopeType.GLOBAL),
            content="test content",
        )
        text = DreamConsolidator.format_entries([entry])
        assert entry.memory_id in text
        assert "semantic" in text
