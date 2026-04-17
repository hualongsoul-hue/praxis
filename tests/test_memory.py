"""S6 记忆系统（存储与检索）验证测试。"""

import asyncio
import math
from datetime import datetime, timezone, timedelta

import pytest

from praxis.config.schemas import PersistenceConfig
from praxis.models.memory import (
    EpisodicMemory,
    MemoryEntry,
    MemoryIndexEntry,
    MemoryScope,
    MemorySearchResult,
    MemoryStatus,
    MemoryType,
    MemoryVersion,
    ProceduralMemory,
    ScopeType,
    SemanticMemory,
    SemanticMode,
    WorkingMemory,
    WorkingMemoryMessage,
)
from praxis.memory.retention import RetentionManager
from praxis.memory.retrieval import MemoryRetriever
from praxis.memory.scope import ScopedMemoryStore
from praxis.memory.scratchpad import Scratchpad
from praxis.memory.vector_store import VectorStore, cosine_similarity
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


# ── Task 7.1: 四类认知记忆数据模型 ─────────────────────────────────────────


class TestMemoryModels:
    """Task 7.1: 数据模型验证。"""

    def test_semantic_memory_collection(self) -> None:
        mem = SemanticMemory(
            scope=MemoryScope(scope_type=ScopeType.USER, scope_id="u1"),
            content="用户偏好 Python 开发",
            tags=["preference", "language"],
        )
        assert mem.memory_type == MemoryType.SEMANTIC
        assert mem.semantic_mode == SemanticMode.COLLECTION
        assert mem.status == MemoryStatus.ACTIVE

    def test_semantic_memory_profile(self) -> None:
        mem = SemanticMemory(
            scope=MemoryScope(scope_type=ScopeType.PROJECT, scope_id="praxis"),
            content="PostgreSQL 配置",
            semantic_mode=SemanticMode.PROFILE,
            profile_schema={"database": "str", "port": "int"},
        )
        assert mem.semantic_mode == SemanticMode.PROFILE
        assert mem.profile_schema is not None

    def test_episodic_memory(self) -> None:
        mem = EpisodicMemory(
            scope=MemoryScope(scope_type=ScopeType.SESSION, scope_id="s1"),
            content="修复了认证模块的 Bug",
            context_description="用户报告登录失败",
            reasoning="检查日志发现 Token 过期逻辑错误",
            action_taken="修改 Token 刷新逻辑",
            outcome="登录功能恢复正常",
        )
        assert mem.memory_type == MemoryType.EPISODIC
        assert mem.outcome == "登录功能恢复正常"

    def test_procedural_memory(self) -> None:
        mem = ProceduralMemory(
            scope=MemoryScope(scope_type=ScopeType.GLOBAL),
            content="PR 审查流程",
            steps=["lint", "test", "review", "merge"],
            applicable_scenarios=["代码提交", "功能合并"],
        )
        assert mem.memory_type == MemoryType.PROCEDURAL
        assert len(mem.steps) == 4

    def test_working_memory(self) -> None:
        wm = WorkingMemory(session_id="sess-1")
        wm.append(WorkingMemoryMessage(role="user", content="你好"))
        wm.append(WorkingMemoryMessage(role="assistant", content="你好！"))
        assert len(wm.get_recent()) == 2
        assert len(wm.get_recent(1)) == 1
        wm.clear()
        assert len(wm.messages) == 0

    def test_memory_scope_serialization(self) -> None:
        scope = MemoryScope(scope_type=ScopeType.PROJECT, scope_id="praxis")
        assert scope.to_string() == "project/praxis"
        parsed = MemoryScope.from_string("project/praxis")
        assert parsed.scope_type == ScopeType.PROJECT
        assert parsed.scope_id == "praxis"

    def test_memory_scope_global(self) -> None:
        scope = MemoryScope(scope_type=ScopeType.GLOBAL)
        assert scope.to_string() == "global"
        parsed = MemoryScope.from_string("global")
        assert parsed.scope_type == ScopeType.GLOBAL

    def test_memory_entry_serialization(self) -> None:
        entry = MemoryEntry(
            memory_type=MemoryType.SEMANTIC,
            scope=MemoryScope(scope_type=ScopeType.USER, scope_id="u1"),
            content="测试内容",
        )
        data = entry.model_dump(mode="json")
        restored = MemoryEntry.model_validate(data)
        assert restored.memory_id == entry.memory_id
        assert restored.content == "测试内容"


# ── Task 7.2: 多作用域记忆隔离 ─────────────────────────────────────────────


class TestScopedMemoryStore:
    """Task 7.2: 作用域隔离验证。"""

    async def test_save_and_load(self, scoped: ScopedMemoryStore) -> None:
        scope = MemoryScope(scope_type=ScopeType.USER, scope_id="u1")
        entry = MemoryEntry(
            memory_type=MemoryType.SEMANTIC,
            scope=scope,
            content="用户偏好",
        )
        mid = await scoped.save(entry)
        loaded = await scoped.load(scope, mid)
        assert loaded is not None
        assert loaded.content == "用户偏好"

    async def test_scope_isolation(self, scoped: ScopedMemoryStore) -> None:
        scope_a = MemoryScope(scope_type=ScopeType.USER, scope_id="u1")
        scope_b = MemoryScope(scope_type=ScopeType.USER, scope_id="u2")
        entry_a = MemoryEntry(
            memory_type=MemoryType.SEMANTIC,
            scope=scope_a,
            content="用户A偏好",
        )
        entry_b = MemoryEntry(
            memory_type=MemoryType.SEMANTIC,
            scope=scope_b,
            content="用户B偏好",
        )
        await scoped.save(entry_a)
        await scoped.save(entry_b)

        list_a = await scoped.list_scope(scope_a)
        list_b = await scoped.list_scope(scope_b)
        assert len(list_a) == 1
        assert len(list_b) == 1
        assert list_a[0].content == "用户A偏好"
        assert list_b[0].content == "用户B偏好"

    async def test_compound_query(self, scoped: ScopedMemoryStore) -> None:
        scope_proj = MemoryScope(scope_type=ScopeType.PROJECT, scope_id="praxis")
        scope_user = MemoryScope(scope_type=ScopeType.USER, scope_id="u1")
        await scoped.save(MemoryEntry(
            memory_type=MemoryType.SEMANTIC,
            scope=scope_proj,
            content="项目使用 PostgreSQL",
            tags=["db"],
        ))
        await scoped.save(MemoryEntry(
            memory_type=MemoryType.SEMANTIC,
            scope=scope_user,
            content="用户偏好 Python",
            tags=["lang"],
        ))

        results = await scoped.query([scope_proj, scope_user])
        assert len(results) == 2

        results_tagged = await scoped.query([scope_proj, scope_user], tags=["db"])
        assert len(results_tagged) == 1

    async def test_type_filter(self, scoped: ScopedMemoryStore) -> None:
        scope = MemoryScope(scope_type=ScopeType.GLOBAL)
        await scoped.save(MemoryEntry(
            memory_type=MemoryType.SEMANTIC, scope=scope, content="fact",
        ))
        await scoped.save(MemoryEntry(
            memory_type=MemoryType.PROCEDURAL, scope=scope, content="workflow",
        ))

        semantic = await scoped.list_scope(scope, memory_type=MemoryType.SEMANTIC)
        assert len(semantic) == 1
        assert semantic[0].content == "fact"

    async def test_clear_scope(self, scoped: ScopedMemoryStore) -> None:
        scope = MemoryScope(scope_type=ScopeType.SESSION, scope_id="s1")
        await scoped.save(MemoryEntry(
            memory_type=MemoryType.WORKING, scope=scope, content="msg1",
        ))
        await scoped.save(MemoryEntry(
            memory_type=MemoryType.WORKING, scope=scope, content="msg2",
        ))
        count = await scoped.clear_scope(scope)
        assert count == 2
        assert len(await scoped.list_scope(scope)) == 0


# ── Task 7.3: 向量嵌入存储与语义搜索 ───────────────────────────────────────


class TestVectorStore:
    """Task 7.3: 向量存储与语义搜索验证。"""

    def test_cosine_similarity(self) -> None:
        assert cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)
        assert cosine_similarity([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)
        assert cosine_similarity([1, 0, 0], [-1, 0, 0]) == pytest.approx(-1.0)
        assert cosine_similarity([0, 0, 0], [1, 0, 0]) == pytest.approx(0.0)

    async def test_add_and_search(self, scoped: ScopedMemoryStore) -> None:
        call_count = 0

        async def mock_embed(text: str) -> list[float]:
            nonlocal call_count
            call_count += 1
            if "PostgreSQL" in text or "数据库" in text:
                return [0.9, 0.1, 0.0]
            if "Python" in text or "编程" in text:
                return [0.1, 0.9, 0.0]
            return [0.3, 0.3, 0.4]

        vs = VectorStore(scoped, embed_func=mock_embed)
        scope = MemoryScope(scope_type=ScopeType.PROJECT, scope_id="test")

        await vs.add(MemoryEntry(
            memory_type=MemoryType.SEMANTIC, scope=scope,
            content="项目使用 PostgreSQL 数据库",
        ))
        await vs.add(MemoryEntry(
            memory_type=MemoryType.SEMANTIC, scope=scope,
            content="项目使用 Python 编程语言",
        ))

        results = await vs.search("数据库配置", scopes=[scope])
        assert len(results) > 0
        assert "PostgreSQL" in results[0][0].content

    async def test_scope_filter_in_search(self, scoped: ScopedMemoryStore) -> None:
        async def mock_embed(text: str) -> list[float]:
            return [0.5, 0.5]

        vs = VectorStore(scoped, embed_func=mock_embed)
        scope_a = MemoryScope(scope_type=ScopeType.PROJECT, scope_id="a")
        scope_b = MemoryScope(scope_type=ScopeType.PROJECT, scope_id="b")

        await vs.add(MemoryEntry(
            memory_type=MemoryType.SEMANTIC, scope=scope_a, content="data in A",
        ))
        await vs.add(MemoryEntry(
            memory_type=MemoryType.SEMANTIC, scope=scope_b, content="data in B",
        ))

        results_a = await vs.search("data", scopes=[scope_a])
        assert len(results_a) == 1
        assert results_a[0][0].scope.scope_id == "a"

    async def test_inactive_excluded(self, scoped: ScopedMemoryStore) -> None:
        async def mock_embed(text: str) -> list[float]:
            return [0.5, 0.5]

        vs = VectorStore(scoped, embed_func=mock_embed)
        scope = MemoryScope(scope_type=ScopeType.GLOBAL)

        entry = MemoryEntry(
            memory_type=MemoryType.SEMANTIC, scope=scope,
            content="inactive entry", status=MemoryStatus.INACTIVE,
        )
        await vs.add(entry)
        results = await vs.search("inactive")
        assert len(results) == 0


# ── Task 7.4: 元数据过滤+重排序+渐进式检索 ─────────────────────────────────


class TestRetrieval:
    """Task 7.4: 检索系统验证。"""

    async def test_get_memory_index(self, scoped: ScopedMemoryStore) -> None:
        scope = MemoryScope(scope_type=ScopeType.GLOBAL)
        await scoped.save(MemoryEntry(
            memory_type=MemoryType.SEMANTIC, scope=scope,
            content="长内容" * 100,
            summary="短摘要",
            tags=["test"],
        ))

        async def mock_embed(text: str) -> list[float]:
            return [0.5, 0.5]

        vs = VectorStore(scoped, embed_func=mock_embed)
        retriever = MemoryRetriever(scoped, vs)
        index = await retriever.get_memory_index(scopes=[scope])
        assert len(index) == 1
        assert index[0].summary == "短摘要"

    async def test_search_with_tag_filter(self, scoped: ScopedMemoryStore) -> None:
        scope = MemoryScope(scope_type=ScopeType.GLOBAL)

        async def mock_embed(text: str) -> list[float]:
            return [0.5, 0.5]

        vs = VectorStore(scoped, embed_func=mock_embed)

        await vs.add(MemoryEntry(
            memory_type=MemoryType.SEMANTIC, scope=scope,
            content="Python facts", tags=["lang"],
        ))
        await vs.add(MemoryEntry(
            memory_type=MemoryType.SEMANTIC, scope=scope,
            content="DB facts", tags=["db"],
        ))

        retriever = MemoryRetriever(scoped, vs)
        results = await retriever.search_memory("facts", scopes=[scope], tags=["db"])
        assert len(results) == 1
        assert results[0].entry.content == "DB facts"

    async def test_load_memory_detail(self, scoped: ScopedMemoryStore) -> None:
        scope = MemoryScope(scope_type=ScopeType.GLOBAL)
        entry = MemoryEntry(
            memory_type=MemoryType.SEMANTIC, scope=scope,
            content="详细内容",
        )
        await scoped.save(entry)

        async def mock_embed(text: str) -> list[float]:
            return [0.5, 0.5]

        vs = VectorStore(scoped, embed_func=mock_embed)
        retriever = MemoryRetriever(scoped, vs)
        loaded = await retriever.load_memory_detail(scope, entry.memory_id)
        assert loaded is not None
        assert loaded.access_count == 1

    def test_rerank_scoring(self) -> None:
        now = datetime.now(timezone.utc)
        entry_recent = MemoryEntry(
            memory_type=MemoryType.SEMANTIC,
            scope=MemoryScope(scope_type=ScopeType.GLOBAL),
            content="recent",
            updated_at=now,
            access_count=5,
            confidence=0.9,
        )
        entry_old = MemoryEntry(
            memory_type=MemoryType.SEMANTIC,
            scope=MemoryScope(scope_type=ScopeType.GLOBAL),
            content="old",
            updated_at=now - timedelta(days=30),
            access_count=0,
            confidence=0.5,
        )
        candidates = [
            (entry_old, 0.8),
            (entry_recent, 0.7),
        ]
        reranked = MemoryRetriever.rerank(candidates)
        assert reranked[0][0].content == "recent"


# ── Task 7.5: 工作记忆 Scratchpad ──────────────────────────────────────────


class TestScratchpad:
    """Task 7.5: Scratchpad 验证。"""

    async def test_write_and_read(self, store: PersistenceStore) -> None:
        sp = Scratchpad(store, "sess-1")
        await sp.write("progress.json", {"step": 1, "done": False})
        content = await sp.read("progress.json")
        assert content == {"step": 1, "done": False}

    async def test_read_nonexistent(self, store: PersistenceStore) -> None:
        sp = Scratchpad(store, "sess-1")
        assert await sp.read("no_such_key") is None

    async def test_delete(self, store: PersistenceStore) -> None:
        sp = Scratchpad(store, "sess-1")
        await sp.write("todos.json", [{"id": 1}])
        await sp.delete("todos.json")
        assert await sp.read("todos.json") is None

    async def test_list_keys(self, store: PersistenceStore) -> None:
        sp = Scratchpad(store, "sess-1")
        await sp.write("progress.json", {})
        await sp.write("todos.json", [])
        keys = await sp.list_keys()
        assert set(keys) == {"progress.json", "todos.json"}

    async def test_export_import_state(self, store: PersistenceStore) -> None:
        sp1 = Scratchpad(store, "sess-1")
        await sp1.write("progress.json", {"step": 3})
        await sp1.write("todos.json", [1, 2, 3])

        state = await sp1.export_state()
        assert "progress.json" in state

        sp2 = Scratchpad(store, "sess-2")
        await sp2.import_state(state)
        assert await sp2.read("progress.json") == {"step": 3}

    async def test_clear(self, store: PersistenceStore) -> None:
        sp = Scratchpad(store, "sess-1")
        await sp.write("a", 1)
        await sp.write("b", 2)
        count = await sp.clear()
        assert count == 2
        assert len(await sp.list_keys()) == 0

    async def test_session_isolation(self, store: PersistenceStore) -> None:
        sp1 = Scratchpad(store, "sess-1")
        sp2 = Scratchpad(store, "sess-2")
        await sp1.write("data", "session1")
        await sp2.write("data", "session2")
        assert await sp1.read("data") == "session1"
        assert await sp2.read("data") == "session2"


# ── Task 7.6: 记忆生命周期管理 ─────────────────────────────────────────────


class TestLifecycle:
    """Task 7.6: 生命周期管理验证。"""

    async def test_mark_inactive(self, store: PersistenceStore) -> None:
        scoped = ScopedMemoryStore(store)
        lm = RetentionManager(scoped, store)
        scope = MemoryScope(scope_type=ScopeType.GLOBAL)
        entry = MemoryEntry(
            memory_type=MemoryType.SEMANTIC, scope=scope,
            content="旧记忆",
        )
        await scoped.save(entry)
        await lm.mark_inactive(entry, "过时信息")

        loaded = await scoped.load(scope, entry.memory_id)
        assert loaded is not None
        assert loaded.status == MemoryStatus.INACTIVE

    async def test_supersede(self, store: PersistenceStore) -> None:
        scoped = ScopedMemoryStore(store)
        lm = RetentionManager(scoped, store)
        scope = MemoryScope(scope_type=ScopeType.GLOBAL)

        old = MemoryEntry(
            memory_type=MemoryType.SEMANTIC, scope=scope,
            content="用户使用 PostgreSQL",
        )
        await scoped.save(old)

        new = MemoryEntry(
            memory_type=MemoryType.SEMANTIC, scope=scope,
            content="用户迁移到 MySQL",
        )
        new_id = await lm.supersede(old, new, "数据库迁移")

        old_loaded = await scoped.load(scope, old.memory_id)
        assert old_loaded is not None
        assert old_loaded.status == MemoryStatus.SUPERSEDED
        assert old_loaded.superseded_by == new_id

        new_loaded = await scoped.load(scope, new_id)
        assert new_loaded is not None
        assert new_loaded.version == 2

    async def test_version_history(self, store: PersistenceStore) -> None:
        scoped = ScopedMemoryStore(store)
        lm = RetentionManager(scoped, store)
        scope = MemoryScope(scope_type=ScopeType.GLOBAL)

        entry = MemoryEntry(
            memory_type=MemoryType.SEMANTIC, scope=scope,
            content="版本1",
        )
        await scoped.save(entry)
        await lm.save_version(entry, "初始版本")

        history = await lm.get_version_history(entry.memory_id)
        assert len(history) == 1
        assert history[0].version == 1
        assert history[0].content == "版本1"

    def test_compute_decay(self) -> None:
        lm = RetentionManager.__new__(RetentionManager)
        lm.decay_half_life_days = 30.0

        recent = MemoryEntry(
            memory_type=MemoryType.SEMANTIC,
            scope=MemoryScope(scope_type=ScopeType.GLOBAL),
            content="recent",
            updated_at=datetime.now(timezone.utc),
        )
        old = MemoryEntry(
            memory_type=MemoryType.SEMANTIC,
            scope=MemoryScope(scope_type=ScopeType.GLOBAL),
            content="old",
            updated_at=datetime.now(timezone.utc) - timedelta(days=30),
        )

        decay_recent = lm.compute_decay(recent)
        decay_old = lm.compute_decay(old)
        assert decay_recent > decay_old
        assert decay_old == pytest.approx(0.5, abs=0.05)

    async def test_decay_sweep(self, store: PersistenceStore) -> None:
        scoped = ScopedMemoryStore(store)
        lm = RetentionManager(
            scoped, store,
            inactivity_threshold_days=0.001,
            min_access_count=0,
        )
        scope = MemoryScope(scope_type=ScopeType.GLOBAL)

        entry = MemoryEntry(
            memory_type=MemoryType.SEMANTIC, scope=scope,
            content="极旧记忆",
            access_count=0,
            confidence=0.01,
            created_at=datetime.now(timezone.utc) - timedelta(days=365),
            updated_at=datetime.now(timezone.utc) - timedelta(days=365),
            last_accessed_at=datetime.now(timezone.utc) - timedelta(days=365),
        )
        await scoped.save(entry)

        count = await lm.run_decay_sweep(scope)
        assert count == 1
