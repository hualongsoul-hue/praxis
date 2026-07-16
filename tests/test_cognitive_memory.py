"""S6 记忆系统端到端验证。

覆盖：
- 四类认知记忆模型（含 EpisodicMemory/ProceduralMemory 结构化字段）
- SemanticProfile 档案模式
- ScopedMemoryStore 多态反序列化 / 复合作用域查询 / 跨域隔离
- VectorStore 语义搜索 + INACTIVE 排除
- MemoryRetriever 三层渐进式 + 综合重排
- RetentionManager 动态遗忘 / 版本链 / 审计
- Scratchpad 白名单强校验
- ProfileManager 档案合并 / 版本递增
- ProjectMemoryLoader praxis.md 解析 / 去重
- MemoryExtractor 按类型构造子类
- MemoryConsolidator ADD/UPDATE/NOOP + 失败保守 ADD
- DreamConsolidator + DreamScheduler
- BackgroundWorker 游标推进 / 失败重试 / 不丢消息
- CognitiveMemory 门面闭环：start/stop、append→worker、save/search、export/import
"""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from praxis.config.schemas import MemoryConfig, PersistenceConfig
from praxis.gateway.router import GatewayRouter
from praxis.memory.consolidator import MemoryConsolidator
from praxis.memory.core import CognitiveMemory
from praxis.memory.dream import DreamConsolidator, DreamReport, DreamScheduler
from praxis.memory.extractor import MemoryExtractor
from praxis.memory.profile import ProfileManager
from praxis.memory.project_loader import ProjectMemoryLoader
from praxis.memory.retention import RetentionManager
from praxis.memory.retriever import MemoryRetriever
from praxis.memory.scratchpad import Scratchpad
from praxis.memory.store import ProfileStore, ScopedMemoryStore, entry_from_dict
from praxis.memory.vector import VectorStore, cosine_similarity
from praxis.memory.worker import BackgroundWorker
from praxis.models.memory import (
    ConsolidationAction,
    EpisodicMemory,
    MemoryScope,
    MemoryStatus,
    MemoryType,
    ProceduralMemory,
    ScopeType,
    SemanticMemory,
    SemanticProfile,
    WorkingMemory,
    WorkingMemoryMessage,
)
from praxis.persistence.store import PersistenceStore, create_store

# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
async def store(tmp_path) -> PersistenceStore:
    config = PersistenceConfig(backend="sqlite", sqlite_path=str(tmp_path / "test.db"))
    s = await create_store(config)
    yield s
    await s.close()


@pytest.fixture
def scoped(store: PersistenceStore) -> ScopedMemoryStore:
    return ScopedMemoryStore(store)


@pytest.fixture
def mock_gateway() -> GatewayRouter:
    gw = MagicMock(spec=GatewayRouter)
    gw.config = MagicMock()
    gw.config.default_model = "mock"
    gw.config.max_budget = None
    gw.router = MagicMock()
    return gw


def make_chat_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.content = content
    return resp


async def mock_embed(text: str) -> list[float]:
    """简单确定性嵌入：按关键词映射到固定向量。"""
    if "PostgreSQL" in text or "数据库" in text or "postgres" in text.lower():
        return [0.9, 0.1, 0.0]
    if "Python" in text or "编程" in text or "python" in text.lower():
        return [0.1, 0.9, 0.0]
    if "MySQL" in text or "mysql" in text.lower():
        return [0.85, 0.15, 0.0]
    return [0.3, 0.3, 0.4]


# ── 认知记忆模型 ────────────────────────────────────────────────────────────


class TestMemoryModels:
    def test_semantic_memory(self) -> None:
        mem = SemanticMemory(
            scope=MemoryScope(scope_type=ScopeType.USER, scope_id="u1"),
            content="用户偏好 Python",
            tags=["preference"],
        )
        assert mem.memory_type == MemoryType.SEMANTIC
        assert mem.status == MemoryStatus.ACTIVE

    def test_episodic_structured_fields(self) -> None:
        mem = EpisodicMemory(
            scope=MemoryScope(scope_type=ScopeType.SESSION, scope_id="s1"),
            content="修复登录 Bug",
            context_description="用户报告登录失败",
            reasoning="Token 过期逻辑错误",
            action_taken="修改刷新逻辑",
            outcome="登录恢复",
        )
        assert mem.memory_type == MemoryType.EPISODIC
        assert mem.outcome == "登录恢复"

    def test_procedural_structured_fields(self) -> None:
        mem = ProceduralMemory(
            scope=MemoryScope(scope_type=ScopeType.GLOBAL),
            content="PR 审查",
            steps=["lint", "test", "review"],
            applicable_scenarios=["合并"],
        )
        assert len(mem.steps) == 3
        assert mem.applicable_scenarios == ["合并"]

    def test_working_memory_message_has_id(self) -> None:
        msg = WorkingMemoryMessage(role="user", content="hi")
        assert msg.message_id
        assert len(msg.message_id) >= 8

    def test_working_memory_max_truncation(self) -> None:
        wm = WorkingMemory(session_id="s", max_messages=3)
        for i in range(5):
            wm.append(WorkingMemoryMessage(role="user", content=str(i)))
        assert len(wm.messages) == 3
        assert wm.messages[0].content == "2"

    def test_semantic_profile(self) -> None:
        profile = SemanticProfile(
            scope=MemoryScope(scope_type=ScopeType.PROJECT, scope_id="x"),
            schema_name="db_config",
            fields={"database": "pg", "port": 5432},
        )
        assert profile.version == 1
        assert profile.fields["port"] == 5432

    def test_scope_round_trip(self) -> None:
        scope = MemoryScope(scope_type=ScopeType.PROJECT, scope_id="praxis")
        assert MemoryScope.from_string(scope.to_string()) == scope
        assert MemoryScope.from_string("global").scope_type == ScopeType.GLOBAL


# ── ScopedMemoryStore 多态反序列化 ─────────────────────────────────────────


class TestScopedStore:
    async def test_round_trip_episodic_preserves_fields(
        self, scoped: ScopedMemoryStore,
    ) -> None:
        scope = MemoryScope(scope_type=ScopeType.SESSION, scope_id="s1")
        entry = EpisodicMemory(
            scope=scope,
            content="测试",
            context_description="上下文",
            reasoning="推理",
            action_taken="动作",
            outcome="结果",
        )
        await scoped.save(entry)
        loaded = await scoped.load(scope, entry.memory_id)
        assert isinstance(loaded, EpisodicMemory)
        assert loaded.reasoning == "推理"
        assert loaded.outcome == "结果"

    async def test_round_trip_procedural_preserves_fields(
        self, scoped: ScopedMemoryStore,
    ) -> None:
        scope = MemoryScope(scope_type=ScopeType.GLOBAL)
        entry = ProceduralMemory(
            scope=scope,
            content="流程",
            steps=["a", "b"],
            applicable_scenarios=["场景1"],
        )
        await scoped.save(entry)
        loaded = await scoped.load(scope, entry.memory_id)
        assert isinstance(loaded, ProceduralMemory)
        assert loaded.steps == ["a", "b"]

    async def test_scope_isolation(self, scoped: ScopedMemoryStore) -> None:
        s_a = MemoryScope(scope_type=ScopeType.USER, scope_id="u1")
        s_b = MemoryScope(scope_type=ScopeType.USER, scope_id="u2")
        await scoped.save(SemanticMemory(scope=s_a, content="a"))
        await scoped.save(SemanticMemory(scope=s_b, content="b"))
        assert len(await scoped.list_scope(s_a)) == 1
        assert len(await scoped.list_scope(s_b)) == 1

    async def test_compound_query_with_tags(self, scoped: ScopedMemoryStore) -> None:
        s1 = MemoryScope(scope_type=ScopeType.PROJECT, scope_id="p")
        s2 = MemoryScope(scope_type=ScopeType.USER, scope_id="u")
        await scoped.save(SemanticMemory(scope=s1, content="x", tags=["a"]))
        await scoped.save(SemanticMemory(scope=s2, content="y", tags=["b"]))
        all_results = await scoped.query([s1, s2])
        assert len(all_results) == 2
        tagged = await scoped.query([s1, s2], tags=["a"])
        assert len(tagged) == 1

    async def test_find_by_id_cross_scope(
        self, scoped: ScopedMemoryStore,
    ) -> None:
        s = MemoryScope(scope_type=ScopeType.USER, scope_id="u1")
        entry = SemanticMemory(scope=s, content="z")
        await scoped.save(entry)
        found = await scoped.find_by_id(entry.memory_id)
        assert found is not None
        assert found.content == "z"

    def test_entry_from_dict_dispatches_by_type(self) -> None:
        data = EpisodicMemory(
            scope=MemoryScope(scope_type=ScopeType.GLOBAL),
            content="e",
            reasoning="r",
        ).model_dump(mode="json")
        restored = entry_from_dict(data)
        assert isinstance(restored, EpisodicMemory)
        assert restored.reasoning == "r"


# ── VectorStore 语义搜索 ───────────────────────────────────────────────────


class TestVectorStore:
    def test_cosine_similarity(self) -> None:
        assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
        assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
        assert cosine_similarity([0, 0], [1, 0]) == pytest.approx(0.0)

    async def test_semantic_search_ranks_relevant(
        self, scoped: ScopedMemoryStore,
    ) -> None:
        vs = VectorStore(scoped, embed_func=mock_embed)
        scope = MemoryScope(scope_type=ScopeType.PROJECT, scope_id="p")
        await vs.add(SemanticMemory(scope=scope, content="使用 PostgreSQL 数据库"))
        await vs.add(SemanticMemory(scope=scope, content="使用 Python 编程"))
        results = await vs.search("数据库配置", scopes=[scope])
        assert "PostgreSQL" in results[0][0].content

    async def test_inactive_excluded(self, scoped: ScopedMemoryStore) -> None:
        vs = VectorStore(scoped, embed_func=mock_embed)
        scope = MemoryScope(scope_type=ScopeType.GLOBAL)
        await vs.add(SemanticMemory(
            scope=scope, content="inactive", status=MemoryStatus.INACTIVE,
        ))
        assert len(await vs.search("inactive")) == 0

    async def test_rebuild_index(self, scoped: ScopedMemoryStore) -> None:
        scope = MemoryScope(scope_type=ScopeType.GLOBAL)
        await scoped.save(SemanticMemory(scope=scope, content="pre-existing"))
        vs = VectorStore(scoped, embed_func=mock_embed)
        count = await vs.rebuild_index([scope])
        assert count == 1
        assert len(vs.index) == 1


# ── MemoryRetriever 三层检索 + 综合重排 ─────────────────────────────────────


class TestRetriever:
    async def test_memory_index_summary(self, scoped: ScopedMemoryStore) -> None:
        scope = MemoryScope(scope_type=ScopeType.GLOBAL)
        await scoped.save(SemanticMemory(
            scope=scope, content="长" * 300, summary="短摘要",
        ))
        vs = VectorStore(scoped, embed_func=mock_embed)
        retriever = MemoryRetriever(scoped, vs)
        index = await retriever.get_memory_index(scopes=[scope])
        assert index[0].summary == "短摘要"

    async def test_tag_filter(self, scoped: ScopedMemoryStore) -> None:
        scope = MemoryScope(scope_type=ScopeType.GLOBAL)
        vs = VectorStore(scoped, embed_func=mock_embed)
        await vs.add(SemanticMemory(scope=scope, content="Python facts", tags=["lang"]))
        await vs.add(SemanticMemory(scope=scope, content="DB facts", tags=["db"]))
        retriever = MemoryRetriever(scoped, vs)
        results = await retriever.search_memory(
            "facts", scopes=[scope], tags=["db"],
        )
        assert len(results) == 1
        assert results[0].entry.content == "DB facts"

    async def test_load_detail_increments_access(
        self, scoped: ScopedMemoryStore,
    ) -> None:
        scope = MemoryScope(scope_type=ScopeType.GLOBAL)
        entry = SemanticMemory(scope=scope, content="detail")
        await scoped.save(entry)
        vs = VectorStore(scoped, embed_func=mock_embed)
        retriever = MemoryRetriever(scoped, vs)
        loaded = await retriever.load_memory_detail(scope, entry.memory_id)
        assert loaded is not None and loaded.access_count == 1

    def test_rerank_prefers_fresh_and_popular(self) -> None:
        now = datetime.now(UTC)
        recent = SemanticMemory(
            scope=MemoryScope(scope_type=ScopeType.GLOBAL),
            content="recent", updated_at=now, access_count=5, confidence=0.9,
        )
        old = SemanticMemory(
            scope=MemoryScope(scope_type=ScopeType.GLOBAL),
            content="old",
            updated_at=now - timedelta(days=60),
            access_count=0, confidence=0.5,
        )
        reranked = MemoryRetriever.rerank([(old, 0.8), (recent, 0.7)])
        assert reranked[0][0].content == "recent"


# ── RetentionManager 生命周期 ──────────────────────────────────────────────


class TestRetention:
    async def test_mark_inactive(self, store: PersistenceStore) -> None:
        scoped = ScopedMemoryStore(store)
        rm = RetentionManager(scoped, store)
        scope = MemoryScope(scope_type=ScopeType.GLOBAL)
        entry = SemanticMemory(scope=scope, content="旧")
        await scoped.save(entry)
        await rm.mark_inactive(entry, "过时")
        loaded = await scoped.load(scope, entry.memory_id)
        assert loaded.status == MemoryStatus.INACTIVE

    async def test_supersede_chain(self, store: PersistenceStore) -> None:
        scoped = ScopedMemoryStore(store)
        rm = RetentionManager(scoped, store)
        scope = MemoryScope(scope_type=ScopeType.GLOBAL)
        old = SemanticMemory(scope=scope, content="v1")
        await scoped.save(old)
        new = SemanticMemory(scope=scope, content="v2")
        new_id = await rm.supersede(old, new, "升级")
        old_loaded = await scoped.load(scope, old.memory_id)
        new_loaded = await scoped.load(scope, new_id)
        assert old_loaded.status == MemoryStatus.SUPERSEDED
        assert old_loaded.superseded_by == new_id
        assert new_loaded.version == 2

    async def test_version_history(self, store: PersistenceStore) -> None:
        scoped = ScopedMemoryStore(store)
        rm = RetentionManager(scoped, store)
        entry = SemanticMemory(
            scope=MemoryScope(scope_type=ScopeType.GLOBAL),
            content="v",
        )
        await scoped.save(entry)
        await rm.save_version(entry, "初始")
        history = await rm.get_version_history(entry.memory_id)
        assert len(history) == 1 and history[0].change_reason == "初始"

    def test_decay_half_life(self) -> None:
        rm = RetentionManager.__new__(RetentionManager)
        rm.decay_half_life_days = 30.0
        old = SemanticMemory(
            scope=MemoryScope(scope_type=ScopeType.GLOBAL),
            content="o",
            updated_at=datetime.now(UTC) - timedelta(days=30),
        )
        assert rm.compute_decay(old) == pytest.approx(0.5, abs=0.05)

    async def test_decay_sweep(self, store: PersistenceStore) -> None:
        scoped = ScopedMemoryStore(store)
        rm = RetentionManager(
            scoped, store,
            inactivity_threshold_days=0.001,
            min_access_count=0,
        )
        scope = MemoryScope(scope_type=ScopeType.GLOBAL)
        entry = SemanticMemory(
            scope=scope, content="极旧",
            access_count=0, confidence=0.01,
            created_at=datetime.now(UTC) - timedelta(days=365),
            updated_at=datetime.now(UTC) - timedelta(days=365),
            last_accessed_at=datetime.now(UTC) - timedelta(days=365),
        )
        await scoped.save(entry)
        assert await rm.run_decay_sweep(scope) == 1

    async def test_enforce_capacity_evicts_lowest_relevance(
        self, store: PersistenceStore,
    ) -> None:
        scoped = ScopedMemoryStore(store)
        rm = RetentionManager(scoped, store)
        scope = MemoryScope(scope_type=ScopeType.GLOBAL)
        now = datetime.now(UTC)
        # 高相关性（新、高置信、高访问）
        keep = SemanticMemory(
            scope=scope, content="高相关", confidence=0.95, access_count=20,
            updated_at=now, last_accessed_at=now,
        )
        # 低相关性（旧、低置信、零访问）
        evict = SemanticMemory(
            scope=scope, content="低相关", confidence=0.05, access_count=0,
            updated_at=now - timedelta(days=200),
            last_accessed_at=now - timedelta(days=200),
        )
        await scoped.save(keep)
        await scoped.save(evict)

        n = await rm.enforce_capacity(scope, max_memories=1)
        assert n == 1
        assert (await scoped.load(scope, evict.memory_id)).status == MemoryStatus.INACTIVE
        assert (await scoped.load(scope, keep.memory_id)).status == MemoryStatus.ACTIVE
        # 未超容量时不淘汰
        assert await rm.enforce_capacity(scope, max_memories=10) == 0


# ── Scratchpad 白名单强校验 ────────────────────────────────────────────────


class TestScratchpad:
    async def test_write_read_delete(self, store: PersistenceStore) -> None:
        sp = Scratchpad(store, "s1")
        await sp.write("progress.json", {"step": 1})
        assert await sp.read("progress.json") == {"step": 1}
        await sp.delete("progress.json")
        assert await sp.read("progress.json") is None

    async def test_unknown_key_rejected(self, store: PersistenceStore) -> None:
        sp = Scratchpad(store, "s1")
        with pytest.raises(ValueError):
            await sp.write("random.json", {})
        with pytest.raises(ValueError):
            await sp.read("random.json")

    async def test_export_import_filters_unknown(
        self, store: PersistenceStore,
    ) -> None:
        sp1 = Scratchpad(store, "s1")
        await sp1.write("progress.json", {"x": 1})
        state = await sp1.export_state()
        state["garbage.json"] = "bad"  # 导入不合法 key
        sp2 = Scratchpad(store, "s2")
        await sp2.import_state(state)
        assert await sp2.read("progress.json") == {"x": 1}
        assert "garbage.json" not in await sp2.list_keys()


# ── ProfileManager 档案模式 ────────────────────────────────────────────────


class TestProfile:
    async def test_create_then_merge_update(self, store: PersistenceStore) -> None:
        mgr = ProfileManager(ProfileStore(store))
        scope = MemoryScope(scope_type=ScopeType.PROJECT, scope_id="p")
        await mgr.update(scope, "db", {"engine": "pg", "port": 5432})
        await mgr.update(scope, "db", {"port": 5433, "ssl": True})
        fields = await mgr.get(scope, "db")
        assert fields == {"engine": "pg", "port": 5433, "ssl": True}

    async def test_version_increments(self, store: PersistenceStore) -> None:
        mgr = ProfileManager(ProfileStore(store))
        scope = MemoryScope(scope_type=ScopeType.USER, scope_id="u")
        p1 = await mgr.update(scope, "pref", {"a": 1})
        p2 = await mgr.update(scope, "pref", {"b": 2})
        assert p1.version == 1 and p2.version == 2

    async def test_scope_isolation(self, store: PersistenceStore) -> None:
        mgr = ProfileManager(ProfileStore(store))
        s1 = MemoryScope(scope_type=ScopeType.USER, scope_id="u1")
        s2 = MemoryScope(scope_type=ScopeType.USER, scope_id="u2")
        await mgr.update(s1, "pref", {"v": "u1"})
        await mgr.update(s2, "pref", {"v": "u2"})
        assert (await mgr.get(s1, "pref"))["v"] == "u1"
        assert (await mgr.get(s2, "pref"))["v"] == "u2"


# ── ProjectMemoryLoader ────────────────────────────────────────────────────


class TestProjectLoader:
    async def test_load_sections(
        self, tmp_path, scoped: ScopedMemoryStore,
    ) -> None:
        md = tmp_path / "praxis.md"
        md.write_text(
            "---\nproject: praxis\ntags:\n  - core\n---\n"
            "## 架构\n\nPraxis 采用分层架构。\n\n"
            "## 风格\n\nPython 3.12，严格类型标注。\n",
            encoding="utf-8",
        )
        vs = VectorStore(scoped, embed_func=mock_embed)
        loader = ProjectMemoryLoader(scoped, vs)
        count = await loader.load(str(tmp_path), "fallback")
        assert count == 2

        scope = MemoryScope(scope_type=ScopeType.PROJECT, scope_id="praxis")
        entries = await scoped.list_scope(scope)
        headings = {e.metadata["heading"] for e in entries}
        assert headings == {"架构", "风格"}
        assert all("core" in e.tags for e in entries)

    async def test_idempotent(
        self, tmp_path, scoped: ScopedMemoryStore,
    ) -> None:
        md = tmp_path / "praxis.md"
        md.write_text("## A\n\ncontent\n", encoding="utf-8")
        vs = VectorStore(scoped, embed_func=mock_embed)
        loader = ProjectMemoryLoader(scoped, vs)
        assert await loader.load(str(tmp_path), "default") == 1
        assert await loader.load(str(tmp_path), "default") == 0

    async def test_missing_file_returns_zero(
        self, tmp_path, scoped: ScopedMemoryStore,
    ) -> None:
        vs = VectorStore(scoped, embed_func=mock_embed)
        loader = ProjectMemoryLoader(scoped, vs)
        assert await loader.load(str(tmp_path), "default") == 0


# ── MemoryExtractor 按类型实例化子类 ───────────────────────────────────────


class TestExtractor:
    async def test_semantic_extraction(self, mock_gateway) -> None:
        response = make_chat_response(json.dumps([
            {"content": "用户偏好 Python", "tags": ["lang"], "confidence": 0.9},
        ]))
        with patch("praxis.memory.extractor.chat", AsyncMock(return_value=response)):
            extractor = MemoryExtractor(mock_gateway)
            entries = await extractor.extract(
                [{"role": "user", "content": "I love Python"}],
                MemoryScope(scope_type=ScopeType.USER, scope_id="u1"),
                memory_types=[MemoryType.SEMANTIC],
            )
        assert len(entries) == 1
        assert isinstance(entries[0], SemanticMemory)
        assert entries[0].confidence == 0.9

    async def test_episodic_keeps_structured_fields(self, mock_gateway) -> None:
        response = make_chat_response(json.dumps([
            {
                "content": "选择 Redis 方案",
                "context_description": "讨论缓存方案",
                "reasoning": "性能更高",
                "action_taken": "引入 Redis",
                "outcome": "缓存命中率提升",
                "tags": ["arch"],
                "confidence": 0.85,
            },
        ]))
        with patch("praxis.memory.extractor.chat", AsyncMock(return_value=response)):
            extractor = MemoryExtractor(mock_gateway)
            entries = await extractor.extract(
                [{"role": "user", "content": "..."}],
                MemoryScope(scope_type=ScopeType.GLOBAL),
                memory_types=[MemoryType.EPISODIC],
            )
        assert isinstance(entries[0], EpisodicMemory)
        assert entries[0].reasoning == "性能更高"
        assert entries[0].outcome == "缓存命中率提升"

    async def test_procedural_keeps_steps(self, mock_gateway) -> None:
        response = make_chat_response(json.dumps([
            {
                "content": "部署流程",
                "steps": ["lint", "test", "deploy"],
                "applicable_scenarios": ["生产发布"],
                "tags": [],
            },
        ]))
        with patch("praxis.memory.extractor.chat", AsyncMock(return_value=response)):
            extractor = MemoryExtractor(mock_gateway)
            entries = await extractor.extract(
                [{"role": "user", "content": "..."}],
                MemoryScope(scope_type=ScopeType.GLOBAL),
                memory_types=[MemoryType.PROCEDURAL],
            )
        assert isinstance(entries[0], ProceduralMemory)
        assert entries[0].steps == ["lint", "test", "deploy"]

    async def test_malformed_json_returns_empty(self, mock_gateway) -> None:
        response = make_chat_response("not valid json")
        with patch("praxis.memory.extractor.chat", AsyncMock(return_value=response)):
            extractor = MemoryExtractor(mock_gateway)
            entries = await extractor.extract(
                [{"role": "user", "content": "x"}],
                MemoryScope(scope_type=ScopeType.GLOBAL),
                memory_types=[MemoryType.SEMANTIC],
            )
        assert entries == []


# ── MemoryConsolidator 决策 ────────────────────────────────────────────────


class TestConsolidator:
    async def test_add_when_no_similar(
        self, store: PersistenceStore, mock_gateway,
    ) -> None:
        scoped = ScopedMemoryStore(store)
        vs = VectorStore(scoped, embed_func=mock_embed)
        rm = RetentionManager(scoped, store)
        consolidator = MemoryConsolidator(
            mock_gateway, vs, rm, similarity_threshold=0.99,
        )
        scope = MemoryScope(scope_type=ScopeType.GLOBAL)
        new = SemanticMemory(scope=scope, content="fresh data")
        result = await consolidator.consolidate(new)
        assert result.action == ConsolidationAction.ADD
        assert len(vs.index) == 1

    async def test_update_supersedes(
        self, store: PersistenceStore, mock_gateway,
    ) -> None:
        scoped = ScopedMemoryStore(store)
        vs = VectorStore(scoped, embed_func=mock_embed)
        rm = RetentionManager(scoped, store)
        scope = MemoryScope(scope_type=ScopeType.GLOBAL)
        old = SemanticMemory(scope=scope, content="项目使用 PostgreSQL 数据库")
        await vs.add(old)

        update_resp = make_chat_response(json.dumps({
            "action": "update",
            "reasoning": "替代",
            "merged_content": "项目使用 MySQL 数据库（从 PostgreSQL 迁移）",
        }))
        with patch("praxis.memory.consolidator.chat", AsyncMock(return_value=update_resp)):
            consolidator = MemoryConsolidator(
                mock_gateway, vs, rm, similarity_threshold=0.3,
            )
            new = SemanticMemory(scope=scope, content="项目使用 MySQL 数据库")
            result = await consolidator.consolidate(new)

        assert result.action == ConsolidationAction.UPDATE
        old_loaded = await scoped.load(scope, old.memory_id)
        assert old_loaded.status == MemoryStatus.SUPERSEDED

    async def test_noop_when_llm_says_noop(
        self, store: PersistenceStore, mock_gateway,
    ) -> None:
        scoped = ScopedMemoryStore(store)
        vs = VectorStore(scoped, embed_func=mock_embed)
        rm = RetentionManager(scoped, store)
        scope = MemoryScope(scope_type=ScopeType.GLOBAL)
        await vs.add(SemanticMemory(scope=scope, content="使用 Python 编程"))

        noop_resp = make_chat_response(json.dumps({
            "action": "noop",
            "reasoning": "冗余",
            "merged_content": "",
        }))
        with patch("praxis.memory.consolidator.chat", AsyncMock(return_value=noop_resp)):
            consolidator = MemoryConsolidator(
                mock_gateway, vs, rm, similarity_threshold=0.3,
            )
            result = await consolidator.consolidate(
                SemanticMemory(scope=scope, content="使用 Python 编程语言"),
            )
        assert result.action == ConsolidationAction.NOOP
        assert len(vs.index) == 1  # 不新增

    async def test_llm_failure_falls_back_to_add(
        self, store: PersistenceStore, mock_gateway,
    ) -> None:
        scoped = ScopedMemoryStore(store)
        vs = VectorStore(scoped, embed_func=mock_embed)
        rm = RetentionManager(scoped, store)
        scope = MemoryScope(scope_type=ScopeType.GLOBAL)
        await vs.add(SemanticMemory(scope=scope, content="使用 Python"))

        with patch(
            "praxis.memory.consolidator.chat",
            AsyncMock(side_effect=RuntimeError("LLM down")),
        ):
            consolidator = MemoryConsolidator(
                mock_gateway, vs, rm, similarity_threshold=0.3,
            )
            new = SemanticMemory(scope=scope, content="使用 Python 编程")
            result = await consolidator.consolidate(new)
        assert result.action == ConsolidationAction.ADD
        assert len(vs.index) == 2


# ── DreamConsolidator ──────────────────────────────────────────────────────


class TestDream:
    async def test_should_run_requires_min_sessions(
        self, store: PersistenceStore, mock_gateway,
    ) -> None:
        scoped = ScopedMemoryStore(store)
        rm = RetentionManager(scoped, store)
        dream = DreamConsolidator(
            mock_gateway, scoped, rm, store,
            min_sessions=5, min_hours_since_last=24.0,
        )
        assert not await dream.should_run(session_count=3)
        assert await dream.should_run(session_count=5)

    async def test_run_dream_applies_stale_marking(
        self, store: PersistenceStore, mock_gateway,
    ) -> None:
        scoped = ScopedMemoryStore(store)
        rm = RetentionManager(scoped, store)
        scope = MemoryScope(scope_type=ScopeType.GLOBAL)
        stale = SemanticMemory(scope=scope, content="过时")
        await scoped.save(stale)

        response = make_chat_response(json.dumps({
            "anchored": [],
            "conflicts": [],
            "stale": [stale.memory_id],
            "merge_suggestions": [],
            "summary": "清理 1 条",
        }))
        with patch("praxis.memory.dream.chat", AsyncMock(return_value=response)):
            dream = DreamConsolidator(mock_gateway, scoped, rm, store)
            report = await dream.run_dream([scope])
        assert report.stale_marked == 1
        loaded = await scoped.load(scope, stale.memory_id)
        assert loaded.status == MemoryStatus.INACTIVE

    async def test_run_dream_runs_decay_when_enabled(
        self, store: PersistenceStore, mock_gateway,
    ) -> None:
        """梦境周期应在 decay_enabled 时执行相关性衰减遗忘。"""
        scoped = ScopedMemoryStore(store)
        rm = RetentionManager(
            scoped, store, inactivity_threshold_days=0.001, min_access_count=0,
        )
        scope = MemoryScope(scope_type=ScopeType.GLOBAL)
        old = SemanticMemory(
            scope=scope, content="极旧记忆",
            access_count=0, confidence=0.01,
            created_at=datetime.now(UTC) - timedelta(days=365),
            updated_at=datetime.now(UTC) - timedelta(days=365),
            last_accessed_at=datetime.now(UTC) - timedelta(days=365),
        )
        await scoped.save(old)

        response = make_chat_response(json.dumps({
            "anchored": [], "conflicts": [], "stale": [],
            "merge_suggestions": [], "summary": "无",
        }))
        with patch("praxis.memory.dream.chat", AsyncMock(return_value=response)):
            dream = DreamConsolidator(mock_gateway, scoped, rm, store, decay_enabled=True)
            report = await dream.run_dream([scope])
        assert report.decayed_count == 1
        loaded = await scoped.load(scope, old.memory_id)
        assert loaded.status == MemoryStatus.INACTIVE

    async def test_run_dream_skips_decay_when_disabled(
        self, store: PersistenceStore, mock_gateway,
    ) -> None:
        scoped = ScopedMemoryStore(store)
        rm = RetentionManager(
            scoped, store, inactivity_threshold_days=0.001, min_access_count=0,
        )
        scope = MemoryScope(scope_type=ScopeType.GLOBAL)
        old = SemanticMemory(
            scope=scope, content="极旧记忆", access_count=0, confidence=0.01,
            created_at=datetime.now(UTC) - timedelta(days=365),
            updated_at=datetime.now(UTC) - timedelta(days=365),
            last_accessed_at=datetime.now(UTC) - timedelta(days=365),
        )
        await scoped.save(old)
        response = make_chat_response(json.dumps({
            "anchored": [], "conflicts": [], "stale": [],
            "merge_suggestions": [], "summary": "无",
        }))
        with patch("praxis.memory.dream.chat", AsyncMock(return_value=response)):
            dream = DreamConsolidator(mock_gateway, scoped, rm, store, decay_enabled=False)
            report = await dream.run_dream([scope])
        assert report.decayed_count == 0
        loaded = await scoped.load(scope, old.memory_id)
        assert loaded.status == MemoryStatus.ACTIVE

    async def test_scheduler_triggers_on_condition(
        self, store: PersistenceStore, mock_gateway,
    ) -> None:
        scoped = ScopedMemoryStore(store)
        rm = RetentionManager(scoped, store)
        dream = DreamConsolidator(
            mock_gateway, scoped, rm, store,
            min_sessions=1, min_hours_since_last=0.0,
        )
        dream.run_dream = AsyncMock(return_value=DreamReport())
        session_count = 3
        scheduler = DreamScheduler(
            consolidator=dream,
            scopes=[MemoryScope(scope_type=ScopeType.GLOBAL)],
            check_interval_seconds=0.05,
            session_count_getter=lambda: session_count,
        )
        scheduler.start()
        await asyncio.sleep(0.15)
        await scheduler.stop()
        assert dream.run_dream.await_count >= 1


# ── BackgroundWorker 游标 + 失败重试 ──────────────────────────────────────


class TestBackgroundWorker:
    async def test_notify_and_process_cursor_advance(self) -> None:
        processed: list[str] = []

        async def process_fn(messages, scope):
            for m in messages:
                processed.append(m.message_id)
            return len(messages)

        scope = MemoryScope(scope_type=ScopeType.SESSION, scope_id="s1")
        worker = BackgroundWorker(process_fn, scope, batch_threshold=1)
        for i in range(3):
            worker.notify(WorkingMemoryMessage(role="user", content=str(i)))
        await worker.process_once()
        assert len(processed) == 3
        assert worker.last_processed_message_id == processed[-1]
        assert worker.pending == []

    async def test_failure_keeps_pending(self) -> None:
        calls = {"n": 0}

        async def process_fn(messages, scope):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom")
            return len(messages)

        scope = MemoryScope(scope_type=ScopeType.SESSION, scope_id="s")
        worker = BackgroundWorker(process_fn, scope, batch_threshold=1)
        worker.notify(WorkingMemoryMessage(role="user", content="x"))
        # 首次失败，pending 保留
        await worker.process_once()
        assert len(worker.pending) == 1
        assert worker.last_processed_message_id is None
        # 再次处理成功
        await worker.process_once()
        assert worker.pending == []

    async def test_start_stop_lifecycle(self) -> None:
        async def process_fn(messages, scope):
            return len(messages)

        scope = MemoryScope(scope_type=ScopeType.SESSION, scope_id="s")
        worker = BackgroundWorker(
            process_fn, scope, batch_threshold=100, interval_seconds=0.05,
        )
        worker.start()
        await asyncio.sleep(0.12)
        await worker.stop()
        assert worker.task is None

    async def test_new_messages_during_processing_not_lost(self) -> None:
        """背压：处理期间新追加的消息不应丢失。"""
        started = asyncio.Event()
        finish = asyncio.Event()

        async def process_fn(messages, scope):
            started.set()
            await finish.wait()
            return len(messages)

        scope = MemoryScope(scope_type=ScopeType.SESSION, scope_id="s")
        worker = BackgroundWorker(process_fn, scope, batch_threshold=1)
        worker.notify(WorkingMemoryMessage(role="user", content="a"))

        task = asyncio.create_task(worker.process_once())
        await started.wait()
        # 处理期间注入新消息
        new_msg = WorkingMemoryMessage(role="user", content="b")
        worker.notify(new_msg)
        finish.set()
        await task

        # 第一条应该已处理并从 pending 移除；第二条应保留
        assert len(worker.pending) == 1
        assert worker.pending[0].content == "b"


# ── CognitiveMemory 门面闭环 ──────────────────────────────────────────────────


class TestCognitiveMemory:
    async def test_lifecycle_and_append(
        self, store: PersistenceStore, mock_gateway,
    ) -> None:
        config = MemoryConfig(background_enabled=False, dream_enabled=False)
        ms = CognitiveMemory(store, mock_gateway, session_id="s1", config=config)
        await ms.start()
        try:
            ms.append_message(WorkingMemoryMessage(role="user", content="hi"))
            ms.append_message(WorkingMemoryMessage(role="assistant", content="hello"))
            assert len(ms.get_message_history()) == 2
            # Worker 未启动，但 pending 仍排队
            assert len(ms.worker.pending) == 2
        finally:
            await ms.stop()

    async def test_save_memory_goes_through_consolidation(
        self, store: PersistenceStore, mock_gateway,
    ) -> None:
        config = MemoryConfig(
            background_enabled=False, dream_enabled=False,
            consolidation_similarity_threshold=0.99,
        )
        ms = CognitiveMemory(store, mock_gateway, session_id="s", config=config)
        # 替换 embed 避免真实调用
        ms.vector_store.embed_func = mock_embed
        await ms.start()
        try:
            mid = await ms.save_memory(
                "用户偏好 Python",
                scope=MemoryScope(scope_type=ScopeType.USER, scope_id="u1"),
                tags=["pref"],
            )
            assert mid
            results = await ms.search_memory(
                "Python", scopes=[MemoryScope(scope_type=ScopeType.USER, scope_id="u1")],
            )
            assert len(results) == 1
        finally:
            await ms.stop()

    async def test_update_memory_cross_scope(
        self, store: PersistenceStore, mock_gateway,
    ) -> None:
        config = MemoryConfig(background_enabled=False, dream_enabled=False)
        ms = CognitiveMemory(store, mock_gateway, session_id="s", config=config)
        ms.vector_store.embed_func = mock_embed
        await ms.start()
        try:
            other_scope = MemoryScope(scope_type=ScopeType.PROJECT, scope_id="p")
            mid = await ms.save_memory(
                "旧内容", scope=other_scope,
            )
            await ms.update_memory(mid, "新内容")
            # 旧条目被 SUPERSEDED
            old_loaded = await ms.scoped_store.load(other_scope, mid)
            assert old_loaded.status == MemoryStatus.SUPERSEDED
        finally:
            await ms.stop()

    async def test_export_import_state(
        self, store: PersistenceStore, mock_gateway,
    ) -> None:
        config = MemoryConfig(background_enabled=False, dream_enabled=False)
        ms1 = CognitiveMemory(store, mock_gateway, session_id="s1", config=config)
        await ms1.start()
        ms1.append_message(WorkingMemoryMessage(role="user", content="A"))
        ms1.append_message(WorkingMemoryMessage(role="user", content="B"))
        ms1.dream_session_count = 2
        snapshot = ms1.export_state()
        await ms1.stop()

        ms2 = CognitiveMemory(store, mock_gateway, session_id="s1", config=config)
        await ms2.start()
        try:
            await ms2.import_state(snapshot)
            assert len(ms2.get_message_history()) == 2
            assert ms2.dream_session_count == 2
        finally:
            await ms2.stop()

    async def test_clear_session_increments_dream_count(
        self, store: PersistenceStore, mock_gateway,
    ) -> None:
        config = MemoryConfig(background_enabled=False, dream_enabled=False)
        ms = CognitiveMemory(store, mock_gateway, session_id="s", config=config)
        await ms.start()
        try:
            ms.append_message(WorkingMemoryMessage(role="user", content="x"))
            assert ms.dream_session_count == 0
            await ms.clear_session()
            assert ms.dream_session_count == 1
            assert len(ms.get_message_history()) == 0
            assert ms.worker.pending == []
        finally:
            await ms.stop()

    async def test_profile_interfaces(
        self, store: PersistenceStore, mock_gateway,
    ) -> None:
        config = MemoryConfig(background_enabled=False, dream_enabled=False)
        ms = CognitiveMemory(store, mock_gateway, session_id="s", config=config)
        await ms.start()
        try:
            scope = MemoryScope(scope_type=ScopeType.USER, scope_id="u")
            await ms.update_profile(scope, "pref", {"lang": "zh"})
            await ms.update_profile(scope, "pref", {"theme": "dark"})
            profile = await ms.get_profile(scope, "pref")
            assert profile == {"lang": "zh", "theme": "dark"}
        finally:
            await ms.stop()

    async def test_scratchpad_rejected_unknown_key(
        self, store: PersistenceStore, mock_gateway,
    ) -> None:
        config = MemoryConfig(background_enabled=False, dream_enabled=False)
        ms = CognitiveMemory(store, mock_gateway, session_id="s", config=config)
        await ms.start()
        try:
            await ms.write_scratchpad("progress.json", {"step": 1})
            assert await ms.read_scratchpad("progress.json") == {"step": 1}
            with pytest.raises(ValueError):
                await ms.write_scratchpad("bad.json", 1)
        finally:
            await ms.stop()

    async def test_project_preload(
        self, tmp_path, store: PersistenceStore, mock_gateway,
    ) -> None:
        md = tmp_path / "praxis.md"
        md.write_text("## 规范\n\n使用 Python 3.12。\n", encoding="utf-8")
        config = MemoryConfig(
            background_enabled=False,
            dream_enabled=False,
            load_project_praxis_md=True,
            project_root=str(tmp_path),
            project_name="testproj",
        )
        ms = CognitiveMemory(store, mock_gateway, session_id="s", config=config)
        ms.vector_store.embed_func = mock_embed
        await ms.start()
        try:
            scope = MemoryScope(scope_type=ScopeType.PROJECT, scope_id="testproj")
            entries = await ms.scoped_store.list_scope(scope)
            assert len(entries) == 1
            assert entries[0].metadata["source"] == "project_praxis_md"
            # 二次 start 不重复导入
            await ms.stop()
            ms2 = CognitiveMemory(store, mock_gateway, session_id="s", config=config)
            ms2.vector_store.embed_func = mock_embed
            await ms2.start()
            entries2 = await ms2.scoped_store.list_scope(scope)
            assert len(entries2) == 1
            await ms2.stop()
        finally:
            if ms.worker.task is not None:
                await ms.stop()

    async def test_background_worker_end_to_end(
        self, store: PersistenceStore, mock_gateway,
    ) -> None:
        """启用后台 Worker 验证 append → extract → consolidate 闭环。"""
        extract_resp = make_chat_response(json.dumps([
            {"content": "用户偏好 Python", "tags": ["lang"], "confidence": 0.9},
        ]))
        config = MemoryConfig(
            background_enabled=True,
            dream_enabled=False,
            background_batch_threshold=2,
            background_interval_seconds=0.05,
            consolidation_similarity_threshold=0.99,
        )
        ms = CognitiveMemory(store, mock_gateway, session_id="s", config=config)
        ms.vector_store.embed_func = mock_embed

        with patch(
            "praxis.memory.extractor.chat",
            AsyncMock(return_value=extract_resp),
        ):
            await ms.start()
            try:
                ms.append_message(WorkingMemoryMessage(role="user", content="I like Python"))
                ms.append_message(WorkingMemoryMessage(role="assistant", content="Noted"))
                # 等待 Worker 消费
                for poll_attempt in range(40):  # noqa: B007 - public discard name
                    await asyncio.sleep(0.05)
                    if ms.worker.last_processed_message_id is not None:
                        break
                assert ms.worker.last_processed_message_id is not None
                assert ms.worker.pending == []
            finally:
                await ms.stop()


# ── 记忆工具：S6 热路径接口暴露为 LLM 可调用工具 ──────────────────────────────


class TestMemoryTools:
    async def test_register_and_invoke(
        self, store: PersistenceStore, mock_gateway,
    ) -> None:
        from praxis.tools.builtins.memory_ops import register_memory_tools
        from praxis.tools.registry import ToolRegistry

        config = MemoryConfig(background_enabled=False, dream_enabled=False)
        ms = CognitiveMemory(store, mock_gateway, session_id="s", config=config)
        ms.vector_store.embed_func = mock_embed

        registry = ToolRegistry()
        names = register_memory_tools(registry, ms)
        assert set(names) == {
            "save_memory", "search_memory", "update_memory",
            "delete_memory", "write_scratchpad", "read_scratchpad",
        }
        for name in names:
            assert registry.has_tool(name)
            assert registry.get_metadata(name).category == "memory"

        # save → search 闭环
        save = registry.get_entry("save_memory").handler
        out = await save({"content": "用户偏好 Python", "tags": ["pref"]})
        assert "已保存记忆" in out

        search = registry.get_entry("search_memory").handler
        results = await search({"query": "Python 偏好", "top_k": 5})
        assert "Python" in results

        # scratchpad 写读闭环（仅限白名单 key）
        write = registry.get_entry("write_scratchpad").handler
        await write({"key": "todos.json", "content": "写测试"})
        read = registry.get_entry("read_scratchpad").handler
        assert "写测试" in await read({"key": "todos.json"})
        # 非白名单 key 被优雅拒绝
        assert "不支持" in await write({"key": "bad", "content": "x"})

    async def test_tools_surface_in_session(
        self, store: PersistenceStore, mock_gateway,
    ) -> None:
        """记忆工具应出现在会话的 general 阶段注入工具集中。"""
        from praxis.config.schemas import (
            ContextConfig,
            OrchestratorConfig,
            SessionConfig,
        )
        from praxis.guardrails.engine import GuardrailEngine
        from praxis.guardrails.permissions import PermissionManager
        from praxis.guardrails.rules import RuleEngine
        from praxis.session.core import SessionFactory

        mock_gateway.config.background_enabled = False
        guardrails = GuardrailEngine(RuleEngine(), PermissionManager())
        factory = SessionFactory(
            store=store,
            session_config=SessionConfig(auto_checkpoint=False),
            orchestrator_config=OrchestratorConfig(),
            context_config=ContextConfig(),
            memory_config=MemoryConfig(background_enabled=False, dream_enabled=False),
        )
        session = await factory.create_session(guardrails=guardrails, gateway=mock_gateway)
        try:
            schemas = session.injector.get_tools_for_stage("general")
            tool_names = {s["function"]["name"] for s in schemas}
            assert "save_memory" in tool_names
            assert "search_memory" in tool_names
        finally:
            await session.terminate()
