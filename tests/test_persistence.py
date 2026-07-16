"""S3 持久化引擎验证测试。"""

import asyncio

import pytest

from praxis.config.schemas import PersistenceConfig
from praxis.exceptions import (
    CheckpointCorruptionError,
    CheckpointVersionError,
    PersistenceError,
)
from praxis.persistence.checkpoint import CheckpointManager
from praxis.persistence.namespace import NamespaceManager
from praxis.persistence.store import PersistenceStore, create_store


class TestSqliteBackend:
    """Task 3.5: SQLite 后端完整 CRUD 测试。"""

    @pytest.fixture
    async def store(self, tmp_path: pytest.TempPathFactory) -> PersistenceStore:
        config = PersistenceConfig(
            backend="sqlite",
            sqlite_path=str(tmp_path / "test.db"),  # type: ignore[operator]
        )
        s = await create_store(config)
        yield s  # type: ignore[misc]
        await s.close()

    async def test_save_and_load(self, store: PersistenceStore) -> None:
        await store.save("test_ns", "key1", {"name": "praxis", "version": 1})
        result = await store.load("test_ns", "key1")
        assert result == {"name": "praxis", "version": 1}

    async def test_save_if_absent_is_atomic(self, store: PersistenceStore) -> None:
        outcomes = await asyncio.gather(*(
            store.save_if_absent("locks", "same", {"winner": index})
            for index in range(20)
        ))
        assert outcomes.count(True) == 1
        assert outcomes.count(False) == 19

    async def test_load_nonexistent(self, store: PersistenceStore) -> None:
        result = await store.load("test_ns", "nonexistent")
        assert result is None

    async def test_update(self, store: PersistenceStore) -> None:
        await store.save("test_ns", "key1", {"v": 1})
        await store.save("test_ns", "key1", {"v": 2})
        result = await store.load("test_ns", "key1")
        assert result == {"v": 2}

    async def test_delete(self, store: PersistenceStore) -> None:
        await store.save("test_ns", "key1", {"v": 1})
        await store.delete("test_ns", "key1")
        result = await store.load("test_ns", "key1")
        assert result is None

    async def test_list_keys(self, store: PersistenceStore) -> None:
        await store.save("ns", "alpha", 1)
        await store.save("ns", "alpha:sub", 2)
        await store.save("ns", "beta", 3)
        all_keys = await store.list_keys("ns")
        assert sorted(all_keys) == ["alpha", "alpha:sub", "beta"]
        filtered = await store.list_keys("ns", prefix="alpha")
        assert sorted(filtered) == ["alpha", "alpha:sub"]

    async def test_namespace_isolation(self, store: PersistenceStore) -> None:
        await store.save("ns_a", "key", "value_a")
        await store.save("ns_b", "key", "value_b")
        assert await store.load("ns_a", "key") == "value_a"
        assert await store.load("ns_b", "key") == "value_b"

    async def test_concurrent_writes_and_idempotent_close(self, tmp_path) -> None:
        store = await create_store(PersistenceConfig(
            sqlite_path=str(tmp_path / "concurrent.db"),
        ))
        await asyncio.gather(*(
            store.save("parallel", f"key-{index}", {"index": index})
            for index in range(20)
        ))
        assert len(await store.list_keys("parallel")) == 20
        await asyncio.gather(store.close(), store.close())
        with pytest.raises(PersistenceError, match="已关闭"):
            await store.load("parallel", "key-0")


class TestFilesystemBackend:
    """Task 3.5: 文件系统后端测试。"""

    @pytest.fixture
    async def store(self, tmp_path: pytest.TempPathFactory) -> PersistenceStore:
        config = PersistenceConfig(
            backend="filesystem",
            filesystem_path=str(tmp_path / "fs_store"),  # type: ignore[operator]
        )
        s = await create_store(config)
        yield s  # type: ignore[misc]
        await s.close()

    async def test_save_and_load(self, store: PersistenceStore) -> None:
        await store.save("test_ns", "key1", {"data": "hello"})
        result = await store.load("test_ns", "key1")
        assert result == {"data": "hello"}

    async def test_save_if_absent_is_atomic(self, store: PersistenceStore) -> None:
        outcomes = await asyncio.gather(*(
            store.save_if_absent("locks", "same", {"winner": index})
            for index in range(20)
        ))
        assert outcomes.count(True) == 1
        assert outcomes.count(False) == 19

    async def test_delete(self, store: PersistenceStore) -> None:
        await store.save("test_ns", "key1", "val")
        await store.delete("test_ns", "key1")
        assert await store.load("test_ns", "key1") is None

    async def test_list_keys(self, store: PersistenceStore) -> None:
        await store.save("ns", "a", 1)
        await store.save("ns", "b", 2)
        keys = await store.list_keys("ns")
        assert sorted(keys) == ["a", "b"]

    async def test_keys_round_trip_without_filename_collisions(
        self,
        store: PersistenceStore,
    ) -> None:
        await store.save("ns", "scope/item", "slash")
        await store.save("ns", "scope__item", "underscores")

        assert await store.load("ns", "scope/item") == "slash"
        assert await store.load("ns", "scope__item") == "underscores"
        assert await store.list_keys("ns") == ["scope/item", "scope__item"]

    @pytest.mark.parametrize("namespace", ["../escape", "..\\escape", "/absolute"])
    async def test_rejects_unsafe_namespace(
        self,
        store: PersistenceStore,
        namespace: str,
    ) -> None:
        with pytest.raises(PersistenceError, match="命名空间"):
            await store.save(namespace, "proof", {"unsafe": True})

    @pytest.mark.parametrize("key", ["../escape", "..\\escape", "/absolute"])
    async def test_rejects_unsafe_key(
        self,
        store: PersistenceStore,
        key: str,
    ) -> None:
        with pytest.raises(PersistenceError, match="键"):
            await store.save("safe", key, {"unsafe": True})


class TestCheckpoint:
    """Task 3.6: 检查点管理验证。"""

    @pytest.fixture
    async def manager(self, tmp_path: pytest.TempPathFactory) -> CheckpointManager:
        config = PersistenceConfig(
            backend="sqlite",
            sqlite_path=str(tmp_path / "cp_test.db"),  # type: ignore[operator]
        )
        store = await create_store(config)
        yield CheckpointManager(store)  # type: ignore[misc]
        await store.close()

    async def test_save_and_load(self, manager: CheckpointManager) -> None:
        state = {"messages": ["hello"], "turn": 3, "memory": {}}
        cp_id = await manager.save_checkpoint("sess-1", state, description="turn 3")
        key = f"sess-1:{cp_id}"
        loaded = await manager.load_checkpoint(key)
        assert loaded == state

    async def test_list_by_session(self, manager: CheckpointManager) -> None:
        await manager.save_checkpoint("sess-1", {"turn": 1})
        await manager.save_checkpoint("sess-1", {"turn": 2})
        await manager.save_checkpoint("sess-2", {"turn": 1})
        cps = await manager.list_checkpoints("sess-1")
        assert len(cps) == 2
        assert all(cp.session_id == "sess-1" for cp in cps)

    async def test_corrupted_checkpoint_is_rejected(self, manager: CheckpointManager) -> None:
        checkpoint_id = await manager.save_checkpoint("sess-1", {"turn": 1})
        key = f"sess-1:{checkpoint_id}"
        raw = await manager.store.load("checkpoints", key)
        assert isinstance(raw, dict)
        raw["state"] = {"turn": 999}
        await manager.store.save("checkpoints", key, raw)
        with pytest.raises(CheckpointCorruptionError):
            await manager.load_checkpoint(key)

    async def test_legacy_checkpoint_is_rejected(self, manager: CheckpointManager) -> None:
        key = "sess-1:legacy"
        await manager.store.save(
            "checkpoints",
            key,
            {"checkpoint_id": "legacy", "session_id": "sess-1", "state": {}},
        )
        with pytest.raises(CheckpointVersionError):
            await manager.load_checkpoint(key)


class TestNamespace:
    """Task 3.6: 命名空间隔离验证。"""

    @pytest.fixture
    async def ns_mgr(self, tmp_path: pytest.TempPathFactory) -> NamespaceManager:
        config = PersistenceConfig(
            backend="sqlite",
            sqlite_path=str(tmp_path / "ns_test.db"),  # type: ignore[operator]
        )
        store = await create_store(config)
        yield NamespaceManager(store)  # type: ignore[misc]
        await store.close()

    async def test_clear_namespace(self, ns_mgr: NamespaceManager) -> None:
        await ns_mgr.store.save("to_clear", "k1", "v1")
        await ns_mgr.store.save("to_clear", "k2", "v2")
        await ns_mgr.store.save("keep", "k1", "v1")
        count = await ns_mgr.clear_namespace("to_clear")
        assert count == 2
        assert not await ns_mgr.namespace_exists("to_clear")
        assert await ns_mgr.namespace_exists("keep")

    async def test_key_count(self, ns_mgr: NamespaceManager) -> None:
        await ns_mgr.store.save("counted", "a", 1)
        await ns_mgr.store.save("counted", "b", 2)
        assert await ns_mgr.namespace_key_count("counted") == 2
