"""S14 技能系统单元测试。"""

import textwrap
from pathlib import Path

import pytest
import pytest_asyncio

from praxis.config.schemas import PersistenceConfig
from praxis.models.skills import (
    SkillAuditResult,
    SkillDefinition,
    SkillIndexEntry,
    SkillMetadata,
)
from praxis.persistence.store import PersistenceStore, create_store
from praxis.skills.activation import SkillActivation
from praxis.skills.disclosure import SkillDisclosure
from praxis.skills.discovery import SkillDiscovery
from praxis.skills.manager import SkillManager
from praxis.skills.parser import SkillParser
from praxis.skills.tools_bridge import SkillToolsBridge
from praxis.tools.registry import ToolRegistry


# ── 测试辅助 ────────────────────────────────────────────────────────────────


def make_skill_dir(base: Path, name: str, extra_files: dict[str, str] | None = None) -> Path:
    """在 base 下创建技能目录并写入 SKILL.md。"""
    skill_dir = base / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_md = textwrap.dedent(f"""\
        ---
        name: {name}
        description: {name} 技能描述
        version: "1.0.0"
        tags:
          - test
        tools:
          - read_file
        ---
        # {name}

        这是 {name} 技能的详细指令。
    """)
    (skill_dir / "SKILL.md").write_text(skill_md, encoding="utf-8")
    if extra_files:
        for fname, content in extra_files.items():
            fpath = skill_dir / fname
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text(content, encoding="utf-8")
    return skill_dir


def make_skill_def(name: str = "test-skill", version: str = "1.0.0") -> SkillDefinition:
    """创建测试用 SkillDefinition。"""
    return SkillDefinition(
        skill_id=name,
        metadata=SkillMetadata(
            name=name,
            description=f"{name} 技能描述",
            version=version,
            tags=["test"],
            tools=["read_file"],
        ),
        content=f"# {name}\n\n这是详细指令。",
        base_path=".",
    )


@pytest.fixture
async def store(tmp_path: Path) -> PersistenceStore:
    config = PersistenceConfig(backend="sqlite", sqlite_path=str(tmp_path / "test.db"))
    s = await create_store(config)
    yield s
    await s.close()


# ── Task 10.1: 技能格式解析 ─────────────────────────────────────────────────


class TestSkillParser:
    """技能格式解析测试。"""

    def test_parse_valid_skill_md(self) -> None:
        raw = textwrap.dedent("""\
            ---
            name: pdf-processor
            description: 处理 PDF 文件
            version: "2.0.0"
            tags:
              - pdf
              - document
            tools:
              - read_pdf
              - fill_form
            ---
            # PDF Processor

            详细指令内容。
        """)
        parser = SkillParser()
        metadata, content = parser.parse_skill_md(raw)
        assert metadata is not None
        assert metadata.name == "pdf-processor"
        assert metadata.description == "处理 PDF 文件"
        assert metadata.version == "2.0.0"
        assert "pdf" in metadata.tags
        assert "read_pdf" in metadata.tools
        assert "详细指令内容" in content

    def test_parse_missing_frontmatter(self) -> None:
        parser = SkillParser()
        metadata, content = parser.parse_skill_md("# No frontmatter\nContent")
        assert metadata is None

    def test_parse_incomplete_frontmatter(self) -> None:
        parser = SkillParser()
        metadata, content = parser.parse_skill_md("---\nname: test\nNo closing")
        assert metadata is None

    def test_parse_directory(self, tmp_path: Path) -> None:
        make_skill_dir(tmp_path, "my-skill", {"ref.txt": "参考资料"})
        parser = SkillParser()
        skill = parser.parse_directory(tmp_path / "my-skill")
        assert skill is not None
        assert skill.skill_id == "my-skill"
        assert "ref.txt" in skill.files

    def test_parse_directory_no_skill_md(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        parser = SkillParser()
        assert parser.parse_directory(empty_dir) is None

    def test_discover_scripts(self, tmp_path: Path) -> None:
        make_skill_dir(tmp_path, "scripted", {"run.py": "print('hi')"})
        parser = SkillParser()
        skill = parser.parse_directory(tmp_path / "scripted")
        assert skill is not None
        assert "run.py" in skill.scripts

    def test_validate_directory_valid(self, tmp_path: Path) -> None:
        make_skill_dir(tmp_path, "valid")
        parser = SkillParser()
        issues = parser.validate_directory(tmp_path / "valid")
        assert issues == []

    def test_validate_directory_missing_skill_md(self, tmp_path: Path) -> None:
        d = tmp_path / "no-skill"
        d.mkdir()
        parser = SkillParser()
        issues = parser.validate_directory(d)
        assert len(issues) == 1
        assert "SKILL.md" in issues[0]


# ── Task 10.2: 渐进式披露 ───────────────────────────────────────────────────


class TestSkillDisclosure:
    """渐进式披露测试。"""

    def test_get_skill_index(self) -> None:
        disc = SkillDisclosure()
        disc.register(make_skill_def("skill-a"))
        disc.register(make_skill_def("skill-b"))
        index = disc.get_skill_index()
        assert len(index) == 2
        names = {e.name for e in index}
        assert names == {"skill-a", "skill-b"}

    def test_load_skill(self) -> None:
        disc = SkillDisclosure()
        disc.register(make_skill_def("skill-x"))
        loaded = disc.load_skill("skill-x")
        assert loaded is not None
        assert loaded.metadata.name == "skill-x"

    def test_load_skill_not_found(self) -> None:
        disc = SkillDisclosure()
        assert disc.load_skill("nope") is None

    def test_load_skill_file(self, tmp_path: Path) -> None:
        make_skill_dir(tmp_path, "with-file", {"docs/guide.txt": "使用指南"})
        parser = SkillParser()
        skill = parser.parse_directory(tmp_path / "with-file")
        assert skill is not None

        disc = SkillDisclosure()
        disc.register(skill)

        content = disc.load_skill_file("with-file", "docs/guide.txt")
        assert content == "使用指南"

    def test_load_skill_file_not_in_list(self) -> None:
        disc = SkillDisclosure()
        disc.register(make_skill_def("no-files"))
        assert disc.load_skill_file("no-files", "missing.txt") is None

    def test_unregister(self) -> None:
        disc = SkillDisclosure()
        disc.register(make_skill_def("temp"))
        assert disc.unregister("temp") is True
        assert disc.get_skill_index() == []
        assert disc.unregister("temp") is False

    def test_list_skill_files(self, tmp_path: Path) -> None:
        make_skill_dir(tmp_path, "multi", {"a.txt": "a", "b.txt": "b"})
        parser = SkillParser()
        skill = parser.parse_directory(tmp_path / "multi")
        assert skill is not None
        disc = SkillDisclosure()
        disc.register(skill)
        files = disc.list_skill_files("multi")
        assert len(files) == 2


# ── Task 10.3: 技能发现与安装 ───────────────────────────────────────────────


class TestSkillDiscovery:
    """技能发现与安全审计测试。"""

    def test_discover_skills(self, tmp_path: Path) -> None:
        make_skill_dir(tmp_path, "alpha")
        make_skill_dir(tmp_path, "beta")
        make_skill_dir(tmp_path, "gamma")

        disc = SkillDiscovery()
        skills = disc.discover_skills([str(tmp_path)])
        assert len(skills) == 3
        ids = {s.skill_id for s in skills}
        assert ids == {"alpha", "beta", "gamma"}

    def test_discover_empty_path(self, tmp_path: Path) -> None:
        disc = SkillDiscovery()
        skills = disc.discover_skills([str(tmp_path / "nonexistent")])
        assert skills == []

    def test_discover_dedup(self, tmp_path: Path) -> None:
        make_skill_dir(tmp_path, "dup")
        disc = SkillDiscovery()
        skills = disc.discover_skills([str(tmp_path), str(tmp_path)])
        assert len(skills) == 1

    def test_audit_safe_skill(self, tmp_path: Path) -> None:
        make_skill_dir(tmp_path, "safe")
        disc = SkillDiscovery()
        skill = disc.parser.parse_directory(tmp_path / "safe")
        assert skill is not None
        result = disc.audit_skill(skill)
        assert result.safe is True
        assert result.has_sensitive_ops is False

    def test_audit_risky_skill(self, tmp_path: Path) -> None:
        make_skill_dir(tmp_path, "risky", {
            "danger.py": "import subprocess\nsubprocess.run(['rm', '-rf', '/'])",
        })
        disc = SkillDiscovery()
        skill = disc.parser.parse_directory(tmp_path / "risky")
        assert skill is not None
        result = disc.audit_skill(skill)
        assert result.safe is False
        assert result.has_sensitive_ops is True
        assert result.has_scripts is True

    def test_audit_network_refs(self, tmp_path: Path) -> None:
        make_skill_dir(tmp_path, "network", {
            "fetch.py": "import httpx\nhttpx.get('https://example.com')",
        })
        disc = SkillDiscovery()
        skill = disc.parser.parse_directory(tmp_path / "network")
        assert skill is not None
        result = disc.audit_skill(skill)
        assert result.has_network_refs is True


# ── Task 10.4: 技能触发与激活 ───────────────────────────────────────────────


class TestSkillActivation:
    """技能触发与激活测试。"""

    def test_activate_deactivate(self) -> None:
        act = SkillActivation()
        skill = make_skill_def("test")
        act.activate(skill)
        assert act.is_active("test")
        assert act.list_active() == ["test"]
        assert act.deactivate("test") is True
        assert not act.is_active("test")

    def test_evaluate_relevance_exact_match(self) -> None:
        act = SkillActivation()
        index = [
            SkillIndexEntry(skill_id="pdf-processor", name="pdf-processor", description="处理 PDF 文件"),
            SkillIndexEntry(skill_id="csv-parser", name="csv-parser", description="解析 CSV 数据"),
        ]
        ranked = act.evaluate_relevance("处理 PDF 文件", index)
        assert ranked[0][0].skill_id == "pdf-processor"
        assert ranked[0][1] > ranked[1][1]

    def test_evaluate_relevance_keyword_match(self) -> None:
        act = SkillActivation()
        index = [
            SkillIndexEntry(skill_id="pdf-tool", name="pdf-tool", description="PDF 文件处理工具"),
            SkillIndexEntry(skill_id="other", name="other", description="其他功能"),
        ]
        ranked = act.evaluate_relevance("pdf", index)
        assert ranked[0][0].skill_id == "pdf-tool"

    def test_auto_activate(self) -> None:
        act = SkillActivation()
        skills = {
            "pdf-processor": make_skill_def("pdf-processor"),
            "csv-parser": make_skill_def("csv-parser"),
        }
        index = [
            SkillIndexEntry(skill_id="pdf-processor", name="pdf-processor", description="处理 PDF"),
            SkillIndexEntry(skill_id="csv-parser", name="csv-parser", description="解析 CSV"),
        ]
        activated = act.auto_activate("处理 PDF 文件", index, skills, threshold=0.1)
        assert "pdf-processor" in activated

    def test_auto_activate_max_limit(self) -> None:
        act = SkillActivation()
        skills = {f"s{i}": make_skill_def(f"s{i}") for i in range(5)}
        index = [
            SkillIndexEntry(skill_id=f"s{i}", name=f"s{i}", description=f"s{i} desc")
            for i in range(5)
        ]
        activated = act.auto_activate("s0 s1 s2 s3 s4", index, skills, threshold=0.0, max_activate=2)
        assert len(activated) <= 2


# ── Task 10.5: 技能与工具系统协同 ───────────────────────────────────────────


class TestSkillToolsBridge:
    """技能与工具系统协同测试。"""

    def test_check_dependencies_met(self) -> None:
        reg = ToolRegistry()
        from praxis.models.tools import ToolDefinition, ToolMetadata

        reg.register(
            ToolDefinition(name="read_file", description="读取文件", parameters={}),
            handler=lambda args: "ok",
        )
        bridge = SkillToolsBridge(reg)
        skill = make_skill_def("test")
        missing = bridge.check_tool_dependencies(skill)
        assert missing == []

    def test_check_dependencies_missing(self) -> None:
        reg = ToolRegistry()
        bridge = SkillToolsBridge(reg)
        skill = make_skill_def("test")
        missing = bridge.check_tool_dependencies(skill)
        assert "read_file" in missing

    def test_register_skill_scripts(self, tmp_path: Path) -> None:
        make_skill_dir(tmp_path, "scripted", {"run.py": "print('hello')"})
        parser = SkillParser()
        skill = parser.parse_directory(tmp_path / "scripted")
        assert skill is not None

        reg = ToolRegistry()
        bridge = SkillToolsBridge(reg)
        tools = bridge.register_skill_scripts(skill)
        assert len(tools) == 1
        assert reg.has_tool(tools[0])

    def test_unregister_skill_scripts(self, tmp_path: Path) -> None:
        make_skill_dir(tmp_path, "scripted", {"run.py": "print('hello')"})
        parser = SkillParser()
        skill = parser.parse_directory(tmp_path / "scripted")
        assert skill is not None

        reg = ToolRegistry()
        bridge = SkillToolsBridge(reg)
        tools = bridge.register_skill_scripts(skill)
        bridge.unregister_skill_scripts("scripted")
        assert not reg.has_tool(tools[0])

    def test_list_skill_tools(self, tmp_path: Path) -> None:
        make_skill_dir(tmp_path, "scripted", {"run.py": "x = 1"})
        parser = SkillParser()
        skill = parser.parse_directory(tmp_path / "scripted")
        assert skill is not None

        reg = ToolRegistry()
        bridge = SkillToolsBridge(reg)
        bridge.register_skill_scripts(skill)
        assert len(bridge.list_skill_tools("scripted")) == 1

    async def test_script_reader_handler(self, tmp_path: Path) -> None:
        script = tmp_path / "test.py"
        script.write_text("x = 42", encoding="utf-8")
        handler = SkillToolsBridge.create_script_reader(script)
        result = await handler({})
        assert "x = 42" in result


# ── Task 10.6: 技能生命周期管理 ─────────────────────────────────────────────


class TestSkillLifecycle:
    """技能生命周期管理测试。"""

    async def test_initialize(self, tmp_path: Path, store: PersistenceStore) -> None:
        make_skill_dir(tmp_path / "skills", "alpha")
        make_skill_dir(tmp_path / "skills", "beta")
        reg = ToolRegistry()
        mgr = SkillManager(reg, store)
        discovered = await mgr.initialize([str(tmp_path / "skills")])
        assert len(discovered) == 2
        index = mgr.get_skill_index()
        assert len(index) == 2

    async def test_max_skills_in_context_truncates_index(
        self, store: PersistenceStore,
    ) -> None:
        reg = ToolRegistry()
        mgr = SkillManager(reg, store, max_skills_in_context=1)
        mgr.register_skill(make_skill_def("a"))
        mgr.register_skill(make_skill_def("b"))
        mgr.register_skill(make_skill_def("c"))
        assert len(mgr.get_skill_index()) == 1

    async def test_build_skill_manager_from_config(
        self, tmp_path: Path, store: PersistenceStore,
    ) -> None:
        from praxis.config.schemas import SkillsConfig
        from praxis.skills.manager import build_skill_manager

        make_skill_dir(tmp_path / "skills", "alpha")
        reg = ToolRegistry()
        mgr = await build_skill_manager(
            SkillsConfig(
                skill_paths=[str(tmp_path / "skills")],
                auto_discover=True,
                max_skills_in_context=5,
            ),
            reg, store,
        )
        assert mgr.max_skills_in_context == 5
        assert len(mgr.get_skill_index()) == 1

    async def test_build_skill_manager_no_autodiscover(
        self, store: PersistenceStore,
    ) -> None:
        from praxis.config.schemas import SkillsConfig
        from praxis.skills.manager import build_skill_manager

        reg = ToolRegistry()
        mgr = await build_skill_manager(
            SkillsConfig(auto_discover=False), reg, store,
        )
        assert mgr.get_skill_index() == []

    async def test_register_unregister(self, store: PersistenceStore) -> None:
        reg = ToolRegistry()
        mgr = SkillManager(reg, store)
        mgr.register_skill(make_skill_def("temp"))
        assert len(mgr.get_skill_index()) == 1
        assert mgr.unregister_skill("temp") is True
        assert len(mgr.get_skill_index()) == 0

    async def test_hot_reload_index_updated(self, store: PersistenceStore) -> None:
        reg = ToolRegistry()
        mgr = SkillManager(reg, store)
        mgr.register_skill(make_skill_def("first"))
        assert len(mgr.get_skill_index()) == 1
        mgr.register_skill(make_skill_def("second"))
        assert len(mgr.get_skill_index()) == 2

    async def test_version_management(self, store: PersistenceStore) -> None:
        reg = ToolRegistry()
        mgr = SkillManager(reg, store)
        mgr.register_skill(make_skill_def("my-skill", version="1.0.0"))
        mgr.register_skill(make_skill_def("my-skill", version="2.0.0"))
        versions = mgr.list_versions("my-skill")
        assert "1.0.0" in versions
        assert "2.0.0" in versions

    async def test_upgrade_and_rollback(self, store: PersistenceStore) -> None:
        reg = ToolRegistry()
        mgr = SkillManager(reg, store)
        mgr.register_skill(make_skill_def("up", version="1.0.0"))
        v2 = make_skill_def("up", version="2.0.0")
        assert mgr.upgrade_skill(v2) is True
        current = mgr.load_skill("up")
        assert current is not None
        assert current.metadata.version == "2.0.0"
        assert mgr.rollback_skill("up", "1.0.0") is True
        rolled = mgr.load_skill("up")
        assert rolled is not None
        assert rolled.metadata.version == "1.0.0"

    async def test_usage_stats(self, store: PersistenceStore) -> None:
        reg = ToolRegistry()
        mgr = SkillManager(reg, store)
        mgr.register_skill(make_skill_def("stat"))
        mgr.record_trigger("stat", success=True)
        mgr.record_trigger("stat", success=False)
        stats = mgr.get_usage_stats("stat")
        assert stats["trigger_count"] == 2
        assert stats["success_count"] == 1

    async def test_activate_deactivate(self, store: PersistenceStore) -> None:
        reg = ToolRegistry()
        mgr = SkillManager(reg, store)
        mgr.register_skill(make_skill_def("act"))
        assert mgr.activate_skill("act") is True
        assert mgr.deactivate_skill("act") is True

    async def test_evaluate_relevance(self, store: PersistenceStore) -> None:
        reg = ToolRegistry()
        mgr = SkillManager(reg, store)
        mgr.register_skill(make_skill_def("pdf-processor"))
        mgr.register_skill(make_skill_def("csv-parser"))
        ranked = mgr.evaluate_relevance("处理 pdf-processor 文件")
        assert ranked[0][0].skill_id == "pdf-processor"

    async def test_index_cache_persistence(self, store: PersistenceStore) -> None:
        reg = ToolRegistry()
        mgr = SkillManager(reg, store)
        mgr.register_skill(make_skill_def("cached"))
        await mgr.save_index_cache()
        loaded = await mgr.load_index_cache()
        assert len(loaded) == 1
        assert loaded[0].skill_id == "cached"
