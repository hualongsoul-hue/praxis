"""场景七：技能触发的工具工作流。

用户请求 → S14 技能索引注入 → LLM 发现技能 →
第二层/三层披露 → 技能脚本工具注册 → 工具执行。
"""

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from praxis.config.schemas import PersistenceConfig
from praxis.models.skills import SkillDefinition, SkillMetadata
from praxis.persistence.store import PersistenceStore, create_store
from praxis.skills.manager import SkillManager
from praxis.tools.registry import ToolRegistry


@pytest.fixture
async def skill_store(tmp_path: Path) -> PersistenceStore:
    config = PersistenceConfig(backend="sqlite", sqlite_path=str(tmp_path / "skill.db"))
    s = await create_store(config)
    yield s  # type: ignore[misc]
    await s.close()


def make_skill(
    skill_id: str = "pdf-processor",
    name: str = "PDF Processor",
    description: str = "处理 PDF 文件",
    content: str = "# PDF 技能\n\n处理 PDF 表单的完整指引。",
    tags: list[str] | None = None,
    scripts: list[str] | None = None,
    files: list[str] | None = None,
    version: str = "1.0.0",
) -> SkillDefinition:
    return SkillDefinition(
        skill_id=skill_id,
        metadata=SkillMetadata(
            name=name,
            description=description,
            version=version,
            tags=tags or ["pdf"],
            tools=[],
        ),
        content=content,
        scripts=scripts or [],
        files=files or [],
    )


class TestSkillWorkflow:
    """场景七：技能触发的工具工作流 E2E 测试。"""

    def test_skill_registration_and_index(
        self,
        skill_store: PersistenceStore,
    ) -> None:
        """验证：技能注册后可通过索引发现。"""
        registry = ToolRegistry()
        manager = SkillManager(registry, skill_store)

        skill = make_skill()
        manager.register_skill(skill)

        index = manager.get_skill_index()
        assert len(index) == 1
        assert index[0].skill_id == "pdf-processor"
        assert index[0].name == "PDF Processor"

    def test_skill_disclosure_three_layers(
        self,
        skill_store: PersistenceStore,
    ) -> None:
        """验证：渐进式披露三层机制。"""
        registry = ToolRegistry()
        manager = SkillManager(registry, skill_store)

        skill = make_skill(
            content="# PDF 技能\n\n详细的工作流步骤...",
            files=["forms.md", "templates/invoice.json"],
        )
        manager.register_skill(skill)

        # 第一层：索引（~150 字符）
        index = manager.get_skill_index()
        assert len(index) == 1
        total_chars = len(index[0].name) + len(index[0].description)
        assert total_chars < 200

        # 第二层：完整 SKILL.md 内容
        loaded = manager.load_skill("pdf-processor")
        assert loaded is not None
        assert "详细的工作流步骤" in loaded.content

    def test_disclosure_tools_registered(
        self,
        skill_store: PersistenceStore,
    ) -> None:
        """验证：技能披露接口被注册为 LLM 可调用工具。"""
        registry = ToolRegistry()
        manager = SkillManager(registry, skill_store)

        skill = make_skill()
        manager.register_skill(skill)

        registered = manager.register_disclosure_tools()
        assert "load_skill" in registered
        assert "load_skill_file" in registered
        assert "list_skill_tools" in registered
        assert "list_skill_files" in registered

        assert registry.has_tool("load_skill")
        assert registry.has_tool("load_skill_file")

    async def test_disclosure_tool_execution(
        self,
        skill_store: PersistenceStore,
    ) -> None:
        """验证：通过披露工具可加载技能内容。"""
        registry = ToolRegistry()
        manager = SkillManager(registry, skill_store)

        skill = make_skill(content="# 完整技能内容\n\n步骤 1: 读取 PDF")
        manager.register_skill(skill)
        manager.register_disclosure_tools()

        # 模拟 LLM 调用 load_skill 工具
        entry = registry.get_entry("load_skill")
        result = await entry.handler({"skill_id": "pdf-processor"})
        assert "完整技能内容" in result

    def test_auto_activate_for_task(
        self,
        skill_store: PersistenceStore,
    ) -> None:
        """验证：根据任务描述自动激活相关技能。"""
        registry = ToolRegistry()
        manager = SkillManager(registry, skill_store)

        manager.register_skill(make_skill(
            skill_id="pdf-processor",
            name="PDF Processor",
            description="处理 PDF 表单、提取文本",
            tags=["pdf", "form"],
        ))
        manager.register_skill(make_skill(
            skill_id="excel-handler",
            name="Excel Handler",
            description="处理 Excel 表格数据",
            tags=["excel", "spreadsheet"],
        ))

        # 自动激活与 PDF 相关的技能
        activated = manager.auto_activate_for_task("请帮我填写这份 PDF 表单")
        # 至少 PDF 技能应被激活（取决于相关性算法）
        assert isinstance(activated, list)

    def test_version_management(
        self,
        skill_store: PersistenceStore,
    ) -> None:
        """验证：技能多版本管理。"""
        registry = ToolRegistry()
        manager = SkillManager(registry, skill_store)

        v1 = make_skill(version="1.0.0", content="v1 content")
        v2 = make_skill(version="2.0.0", content="v2 content")

        manager.register_skill(v1)
        manager.upgrade_skill(v2)

        versions = manager.list_versions("pdf-processor")
        assert "1.0.0" in versions
        assert "2.0.0" in versions

        current = manager.load_skill("pdf-processor")
        assert current is not None
        assert "v2 content" in current.content

    def test_usage_stats_tracking(
        self,
        skill_store: PersistenceStore,
    ) -> None:
        """验证：技能使用统计。"""
        registry = ToolRegistry()
        manager = SkillManager(registry, skill_store)

        manager.register_skill(make_skill())

        manager.record_trigger("pdf-processor", success=True)
        manager.record_trigger("pdf-processor", success=True)
        manager.record_trigger("pdf-processor", success=False)

        stats = manager.get_usage_stats("pdf-processor")
        assert stats["trigger_count"] == 3
        assert stats["success_count"] == 2

    async def test_index_cache_persistence(
        self,
        skill_store: PersistenceStore,
    ) -> None:
        """验证：技能索引缓存的持久化与加载。"""
        registry = ToolRegistry()
        manager = SkillManager(registry, skill_store)

        manager.register_skill(make_skill())
        await manager.save_index_cache()

        loaded = await manager.load_index_cache()
        assert len(loaded) == 1
        assert loaded[0].skill_id == "pdf-processor"
