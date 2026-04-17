"""技能生命周期管理。

register_skill/unregister_skill 运行时热加载，
版本管理（多版本共存、升级、回退），
使用统计（S2 记录触发次数/成功率/上下文消耗），
索引缓存通过 S3 持久化。
"""

from typing import Any

from praxis.models.skills import SkillDefinition, SkillIndexEntry
from praxis.persistence.store import PersistenceStore
from praxis.skills.activation import SkillActivation
from praxis.skills.disclosure import SkillDisclosure
from praxis.skills.discovery import SkillDiscovery
from praxis.skills.tools_bridge import SkillToolsBridge
from praxis.telemetry.logger import get_logger
from praxis.telemetry.metrics import emit_metric
from praxis.tools.registry import ToolRegistry

log = get_logger("skills.lifecycle")

SKILLS_NAMESPACE = "skills"
INDEX_KEY = "skill_index"


class SkillLifecycleManager:
    """技能生命周期管理器。

    统一管理技能的发现、注册、激活、注销和索引缓存。
    """

    def __init__(
        self,
        registry: ToolRegistry,
        store: PersistenceStore,
    ) -> None:
        self.discovery = SkillDiscovery()
        self.disclosure = SkillDisclosure()
        self.activation = SkillActivation()
        self.bridge = SkillToolsBridge(registry)
        self.store = store
        self.skills: dict[str, SkillDefinition] = {}
        self.versions: dict[str, dict[str, SkillDefinition]] = {}
        self.usage_stats: dict[str, dict[str, int]] = {}

    async def initialize(self, paths: list[str]) -> list[SkillDefinition]:
        """启动时初始化：发现并注册所有技能。

        Args:
            paths: 技能搜索路径列表。

        Returns:
            发现并注册的技能列表。
        """
        discovered = self.discovery.discover_skills(paths)
        for skill in discovered:
            self.register_skill(skill)

        await self.save_index_cache()

        log.info("技能系统初始化完成", count=len(discovered))
        return discovered

    def register_skill(self, skill: SkillDefinition) -> None:
        """注册技能。

        支持运行时热加载，注册后立即可通过索引发现。

        Args:
            skill: 技能定义。
        """
        self.skills[skill.skill_id] = skill
        self.disclosure.register(skill)

        if skill.skill_id not in self.versions:
            self.versions[skill.skill_id] = {}
        self.versions[skill.skill_id][skill.metadata.version] = skill

        if skill.skill_id not in self.usage_stats:
            self.usage_stats[skill.skill_id] = {
                "trigger_count": 0,
                "success_count": 0,
            }

        emit_metric("skill_registered", 1.0, {"skill_id": skill.skill_id}, "counter")
        log.info(
            "技能已注册",
            skill_id=skill.skill_id,
            version=skill.metadata.version,
        )

    def unregister_skill(self, skill_id: str) -> bool:
        """注销技能。

        Args:
            skill_id: 技能标识符。

        Returns:
            是否成功注销。
        """
        if skill_id not in self.skills:
            return False

        self.disclosure.unregister(skill_id)
        self.activation.deactivate(skill_id)
        self.bridge.unregister_skill_scripts(skill_id)
        del self.skills[skill_id]

        log.info("技能已注销", skill_id=skill_id)
        return True

    def get_skill_index(self) -> list[SkillIndexEntry]:
        """获取所有已注册技能的轻量索引。"""
        return self.disclosure.get_skill_index()

    def load_skill(self, skill_id: str) -> SkillDefinition | None:
        """加载完整技能内容。"""
        return self.disclosure.load_skill(skill_id)

    def load_skill_file(self, skill_id: str, filename: str) -> str | None:
        """加载技能附属文件。"""
        return self.disclosure.load_skill_file(skill_id, filename)

    def list_skill_tools(self, skill_id: str) -> list[str]:
        """列出技能附带的脚本工具。"""
        return self.bridge.list_skill_tools(skill_id)

    def evaluate_relevance(
        self,
        task_description: str,
    ) -> list[tuple[SkillIndexEntry, float]]:
        """评估任务与技能的相关性。"""
        index = self.get_skill_index()
        return self.activation.evaluate_relevance(task_description, index)

    def activate_skill(self, skill_id: str) -> bool:
        """手动激活技能。"""
        skill = self.skills.get(skill_id)
        if skill is None:
            return False
        self.activation.activate(skill)
        self.bridge.register_skill_scripts(skill)
        return True

    def deactivate_skill(self, skill_id: str) -> bool:
        """停用技能。"""
        self.bridge.unregister_skill_scripts(skill_id)
        return self.activation.deactivate(skill_id)

    def get_version(self, skill_id: str, version: str) -> SkillDefinition | None:
        """获取指定版本的技能。"""
        versions = self.versions.get(skill_id, {})
        return versions.get(version)

    def list_versions(self, skill_id: str) -> list[str]:
        """列出技能的所有版本。"""
        return list(self.versions.get(skill_id, {}).keys())

    def upgrade_skill(self, new_skill: SkillDefinition) -> bool:
        """升级技能到新版本。

        Args:
            new_skill: 新版本技能定义。

        Returns:
            是否成功升级。
        """
        sid = new_skill.skill_id
        if sid not in self.skills:
            return False

        was_active = self.activation.is_active(sid)
        if was_active:
            self.deactivate_skill(sid)

        self.register_skill(new_skill)

        if was_active:
            self.activate_skill(sid)

        log.info("技能已升级", skill_id=sid, version=new_skill.metadata.version)
        return True

    def rollback_skill(self, skill_id: str, version: str) -> bool:
        """回退技能到指定版本。"""
        target = self.get_version(skill_id, version)
        if target is None:
            return False
        return self.upgrade_skill(target)

    def record_trigger(self, skill_id: str, success: bool) -> None:
        """记录技能触发事件。

        Args:
            skill_id: 技能标识符。
            success: 是否成功。
        """
        stats = self.usage_stats.get(skill_id)
        if stats is None:
            return
        stats["trigger_count"] += 1
        if success:
            stats["success_count"] += 1

        emit_metric(
            "skill_trigger",
            1.0,
            {"skill_id": skill_id, "success": str(success).lower()},
            "counter",
        )

    def get_usage_stats(self, skill_id: str) -> dict[str, int]:
        """获取技能使用统计。"""
        return dict(self.usage_stats.get(skill_id, {}))

    async def save_index_cache(self) -> None:
        """将索引缓存持久化到 S3。"""
        index = self.get_skill_index()
        data = [entry.model_dump(mode="json") for entry in index]
        await self.store.save(SKILLS_NAMESPACE, INDEX_KEY, data)

    async def load_index_cache(self) -> list[SkillIndexEntry]:
        """从 S3 加载索引缓存。"""
        data = await self.store.load(SKILLS_NAMESPACE, INDEX_KEY)
        if data is None or not isinstance(data, list):
            return []
        return [SkillIndexEntry.model_validate(item) for item in data]
