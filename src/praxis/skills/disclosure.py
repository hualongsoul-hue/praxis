"""渐进式披露（Progressive Disclosure）。

三层按需加载：
第一层：name + description（~150 字符索引）——始终加载到系统提示
第二层：完整 SKILL.md 正文——Agent 判断相关时加载
第三层：附属文件——深度探索时按需加载
"""


from praxis.models.skills import SkillDefinition, SkillIndexEntry
from praxis.skills.paths import resolve_skill_path
from praxis.telemetry.logger import get_logger

log = get_logger("skills.disclosure")


class SkillDisclosure:
    """渐进式披露管理器。"""

    def __init__(self) -> None:
        self.skills: dict[str, SkillDefinition] = {}

    def register(self, skill: SkillDefinition) -> None:
        """注册技能到披露系统。"""
        self.skills[skill.skill_id] = skill

    def unregister(self, skill_id: str) -> bool:
        """从披露系统注销技能。"""
        if skill_id in self.skills:
            del self.skills[skill_id]
            return True
        return False

    def get_skill_index(self) -> list[SkillIndexEntry]:
        """第一层：获取所有技能的轻量索引。

        Returns:
            索引条目列表（name + description，约 150 字符/条目）。
        """
        entries: list[SkillIndexEntry] = []
        for skill in self.skills.values():
            entries.append(SkillIndexEntry(
                skill_id=skill.skill_id,
                name=skill.metadata.name,
                description=skill.metadata.description[:150],
            ))
        return entries

    def load_skill(self, skill_id: str) -> SkillDefinition | None:
        """第二层：加载完整技能内容（SKILL.md 正文）。

        Args:
            skill_id: 技能标识符。

        Returns:
            完整技能定义，未找到时返回 None。
        """
        skill = self.skills.get(skill_id)
        if skill is None:
            log.warning("技能未找到", skill_id=skill_id)
        return skill

    def load_skill_file(self, skill_id: str, filename: str) -> str | None:
        """第三层：按需加载技能附属文件。

        Args:
            skill_id: 技能标识符。
            filename: 附属文件相对路径。

        Returns:
            文件内容文本，未找到时返回 None。
        """
        skill = self.skills.get(skill_id)
        if skill is None:
            log.warning("技能未找到", skill_id=skill_id)
            return None

        if filename not in skill.files:
            log.warning("附属文件不在技能目录中", skill_id=skill_id, file_name=filename)
            return None

        file_path = resolve_skill_path(skill.base_path, filename)
        if not file_path.is_file():
            log.warning("附属文件不存在", path=str(file_path))
            return None

        return file_path.read_text(encoding="utf-8")

    def list_skill_files(self, skill_id: str) -> list[str]:
        """列出技能的所有附属文件。

        Args:
            skill_id: 技能标识符。

        Returns:
            文件相对路径列表。
        """
        skill = self.skills.get(skill_id)
        if skill is None:
            return []
        return list(skill.files)
