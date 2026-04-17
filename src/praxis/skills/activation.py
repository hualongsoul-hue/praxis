"""技能触发与激活。

自动触发（语义匹配任务描述）和手动触发，
evaluate_relevance 接口，多技能并行激活，
激活事件通过 S2 记录。
"""

from praxis.models.skills import SkillDefinition, SkillIndexEntry
from praxis.telemetry.logger import get_logger
from praxis.telemetry.metrics import emit_metric

log = get_logger("skills.activation")


class SkillActivation:
    """技能触发与激活管理器。"""

    def __init__(self) -> None:
        self.active_skills: dict[str, SkillDefinition] = {}

    def activate(self, skill: SkillDefinition) -> None:
        """手动激活技能。

        Args:
            skill: 技能定义。
        """
        self.active_skills[skill.skill_id] = skill
        emit_metric(
            "skill_activated",
            1.0,
            {"skill_id": skill.skill_id},
            "counter",
        )
        log.info("技能已激活", skill_id=skill.skill_id)

    def deactivate(self, skill_id: str) -> bool:
        """停用技能。"""
        if skill_id in self.active_skills:
            del self.active_skills[skill_id]
            log.info("技能已停用", skill_id=skill_id)
            return True
        return False

    def is_active(self, skill_id: str) -> bool:
        """检查技能是否处于激活状态。"""
        return skill_id in self.active_skills

    def list_active(self) -> list[str]:
        """列出所有激活的技能 ID。"""
        return list(self.active_skills.keys())

    def evaluate_relevance(
        self,
        task_description: str,
        skill_index: list[SkillIndexEntry],
    ) -> list[tuple[SkillIndexEntry, float]]:
        """评估任务与技能的相关性。

        基于关键词匹配评估相关性分数（0.0~1.0），
        按得分降序排列。

        Args:
            task_description: 当前任务描述。
            skill_index: 技能索引列表。

        Returns:
            (索引条目, 相关性分数) 元组列表，按得分降序。
        """
        task_lower = task_description.lower()
        task_tokens = set(task_lower.split())

        scored: list[tuple[SkillIndexEntry, float]] = []
        for entry in skill_index:
            score = self.compute_relevance(task_lower, task_tokens, entry)
            scored.append((entry, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    @staticmethod
    def compute_relevance(
        task_lower: str,
        task_tokens: set[str],
        entry: SkillIndexEntry,
    ) -> float:
        """计算单个技能与任务的相关性分数。

        综合名称匹配、描述关键词重叠和精确包含。

        Args:
            task_lower: 小写任务描述。
            task_tokens: 任务描述分词集合。
            entry: 技能索引条目。

        Returns:
            0.0~1.0 之间的相关性分数。
        """
        score = 0.0
        name_lower = entry.name.lower()
        desc_lower = entry.description.lower()

        if name_lower in task_lower:
            score += 0.5

        name_tokens = set(name_lower.replace("-", " ").replace("_", " ").split())
        name_overlap = len(task_tokens & name_tokens)
        if name_tokens:
            score += 0.3 * (name_overlap / len(name_tokens))

        desc_tokens = set(desc_lower.replace("-", " ").replace("_", " ").split())
        desc_overlap = len(task_tokens & desc_tokens)
        if desc_tokens:
            score += 0.2 * (desc_overlap / len(desc_tokens))

        return min(1.0, score)

    def auto_activate(
        self,
        task_description: str,
        skill_index: list[SkillIndexEntry],
        available_skills: dict[str, SkillDefinition],
        threshold: float = 0.3,
        max_activate: int = 3,
    ) -> list[str]:
        """自动激活与任务相关的技能。

        Args:
            task_description: 任务描述。
            skill_index: 技能索引。
            available_skills: 可用技能映射。
            threshold: 最低相关性阈值。
            max_activate: 最多同时激活数。

        Returns:
            新激活的技能 ID 列表。
        """
        ranked = self.evaluate_relevance(task_description, skill_index)
        activated: list[str] = []

        for entry, score in ranked:
            if score < threshold:
                break
            if len(activated) >= max_activate:
                break
            if entry.skill_id in self.active_skills:
                continue
            skill = available_skills.get(entry.skill_id)
            if skill is not None:
                self.activate(skill)
                activated.append(entry.skill_id)

        return activated
