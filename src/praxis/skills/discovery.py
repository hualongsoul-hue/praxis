"""技能发现与安装。

从配置路径扫描技能目录，版本控制共享支持，
安装安全审计（代码依赖、网络连接、敏感操作检查）。
"""

import re
from collections.abc import Sequence
from pathlib import Path

from praxis.models.skills import SkillAuditResult, SkillDefinition
from praxis.skills.parser import SCRIPT_EXTENSIONS, SkillParser
from praxis.skills.paths import resolve_skill_path
from praxis.telemetry.logger import get_logger
from praxis.telemetry.metrics import emit_metric

log = get_logger("skills.discovery")

NETWORK_PATTERNS = re.compile(
    r"(https?://|requests\.|urllib\.|httpx\.|aiohttp\.|socket\.)",
    re.IGNORECASE,
)
SENSITIVE_PATTERNS = re.compile(
    r"(os\.system|subprocess\.|eval\(|exec\(|__import__|shutil\.rmtree)",
    re.IGNORECASE,
)


class SkillDiscovery:
    """技能发现管理器。"""

    def __init__(self) -> None:
        self.parser = SkillParser()

    def discover_skills(self, paths: Sequence[str]) -> list[SkillDefinition]:
        """从多个路径发现技能。

        Args:
            paths: 技能搜索路径列表。

        Returns:
            发现的技能定义列表。
        """
        skills: list[SkillDefinition] = []
        seen_ids: set[str] = set()

        for raw_path in paths:
            expanded = Path(raw_path).expanduser()
            if not expanded.is_dir():
                continue

            for child in sorted(expanded.iterdir()):
                if not child.is_dir():
                    continue
                skill = self.parser.parse_directory(child)
                if skill is not None and skill.skill_id not in seen_ids:
                    seen_ids.add(skill.skill_id)
                    skills.append(skill)

        emit_metric(
            "skills_discovered",
            float(len(skills)),
            {"paths_count": str(len(paths))},
            "counter",
        )
        log.info("技能发现完成", count=len(skills), paths=len(paths))
        return skills

    def audit_skill(self, skill: SkillDefinition) -> SkillAuditResult:
        """安全审计技能。

        检查代码依赖、网络连接、敏感操作。

        Args:
            skill: 技能定义。

        Returns:
            审计结果。
        """
        warnings: list[str] = []
        has_scripts = len(skill.scripts) > 0
        has_network = False
        has_sensitive = False

        base = Path(skill.base_path)

        for rel_path in skill.files + [""]:
            if rel_path:
                try:
                    file_path = resolve_skill_path(base, rel_path)
                except Exception:
                    has_sensitive = True
                    warnings.append(f"检测到越界资源路径: {rel_path}")
                    continue
            else:
                file_path = resolve_skill_path(base, "SKILL.md")

            if not file_path.is_file():
                continue

            suffix = file_path.suffix
            if suffix not in SCRIPT_EXTENSIONS and suffix != ".md":
                continue

            content = file_path.read_text(encoding="utf-8", errors="ignore")

            if NETWORK_PATTERNS.search(content):
                has_network = True
                warnings.append(f"检测到网络引用: {rel_path or 'SKILL.md'}")

            if suffix in SCRIPT_EXTENSIONS and SENSITIVE_PATTERNS.search(content):
                has_sensitive = True
                warnings.append(f"检测到敏感操作: {rel_path or 'SKILL.md'}")

        safe = not has_sensitive

        log.info(
            "技能安全审计",
            skill_id=skill.skill_id,
            safe=safe,
            warnings_count=len(warnings),
        )

        return SkillAuditResult(
            skill_id=skill.skill_id,
            safe=safe,
            warnings=warnings,
            has_scripts=has_scripts,
            has_network_refs=has_network,
            has_sensitive_ops=has_sensitive,
        )
