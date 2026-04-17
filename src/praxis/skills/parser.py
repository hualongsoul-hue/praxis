"""技能格式解析。

解析 SKILL.md（YAML frontmatter + Markdown 正文），
目录结构验证，附属文件发现。
"""

from pathlib import Path

import yaml

from praxis.models.skills import SkillDefinition, SkillMetadata
from praxis.telemetry.logger import get_logger

log = get_logger("skills.parser")

SKILL_FILENAME = "SKILL.md"
SCRIPT_EXTENSIONS = frozenset({".py", ".sh", ".bash", ".ps1"})


class SkillParser:
    """技能格式解析器。

    从技能目录解析 SKILL.md 和附属文件。
    """

    def parse_directory(self, skill_dir: Path) -> SkillDefinition | None:
        """解析技能目录。

        Args:
            skill_dir: 技能目录路径。

        Returns:
            技能定义，或 None（目录无效）。
        """
        skill_file = skill_dir / SKILL_FILENAME
        if not skill_file.is_file():
            log.warning("目录中未找到 SKILL.md", path=str(skill_dir))
            return None

        raw_text = skill_file.read_text(encoding="utf-8")
        metadata, content = self.parse_skill_md(raw_text)
        if metadata is None:
            log.warning("SKILL.md 解析失败", path=str(skill_file))
            return None

        skill_id = metadata.name or skill_dir.name
        files = self.discover_files(skill_dir)
        scripts = self.discover_scripts(skill_dir)

        log.info(
            "技能解析完成",
            skill_id=skill_id,
            files_count=len(files),
            scripts_count=len(scripts),
        )

        return SkillDefinition(
            skill_id=skill_id,
            metadata=metadata,
            content=content,
            base_path=str(skill_dir),
            files=files,
            scripts=scripts,
        )

    @staticmethod
    def parse_skill_md(raw_text: str) -> tuple[SkillMetadata | None, str]:
        """解析 SKILL.md 内容。

        Args:
            raw_text: SKILL.md 原始文本。

        Returns:
            (元数据, Markdown 正文)，解析失败时元数据为 None。
        """
        stripped = raw_text.strip()
        if not stripped.startswith("---"):
            return None, raw_text

        end_index = stripped.find("---", 3)
        if end_index == -1:
            return None, raw_text

        frontmatter_text = stripped[3:end_index].strip()
        content = stripped[end_index + 3:].strip()

        try:
            data = yaml.safe_load(frontmatter_text)
            if not isinstance(data, dict):
                return None, raw_text
        except yaml.YAMLError:
            return None, raw_text

        name = data.pop("name", "")
        description = data.pop("description", "")
        version = data.pop("version", "1.0.0")
        author = data.pop("author", "")
        tags = data.pop("tags", [])
        tools = data.pop("tools", [])

        if not isinstance(tags, list):
            tags = []
        if not isinstance(tools, list):
            tools = []

        metadata = SkillMetadata(
            name=str(name),
            description=str(description),
            version=str(version),
            author=str(author),
            tags=[str(t) for t in tags],
            tools=[str(t) for t in tools],
            extra=data,
        )
        return metadata, content

    @staticmethod
    def discover_files(skill_dir: Path) -> list[str]:
        """发现技能目录中的附属文件。

        Args:
            skill_dir: 技能目录。

        Returns:
            相对路径列表（排除 SKILL.md 本身）。
        """
        files: list[str] = []
        for path in skill_dir.rglob("*"):
            if path.is_file() and path.name != SKILL_FILENAME:
                rel = path.relative_to(skill_dir)
                files.append(rel.as_posix())
        return sorted(files)

    @staticmethod
    def discover_scripts(skill_dir: Path) -> list[str]:
        """发现技能目录中的可执行脚本。

        Args:
            skill_dir: 技能目录。

        Returns:
            脚本相对路径列表。
        """
        scripts: list[str] = []
        for path in skill_dir.rglob("*"):
            if path.is_file() and path.suffix in SCRIPT_EXTENSIONS:
                rel = path.relative_to(skill_dir)
                scripts.append(rel.as_posix())
        return sorted(scripts)

    @staticmethod
    def validate_directory(skill_dir: Path) -> list[str]:
        """验证技能目录结构。

        Args:
            skill_dir: 技能目录。

        Returns:
            问题列表，空列表表示有效。
        """
        issues: list[str] = []

        if not skill_dir.is_dir():
            issues.append(f"路径不是目录: {skill_dir}")
            return issues

        skill_file = skill_dir / SKILL_FILENAME
        if not skill_file.is_file():
            issues.append(f"缺少 {SKILL_FILENAME}")
            return issues

        raw_text = skill_file.read_text(encoding="utf-8")
        if not raw_text.strip().startswith("---"):
            issues.append("SKILL.md 缺少 YAML frontmatter")

        return issues
