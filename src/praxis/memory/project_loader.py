"""项目级记忆预加载器（F6.4）。

从项目根目录的 `praxis.md` 文件加载 project 作用域的语义记忆。
格式：
  - 可选 YAML frontmatter（`---` 分隔），字段包括 `project`、`tags`
  - 正文每个 `## <heading>` 二级标题段作为独立记忆条目
"""

import re
from pathlib import Path
from typing import Any

import yaml

from praxis.memory.store import ScopedMemoryStore
from praxis.memory.vector import VectorStore
from praxis.models.memory import (
    MemoryScope,
    MemoryType,
    ScopeType,
    SemanticMemory,
)
from praxis.telemetry.logger import get_logger
from praxis.telemetry.metrics import emit_metric

log = get_logger("memory.project_loader")

PRAXIS_MD_FILENAME = "praxis.md"
PRELOAD_SOURCE = "project_praxis_md"

FRONTMATTER_PATTERN = re.compile(
    r"^---\s*\n(.*?)\n---\s*\n(.*)$",
    re.DOTALL,
)


class ProjectMemoryLoader:
    """从项目根目录加载 praxis.md 为项目级语义记忆。"""

    def __init__(
        self,
        scoped_store: ScopedMemoryStore,
        vector_store: VectorStore,
    ) -> None:
        self.scoped_store = scoped_store
        self.vector_store = vector_store

    async def load(
        self,
        project_root: str | Path,
        default_project_name: str,
    ) -> int:
        """加载 praxis.md，若已导入过则跳过。返回新增条目数。"""
        root = Path(project_root)
        md_path = root / PRAXIS_MD_FILENAME
        if not md_path.exists() or not md_path.is_file():
            log.info("未发现 praxis.md，跳过预加载", path=str(md_path))
            return 0

        try:
            text = md_path.read_text(encoding="utf-8")
        except OSError as exc:
            log.warning("praxis.md 读取失败", error=str(exc))
            return 0

        frontmatter, body = self.split_frontmatter(text)
        project_name = frontmatter.get("project") or default_project_name
        common_tags = frontmatter.get("tags") or []
        if not isinstance(common_tags, list):
            common_tags = []

        scope = MemoryScope(scope_type=ScopeType.PROJECT, scope_id=str(project_name))

        # 去重：仅当目标作用域下无预加载记录时才导入
        existing = await self.scoped_store.list_scope(scope, MemoryType.SEMANTIC)
        if any(e.metadata.get("source") == PRELOAD_SOURCE for e in existing):
            log.info("praxis.md 已导入，跳过重复加载", scope=scope.to_string())
            return 0

        sections = self.split_sections(body)
        if not sections:
            log.info("praxis.md 无有效内容段", path=str(md_path))
            return 0

        count = 0
        for heading, content in sections:
            entry = SemanticMemory(
                scope=scope,
                content=content,
                summary=heading[:150],
                tags=[str(t) for t in common_tags],
                metadata={
                    "source": PRELOAD_SOURCE,
                    "heading": heading,
                    "file": str(md_path),
                },
            )
            await self.vector_store.add(entry)
            count += 1

        emit_metric(
            "memory_project_preload",
            float(count),
            {"project": project_name},
            "counter",
        )
        log.info("praxis.md 预加载完成", scope=scope.to_string(), count=count)
        return count

    @staticmethod
    def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
        """拆分 YAML frontmatter 与正文。"""
        match = FRONTMATTER_PATTERN.match(text)
        if match is None:
            return {}, text
        try:
            data = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError:
            return {}, text
        if not isinstance(data, dict):
            return {}, text
        return data, match.group(2)

    @staticmethod
    def split_sections(body: str) -> list[tuple[str, str]]:
        """按二级标题拆分段落。"""
        sections: list[tuple[str, str]] = []
        current_heading: str | None = None
        current_lines: list[str] = []

        for line in body.splitlines():
            if line.startswith("## "):
                if current_heading is not None and current_lines:
                    content = "\n".join(current_lines).strip()
                    if content:
                        sections.append((current_heading, content))
                current_heading = line[3:].strip()
                current_lines = []
            elif current_heading is not None:
                current_lines.append(line)

        if current_heading is not None and current_lines:
            content = "\n".join(current_lines).strip()
            if content:
                sections.append((current_heading, content))
        return sections
