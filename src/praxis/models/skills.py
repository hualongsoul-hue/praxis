"""技能系统数据模型——S14 跨组件共享类型。"""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class SkillMetadata(BaseModel):
    """技能元数据（来自 SKILL.md frontmatter）。"""

    name: str
    description: str = ""
    version: str = "1.0.0"
    author: str = ""
    tags: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class SkillDefinition(BaseModel):
    """完整技能定义。"""

    skill_id: str
    metadata: SkillMetadata
    content: str = ""
    base_path: str = ""
    files: list[str] = Field(default_factory=list)
    scripts: list[str] = Field(default_factory=list)
    registered_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SkillIndexEntry(BaseModel):
    """技能轻量索引条目（~150 字符）——渐进式披露第一层。"""

    skill_id: str
    name: str
    description: str


class SkillAuditResult(BaseModel):
    """技能安装安全审计结果。"""

    skill_id: str
    safe: bool = True
    warnings: list[str] = Field(default_factory=list)
    has_scripts: bool = False
    has_network_refs: bool = False
    has_sensitive_ops: bool = False
