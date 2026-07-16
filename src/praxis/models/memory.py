"""记忆系统数据模型——S6 四类认知记忆及辅助类型。

语义记忆（集合模式 + 档案模式）、情景记忆（学习范例）、
程序记忆（行为准则）、工作记忆（消息序列）。
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

# ── 枚举类型 ────────────────────────────────────────────────────────────────


class MemoryType(StrEnum):
    """认知记忆类型。"""

    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"
    WORKING = "working"


class MemoryStatus(StrEnum):
    """记忆生命周期状态（不可变审计）。"""

    ACTIVE = "active"
    INACTIVE = "inactive"
    SUPERSEDED = "superseded"


class ScopeType(StrEnum):
    """记忆作用域类型。"""

    SESSION = "session"
    PROJECT = "project"
    USER = "user"
    GLOBAL = "global"


class ConsolidationAction(StrEnum):
    """记忆整合决策。"""

    ADD = "add"
    UPDATE = "update"
    NOOP = "noop"


# ── 记忆作用域 ──────────────────────────────────────────────────────────────


class MemoryScope(BaseModel):
    """记忆作用域定义。

    格式：<type>/<id>，如 session/abc123、project/praxis、user/u1、global
    """

    scope_type: ScopeType
    scope_id: str = ""

    def to_string(self) -> str:
        """序列化为字符串。"""
        if self.scope_type == ScopeType.GLOBAL:
            return "global"
        return f"{self.scope_type.value}/{self.scope_id}"

    @staticmethod
    def from_string(scope_str: str) -> "MemoryScope":
        """从字符串解析。"""
        if scope_str == "global":
            return MemoryScope(scope_type=ScopeType.GLOBAL)
        parts = scope_str.split("/", maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"无效的作用域格式: {scope_str}")
        return MemoryScope(scope_type=ScopeType(parts[0]), scope_id=parts[1])


# ── 记忆基础条目 ────────────────────────────────────────────────────────────


class MemoryEntry(BaseModel):
    """记忆条目基础模型。所有认知类型的记忆共享此结构。"""

    memory_id: str = Field(default_factory=lambda: uuid4().hex)
    memory_type: MemoryType
    scope: MemoryScope
    content: str
    summary: str = ""
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    status: MemoryStatus = MemoryStatus.ACTIVE
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    version: int = 1
    superseded_by: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_accessed_at: datetime | None = None
    access_count: int = 0
    embedding: list[float] | None = None


# ── 语义记忆 ────────────────────────────────────────────────────────────────


class SemanticMemory(MemoryEntry):
    """语义记忆——事实与知识（集合模式）。

    档案模式由独立的 SemanticProfile 类表示。
    """

    memory_type: MemoryType = MemoryType.SEMANTIC


# ── 情景记忆 ────────────────────────────────────────────────────────────────


class EpisodicMemory(MemoryEntry):
    """情景记忆——经历与交互。

    捕获成功交互的学习范例：情境上下文 → 推理过程 → 采取行动 → 达成结果。
    """

    memory_type: MemoryType = MemoryType.EPISODIC
    context_description: str = ""
    reasoning: str = ""
    action_taken: str = ""
    outcome: str = ""


# ── 程序记忆 ────────────────────────────────────────────────────────────────


class ProceduralMemory(MemoryEntry):
    """程序记忆——工作流与模式。

    编码系统行为、响应模式、核心指令和经验改进。
    """

    memory_type: MemoryType = MemoryType.PROCEDURAL
    steps: list[str] = Field(default_factory=list)
    applicable_scenarios: list[str] = Field(default_factory=list)


# ── 工作记忆 ────────────────────────────────────────────────────────────────


class WorkingMemoryMessage(BaseModel):
    """工作记忆中的单条消息。"""

    message_id: str = Field(default_factory=lambda: uuid4().hex)
    role: str
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    tool_call_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkingMemory(BaseModel):
    """工作记忆——当前会话上下文。

    维护消息序列和中间推理状态。
    """

    session_id: str
    messages: list[WorkingMemoryMessage] = Field(default_factory=lambda: [])
    max_messages: int = 200

    def append(self, message: WorkingMemoryMessage) -> None:
        """追加消息到工作记忆，超出 max_messages 时丢弃最老消息。"""
        self.messages.append(message)
        if self.max_messages > 0 and len(self.messages) > self.max_messages:
            overflow = len(self.messages) - self.max_messages
            del self.messages[:overflow]

    def get_recent(self, limit: int | None = None) -> list[WorkingMemoryMessage]:
        """获取最近的消息列表。"""
        if limit is None:
            return list(self.messages)
        return list(self.messages[-limit:])

    def clear(self) -> None:
        """清空工作记忆。"""
        self.messages.clear()


# ── 语义档案（F6.1 Profile 模式） ──────────────────────────────────────────


class SemanticProfile(BaseModel):
    """语义档案——档案模式的结构化存储。

    每 (scope, schema_name) 对应唯一档案文档，字段字典就地合并更新。
    """

    profile_id: str = Field(default_factory=lambda: uuid4().hex)
    scope: MemoryScope
    schema_name: str
    fields: dict[str, Any] = Field(default_factory=dict)
    version: int = 1
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ── 记忆索引（渐进式检索第一层） ────────────────────────────────────────────


class MemoryIndexEntry(BaseModel):
    """记忆轻量索引条目（~150 字符），始终加载到系统提示。"""

    memory_id: str
    memory_type: MemoryType
    scope: str
    summary: str
    tags: list[str] = Field(default_factory=list)
    confidence: float = 1.0
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ── 检索结果 ────────────────────────────────────────────────────────────────


class MemorySearchResult(BaseModel):
    """记忆检索结果。"""

    entry: MemoryEntry
    relevance_score: float = 0.0
    source: str = ""


# ── 版本历史 ────────────────────────────────────────────────────────────────


class MemoryVersion(BaseModel):
    """记忆版本记录，用于审计追溯。"""

    memory_id: str
    version: int
    content: str
    status: MemoryStatus
    changed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    change_reason: str = ""
