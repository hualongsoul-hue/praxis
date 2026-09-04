"""工具相关类型定义——跨 S5、S8、S11 共享。"""

from collections.abc import Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FunctionCall(BaseModel):
    """LLM 请求的函数调用。"""

    name: str
    arguments: str


class ToolCall(BaseModel):
    """LLM 返回的工具调用请求。"""

    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class ToolResult(BaseModel):
    """工具执行结果，格式化为 LLM 可读的观察。"""

    tool_call_id: str
    success: bool
    content: str
    error: str | None = None
    error_type: str | None = None
    execution_time_ms: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolExecutionState(StrEnum):
    """Durable state of one tool call at its side-effect boundary."""

    PREPARED = "prepared"
    STARTED = "started"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNCERTAIN = "uncertain"


class ToolExecutionRecord(BaseModel):
    """Checkpointed tool-call ledger entry used to prevent duplicate writes."""

    tool_call_id: str
    tool_name: str
    argument_digest: str
    state: ToolExecutionState
    readonly: bool
    idempotent: bool
    result: ToolResult | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ToolMetadata(BaseModel):
    """工具元数据，供 S8 护栏裁决使用。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    category: str = "general"
    # None 表示"未声明"，由 S8 权限策略的 default_permission 兜底；
    # 显式声明时该声明优先于全局默认。
    permission_level: Literal["auto_approve", "confirm", "deny"] | None = None
    readonly: bool = False
    idempotent: bool = False
    timeout_seconds: float | None = Field(default=None, gt=0)
    tags: Sequence[str] = ()

    @field_validator("tags")
    @classmethod
    def freeze_tags(cls, value: Sequence[str]) -> tuple[str, ...]:
        return tuple(value)


class ToolDefinition(BaseModel):
    """完整工具定义，用于 S5 工具注册。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str
    parameters: dict[str, Any]
    metadata: ToolMetadata = Field(default_factory=ToolMetadata)


class ApprovalRequest(BaseModel):
    """交给宿主应用审批处理器的结构化工具请求。"""

    request_id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str | None = None
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    requested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ApprovalDecision(BaseModel):
    """审批处理器的显式决定。"""

    approved: bool
    reason: str = ""
