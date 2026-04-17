"""工具相关类型定义——跨 S5、S8、S11 共享。"""

from typing import Any, Literal

from pydantic import BaseModel, Field


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
    execution_time_ms: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolMetadata(BaseModel):
    """工具元数据，供 S8 护栏裁决使用。"""

    category: str = "general"
    permission_level: Literal["auto_approve", "confirm", "deny"] = "confirm"
    readonly: bool = False
    timeout_seconds: float = 30.0
    tags: list[str] = Field(default_factory=list)


class ToolDefinition(BaseModel):
    """完整工具定义，用于 S5 工具注册。"""

    name: str
    description: str
    parameters: dict[str, Any]
    metadata: ToolMetadata = Field(default_factory=ToolMetadata)
