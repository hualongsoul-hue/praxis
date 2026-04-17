"""模型响应类型定义——跨 S4、S11 共享。"""

from typing import Literal

from pydantic import BaseModel

from praxis.models.tools import ToolCall


class Usage(BaseModel):
    """Token 用量统计。"""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ModelResponse(BaseModel):
    """LLM 完整响应（S4 从 LiteLLM 响应转换而来）。"""

    id: str
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    usage: Usage
    model: str
    finish_reason: str | None = None
    created: int


class FunctionCallDelta(BaseModel):
    """流式函数调用增量。"""

    name: str | None = None
    arguments: str | None = None


class ToolCallDelta(BaseModel):
    """流式工具调用增量。"""

    index: int
    id: str | None = None
    type: Literal["function"] | None = None
    function: FunctionCallDelta | None = None


class ModelResponseChunk(BaseModel):
    """LLM 流式响应块。"""

    id: str
    delta_content: str | None = None
    delta_tool_calls: list[ToolCallDelta] | None = None
    usage: Usage | None = None
    model: str | None = None
    finish_reason: str | None = None
