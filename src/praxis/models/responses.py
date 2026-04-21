"""模型响应类型定义——跨 S4、S11 共享。"""

from typing import Literal

from pydantic import BaseModel

from praxis.models.tools import ToolCall


class Usage(BaseModel):
    """Token 用量统计。

    - ``reasoning_tokens``：思考阶段消耗（o1/o3/R1/GLM-thinking 等），
      已包含在 ``completion_tokens`` 内，单独暴露用于成本归因与观测。
    - ``cached_prompt_tokens``：prompt 缓存命中数（Claude/OpenAI/DeepSeek 等），
      已计入 ``prompt_tokens``；命中后厂商通常折扣，需据此修正成本。
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    reasoning_tokens: int = 0
    cached_prompt_tokens: int = 0


class ModelResponse(BaseModel):
    """LLM 完整响应（S4 从 LiteLLM 响应转换而来）。"""

    id: str
    content: str | None = None
    reasoning_content: str | None = None
    refusal: str | None = None
    tool_calls: list[ToolCall] | None = None
    usage: Usage
    model: str
    finish_reason: str | None = None
    created: int
    system_fingerprint: str | None = None


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
    delta_reasoning_content: str | None = None
    delta_refusal: str | None = None
    delta_tool_calls: list[ToolCallDelta] | None = None
    usage: Usage | None = None
    model: str | None = None
    finish_reason: str | None = None
    system_fingerprint: str | None = None
