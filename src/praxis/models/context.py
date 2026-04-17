"""上下文引擎数据模型——S7 跨组件共享类型。"""

from typing import Any

from pydantic import BaseModel, Field


class TurnContext(BaseModel):
    """单轮上下文输入。"""

    user_message: str
    system_prompt_override: str | None = None
    developer_instructions: str = ""
    user_instructions: str = ""
    task_stage: str = "general"
    extra: dict[str, Any] = Field(default_factory=dict)


class AssembledPrompt(BaseModel):
    """组装后的完整 Prompt。"""

    messages: list[dict[str, Any]] = Field(default_factory=list)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    token_count: int = 0
    max_tokens: int = 0
    layers_included: list[str] = Field(default_factory=list)


class TokenUsage(BaseModel):
    """上下文 Token 用量。"""

    current_tokens: int = 0
    max_tokens: int = 0
    usage_ratio: float = 0.0
    compaction_needed: bool = False


class CompactionResult(BaseModel):
    """压缩结果。"""

    original_tokens: int = 0
    compacted_tokens: int = 0
    summary: str = ""
    retained_file_refs: list[str] = Field(default_factory=list)


class ContextState(BaseModel):
    """可序列化的上下文状态。"""

    messages: list[dict[str, Any]] = Field(default_factory=list)
    tool_schemas: list[dict[str, Any]] = Field(default_factory=list)
    file_refs: list[str] = Field(default_factory=list)
    compaction_count: int = 0
    total_turns: int = 0
