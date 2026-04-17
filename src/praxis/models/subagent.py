"""子代理协调数据模型——S13 跨组件共享类型。"""

from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class SubagentMode(str, Enum):
    """子代理执行模型。"""

    AGENT_AS_TOOL = "agent_as_tool"
    HANDOFF = "handoff"
    FORK = "fork"


class SubagentStatus(str, Enum):
    """子代理状态。"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    FAILED = "failed"


class SubagentResult(BaseModel):
    """子代理返回结果。"""

    subagent_id: str
    mode: SubagentMode
    status: SubagentStatus
    summary: str = ""
    key_findings: list[str] = Field(default_factory=list)
    total_turns: int = 0
    total_tokens: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class SubagentSpec(BaseModel):
    """子代理创建规格。"""

    subagent_id: str = Field(default_factory=lambda: f"sub-{uuid4().hex[:8]}")
    task: str
    mode: SubagentMode = SubagentMode.AGENT_AS_TOOL
    tool_names: list[str] = Field(default_factory=list)
    context_summary: str = ""
    max_turns: int = 50
    max_tokens: int = 0
    timeout_seconds: float = 300.0
    system_prompt_override: str = ""


class ConflictMarker(BaseModel):
    """结果冲突标记。"""

    field: str
    values: list[str] = Field(default_factory=list)
    subagent_ids: list[str] = Field(default_factory=list)
    description: str = ""
