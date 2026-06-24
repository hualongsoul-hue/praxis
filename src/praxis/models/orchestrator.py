"""编排循环数据模型——S11 跨组件共享类型。"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class LoopPhase(str, Enum):
    """循环阶段。"""

    IDLE = "idle"
    PLANNING = "planning"
    ASSEMBLING = "assembling"
    LLM_CALLING = "llm_calling"
    PARSING = "parsing"
    TOOL_EXECUTING = "tool_executing"
    VERIFYING = "verifying"
    TERMINATING = "terminating"


class TerminationReason(str, Enum):
    """终止原因。"""

    NATURAL = "natural"
    TRIPWIRE = "tripwire"
    USER_ABORT = "user_abort"
    SAFETY_REFUSAL = "safety_refusal"
    MAX_TURNS = "max_turns"
    TOKEN_EXHAUSTED = "token_exhausted"
    HANDOFF = "handoff"


class StrategyMode(str, Enum):
    """循环策略模式。"""

    REACT = "react"
    PLAN_AND_EXECUTE = "plan-and-execute"


class AgentEvent(BaseModel):
    """编排循环事件。"""

    event_type: str
    turn: int = 0
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = 0.0


class AgentResponse(BaseModel):
    """Agent 最终响应。"""

    content: str = ""
    tool_calls_made: int = 0
    total_turns: int = 0
    termination_reason: TerminationReason = TerminationReason.NATURAL
    events: list[AgentEvent] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LoopState(BaseModel):
    """循环运行状态。"""

    phase: LoopPhase = LoopPhase.IDLE
    current_turn: int = 0
    total_tool_calls: int = 0
    aborted: bool = False
    termination_reason: TerminationReason | None = None
