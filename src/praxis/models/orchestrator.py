"""Strict public orchestration state, event, and response models."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal, cast
from uuid import uuid4

from pydantic import ConfigDict, Field, field_validator, model_validator

from praxis.models.base import SafeBaseModel


class LoopPhase(StrEnum):
    IDLE = "idle"
    PLANNING = "planning"
    ASSEMBLING = "assembling"
    LLM_CALLING = "llm_calling"
    PARSING = "parsing"
    TOOL_EXECUTING = "tool_executing"
    VERIFYING = "verifying"
    TERMINATING = "terminating"


class TerminationReason(StrEnum):
    NATURAL = "natural"
    TRIPWIRE = "tripwire"
    USER_ABORT = "user_abort"
    SAFETY_REFUSAL = "safety_refusal"
    MAX_TURNS = "max_turns"
    TOKEN_EXHAUSTED = "token_exhausted"
    HANDOFF = "handoff"


class StrategyMode(StrEnum):
    REACT = "react"
    PLAN_AND_EXECUTE = "plan-and-execute"


class EventType(StrEnum):
    PLAN_CREATED = "plan_created"
    TURN_START = "turn_start"
    LLM_REQUEST = "llm_request"
    CONTENT_DELTA = "content_delta"
    REASONING_DELTA = "reasoning_delta"
    LLM_RESPONSE = "llm_response"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_END = "tool_call_end"
    TOOL_RETRY = "tool_retry"
    VERIFICATION_RESULT = "verification_result"
    GAV_FEEDBACK = "gav_feedback"
    TURN_END = "turn_end"
    TERMINATION = "termination"


class EventPayload(SafeBaseModel):
    """Frozen mapping-like base for every event-specific payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    def __getitem__(self, key: str) -> Any:
        if key not in type(self).model_fields:
            raise KeyError(key)
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return self[key] if key in type(self).model_fields else default

    def items(self) -> list[tuple[str, Any]]:
        return list(self.model_dump(mode="python").items())


class EmptyEventPayload(EventPayload):
    pass


class PlanCreatedPayload(EventPayload):
    request_id: str = ""
    step_count: int = Field(default=0, ge=0)


class ModelRequestPayload(EventPayload):
    token_count: int = Field(default=0, ge=0)


class TextDeltaPayload(EventPayload):
    text: str = Field(max_length=1_000_000)

    @field_validator("text")
    @classmethod
    def redact_text(cls, value: str) -> str:
        from praxis.telemetry.redaction import DEFAULT_REDACTION_POLICY

        return DEFAULT_REDACTION_POLICY.redact_text(value, max_length=1_000_000)


class ModelResponsePayload(EventPayload):
    has_content: bool = False
    has_reasoning: bool = False
    has_refusal: bool = False
    tool_call_count: int = Field(default=0, ge=0)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    cached_prompt_tokens: int = Field(default=0, ge=0)


class ToolCallStartPayload(EventPayload):
    tool_name: str = "unknown"
    tool_call_id: str = ""
    argument_keys: tuple[str, ...] = ()
    arguments_summary: str = Field(default="{}", max_length=4096)

    @field_validator("arguments_summary")
    @classmethod
    def redact_arguments_summary(cls, value: str) -> str:
        from praxis.telemetry.redaction import DEFAULT_REDACTION_POLICY

        return DEFAULT_REDACTION_POLICY.redact_text(value, max_length=4096)


class ToolCallEndPayload(EventPayload):
    tool_name: str = "unknown"
    tool_call_id: str = ""
    success: bool | None = None
    skipped: bool = False
    tripwire: bool = False
    reason: str = Field(default="", max_length=4096)
    content_summary: str = Field(default="", max_length=4096)
    error_type: str = Field(default="", max_length=512)
    execution_time_ms: float | None = Field(default=None, ge=0)

    @field_validator("reason", "content_summary")
    @classmethod
    def redact_result_text(cls, value: str) -> str:
        from praxis.telemetry.redaction import DEFAULT_REDACTION_POLICY

        return DEFAULT_REDACTION_POLICY.redact_text(value, max_length=4096)


class ToolRetryPayload(EventPayload):
    tool_name: str = "unknown"
    tool_call_id: str = ""
    attempt: int = Field(default=0, ge=0)
    delay_seconds: float = Field(default=0.0, ge=0)
    error_type: str = Field(default="", max_length=512)


class VerificationResultPayload(EventPayload):
    verifier: str = ""
    status: str = ""
    passed: bool | None = None


class GavFeedbackPayload(EventPayload):
    passed: bool = False
    retry_hint: str = Field(default="", max_length=4096)
    feedback: str = Field(default="", max_length=4096)

    @field_validator("retry_hint", "feedback")
    @classmethod
    def redact_feedback(cls, value: str) -> str:
        from praxis.telemetry.redaction import DEFAULT_REDACTION_POLICY

        return DEFAULT_REDACTION_POLICY.redact_text(value, max_length=4096)


class TerminationPayload(EventPayload):
    reason: TerminationReason = TerminationReason.NATURAL
    content: str = Field(default="", max_length=1_000_000)

    @field_validator("content")
    @classmethod
    def redact_content(cls, value: str) -> str:
        from praxis.telemetry.redaction import DEFAULT_REDACTION_POLICY

        return DEFAULT_REDACTION_POLICY.redact_text(value, max_length=1_000_000)


AgentEventPayload = (
    EmptyEventPayload
    | PlanCreatedPayload
    | ModelRequestPayload
    | TextDeltaPayload
    | ModelResponsePayload
    | ToolCallStartPayload
    | ToolCallEndPayload
    | ToolRetryPayload
    | VerificationResultPayload
    | GavFeedbackPayload
    | TerminationPayload
)

EVENT_PAYLOAD_MODELS: dict[EventType, type[EventPayload]] = {
    EventType.PLAN_CREATED: PlanCreatedPayload,
    EventType.TURN_START: EmptyEventPayload,
    EventType.LLM_REQUEST: ModelRequestPayload,
    EventType.CONTENT_DELTA: TextDeltaPayload,
    EventType.REASONING_DELTA: TextDeltaPayload,
    EventType.LLM_RESPONSE: ModelResponsePayload,
    EventType.TOOL_CALL_START: ToolCallStartPayload,
    EventType.TOOL_CALL_END: ToolCallEndPayload,
    EventType.TOOL_RETRY: ToolRetryPayload,
    EventType.VERIFICATION_RESULT: VerificationResultPayload,
    EventType.GAV_FEEDBACK: GavFeedbackPayload,
    EventType.TURN_END: EmptyEventPayload,
    EventType.TERMINATION: TerminationPayload,
}


class AgentEvent(SafeBaseModel):
    """Versioned, correlated, ordered event with an event-specific payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    event_id: str = Field(default_factory=lambda: uuid4().hex)
    event_type: EventType
    runtime_id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str = Field(default_factory=lambda: uuid4().hex[:16])
    run_id: str = Field(default_factory=lambda: uuid4().hex)
    sequence: int = Field(default=0, ge=0)
    turn: int = Field(default=0, ge=0)
    data: AgentEventPayload
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="before")
    @classmethod
    def validate_payload_for_event(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        raw = cast(dict[str, Any], value)
        event_type = EventType(raw.get("event_type"))
        payload_type = EVENT_PAYLOAD_MODELS[event_type]
        payload = raw.get("data", {})
        if not isinstance(payload, payload_type):
            payload = payload_type.model_validate(payload)
        return {**raw, "event_type": event_type, "data": payload}


class AgentResponse(SafeBaseModel):
    content: str = ""
    tool_calls_made: int = 0
    total_turns: int = 0
    termination_reason: TerminationReason = TerminationReason.NATURAL
    events: list[AgentEvent] = Field(default_factory=lambda: list[AgentEvent]())
    metadata: dict[str, Any] = Field(default_factory=dict)


class LoopState(SafeBaseModel):
    phase: LoopPhase = LoopPhase.IDLE
    current_turn: int = 0
    total_tool_calls: int = 0
    aborted: bool = False
    termination_reason: TerminationReason | None = None
