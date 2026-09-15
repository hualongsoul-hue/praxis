"""Provider output exhaustion through native complete and streaming sessions."""

from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest

from praxis.config.schemas import (
    ContextConfig,
    GatewayConfig,
    MemoryConfig,
    ModelCapabilities,
    OrchestratorConfig,
    PersistenceConfig,
    SessionConfig,
)
from praxis.guardrails.engine import GuardrailEngine
from praxis.guardrails.permissions import PermissionManager
from praxis.guardrails.rules import GuardrailRule, RuleEngine, RuleTarget
from praxis.models.orchestrator import AgentEvent, TerminationReason
from praxis.models.responses import (
    FunctionCallDelta,
    ModelResponse,
    ModelResponseChunk,
    ToolCallDelta,
    Usage,
)
from praxis.models.tools import FunctionCall, ToolCall, ToolDefinition, ToolMetadata
from praxis.persistence.store import PersistenceStore, create_store
from praxis.session.core import Session, SessionFactory
from praxis.tools.registry import ToolRegistry


class ScriptedOutputGateway:
    """Deterministic provider boundary; all session and tool components remain real."""

    def __init__(self) -> None:
        self.config = GatewayConfig()
        self.responses: list[ModelResponse] = []
        self.calls = 0
        self.on_response: Callable[[], None] | None = None

    async def complete(
        self, messages: list[dict[str, Any]], *, model: str | None = None,
        tools: list[dict[str, Any]] | None = None, **kwargs: Any,
    ) -> ModelResponse:
        self.calls += 1
        assert self.responses, "unexpected extra model request"
        if self.on_response is not None:
            self.on_response()
        return self.responses.pop(0)

    async def stream(
        self, messages: list[dict[str, Any]], *, model: str | None = None,
        tools: list[dict[str, Any]] | None = None, **kwargs: Any,
    ) -> AsyncIterator[ModelResponseChunk]:
        response = await self.complete(messages, model=model, tools=tools, **kwargs)
        # Keep content/tools and the terminal reason in different chunks, as a
        # provider can announce truncation only after sending partial output.
        yield ModelResponseChunk(
            id=response.id, delta_content=response.content, model=response.model,
            delta_tool_calls=[ToolCallDelta(
                index=index, id=call.id,
                function=FunctionCallDelta(
                    name=call.function.name, arguments=call.function.arguments,
                ),
            ) for index, call in enumerate(response.tool_calls or [])],
        )
        yield ModelResponseChunk(
            id=response.id, finish_reason=response.finish_reason, usage=response.usage,
        )

    def capabilities(self, model_name: str | None = None) -> ModelCapabilities:
        return ModelCapabilities()

    async def health(self) -> bool:
        return True

    async def close(self) -> None:
        pass


def provider_response(
    finish_reason: str, content: str = "partial answer",
    tool_calls: list[ToolCall] | None = None,
) -> ModelResponse:
    return ModelResponse(
        id="output-response", content=content, finish_reason=finish_reason,
        tool_calls=tool_calls, model="default", created=0,
        usage=Usage(prompt_tokens=2, completion_tokens=1, total_tokens=3),
    )


def write_call(arguments: str, call_id: str = "write-1") -> ToolCall:
    return ToolCall(
        id=call_id, function=FunctionCall(name="write_value", arguments=arguments),
    )


@pytest.fixture
async def output_store(tmp_path: Path) -> AsyncIterator[PersistenceStore]:
    store = await create_store(PersistenceConfig(sqlite_path=str(tmp_path / "output.db")))
    try:
        yield store
    finally:
        await store.close()


@pytest.fixture
def gateway() -> ScriptedOutputGateway:
    return ScriptedOutputGateway()


@pytest.fixture
def written_values() -> list[str]:
    return []


@pytest.fixture
async def output_session(
    output_store: PersistenceStore, gateway: ScriptedOutputGateway, written_values: list[str],
) -> AsyncIterator[Session]:
    async def write_value(arguments: dict[str, Any]) -> str:
        written_values.append(arguments["value"])
        return "written"

    registry = ToolRegistry()
    registry.register(ToolDefinition(
        name="write_value", description="Write one value",
        parameters={
            "type": "object", "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        metadata=ToolMetadata(permission_level="auto_approve"),
    ), write_value)
    session = await SessionFactory(
        store=output_store, session_config=SessionConfig(),
        orchestrator_config=OrchestratorConfig(max_turns=5), context_config=ContextConfig(),
        memory_config=MemoryConfig(background_enabled=False, dream_enabled=False),
    ).create_session(
        GuardrailEngine(RuleEngine(), PermissionManager()), gateway, registry=registry,
    )
    try:
        yield session
    finally:
        await session.terminate()


async def run_output(session: Session, streaming: bool) -> list[AgentEvent]:
    if streaming:
        return [event async for event in session.run_turn_stream("request")]
    return (await session.run_turn("request")).events


@pytest.mark.parametrize("streaming", [False, True])
async def test_provider_length_preserves_partial_text_and_marks_token_exhaustion(
    output_session: Session, gateway: ScriptedOutputGateway, streaming: bool,
) -> None:
    gateway.responses = [provider_response("length")]
    events = await run_output(output_session, streaming)
    assert output_session.loop.assembler.get_token_usage().usage_ratio < 0.95
    assert events[-1].data["reason"] == TerminationReason.TOKEN_EXHAUSTED.value
    assert events[-1].data["content"] == "partial answer"
    assert output_session.assembler.conversation_history[-1]["content"] == "partial answer"
    assert gateway.calls == 1


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("arguments", ['{"value":"partial', '{"value":"complete"}'])
async def test_provider_length_never_executes_partial_or_complete_tools(
    output_session: Session, gateway: ScriptedOutputGateway, written_values: list[str],
    streaming: bool, arguments: str,
) -> None:
    gateway.responses = [
        provider_response("length", tool_calls=[write_call(arguments)]),
        provider_response("stop", "unexpected continuation"),
    ]
    events = await run_output(output_session, streaming)
    assert written_values == []
    assert output_session.tool_execution_ledger == {}
    assert events[-1].data["reason"] == "token_exhausted"
    assert gateway.calls == 1


@pytest.mark.parametrize("streaming", [False, True])
async def test_provider_stop_still_completes_naturally(
    output_session: Session, gateway: ScriptedOutputGateway, streaming: bool,
) -> None:
    gateway.responses = [provider_response("stop", "complete answer")]
    events = await run_output(output_session, streaming)
    assert events[-1].data["reason"] == "natural"
    assert events[-1].data["content"] == "complete answer"
    assert gateway.calls == 1


@pytest.mark.parametrize("streaming", [False, True])
async def test_standard_tool_calls_still_execute_and_continue_native_loop(
    output_session: Session, gateway: ScriptedOutputGateway, written_values: list[str],
    streaming: bool,
) -> None:
    gateway.responses = [
        provider_response("tool_calls", tool_calls=[
            write_call('{"value":"first"}'), write_call('{"value":"second"}', "write-2"),
        ]),
        provider_response("stop", "done"),
    ]
    events = await run_output(output_session, streaming)
    assert written_values == ["first", "second"]
    assert events[-1].data["reason"] == "natural"
    assert events[-1].data["content"] == "done"
    assert gateway.calls == 2


@pytest.mark.parametrize("streaming", [False, True])
async def test_user_abort_has_priority_over_provider_length(
    output_session: Session, gateway: ScriptedOutputGateway, written_values: list[str],
    streaming: bool,
) -> None:
    gateway.responses = [provider_response("length", tool_calls=[write_call('{"value":"x"}')])]
    gateway.on_response = output_session.abort
    events = await run_output(output_session, streaming)
    assert events[-1].data["reason"] == "user_abort"
    assert written_values == []


@pytest.mark.parametrize("streaming", [False, True])
async def test_content_filter_still_refuses_without_tools(
    output_session: Session, gateway: ScriptedOutputGateway, written_values: list[str],
    streaming: bool,
) -> None:
    gateway.responses = [
        provider_response("content_filter", tool_calls=[write_call('{"value":"x"}')]),
    ]
    events = await run_output(output_session, streaming)
    assert events[-1].data["reason"] == "safety_refusal"
    assert written_values == []


@pytest.mark.parametrize("streaming", [False, True])
async def test_provider_length_does_not_bypass_output_guardrail(
    output_session: Session, gateway: ScriptedOutputGateway, streaming: bool,
) -> None:
    output_session.loop.guardrails.register_rule(GuardrailRule(
        name="block_output", description="Reject test content", target=RuleTarget.OUTPUT,
        patterns=("blocked output",), tripwire=True,
    ))
    gateway.responses = [provider_response("length", "blocked output")]
    events = await run_output(output_session, streaming)
    assert events[-1].data["reason"] == "tripwire"
    assert all(m.get("content") != "blocked output" for m in output_session.assembler.conversation_history)


@pytest.mark.parametrize("streaming", [False, True])
async def test_input_tripwire_prevents_any_provider_request(
    output_session: Session, gateway: ScriptedOutputGateway, streaming: bool,
) -> None:
    output_session.loop.guardrails.register_rule(GuardrailRule(
        name="block_input", description="Reject request", target=RuleTarget.INPUT,
        patterns=("request",), tripwire=True,
    ))
    gateway.responses = [provider_response("length")]
    events = await run_output(output_session, streaming)
    assert events[-1].data["reason"] == "tripwire"
    assert gateway.calls == 0
