"""Structured provider refusals must terminate real native sessions before actions."""

from typing import Any

import pytest

from praxis.guardrails.rules import GuardrailRule, RuleTarget
from praxis.models.tools import FunctionCall, ToolCall, ToolDefinition, ToolMetadata
from praxis.session.core import Session
from tests import test_output_limits as output_support
from tests.test_output_limits import (
    ScriptedOutputGateway,
    provider_response,
    run_output,
    write_call,
)

# Reuse the actual Session fixture; only the external provider is scripted.
gateway = output_support.gateway
output_session = output_support.output_session
output_store = output_support.output_store
written_values = output_support.written_values


@pytest.mark.parametrize("streaming", [False, True])
async def test_refusal_only_stop_preserves_text_in_native_response_and_history(
    output_session: Session, gateway: ScriptedOutputGateway, streaming: bool,
) -> None:
    response = provider_response("stop", "")
    response.refusal = "I cannot perform this request."
    response.reasoning_content = "private reasoning, not an answer"
    gateway.responses = [response]
    if streaming:
        events = await run_output(output_session, True)
    else:
        result = await output_session.run_turn("request")
        assert result.termination_reason.value == "safety_refusal"
        assert result.content == "I cannot perform this request."
        events = result.events
    assert events[-1].event_type == "termination"
    assert events[-1].data["reason"] == "safety_refusal"
    assert events[-1].data["content"] == "I cannot perform this request."
    assert output_session.assembler.conversation_history[-1] == {
        "role": "assistant", "content": "I cannot perform this request.",
    }
    assert output_session.memory is not None
    assert output_session.memory.get_message_history()[-1].content == "I cannot perform this request."
    assert "".join(e.data["text"] for e in events if e.event_type == "content_delta") == (
        "I cannot perform this request." if streaming else ""
    )
    assert sum(e.event_type == "termination" for e in events) == 1
    assert gateway.calls == 1


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("finish_reason", ["stop", "tool_calls", "length", "content_filter"])
@pytest.mark.parametrize("arguments", ['{"value":"complete"}', '{"value":"partial'])
async def test_refusal_blocks_all_tool_side_effects_and_handoffs(
    output_session: Session, gateway: ScriptedOutputGateway, written_values: list[str],
    streaming: bool, finish_reason: str, arguments: str,
) -> None:
    handoffs: list[str] = []

    async def handoff(arguments: dict[str, Any]) -> str:
        handoffs.append("other_agent")
        return "transferred"

    output_session.registry.register(ToolDefinition(
        name="handoff_to_other_agent", description="Transfer to other agent",
        parameters={"type": "object", "properties": {}},
        metadata=ToolMetadata(permission_level="auto_approve"),
    ), handoff)
    response = provider_response(finish_reason, "", tool_calls=[
        write_call(arguments), write_call('{"value":"second"}', "write-2"),
        ToolCall(id="handoff-1", function=FunctionCall(
            name="handoff_to_other_agent", arguments="{}",
        )),
    ])
    response.refusal = "I cannot perform this request."
    gateway.responses = [response, provider_response("stop", "unexpected continuation")]
    events = await run_output(output_session, streaming)
    assert written_values == []
    assert handoffs == []
    assert output_session.tool_execution_ledger == {}
    assert output_session.loop.state.total_tool_calls == 0
    assert events[-1].data["reason"] == "safety_refusal"
    assert events[-1].data["content"] == "I cannot perform this request."
    assert not any("tool_calls" in message for message in output_session.assembler.conversation_history)
    assert gateway.calls == 1


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("finish_reason", ["stop", "length", "content_filter"])
async def test_user_abort_has_priority_over_structured_refusal(
    output_session: Session, gateway: ScriptedOutputGateway, written_values: list[str],
    streaming: bool, finish_reason: str,
) -> None:
    response = provider_response(finish_reason, "", [write_call('{"value":"x"}')])
    response.refusal = "Refused."
    gateway.responses = [response]
    gateway.on_response = output_session.abort
    events = await run_output(output_session, streaming)
    assert events[-1].data["reason"] == "user_abort"
    assert written_values == []
    assert gateway.calls == 1


@pytest.mark.parametrize("streaming", [False, True])
async def test_input_guardrail_prevents_refusal_provider_request(
    output_session: Session, gateway: ScriptedOutputGateway, streaming: bool,
) -> None:
    output_session.loop.guardrails.register_rule(GuardrailRule(
        name="block_input", description="Reject request", target=RuleTarget.INPUT,
        patterns=("request",), tripwire=True,
    ))
    response = provider_response("stop", "")
    response.refusal = "Refused."
    gateway.responses = [response]
    events = await run_output(output_session, streaming)
    assert events[-1].data["reason"] == "tripwire"
    assert gateway.calls == 0


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("refusal", [None, ""])
async def test_empty_refusal_does_not_block_native_multi_tool_loop(
    output_session: Session, gateway: ScriptedOutputGateway, written_values: list[str],
    streaming: bool, refusal: str | None,
) -> None:
    response = provider_response("tool_calls", "", tool_calls=[
        write_call('{"value":"first"}'), write_call('{"value":"second"}', "write-2"),
    ])
    response.refusal = refusal
    final = provider_response("stop", "I cannot do something else, but this is done.")
    final.refusal = refusal
    gateway.responses = [response, final]
    events = await run_output(output_session, streaming)
    assert written_values == ["first", "second"]
    assert events[-1].data["reason"] == "natural"
    assert events[-1].data["content"] == "I cannot do something else, but this is done."
    assert gateway.calls == 2


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(("content", "refusal", "expected"), [
    ("Some context.", "Refused.", "Some context.\nRefused."),
    ("Refused.", "Refused.", "Refused."),
])
async def test_mixed_refusal_projects_each_field_once_without_reasoning(
    output_session: Session, gateway: ScriptedOutputGateway, streaming: bool,
    content: str, refusal: str, expected: str,
) -> None:
    response = provider_response("stop", content)
    response.refusal = refusal
    response.reasoning_content = "private reasoning"
    gateway.responses = [response]
    events = await run_output(output_session, streaming)
    assert events[-1].data["reason"] == "safety_refusal"
    assert events[-1].data["content"] == expected
    assert output_session.assembler.conversation_history[-1]["content"] == expected
    assert "".join(e.data["text"] for e in events if e.event_type == "content_delta") == (
        expected if streaming else ""
    )
    assert gateway.calls == 1


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("blocked_field", ["content", "refusal"])
async def test_refusal_does_not_bypass_output_guardrails(
    output_session: Session, gateway: ScriptedOutputGateway, streaming: bool,
    blocked_field: str,
) -> None:
    output_session.loop.guardrails.register_rule(GuardrailRule(
        name="block_output", description="Reject output", target=RuleTarget.OUTPUT,
        patterns=("blocked output",), tripwire=True,
    ))
    response = provider_response("stop", "blocked output" if blocked_field == "content" else "")
    response.refusal = "blocked output" if blocked_field == "refusal" else "Refused."
    gateway.responses = [response]
    events = await run_output(output_session, streaming)
    assert events[-1].data["reason"] == "tripwire"
    assert "blocked output" not in str(output_session.assembler.conversation_history)
    assert gateway.calls == 1
