"""Native tool execution and provider/checkpoint history must agree on parsed values."""

import json
from copy import deepcopy
from typing import Any

import pytest

from praxis.config.schemas import ContextConfig, MemoryConfig, OrchestratorConfig, SessionConfig
from praxis.guardrails.permissions import PermissionPolicy, PermissionRule
from praxis.models.guardrails import VerdictType
from praxis.models.responses import ModelResponse
from praxis.models.tools import FunctionCall, ToolCall, ToolDefinition, ToolMetadata
from praxis.persistence.store import PersistenceStore
from praxis.session.checkpoint import CheckpointManager
from praxis.session.core import Session, SessionFactory
from praxis.session.resume import SessionResumer
from tests import test_output_limits as support
from tests.test_output_limits import (
    ScriptedOutputGateway,
    provider_response,
    run_output,
    write_call,
)

output_session = support.output_session
output_store = support.output_store
written_values = support.written_values


def reject_nonfinite_constant(value: str) -> None:
    raise ValueError("Non-finite JSON constant")


class StrictHistoryGateway(ScriptedOutputGateway):
    """Only the external model is scripted; reject invalid protocol JSON like a provider."""

    def __init__(self) -> None:
        super().__init__()
        self.requests: list[list[dict[str, Any]]] = []

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        self.requests.append(deepcopy(messages))
        for message in messages:
            for call in message.get("tool_calls", []):
                arguments = json.loads(
                    call["function"]["arguments"],
                    parse_constant=reject_nonfinite_constant,
                )
                assert isinstance(arguments, dict), "tool arguments must be a JSON object"
        return await super().complete(messages, model=model, tools=tools, **kwargs)


@pytest.fixture
def gateway() -> StrictHistoryGateway:
    return StrictHistoryGateway()


@pytest.mark.parametrize("streaming", [False, True])
async def test_repaired_arguments_reach_next_native_request_as_executed(
    output_session: Session,
    gateway: StrictHistoryGateway,
    written_values: list[str],
    streaming: bool,
) -> None:
    gateway.responses = [
        provider_response("tool_calls", "", [write_call('{"value": solar_return}')]),
        provider_response("stop", "finished"),
    ]
    try:
        events = await run_output(output_session, streaming)
    finally:
        assert written_values == ["solar_return"]
    assert events[-1].data["reason"] == "natural"
    assert events[-1].data["content"] == "finished"
    assert gateway.calls == 2 and len(gateway.requests) == 2
    assistant = next(message for message in gateway.requests[1] if message.get("tool_calls"))
    call = assistant["tool_calls"][0]
    assert call["id"] == "write-1" and call["function"]["name"] == "write_value"
    assert json.loads(call["function"]["arguments"]) == {"value": "solar_return"}


@pytest.mark.parametrize("streaming", [False, True])
async def test_multiple_calls_preserve_json_types_unicode_identity_and_order(
    output_session: Session,
    gateway: StrictHistoryGateway,
    streaming: bool,
) -> None:
    executed: list[dict[str, Any]] = []

    async def capture(arguments: dict[str, Any]) -> str:
        executed.append(deepcopy(arguments))
        return "captured"

    output_session.registry.register(
        ToolDefinition(
            name="capture",
            description="Capture structured values",
            parameters={"type": "object", "properties": {}},
            metadata=ToolMetadata(permission_level="auto_approve"),
        ),
        capture,
    )
    raw = [
        '{"value":solar_return,"text":"星盘\\n第二行","items":[1,true,null,{"x":2.5}]}',
        '{"value":"保持原值","active":false,"count":0}',
    ]
    expected = [
        {"value": "solar_return", "text": "星盘\n第二行", "items": [1, True, None, {"x": 2.5}]},
        {"value": "保持原值", "active": False, "count": 0},
    ]
    gateway.responses = [
        provider_response(
            "tool_calls",
            "",
            [
                ToolCall(id=f"call-{index}", function=FunctionCall(name="capture", arguments=value))
                for index, value in enumerate(raw)
            ],
        ),
        provider_response("stop", "finished"),
    ]
    events = await run_output(output_session, streaming)
    assert executed == expected
    calls = next(
        message["tool_calls"] for message in gateway.requests[1] if message.get("tool_calls")
    )
    assert [call["id"] for call in calls] == ["call-0", "call-1"]
    assert [call["function"]["name"] for call in calls] == ["capture", "capture"]
    values = [json.loads(call["function"]["arguments"]) for call in calls]
    assert values == executed
    assert type(values[0]["items"][0]) is int and values[0]["items"][1] is True
    assert values[0]["items"][2] is None and type(values[0]["items"][3]["x"]) is float
    assert values[1]["active"] is False and type(values[1]["count"]) is int
    assert events[-1].data["reason"] == "natural" and gateway.calls == 2


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("raw", ["", "???", "[]", "42", "null"])
async def test_nonobject_arguments_keep_existing_required_field_error(
    output_session: Session,
    gateway: StrictHistoryGateway,
    written_values: list[str],
    streaming: bool,
    raw: str,
) -> None:
    gateway.responses = [
        provider_response("tool_calls", "", [write_call(raw)]),
        provider_response("stop", "required value unavailable"),
    ]
    events = await run_output(output_session, streaming)
    assert written_values == []
    assert gateway.calls == 2
    result = next(message for message in gateway.requests[1] if message["role"] == "tool")
    assert "参数校验失败" in result["content"] and "required" in result["content"]
    call = next(
        message["tool_calls"][0] for message in gateway.requests[1] if message.get("tool_calls")
    )
    assert json.loads(call["function"]["arguments"]) == {}
    assert events[-1].data["reason"] == "natural"


@pytest.mark.parametrize("streaming", [False, True])
async def test_repaired_arguments_do_not_bypass_permission_denial(
    output_session: Session,
    gateway: StrictHistoryGateway,
    written_values: list[str],
    streaming: bool,
) -> None:
    output_session.loop.guardrails.permission_manager.policy = PermissionPolicy(
        rules=(PermissionRule(tool_name="write_value", permission=VerdictType.DENY),)
    )
    gateway.responses = [
        provider_response("tool_calls", "", [write_call('{"value": solar_return}')]),
        provider_response("stop", "denied"),
    ]
    events = await run_output(output_session, streaming)
    assert written_values == [] and output_session.tool_execution_ledger == {}
    result = next(message for message in gateway.requests[1] if message["role"] == "tool")
    assert "护栏拒绝" in result["content"] and result["tool_call_id"] == "write-1"
    assert events[-1].data["reason"] == "natural" and gateway.calls == 2


@pytest.mark.parametrize("streaming", [False, True])
async def test_checkpoint_resume_reuses_repaired_arguments_and_execution_ledger(
    output_session: Session,
    output_store: PersistenceStore,
    gateway: StrictHistoryGateway,
    written_values: list[str],
    streaming: bool,
) -> None:
    gateway.responses = [
        provider_response("tool_calls", "", [write_call('{"value": solar_return}')]),
        provider_response("stop", "finished"),
    ]
    await run_output(output_session, streaming)
    checkpoints = CheckpointManager(output_store)
    checkpoint = await checkpoints.load_latest(output_session.session_id)
    assert checkpoint is not None
    snapshot = await checkpoints.extract_snapshot(checkpoint)
    history = snapshot.context_state["messages"]
    call = next(message["tool_calls"][0] for message in history if message.get("tool_calls"))
    assert json.loads(call["function"]["arguments"]) == {"value": "solar_return"}
    await output_session.terminate()

    restored_gateway = StrictHistoryGateway()
    restored_gateway.responses = [
        provider_response("tool_calls", "", [write_call('{ "value" : "solar_return" }')]),
        provider_response("stop", "restored without another write"),
    ]
    factory = SessionFactory(
        store=output_store,
        session_config=SessionConfig(),
        orchestrator_config=OrchestratorConfig(max_turns=5),
        context_config=ContextConfig(),
        memory_config=MemoryConfig(background_enabled=False, dream_enabled=False),
    )
    restored = await SessionResumer(factory, checkpoints).resume_session(
        output_session.session_id,
        output_session.loop.guardrails,
        restored_gateway,
        registry=output_session.registry,
    )
    assert restored is not None
    try:
        events = await run_output(restored, streaming)
        assert events[-1].data["reason"] == "natural"
        assert written_values == ["solar_return"]
        assert restored_gateway.calls == 2
        assert len(restored.tool_execution_ledger) == 1
    finally:
        await restored.terminate()


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", "1e999"])
async def test_nonfinite_parsed_numbers_are_rejected_before_tool_side_effects(
    output_session: Session,
    gateway: StrictHistoryGateway,
    streaming: bool,
    value: str,
) -> None:
    executed: list[dict[str, Any]] = []

    async def capture_number(arguments: dict[str, Any]) -> str:
        executed.append(arguments)
        return "written"

    output_session.registry.register(
        ToolDefinition(
            name="capture_number",
            description="Write a numeric value",
            parameters={"type": "object", "properties": {"value": {"type": "number"}}},
            metadata=ToolMetadata(permission_level="auto_approve"),
        ),
        capture_number,
    )
    gateway.responses = [
        provider_response(
            "tool_calls",
            "",
            [
                ToolCall(
                    id="numeric-1",
                    function=FunctionCall(
                        name="capture_number", arguments='{"value":' + value + "}"
                    ),
                )
            ],
        )
    ]
    with pytest.raises(ValueError):
        await run_output(output_session, streaming)
    assert executed == [], "Invalid JSON numbers must fail before any tool side effect"
    assert output_session.tool_execution_ledger == {}
    assert gateway.calls == 1 and len(gateway.requests) == 1
    assert not any(
        message.get("tool_calls") for message in output_session.assembler.conversation_history
    )
