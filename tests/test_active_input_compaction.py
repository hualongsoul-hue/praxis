"""Native compaction must preserve the current input, not equal old messages."""

import asyncio
from copy import deepcopy

import pytest

from praxis.config.schemas import ContextConfig, MemoryConfig, SkillsConfig, VerificationConfig
from praxis.context.compaction import ContextCompactor
from praxis.gateway.tasks import SUMMARIZE_SYSTEM_PROMPT
from praxis.models.responses import FunctionCallDelta, ModelResponseChunk, ToolCallDelta
from praxis.models.tools import ToolDefinition, ToolMetadata
from praxis.runtime import PraxisRuntime
from praxis.session.checkpoint import CheckpointManager
from praxis.session.core import Session, SessionFactory
from praxis.session.resume import SessionResumer
from tests.test_output_limits import provider_response, write_call
from tests.test_runtime import FakeGateway, runtime_config

TEXT = "日常记录天气和阅读进度。" * 500 + "最后仅谈一般休息，不推测任何身体症状。"


class CompactionGateway(FakeGateway):
    def __init__(self):
        super().__init__()
        self.requests = []
        self.summaries = []
        self.active = None
        self.native = None

    async def complete(self, messages, **kwargs):
        if messages[0]["content"] == SUMMARIZE_SYSTEM_PROMPT:
            self.summaries.append(deepcopy(messages))
            return provider_response("stop", "旧对话摘要")
        assert self.native is not None
        history = self.native.assembler.conversation_history
        if not self.requests:
            self.active = next(
                message for message in reversed(history) if message["role"] == "user"
            )
        elif len(self.requests) == 1:
            assert sum(message is self.active for message in history) == 1
            assert self.active["content"] == TEXT
        self.requests.append(deepcopy(messages))
        if len(self.requests) == 1:
            return provider_response("tool_calls", "", [write_call('{"value":"one"}')])
        return provider_response("stop", "完成")

    async def stream(self, messages, **kwargs):
        response = await self.complete(messages, **kwargs)
        yield ModelResponseChunk(
            id=response.id,
            delta_content=response.content,
            delta_tool_calls=[
                ToolCallDelta(
                    index=index,
                    id=call.id,
                    function=FunctionCallDelta(
                        name=call.function.name, arguments=call.function.arguments
                    ),
                )
                for index, call in enumerate(response.tool_calls or [])
            ],
        )
        yield ModelResponseChunk(
            id=response.id,
            finish_reason=response.finish_reason,
            usage=response.usage,
        )


@pytest.mark.parametrize("streaming", [False, True])
async def test_runtime_keeps_active_tail_once_and_restores_next_turn(tmp_path, streaming):
    context = ContextConfig(compaction_threshold=0.000001, compaction_min_history=1)
    config = runtime_config(tmp_path).model_copy(
        update={
            "context": context,
            "memory": MemoryConfig(
                background_enabled=False, dream_enabled=False, load_project_praxis_md=False
            ),
            "skills": SkillsConfig(auto_discover=False, skill_paths=()),
            "verification": VerificationConfig(inferential_enabled=False),
        }
    )
    gateway = CompactionGateway()
    written = []

    async def write_value(arguments):
        written.append(arguments["value"])
        return "one"

    async with PraxisRuntime(config, gateway=gateway) as runtime:
        async with runtime.session() as session:
            native = session.require_runner()
            assert isinstance(native, Session)
            gateway.native = native
            native.registry.register(
                ToolDefinition(
                    name="write_value",
                    description="Record a synthetic value",
                    parameters={
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                    },
                    metadata=ToolMetadata(permission_level="auto_approve"),
                ),
                write_value,
            )
            old = {"role": "user", "content": TEXT}
            native.assembler.conversation_history.extend(
                [old, {"role": "assistant", "content": "旧回复"}]
            )
            if streaming:
                events = [event async for event in session.run_stream(TEXT)]
                assert events[-1].data["reason"] == "natural"
            else:
                assert (await session.run(TEXT)).content == "完成"
            assert written == ["one"]
            assert not any(message is old for message in native.assembler.conversation_history)
            assert (
                sum(message is gateway.active for message in native.assembler.conversation_history)
                == 1
            )
            assert sum(message.get("content") == TEXT for message in gateway.requests[1]) == 1
            assert len(gateway.summaries) == 2
            assert TEXT in gateway.summaries[0][1]["content"]
            assert TEXT not in gateway.summaries[1][1]["content"]
            assert runtime.store is not None and runtime.guardrails is not None
            checkpoint = await CheckpointManager(runtime.store).load_latest(native.session_id)
            assert checkpoint is not None and TEXT in str(checkpoint.state)
            restored = await SessionResumer(
                SessionFactory(
                    runtime.store,
                    config.session,
                    config.orchestrator,
                    context_config=context,
                    memory_config=config.memory,
                ),
                CheckpointManager(runtime.store),
            ).resume_session(
                native.session_id, runtime.guardrails, gateway, registry=native.registry
            )
            assert restored is not None
            try:
                gateway.native = restored
                assert (await restored.run_turn("只继续刚才最后的限制。")).content == "完成"
                assert written == ["one"]
            finally:
                await restored.terminate()


@pytest.mark.parametrize("structured", [False, True])
async def test_compactor_identity_not_equality_and_complete_group_order(structured):
    gateway = CompactionGateway()
    active = {"role": "user", "content": [{"type": "text", "text": TEXT}] if structured else TEXT}
    old = {"role": "user", "content": TEXT}
    call = {
        "role": "assistant",
        "content": "architecture",
        "tool_calls": [
            write_call('{"value":"one"}', "a").model_dump(),
            write_call('{"value":"two"}', "b").model_dump(),
        ],
    }
    result_a = {"role": "tool", "tool_call_id": "a", "content": "one"}
    result_b = {"role": "tool", "tool_call_id": "b", "content": "two"}
    messages = [old, call, result_a, result_b, active]
    await ContextCompactor(ContextConfig(), gateway).compact(messages, [], protected_message=active)
    assert messages[0]["role"] == "system"
    assert all(
        actual is expected
        for actual, expected in zip(messages[1:], [call, result_a, result_b, active], strict=True)
    )
    assert all(message is not old for message in messages)
    assert len(gateway.summaries) == 1


async def test_missing_protected_identity_fails_before_mutation_or_model():
    gateway = CompactionGateway()
    old = {"role": "user", "content": TEXT}
    messages = [old]
    with pytest.raises(ValueError, match="protected message is not in context history"):
        await ContextCompactor(ContextConfig(), gateway).compact(
            messages, [], protected_message=dict(old)
        )
    assert len(messages) == 1 and messages[0] is old
    assert gateway.summaries == []


@pytest.mark.parametrize("outcome", ["empty", "error", "cancel"])
async def test_failed_summary_keeps_original_active_and_history(outcome):
    class FailingGateway(FakeGateway):
        async def complete(self, messages, **kwargs):
            if outcome == "error":
                raise RuntimeError("synthetic summary unavailable")
            if outcome == "cancel":
                raise asyncio.CancelledError()
            return provider_response("stop", "   ")

    active = {"role": "user", "content": TEXT}
    old = {"role": "assistant", "content": "普通旧消息"}
    messages = [old, active]
    compactor = ContextCompactor(ContextConfig(), FailingGateway())
    if outcome == "empty":
        result = await compactor.compact(messages, ["a"], protected_message=active)
        assert result.summary == "" and result.retained_file_refs == ["a"]
    else:
        with pytest.raises(RuntimeError if outcome == "error" else asyncio.CancelledError):
            await compactor.compact(messages, ["a"], protected_message=active)
    assert len(messages) == 2 and messages[0] is old and messages[1] is active
