"""Native tool bodies retain complete structured data independently of log limits."""

import json
from copy import deepcopy

from praxis.config.schemas import ContextConfig, MemoryConfig
from praxis.models.tools import FunctionCall, ToolCall, ToolDefinition, ToolMetadata
from praxis.runtime import PraxisRuntime
from praxis.session.checkpoint import CheckpointManager
from praxis.session.core import Session
from praxis.telemetry.redaction import DEFAULT_REDACTION_POLICY
from tests.test_runtime import FakeGateway, runtime_config


class LargeToolGateway(FakeGateway):
    def __init__(self):
        super().__init__()
        self.requests = []

    async def complete(self, messages, **kwargs):
        self.requests.append(deepcopy(messages))
        response = await super().complete(messages, **kwargs)
        if len(self.requests) == 1:
            return response.model_copy(update={
                "content": "", "finish_reason": "tool_calls",
                "tool_calls": [ToolCall(id="large", function=FunctionCall(name="structured", arguments="{}"))],
            })
        return response


async def test_native_large_json_reaches_next_request_and_checkpoint_with_tail_redaction(tmp_path):
    secret = "sk-" + "Z" * 32
    payload = json.dumps({"facts": "complete-fact " * 6000, "tail": "last-fact", "credential_note": "Bearer " + secret})
    assert len(payload) > 65_536
    executions = []

    async def structured(arguments):
        executions.append(arguments)
        return payload

    config = runtime_config(tmp_path).model_copy(update={
        "context": ContextConfig(masking_token_threshold=32768),
        "memory": MemoryConfig(background_enabled=False, dream_enabled=False),
    })
    gateway = LargeToolGateway()
    async with PraxisRuntime(config, gateway=gateway) as runtime:
        async with runtime.session() as session:
            assert isinstance(session.runner, Session)
            session.runner.registry.register(ToolDefinition(
                name="structured", description="structured", parameters={"type": "object"},
                metadata=ToolMetadata(permission_level="auto_approve", readonly=True),
            ), structured)
            assert (await session.run("read all facts")).content == "unused"
            content = next(item["content"] for item in gateway.requests[-1] if item["role"] == "tool")
            restored = json.loads(content)
            assert restored["facts"] == "complete-fact " * 6000
            assert restored["tail"] == "last-fact" and secret not in content
            checkpoint_id = await session.runner.save_auto_checkpoint()
            checkpoint = await CheckpointManager(runtime.store).load_checkpoint(session.runner.session_id, checkpoint_id)
            assert checkpoint is not None
            persisted = next(item["content"] for item in checkpoint.state["context_state"]["messages"] if item["role"] == "tool")
            assert persisted == content
            assert checkpoint.state["tool_execution_ledger"]["large"]["result"]["content"] == content
    assert len(executions) == 1 and len(gateway.requests) == 2
    summary = DEFAULT_REDACTION_POLICY.summarize({"content": payload}, max_length=2048)
    assert len(summary) < 2100 and secret not in summary
