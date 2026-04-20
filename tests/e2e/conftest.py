"""端到端场景测试公共 fixture。

通过 mock gateway 控制 LLM 返回，使用真实组件实例验证完整调用链。
"""

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from praxis.config.schemas import (
    ContextConfig,
    OrchestratorConfig,
    PersistenceConfig,
    SessionConfig,
    ToolsConfig,
)
from praxis.context.assembler import PromptAssembler
from praxis.context.tool_injection import ToolInjector
from praxis.gateway.router import GatewayRouter
from praxis.guardrails.engine import GuardrailEngine
from praxis.guardrails.permissions import PermissionManager
from praxis.guardrails.rules import RuleEngine
from praxis.memory.core import CognitiveMemory
from praxis.models.responses import ModelResponse, Usage
from praxis.models.tools import FunctionCall, ToolCall, ToolDefinition, ToolResult
from praxis.orchestrator.events import EventEmitter
from praxis.orchestrator.loop import OrchestrationLoop
from praxis.orchestrator.parser import OutputParser
from praxis.orchestrator.strategy import LoopStrategy
from praxis.orchestrator.termination import TerminationManager
from praxis.orchestrator.tool_coordination import ToolCoordinator
from praxis.persistence.store import PersistenceStore, create_store
from praxis.recovery.circuit_breaker import CircuitBreakerRegistry
from praxis.recovery.retry import RetryPolicy
from praxis.session.checkpoint import CheckpointManager
from praxis.session.core import Session, SessionFactory
from praxis.tools.executor import ToolExecutor
from praxis.tools.registry import ToolHandler, ToolRegistry
from praxis.tools.sandbox import Sandbox


# ── 通用辅助 ──────────────────────────────────────────────────────────────


def make_model_response(
    content: str = "",
    tool_calls: list[ToolCall] | None = None,
    finish_reason: str = "stop",
    model: str = "test-model",
) -> ModelResponse:
    """构造 ModelResponse。"""
    return ModelResponse(
        id="resp-e2e",
        content=content,
        tool_calls=tool_calls,
        usage=Usage(prompt_tokens=50, completion_tokens=20, total_tokens=70),
        model=model,
        finish_reason=finish_reason,
        created=1700000000,
    )


def make_tool_call(
    name: str,
    arguments: str = "{}",
    tc_id: str = "tc-1",
) -> ToolCall:
    """构造 ToolCall。"""
    return ToolCall(
        id=tc_id,
        type="function",
        function=FunctionCall(name=name, arguments=arguments),
    )


def register_tool(
    registry: ToolRegistry,
    name: str,
    handler: ToolHandler,
    description: str = "",
    parameters: dict[str, Any] | None = None,
    readonly: bool = True,
) -> None:
    """向注册表注册一个工具。"""
    from praxis.models.tools import ToolMetadata

    defn = ToolDefinition(
        name=name,
        description=description or f"{name} tool",
        parameters=parameters or {"type": "object", "properties": {}},
        metadata=ToolMetadata(readonly=readonly),
    )
    registry.register(defn, handler)


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
async def store(tmp_path: Path) -> PersistenceStore:
    config = PersistenceConfig(backend="sqlite", sqlite_path=str(tmp_path / "e2e.db"))
    s = await create_store(config)
    yield s  # type: ignore[misc]
    await s.close()


@pytest.fixture
def rule_engine() -> RuleEngine:
    return RuleEngine()


@pytest.fixture
def permission_manager() -> PermissionManager:
    return PermissionManager()


@pytest.fixture
def guardrails(rule_engine: RuleEngine, permission_manager: PermissionManager) -> GuardrailEngine:
    return GuardrailEngine(rule_engine, permission_manager)


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry()


@pytest.fixture
def orchestrator_config() -> OrchestratorConfig:
    return OrchestratorConfig(max_turns=10)


@pytest.fixture
def context_config() -> ContextConfig:
    return ContextConfig()


@pytest.fixture(autouse=True)
def mock_litellm_metering():
    """自动 mock litellm 的 Token 计量函数，避免 E2E 测试依赖真实模型。"""
    with patch("praxis.context.assembler.get_max_tokens", return_value=128000), \
         patch("praxis.context.assembler.get_token_count", return_value=100):
        yield


@pytest.fixture
def mock_gateway() -> GatewayRouter:
    """创建 mock Gateway（控制 LLM 返回）。"""
    gw = MagicMock(spec=GatewayRouter)
    gw.config = MagicMock()
    gw.config.default_model = "test-model"
    gw.router = MagicMock()
    return gw


def build_loop(
    gateway: GatewayRouter,
    registry: ToolRegistry,
    guardrails: GuardrailEngine,
    orchestrator_config: OrchestratorConfig,
    context_config: ContextConfig,
    memory: CognitiveMemory | None = None,
) -> OrchestrationLoop:
    """用真实组件组装完整 OrchestrationLoop。"""
    sandbox = Sandbox(ToolsConfig())
    executor = ToolExecutor(registry, sandbox)
    injector = ToolInjector(registry)
    assembler = PromptAssembler(context_config, model="test-model")
    circuits = CircuitBreakerRegistry()
    retry_policy = RetryPolicy()
    emitter = EventEmitter()
    parser = OutputParser()
    termination = TerminationManager(orchestrator_config)
    strategy = LoopStrategy()

    coordinator = ToolCoordinator(
        executor=executor,
        registry=registry,
        guardrails=guardrails,
        circuit_registry=circuits,
        retry_policy=retry_policy,
        emitter=emitter,
    )

    return OrchestrationLoop(
        config=orchestrator_config,
        gateway=gateway,
        assembler=assembler,
        tool_coordinator=coordinator,
        guardrails=guardrails,
        termination=termination,
        strategy=strategy,
        parser=parser,
        emitter=emitter,
        tool_injector=injector,
        memory=memory,
    )
