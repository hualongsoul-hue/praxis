"""场景五：子代理任务委托。

主 Agent → spawn_research_agent → S13 创建子代理 →
S12 创建子会话 → 独立 S4~S11 → 子代理执行 → 结果压缩 → 返回主 Agent。
"""

from pathlib import Path
from typing import Any

import pytest

from praxis.config.schemas import (
    ContextConfig,
    OrchestratorConfig,
    PersistenceConfig,
    SessionConfig,
)
from praxis.guardrails.engine import GuardrailEngine
from praxis.guardrails.permissions import PermissionManager
from praxis.guardrails.rules import RuleEngine
from praxis.models.subagent import SubagentMode, SubagentSpec, SubagentStatus
from praxis.persistence.store import PersistenceStore, create_store
from praxis.gateway.router import GatewayRouter
from praxis.subagent.isolation import IsolatedContext
from praxis.tools.registry import ToolRegistry

from tests.e2e.conftest import register_tool


@pytest.fixture
async def sub_store(tmp_path: Path) -> PersistenceStore:
    config = PersistenceConfig(backend="sqlite", sqlite_path=str(tmp_path / "sub.db"))
    s = await create_store(config)
    yield s  # type: ignore[misc]
    await s.close()


class TestSubagentDelegation:
    """场景五：子代理任务委托 E2E 测试。"""

    def test_isolated_context_creates_independent_session(
        self,
        sub_store: PersistenceStore,
        mock_gateway: GatewayRouter,
    ) -> None:
        """验证：子代理获得独立的组件实例集。"""
        guardrails = GuardrailEngine(RuleEngine(), PermissionManager())

        parent_registry = ToolRegistry()

        async def read_handler(args: dict[str, Any]) -> str:
            return "file content"

        async def grep_handler(args: dict[str, Any]) -> str:
            return "search results"

        async def deploy_handler(args: dict[str, Any]) -> str:
            return "deployed"

        register_tool(parent_registry, "read_file", read_handler)
        register_tool(parent_registry, "grep_search", grep_handler)
        register_tool(parent_registry, "deploy", deploy_handler)

        isolation = IsolatedContext(
            store=sub_store,
            gateway=mock_gateway,
            orchestrator_config=OrchestratorConfig(),
            context_config=ContextConfig(),
        )

        spec = SubagentSpec(
            task="分析 X 模块的性能瓶颈",
            tool_names=["read_file", "grep_search"],
            context_summary="项目使用 Python + asyncio",
            max_turns=20,
        )

        session = isolation.create_isolated_session(
            spec=spec,
            guardrails=guardrails,
            parent_registry=parent_registry,
        )

        # 子代理注册表仅包含指定工具
        assert session.registry.has_tool("read_file")
        assert session.registry.has_tool("grep_search")
        assert not session.registry.has_tool("deploy")

    def test_subagent_session_is_independent(
        self,
        sub_store: PersistenceStore,
        mock_gateway: GatewayRouter,
    ) -> None:
        """验证：子代理会话 ID 与主会话不同，状态互相隔离。"""
        guardrails = GuardrailEngine(RuleEngine(), PermissionManager())
        parent_registry = ToolRegistry()

        isolation = IsolatedContext(
            store=sub_store,
            gateway=mock_gateway,
            orchestrator_config=OrchestratorConfig(),
            context_config=ContextConfig(),
        )

        spec1 = SubagentSpec(task="task1", tool_names=[])
        spec2 = SubagentSpec(task="task2", tool_names=[])

        s1 = isolation.create_isolated_session(spec1, guardrails, parent_registry)
        s2 = isolation.create_isolated_session(spec2, guardrails, parent_registry)

        assert s1.session_id != s2.session_id
        # 修改 s1 不影响 s2
        s1.assembler.conversation_history.append({"role": "user", "content": "test"})
        assert len(s2.assembler.conversation_history) == 0

    def test_subagent_max_turns_respected(
        self,
        sub_store: PersistenceStore,
        mock_gateway: GatewayRouter,
    ) -> None:
        """验证：子代理会话遵守 max_turns 限制。"""
        guardrails = GuardrailEngine(RuleEngine(), PermissionManager())

        isolation = IsolatedContext(
            store=sub_store,
            gateway=mock_gateway,
            orchestrator_config=OrchestratorConfig(max_turns=100),
            context_config=ContextConfig(),
        )

        spec = SubagentSpec(task="任务", tool_names=[], max_turns=10)
        session = isolation.create_isolated_session(
            spec, guardrails, ToolRegistry(),
        )
        # 子代理的 loop 配置应使用 spec.max_turns
        assert session.loop.config.max_turns == 10

    def test_subagent_result_model(self) -> None:
        """验证：SubagentResult 模型正确序列化。"""
        from praxis.models.subagent import SubagentResult

        result = SubagentResult(
            subagent_id="sub-abc123",
            mode=SubagentMode.AGENT_AS_TOOL,
            status=SubagentStatus.COMPLETED,
            summary="分析完成，发现 3 个性能瓶颈",
            key_findings=["数据库查询 N+1", "缺少缓存", "同步 IO"],
            total_turns=5,
            total_tokens=15000,
        )
        data = result.model_dump()
        assert data["status"] == "completed"
        assert len(data["key_findings"]) == 3
