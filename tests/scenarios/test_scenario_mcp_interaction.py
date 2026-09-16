"""场景八：MCP 服务器交互（Elicitation + Sampling）。

Agent 通过 MCP Server 执行外部任务 →
Server 需要 LLM 分析（Sampling） → 用户确认（Elicitation）。
"""

from unittest.mock import AsyncMock, MagicMock

from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

from praxis.gateway.router import GatewayRouter
from praxis.models.mcp import (
    MCPElicitationRequest,
    MCPElicitationResponse,
    MCPSamplingRequest,
    MCPServerStatus,
)
from praxis.tools.mcp.connection import MCPConnectionManager
from praxis.tools.mcp.elicitation import ElicitationManager
from praxis.tools.mcp.sampling import SamplingManager
from praxis.tools.mcp.tools import MCPToolsBridge
from praxis.tools.registry import ToolRegistry


def make_mock_session() -> MagicMock:
    """创建 mock MCP ClientSession。"""
    session = MagicMock()

    mock_tool = Tool(
        name="book_flight", description="预订航班",
        input_schema={
            "type": "object",
            "properties": {
                "origin": {"type": "string"},
                "dest": {"type": "string"},
            },
        },
    )
    tools_result = ListToolsResult(tools=[mock_tool])
    session.list_tools = AsyncMock(return_value=tools_result)

    call_result = CallToolResult(content=[
        TextContent(type="text", text="航班已预订: NYC → BCN, 确认号 FL-2025")
    ])
    session.call_tool = AsyncMock(return_value=call_result)

    return session


class TestMCPInteraction:
    """场景八：MCP 服务器交互 E2E 测试。"""

    async def test_mcp_tool_discovery_and_registration(self) -> None:
        """验证：MCP Server 工具被发现并注册到统一注册表。"""
        registry = ToolRegistry()
        bridge = MCPToolsBridge(registry)
        session = make_mock_session()

        tools = await bridge.discover_tools("travel", session)

        assert len(tools) == 1
        assert tools[0].name == "book_flight"
        assert tools[0].server_name == "travel"
        assert registry.has_tool("mcp_travel_book_flight")

    async def test_mcp_tool_execution_proxied(self) -> None:
        """验证：MCP 工具执行被透明代理到 MCP Server。"""
        registry = ToolRegistry()
        bridge = MCPToolsBridge(registry)
        session = make_mock_session()

        await bridge.discover_tools("travel", session)

        entry = registry.get_entry("mcp_travel_book_flight")
        result = await entry.handler({"origin": "NYC", "dest": "BCN"})
        assert "FL-2025" in result

        session.call_tool.assert_called_once_with(
            "book_flight",
            {"origin": "NYC", "dest": "BCN"},
        )

    async def test_elicitation_with_handler(self) -> None:
        """验证：Elicitation 请求被转发给用户处理器。"""
        manager = ElicitationManager()

        # 注册 mock 用户处理器
        async def mock_handler(req: MCPElicitationRequest) -> MCPElicitationResponse:
            return MCPElicitationResponse(
                accepted=True,
                data={"confirmBooking": True, "seatPref": "window"},
            )

        manager.set_handler(mock_handler)

        request = MCPElicitationRequest(
            server_name="travel",
            message="确认预订 BCN 航班",
            schema={
                "type": "object",
                "properties": {
                    "confirmBooking": {"type": "boolean"},
                    "seatPref": {"type": "string", "enum": ["window", "aisle", "middle"]},
                },
            },
        )

        response = await manager.handle_elicitation(request)
        assert response.accepted is True
        assert response.data["confirmBooking"] is True
        assert response.data["seatPref"] == "window"

    async def test_elicitation_without_handler_auto_rejects(self) -> None:
        """验证：无处理器时 Elicitation 自动拒绝。"""
        manager = ElicitationManager()

        request = MCPElicitationRequest(
            server_name="travel",
            message="确认操作",
        )

        response = await manager.handle_elicitation(request)
        assert response.accepted is False

    async def test_sampling_manager(self) -> None:
        """验证：Sampling 请求处理。"""
        mock_router = MagicMock(spec=GatewayRouter)
        mock_router.config = MagicMock()
        mock_router.config.max_budget = None
        mock_router.config.default_model = "test-model"

        from praxis.models.responses import ModelResponse, Usage

        mock_router.complete = AsyncMock(
            return_value=ModelResponse(
            id="resp-sampling",
            content="推荐航班: BA-447, 直飞, 价格最优",
            usage=Usage(prompt_tokens=30, completion_tokens=50, total_tokens=80),
            model="test-model",
            created=1700000000,
            )
        )

        manager = SamplingManager(mock_router)

        request = MCPSamplingRequest(
            server_name="travel",
            messages=[{"role": "user", "content": "分析 47 个航班选项，推荐最优"}],
        )

        response = await manager.handle_sampling(request)
        assert "推荐航班" in response["content"]

    async def test_sampling_rejected_by_reviewer(self) -> None:
        """验证：Human-in-the-loop 审核拒绝 Sampling 请求。"""
        mock_router = MagicMock(spec=GatewayRouter)
        manager = SamplingManager(mock_router)

        async def reject_handler(messages: list, server: str) -> bool:
            return False

        manager.set_review_handler(reject_handler)

        request = MCPSamplingRequest(
            server_name="travel",
            messages=[{"role": "user", "content": "test"}],
        )

        response = await manager.handle_sampling(request)
        assert response["content"] == "用户拒绝了此请求。"

    async def test_connection_manager_status(self) -> None:
        """验证：MCP 连接管理器的服务器状态跟踪。"""
        registry = ToolRegistry()
        tools_bridge = MCPToolsBridge(registry)
        resources_bridge = MagicMock()
        prompts_bridge = MagicMock()
        roots_mgr = MagicMock()

        manager = MCPConnectionManager(
            tools_bridge=tools_bridge,
            resources_bridge=resources_bridge,
            prompts_bridge=prompts_bridge,
            roots_mgr=roots_mgr,
        )

        # 未连接的服务器状态应为 DISCONNECTED
        status = manager.get_server_status("travel-server")
        assert status == MCPServerStatus.DISCONNECTED

        # 列出已连接服务器应为空
        assert manager.list_connected_servers() == []

    async def test_full_mcp_flow_with_tools_and_elicitation(self) -> None:
        """验证：完整 MCP 流程——工具发现 → 执行 → Elicitation 确认。"""
        registry = ToolRegistry()
        bridge = MCPToolsBridge(registry)
        elicitation = ElicitationManager()

        session = make_mock_session()
        await bridge.discover_tools("travel", session)

        # 模拟用户确认
        async def confirm_handler(req: MCPElicitationRequest) -> MCPElicitationResponse:
            return MCPElicitationResponse(accepted=True, data={"confirmed": True})

        elicitation.set_handler(confirm_handler)

        # 步骤 1: 执行 MCP 工具
        entry = registry.get_entry("mcp_travel_book_flight")
        tool_result = await entry.handler({"origin": "NYC", "dest": "BCN"})
        assert "FL-2025" in tool_result

        # 步骤 2: 服务器请求用户确认
        elicit_req = MCPElicitationRequest(
            server_name="travel",
            message="确认预订?",
            schema={"type": "object", "properties": {"confirmed": {"type": "boolean"}}},
        )
        elicit_resp = await elicitation.handle_elicitation(elicit_req)
        assert elicit_resp.accepted is True
