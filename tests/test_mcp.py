"""S5 MCP 完整集成单元测试。"""

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from praxis.models.mcp import (
    MCPElicitationRequest,
    MCPElicitationResponse,
    MCPPromptInfo,
    MCPPromptMessage,
    MCPResourceContent,
    MCPResourceInfo,
    MCPSamplingRequest,
    MCPServerCapabilities,
    MCPServerConfig,
    MCPServerStatus,
    MCPTaskInfo,
    MCPTaskStatus,
    MCPToolInfo,
    MCPToolResult,
    MCPTransportType,
)
from praxis.tools.mcp.auth import MCPAuthManager, OAuthConfig, OAuthToken
from praxis.tools.mcp.elicitation import ElicitationManager
from praxis.tools.mcp.connection import MCPConnectionManager, MCPServerConnection
from praxis.tools.mcp.prompts import MCPPromptsBridge
from praxis.tools.mcp.resources import MCPResourcesBridge
from praxis.tools.mcp.roots import RootsManager
from praxis.tools.mcp.sampling import SamplingManager
from praxis.tools.mcp.tasks import MCPTaskManager
from praxis.tools.mcp.tools import MCPToolsBridge
from praxis.tools.registry import ToolRegistry


# ── Helpers ───────────────────────────────────────────────────────────────


def make_mock_session() -> MagicMock:
    """创建 mock MCP ClientSession。"""
    session = MagicMock()

    # tools
    mock_tool = MagicMock()
    mock_tool.name = "echo"
    mock_tool.description = "Echo tool"
    mock_tool.inputSchema = {"type": "object", "properties": {"msg": {"type": "string"}}}
    tools_result = MagicMock()
    tools_result.tools = [mock_tool]
    session.list_tools = AsyncMock(return_value=tools_result)

    mock_call_content = MagicMock()
    mock_call_content.text = "hello"
    call_result = MagicMock()
    call_result.content = [mock_call_content]
    session.call_tool = AsyncMock(return_value=call_result)

    # resources
    mock_resource = MagicMock()
    mock_resource.uri = "file:///test.txt"
    mock_resource.name = "test.txt"
    mock_resource.description = "Test file"
    mock_resource.mimeType = "text/plain"
    resources_result = MagicMock()
    resources_result.resources = [mock_resource]
    session.list_resources = AsyncMock(return_value=resources_result)

    mock_template = MagicMock()
    mock_template.uriTemplate = "weather://{city}"
    mock_template.name = "weather"
    mock_template.description = "Weather"
    mock_template.mimeType = "application/json"
    templates_result = MagicMock()
    templates_result.resourceTemplates = [mock_template]
    session.list_resource_templates = AsyncMock(return_value=templates_result)

    mock_content = MagicMock()
    mock_content.text = "file content"
    mock_content.blob = None
    mock_content.mimeType = "text/plain"
    read_result = MagicMock()
    read_result.contents = [mock_content]
    session.read_resource = AsyncMock(return_value=read_result)

    session.subscribe_resource = AsyncMock()
    session.unsubscribe_resource = AsyncMock()

    # prompts
    mock_prompt = MagicMock()
    mock_prompt.name = "greet"
    mock_prompt.description = "Greeting prompt"
    mock_arg = MagicMock()
    mock_arg.name = "name"
    mock_arg.description = "User name"
    mock_arg.required = True
    mock_prompt.arguments = [mock_arg]
    prompts_result = MagicMock()
    prompts_result.prompts = [mock_prompt]
    session.list_prompts = AsyncMock(return_value=prompts_result)

    mock_msg = MagicMock()
    mock_msg.role = "user"
    mock_msg_content = MagicMock()
    mock_msg_content.text = "Hello World"
    mock_msg.content = mock_msg_content
    get_prompt_result = MagicMock()
    get_prompt_result.messages = [mock_msg]
    session.get_prompt = AsyncMock(return_value=get_prompt_result)

    mock_completion = MagicMock()
    mock_completion.values = ["Alice", "Bob"]
    complete_result = MagicMock()
    complete_result.completion = mock_completion
    session.complete = AsyncMock(return_value=complete_result)

    # roots
    session.send_roots_list_changed = AsyncMock()

    # capabilities
    caps = MagicMock()
    caps.tools = MagicMock()
    caps.resources = MagicMock()
    caps.prompts = MagicMock()
    session.get_server_capabilities = MagicMock(return_value=caps)

    session.initialize = AsyncMock()

    return session


# ── Task 15.2: MCP Tools ────────────────────────────────────────────────


class TestMCPToolsBridge:
    """MCP Tools 集成测试。"""

    async def test_discover_tools(self) -> None:
        registry = ToolRegistry()
        bridge = MCPToolsBridge(registry)
        session = make_mock_session()

        tools = await bridge.discover_tools("test-server", session)
        assert len(tools) == 1
        assert tools[0].name == "echo"
        assert registry.has_tool("mcp_test-server_echo")

    async def test_call_tool(self) -> None:
        registry = ToolRegistry()
        bridge = MCPToolsBridge(registry)
        session = make_mock_session()
        await bridge.discover_tools("test-server", session)

        result = await bridge.call_tool("test-server", "echo", {"msg": "hi"})
        assert result == "hello"

    async def test_call_tool_disconnected(self) -> None:
        registry = ToolRegistry()
        bridge = MCPToolsBridge(registry)
        with pytest.raises(RuntimeError, match="未连接"):
            await bridge.call_tool("no-server", "echo", {})

    async def test_refresh_tools(self) -> None:
        registry = ToolRegistry()
        bridge = MCPToolsBridge(registry)
        session = make_mock_session()
        await bridge.discover_tools("test-server", session)
        assert registry.has_tool("mcp_test-server_echo")

        # 刷新
        tools = await bridge.refresh_tools("test-server")
        assert len(tools) == 1
        assert registry.has_tool("mcp_test-server_echo")

    async def test_disconnect_server(self) -> None:
        registry = ToolRegistry()
        bridge = MCPToolsBridge(registry)
        session = make_mock_session()
        await bridge.discover_tools("test-server", session)

        bridge.disconnect_server("test-server")
        assert not registry.has_tool("mcp_test-server_echo")


# ── Task 15.3: MCP Resources ────────────────────────────────────────────


class TestMCPResourcesBridge:
    """MCP Resources 集成测试。"""

    async def test_list_resources(self) -> None:
        bridge = MCPResourcesBridge()
        session = make_mock_session()
        bridge.register_session("srv", session)

        resources = await bridge.list_resources("srv")
        assert len(resources) == 1
        assert resources[0].uri == "file:///test.txt"

    async def test_list_resource_templates(self) -> None:
        bridge = MCPResourcesBridge()
        session = make_mock_session()
        bridge.register_session("srv", session)

        templates = await bridge.list_resource_templates("srv")
        assert len(templates) == 1
        assert "weather" in templates[0]["uri_template"]

    async def test_read_resource(self) -> None:
        bridge = MCPResourcesBridge()
        session = make_mock_session()
        bridge.register_session("srv", session)

        content = await bridge.read_resource("srv", "file:///test.txt")
        assert content.text == "file content"

    async def test_subscribe_resource(self) -> None:
        bridge = MCPResourcesBridge()
        session = make_mock_session()
        bridge.register_session("srv", session)

        await bridge.subscribe_resource("srv", "file:///test.txt")
        assert "file:///test.txt" in bridge.subscriptions["srv"]

    async def test_unsubscribe_resource(self) -> None:
        bridge = MCPResourcesBridge()
        session = make_mock_session()
        bridge.register_session("srv", session)
        await bridge.subscribe_resource("srv", "file:///test.txt")
        await bridge.unsubscribe_resource("srv", "file:///test.txt")
        assert "file:///test.txt" not in bridge.subscriptions.get("srv", set())

    async def test_disconnected_raises(self) -> None:
        bridge = MCPResourcesBridge()
        with pytest.raises(RuntimeError):
            await bridge.list_resources("no-server")


# ── Task 15.4: MCP Prompts ──────────────────────────────────────────────


class TestMCPPromptsBridge:
    """MCP Prompts 集成测试。"""

    async def test_list_prompts(self) -> None:
        bridge = MCPPromptsBridge()
        session = make_mock_session()
        bridge.register_session("srv", session)

        prompts = await bridge.list_prompts("srv")
        assert len(prompts) == 1
        assert prompts[0].name == "greet"
        assert len(prompts[0].arguments) == 1

    async def test_get_prompt(self) -> None:
        bridge = MCPPromptsBridge()
        session = make_mock_session()
        bridge.register_session("srv", session)

        messages = await bridge.get_prompt("srv", "greet", {"name": "Alice"})
        assert len(messages) == 1
        assert messages[0].content == "Hello World"

    async def test_complete_argument(self) -> None:
        bridge = MCPPromptsBridge()
        session = make_mock_session()
        bridge.register_session("srv", session)

        completions = await bridge.complete_argument("srv", "greet", "name", "Al")
        assert "Alice" in completions
        assert "Bob" in completions

    async def test_disconnected_raises(self) -> None:
        bridge = MCPPromptsBridge()
        with pytest.raises(RuntimeError):
            await bridge.list_prompts("no-server")


# ── Task 15.5: Elicitation ──────────────────────────────────────────────


class TestElicitationManager:
    """Elicitation 测试。"""

    async def test_no_handler_returns_rejected(self) -> None:
        mgr = ElicitationManager()
        request = MCPElicitationRequest(server_name="srv", message="info?")
        response = await mgr.handle_elicitation(request)
        assert response.accepted is False

    async def test_with_handler(self) -> None:
        mgr = ElicitationManager()

        async def handler(req: MCPElicitationRequest) -> MCPElicitationResponse:
            return MCPElicitationResponse(accepted=True, data={"answer": "yes"})

        mgr.set_handler(handler)
        request = MCPElicitationRequest(server_name="srv", message="info?")
        response = await mgr.handle_elicitation(request)
        assert response.accepted is True
        assert response.data["answer"] == "yes"


# ── Task 15.5: Roots ────────────────────────────────────────────────────


class TestRootsManager:
    """Roots 测试。"""

    def test_set_and_get_roots(self) -> None:
        mgr = RootsManager()
        mgr.set_roots(["/home/user/project"])
        assert mgr.get_roots() == ["/home/user/project"]

    async def test_notify_roots_changed(self) -> None:
        mgr = RootsManager()
        session = make_mock_session()
        mgr.register_session("srv", session)
        await mgr.notify_roots_changed()
        session.send_roots_list_changed.assert_called_once()


# ── Task 15.6: Lifecycle ────────────────────────────────────────────────


class TestMCPConnectionManager:
    """MCP 生命周期管理测试。"""

    async def test_connect_server(self) -> None:
        registry = ToolRegistry()
        tools_bridge = MCPToolsBridge(registry)
        resources_bridge = MCPResourcesBridge()
        prompts_bridge = MCPPromptsBridge()
        roots_mgr = RootsManager()

        mgr = MCPConnectionManager(tools_bridge, resources_bridge, prompts_bridge, roots_mgr)
        config = MCPServerConfig(name="test", transport=MCPTransportType.STDIO, command="echo")
        session = make_mock_session()

        caps = await mgr.connect_server(config, session)
        assert caps.tools is True
        assert caps.resources is True
        assert mgr.get_server_status("test") == MCPServerStatus.CONNECTED
        assert "test" in mgr.list_connected_servers()

    async def test_disconnect_server(self) -> None:
        registry = ToolRegistry()
        tools_bridge = MCPToolsBridge(registry)
        mgr = MCPConnectionManager(
            tools_bridge, MCPResourcesBridge(), MCPPromptsBridge(), RootsManager()
        )
        config = MCPServerConfig(name="test", command="echo")
        session = make_mock_session()
        await mgr.connect_server(config, session)

        mgr.disconnect_server("test")
        assert mgr.get_server_status("test") == MCPServerStatus.DISCONNECTED
        assert "test" not in mgr.list_connected_servers()

    async def test_get_server_capabilities(self) -> None:
        registry = ToolRegistry()
        tools_bridge = MCPToolsBridge(registry)
        mgr = MCPConnectionManager(
            tools_bridge, MCPResourcesBridge(), MCPPromptsBridge(), RootsManager()
        )
        config = MCPServerConfig(name="test", command="echo")
        session = make_mock_session()
        await mgr.connect_server(config, session)

        caps = mgr.get_server_capabilities("test")
        assert caps is not None
        assert caps.tools is True

    async def test_nonexistent_capabilities(self) -> None:
        mgr = MCPConnectionManager(
            MCPToolsBridge(ToolRegistry()),
            MCPResourcesBridge(),
            MCPPromptsBridge(),
            RootsManager(),
        )
        assert mgr.get_server_capabilities("no") is None


# ── Task 15.7: Tasks ────────────────────────────────────────────────────


class TestMCPTaskManager:
    """MCP Tasks 测试。"""

    def test_register_task(self) -> None:
        mgr = MCPTaskManager()
        task = mgr.register_task("t1", "srv")
        assert task.task_id == "t1"
        assert task.status == MCPTaskStatus.PENDING

    def test_update_status(self) -> None:
        mgr = MCPTaskManager()
        mgr.register_task("t1", "srv")
        mgr.update_status("t1", MCPTaskStatus.RUNNING, progress=0.5)
        task = mgr.get_task("t1")
        assert task.status == MCPTaskStatus.RUNNING
        assert task.progress == 0.5

    def test_cancel_task(self) -> None:
        mgr = MCPTaskManager()
        mgr.register_task("t1", "srv")
        mgr.update_status("t1", MCPTaskStatus.RUNNING)
        assert mgr.cancel_task("t1") is True
        assert mgr.get_task("t1").status == MCPTaskStatus.CANCELLED

    def test_cancel_completed_task(self) -> None:
        mgr = MCPTaskManager()
        mgr.register_task("t1", "srv")
        mgr.update_status("t1", MCPTaskStatus.COMPLETED)
        assert mgr.cancel_task("t1") is False

    def test_list_tasks(self) -> None:
        mgr = MCPTaskManager()
        mgr.register_task("t1", "srv1")
        mgr.register_task("t2", "srv2")
        mgr.register_task("t3", "srv1")
        assert len(mgr.list_tasks(server_name="srv1")) == 2
        assert len(mgr.list_tasks()) == 3

    def test_cleanup_completed(self) -> None:
        mgr = MCPTaskManager()
        mgr.register_task("t1", "srv")
        mgr.register_task("t2", "srv")
        mgr.update_status("t1", MCPTaskStatus.COMPLETED)
        count = mgr.cleanup_completed()
        assert count == 1
        assert mgr.get_task("t1") is None
        assert mgr.get_task("t2") is not None


# ── Task 15.7: Auth ─────────────────────────────────────────────────────


class TestMCPAuthManager:
    """MCP Auth 测试。"""

    def test_store_and_get_token(self) -> None:
        elicitation = ElicitationManager()
        mgr = MCPAuthManager(elicitation)
        token = OAuthToken(
            access_token="abc123",
            expires_at=time.time() + 3600,
        )
        mgr.store_token("srv", token)
        assert mgr.get_token("srv") is not None
        assert mgr.is_authenticated("srv") is True

    def test_expired_token(self) -> None:
        elicitation = ElicitationManager()
        mgr = MCPAuthManager(elicitation)
        token = OAuthToken(
            access_token="expired",
            expires_at=time.time() - 100,
        )
        mgr.store_token("srv", token)
        assert mgr.get_token("srv") is None
        assert mgr.is_authenticated("srv") is False

    def test_revoke_token(self) -> None:
        elicitation = ElicitationManager()
        mgr = MCPAuthManager(elicitation)
        token = OAuthToken(access_token="abc123", expires_at=time.time() + 3600)
        mgr.store_token("srv", token)
        mgr.revoke_token("srv")
        assert mgr.get_token("srv") is None

    def test_auth_headers(self) -> None:
        elicitation = ElicitationManager()
        mgr = MCPAuthManager(elicitation)
        token = OAuthToken(access_token="abc123", expires_at=time.time() + 3600)
        mgr.store_token("srv", token)
        headers = mgr.get_auth_headers("srv")
        assert headers["Authorization"] == "Bearer abc123"

    def test_no_token_empty_headers(self) -> None:
        elicitation = ElicitationManager()
        mgr = MCPAuthManager(elicitation)
        assert mgr.get_auth_headers("srv") == {}

    async def test_initiate_auth_flow_no_config(self) -> None:
        elicitation = ElicitationManager()
        mgr = MCPAuthManager(elicitation)
        assert await mgr.initiate_auth_flow("srv") is False

    async def test_initiate_auth_flow_rejected(self) -> None:
        elicitation = ElicitationManager()

        async def reject(req: MCPElicitationRequest) -> MCPElicitationResponse:
            return MCPElicitationResponse(accepted=False)

        elicitation.set_handler(reject)
        mgr = MCPAuthManager(elicitation)
        mgr.set_oauth_config("srv", OAuthConfig(
            authorization_url="https://auth.example.com/authorize",
            token_url="https://auth.example.com/token",
            client_id="client-1",
        ))
        assert await mgr.initiate_auth_flow("srv") is False
