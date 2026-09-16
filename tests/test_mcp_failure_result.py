"""MCP protocol failures must never become successful native tool results."""

import asyncio
from unittest.mock import AsyncMock

import pytest
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

from praxis.config.schemas import ToolsConfig
from praxis.tools.executor import ToolExecutor
from praxis.tools.mcp.tools import MCPToolsBridge
from praxis.tools.policy import ToolPolicy
from praxis.tools.registry import ToolRegistry


@pytest.mark.parametrize("failure", [False, True, "transport"])
async def test_mcp_error_flag_and_transport_failure_are_native_failures(failure, caplog):
    session = AsyncMock()
    session.list_tools.return_value = ListToolsResult(tools=[Tool(
        name="action", input_schema={"type": "object"},
    )])
    if failure == "transport":
        session.call_tool.side_effect = RuntimeError("secret-transport-canary")
    else:
        session.call_tool.return_value = CallToolResult(
            is_error=failure,
            content=[TextContent(type="text", text="secret-transport-canary" if failure else "ok")],
        )
    registry = ToolRegistry()
    bridge = MCPToolsBridge(registry)
    await bridge.discover_tools("local", session)
    result = await ToolExecutor(registry, ToolPolicy(ToolsConfig())).execute(
        "mcp_local_action", {}, "call-1",
    )
    assert result.success is (not failure)
    assert session.call_tool.await_count == 1
    if failure:
        assert result.error and "MCP" in result.error
        assert "secret-transport-canary" not in result.model_dump_json()
        assert "secret-transport-canary" not in caplog.text
    else:
        assert result.content == "ok"


async def test_mcp_cancellation_propagates():
    session = AsyncMock()
    session.call_tool.side_effect = asyncio.CancelledError()
    bridge = MCPToolsBridge(ToolRegistry())
    bridge.server_sessions["local"] = session
    with pytest.raises(asyncio.CancelledError):
        await bridge.call_tool("local", "action", {})


async def test_transport_failure_drops_exception_chain():
    session = AsyncMock()
    session.call_tool.side_effect = RuntimeError("secret-transport-canary")
    bridge = MCPToolsBridge(ToolRegistry())
    bridge.server_sessions["local"] = session
    with pytest.raises(RuntimeError, match="MCP transport failed") as failure:
        await bridge.call_tool("local", "action", {})
    assert failure.value.__context__ is None
    assert failure.value.__cause__ is None
