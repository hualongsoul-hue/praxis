"""Exercise the MCP SDK transport through real TCP, without an external server."""

import asyncio
import socket

import pytest
import uvicorn
from mcp.shared.exceptions import MCPError
from mcp.types import CONNECTION_CLOSED, CreateMessageResult, TextContent

from praxis.models.mcp import MCPElicitationResponse, MCPServerConfig, MCPTransportType
from praxis.tools.mcp.elicitation import ElicitationManager
from praxis.tools.mcp.prompts import MCPPromptsBridge
from praxis.tools.mcp.resources import MCPResourcesBridge
from praxis.tools.mcp.tools import MCPToolsBridge
from praxis.tools.mcp.transport import create_http_transport
from praxis.tools.mcp.wiring import make_elicitation_callback
from praxis.tools.registry import ToolRegistry
from tests.fixtures.mcp_stdio_server import server as fixture_server

pytestmark = pytest.mark.e2e


async def test_http_tools_resources_prompts_and_callbacks_preserve_session_lifecycle() -> None:
    """Catch incompatible HTTP clients, lost headers, and broken v2 field mapping."""
    application = fixture_server.streamable_http_app()
    received_headers: list[dict[bytes, bytes]] = []

    async def record_requests(scope, receive, send):
        if scope["type"] == "http":
            received_headers.append(dict(scope["headers"]))
        await application(scope, receive, send)

    server = uvicorn.Server(uvicorn.Config(
        record_requests, log_level="error", ws="none", lifespan="on",
        timeout_graceful_shutdown=1,
    ))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        serving = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            async with asyncio.timeout(10):
                while not server.started:
                    if serving.done():
                        await serving
                        pytest.fail("MCP HTTP server exited before becoming ready")
                    await asyncio.sleep(0.01)

            config = MCPServerConfig(
                name="http-contract", transport=MCPTransportType.HTTP,
                url=f"http://127.0.0.1:{port}/mcp",
                headers={"X-Praxis-Test": "forwarded"}, timeout=5,
            )
            sampled_messages: list[str] = []

            async def sample(context, params):
                sampled_messages.append(params.messages[0].content.text)
                return CreateMessageResult(
                    role="assistant", model="test-model",
                    content=TextContent(type="text", text="HTTP sampled response"),
                )

            elicitation = ElicitationManager()

            async def approve(request):
                assert request.request_schema["properties"]["answer"]["type"] == "string"
                return MCPElicitationResponse(accepted=True, data={"answer": "HTTP approved"})

            elicitation.set_handler(approve)
            async with create_http_transport(
                config, sample, make_elicitation_callback(config.name, elicitation),
            ) as session:
                registry = ToolRegistry()
                tools = MCPToolsBridge(registry)
                discovered = await tools.discover_tools(config.name, session)
                echo = next(tool for tool in discovered if tool.name == "echo")
                assert echo.input_schema["properties"]["message"]["type"] == "string"
                assert await tools.call_tool(config.name, "echo", {"message": "HTTP echo"}) == (
                    "HTTP echo"
                )
                resources = MCPResourcesBridge()
                resources.register_session(config.name, session)
                listed = await resources.list_resources(config.name)
                assert listed[0].uri == "test://fixture"
                content = await resources.read_resource(config.name, "test://fixture")
                assert content.text == "fixture resource"
                prompts = MCPPromptsBridge()
                prompts.register_session(config.name, session)
                messages = await prompts.get_prompt(config.name, "greet", {"name": "HTTP"})
                assert messages[0].content == "Hello, HTTP!"
                assert await tools.call_tool(config.name, "request_sampling", {}) == (
                    "HTTP sampled response"
                )
                assert await tools.call_tool(config.name, "request_elicitation", {}) == (
                    "HTTP approved"
                )
                await session.send_ping()

            with pytest.raises(MCPError) as closed:
                await session.list_tools()
            assert closed.value.code == CONNECTION_CLOSED
            assert sampled_messages == ["Please sample a response"]
            assert received_headers
            assert all(headers.get(b"x-praxis-test") == b"forwarded" for headers in received_headers)
        finally:
            server.should_exit = True
            await asyncio.wait_for(serving, timeout=10)
