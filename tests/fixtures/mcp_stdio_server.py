"""Self-contained MCP stdio server used by integration tests."""

import os
import signal

from mcp.server.fastmcp import Context, FastMCP
from mcp.types import SamplingMessage, TextContent
from pydantic import BaseModel

server = FastMCP("praxis-stdio-test", log_level="ERROR")


class Approval(BaseModel):
    """Elicitation schema exposed by the fixture server."""

    answer: str


@server.tool()
async def echo(message: str) -> str:
    """Echo a message."""
    return message


@server.tool()
async def terminate_server() -> str:
    """Terminate this fixture so the client's reconnect supervisor can be tested."""
    os.kill(os.getpid(), signal.SIGTERM)
    return "terminating"


@server.tool()
async def request_sampling(ctx: Context) -> str:
    """Ask the connected client to sample a response."""
    result = await ctx.request_context.session.create_message(
        messages=[
            SamplingMessage(
                role="user",
                content=TextContent(type="text", text="Please sample a response"),
            )
        ],
        max_tokens=32,
        related_request_id=ctx.request_id,
    )
    content = result.content
    return content.text if isinstance(content, TextContent) else "unsupported content"


@server.tool()
async def request_elicitation(ctx: Context) -> str:
    """Ask the connected client for structured approval."""
    result = await ctx.elicit("Provide approval", Approval)
    if result.action == "accept" and result.data is not None:
        return result.data.answer
    return result.action


@server.resource("test://fixture")
def fixture_resource() -> str:
    """Return a deterministic resource payload."""
    return "fixture resource"


@server.prompt()
def greet(name: str) -> str:
    """Return a deterministic prompt."""
    return f"Hello, {name}!"


if __name__ == "__main__":
    server.run("stdio")
