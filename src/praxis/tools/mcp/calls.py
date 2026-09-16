"""Small secret-safe failure boundary for native MCP requests."""

from collections.abc import Awaitable


async def invoke_mcp[T](request: Awaitable[T]) -> T:
    """Preserve results and cancellation without retaining transport diagnostics."""
    try:
        return await request
    except Exception:
        pass
    # Outside the handler: __context__ must not retain headers, URLs or server errors.
    raise RuntimeError("MCP transport failed")
