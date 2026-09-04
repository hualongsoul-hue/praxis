"""带逐跳 SSRF 校验和流式字节上限的 Web Fetch。"""

from typing import Any

from praxis.exceptions import ToolPolicyViolationError
from praxis.models.tools import ToolDefinition, ToolMetadata
from praxis.network import (
    TransportFactory,
)
from praxis.tools.policy import ToolPolicy

DEFINITION = ToolDefinition(
    name="web_fetch",
    description="通过安全策略校验的 HTTP/HTTPS GET 获取公开网络内容。",
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string", "minLength": 1},
            "max_bytes": {"type": "integer", "minimum": 1},
        },
        "required": ["url"],
    },
    metadata=ToolMetadata(
        category="network",
        permission_level="confirm",
        readonly=True,
        idempotent=True,
        tags=["network", "http"],
    ),
)

def create_handler(
    policy: ToolPolicy,
    transport_factory: TransportFactory | None = None,
):
    async def handle(args: dict[str, Any]) -> str:
        initial_url = str(args["url"])
        requested_limit = int(args.get("max_bytes", policy.network_max_response_bytes))
        if requested_limit <= 0:
            raise ToolPolicyViolationError("Web Fetch 响应字节上限必须大于零")
        byte_limit = min(requested_limit, policy.network_max_response_bytes)
        result = await policy.fetch_url(
            initial_url,
            transport_factory,
            max_bytes=byte_limit,
            timeout=30.0,
        )
        text = result.body.decode("utf-8", errors="replace")
        suffix = " (已截断)" if result.truncated else ""
        content_type = result.headers.get("content-type", "unknown")
        return (
            f"Status: {result.status_code}\n"
            f"Content-Type: {content_type}\n\n{text}{suffix}"
        )

    return handle
