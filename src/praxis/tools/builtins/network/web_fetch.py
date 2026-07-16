"""带逐跳 SSRF 校验和流式字节上限的 Web Fetch。"""

from typing import Any
from urllib.parse import urljoin

import httpx

from praxis.exceptions import ToolPolicyViolationError
from praxis.models.tools import ToolDefinition, ToolMetadata
from praxis.network import HTTP_REDIRECT_STATUS_CODES
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

def create_handler(policy: ToolPolicy, client: httpx.AsyncClient | None = None):
    async def handle(args: dict[str, Any]) -> str:
        current_url = str(args["url"])
        requested_limit = int(args.get("max_bytes", policy.network_max_response_bytes))
        if requested_limit <= 0:
            raise ToolPolicyViolationError("Web Fetch 响应字节上限必须大于零")
        byte_limit = min(requested_limit, policy.network_max_response_bytes)
        owns_client = client is None
        active_client = client or httpx.AsyncClient(timeout=30.0, follow_redirects=False)
        try:
            for redirect_attempt in range(6):
                await policy.check_url(current_url)
                request = active_client.build_request("GET", current_url)
                response = await active_client.send(request, stream=True)
                try:
                    if response.status_code in HTTP_REDIRECT_STATUS_CODES:
                        location = response.headers.get("location")
                        if not location:
                            raise ToolPolicyViolationError("重定向响应缺少 Location")
                        if redirect_attempt == 5:
                            break
                        current_url = urljoin(current_url, location)
                        continue

                    body = bytearray()
                    truncated = False
                    async for chunk in response.aiter_bytes():
                        remaining = byte_limit - len(body)
                        if remaining <= 0:
                            truncated = True
                            break
                        body.extend(chunk[:remaining])
                        if len(chunk) > remaining:
                            truncated = True
                            break
                    encoding = response.encoding or "utf-8"
                    text = bytes(body).decode(encoding, errors="replace")
                    suffix = " (已截断)" if truncated else ""
                    content_type = response.headers.get("content-type", "unknown")
                    return (
                        f"Status: {response.status_code}\n"
                        f"Content-Type: {content_type}\n\n{text}{suffix}"
                    )
                finally:
                    await response.aclose()
            raise ToolPolicyViolationError("重定向次数超过上限")
        finally:
            if owns_client:
                await active_client.aclose()

    return handle
