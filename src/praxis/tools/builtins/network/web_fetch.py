"""带逐跳 SSRF 校验和流式字节上限的 Web Fetch。"""

import sys
from typing import Any
from urllib.parse import urljoin

import httpx

from praxis.exceptions import ToolPolicyViolationError
from praxis.models.tools import ToolDefinition, ToolMetadata
from praxis.network import (
    HTTP_REDIRECT_STATUS_CODES,
    TransportFactory,
    close_http_resources,
    send_pinned_http_request,
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
        target = await policy.check_url(initial_url)
        for redirect_attempt in range(6):
            try:
                response, client = await send_pinned_http_request(
                    target,
                    transport_factory,
                    timeout=30.0,
                )
            except ValueError as exc:
                raise ToolPolicyViolationError(str(exc)) from None
            try:
                if response.status_code in HTTP_REDIRECT_STATUS_CODES:
                    location = response.headers.get("location")
                    if not location:
                        raise ToolPolicyViolationError("重定向响应缺少 Location")
                    if redirect_attempt == 5:
                        raise ToolPolicyViolationError("重定向次数超过上限")
                    candidate = urljoin(target.original_url, location)
                    target = await policy.check_url(candidate)
                    continue

                body = bytearray()
                truncated = False
                try:
                    async for chunk in response.aiter_bytes():
                        remaining = byte_limit - len(body)
                        if remaining <= 0:
                            truncated = True
                            break
                        body.extend(chunk[:remaining])
                        if len(chunk) > remaining:
                            truncated = True
                            break
                except (httpx.HTTPError, OSError):
                    raise ToolPolicyViolationError("Web Fetch 响应读取失败") from None
                encoding = response.encoding or "utf-8"
                text = bytes(body).decode(encoding, errors="replace")
                suffix = " (已截断)" if truncated else ""
                content_type = response.headers.get("content-type", "unknown")
                return (
                    f"Status: {response.status_code}\n"
                    f"Content-Type: {content_type}\n\n{text}{suffix}"
                )
            finally:
                active_error = sys.exc_info()[0] is not None
                close_failed = await close_http_resources(response, client)
                if close_failed and not active_error:
                    raise ToolPolicyViolationError("Web Fetch 响应关闭失败") from None
        raise ToolPolicyViolationError("重定向次数超过上限")

    return handle
