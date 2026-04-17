"""内置工具：web_fetch。

通过 HTTP 获取 URL 内容。
"""

from typing import Any

import httpx

from praxis.models.tools import ToolDefinition, ToolMetadata
from praxis.tools.sandbox import Sandbox

DEFINITION = ToolDefinition(
    name="web_fetch",
    description="通过 HTTP GET 获取指定 URL 的内容。返回状态码和响应体文本。",
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "要获取的 URL（http 或 https）",
            },
            "max_length": {
                "type": "integer",
                "description": "响应体最大字符数，默认 10000",
            },
        },
        "required": ["url"],
    },
    metadata=ToolMetadata(
        category="network",
        permission_level="confirm",
        readonly=True,
        tags=["network", "http"],
    ),
)


def create_handler(sandbox: Sandbox):
    """创建绑定沙箱的处理函数。"""

    async def handle(args: dict[str, Any]) -> str:
        sandbox.check_network()

        url = args["url"]
        max_length = args.get("max_length", 10000)

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            resp = await client.get(url)

        text = resp.text[:max_length]
        truncated = " (已截断)" if len(resp.text) > max_length else ""

        return f"Status: {resp.status_code}\nContent-Type: {resp.headers.get('content-type', 'unknown')}\n\n{text}{truncated}"

    return handle
