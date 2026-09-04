"""内置工具：web_search。

执行网络搜索。需要外部搜索 API 配置。
当前实现使用 DuckDuckGo HTML 搜索作为无 API Key 的回退方案。
"""

from typing import Any
from urllib.parse import urlencode

from praxis.models.tools import ToolDefinition, ToolMetadata
from praxis.network import TransportFactory
from praxis.tools.policy import ToolPolicy

DEFINITION = ToolDefinition(
    name="web_search",
    description="执行网络搜索，返回搜索结果摘要列表。",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词",
            },
            "max_results": {
                "type": "integer",
                "description": "最大返回结果数，默认 5",
            },
        },
        "required": ["query"],
    },
    metadata=ToolMetadata(
        category="network",
        permission_level="confirm",
        readonly=True,
        tags=["network", "search"],
    ),
)


def create_handler(
    sandbox: ToolPolicy,
    transport_factory: TransportFactory | None = None,
):
    """创建绑定沙箱的处理函数。"""

    async def handle(args: dict[str, Any]) -> str:
        query = args["query"]
        max_results = min(max(int(args.get("max_results", 5)), 1), 20)
        url = "https://html.duckduckgo.com/html/?" + urlencode({"q": str(query)})
        response = await sandbox.fetch_url(
            url,
            transport_factory,
            max_bytes=sandbox.network_max_response_bytes,
            timeout=15.0,
        )
        if response.status_code != 200:
            return f"搜索请求失败: HTTP {response.status_code}"

        text = response.body.decode("utf-8", errors="replace")
        results: list[str] = []
        start = 0
        remaining_results = max_results
        while remaining_results > 0:
            remaining_results -= 1
            idx = text.find('class="result__a"', start)
            if idx == -1:
                break
            href_start = text.rfind('href="', max(0, idx - 200), idx)
            if href_start == -1:
                start = idx + 1
                continue
            href_start += 6
            href_end = text.find('"', href_start)
            url = text[href_start:href_end]

            title_start = text.find(">", idx)
            title_end = text.find("</a>", title_start)
            title = text[title_start + 1:title_end].strip()
            title = title.replace("<b>", "").replace("</b>", "")

            results.append(f"- {title}\n  {url}")
            start = title_end

        if not results:
            return f"未找到 '{query}' 的搜索结果"
        return "\n\n".join(results)

    return handle
