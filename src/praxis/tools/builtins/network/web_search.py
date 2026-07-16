"""内置工具：web_search。

执行网络搜索。需要外部搜索 API 配置。
当前实现使用 DuckDuckGo HTML 搜索作为无 API Key 的回退方案。
"""

from typing import Any

import httpx

from praxis.models.tools import ToolDefinition, ToolMetadata
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


def create_handler(sandbox: ToolPolicy):
    """创建绑定沙箱的处理函数。"""

    async def handle(args: dict[str, Any]) -> str:
        sandbox.check_network()

        query = args["query"]
        max_results = args.get("max_results", 5)

        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                headers={"User-Agent": "Praxis/0.1"},
            )

        if resp.status_code != 200:
            return f"搜索请求失败: HTTP {resp.status_code}"

        text = resp.text
        results: list[str] = []
        start = 0
        for _ in range(max_results):
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
