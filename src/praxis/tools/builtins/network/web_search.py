"""内置工具：web_search。

执行网络搜索。需要外部搜索 API 配置。
当前实现使用 DuckDuckGo HTML 搜索作为无 API Key 的回退方案。
"""

from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlencode

from praxis.exceptions import ToolExecutionError
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


class SearchPageParser(HTMLParser):
    """Parse each anchor independently; unknown/challenge pages fail closed."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[tuple[str, str]] = []
        self.href: str | None = None
        self.title: list[str] = []
        self.empty = False
        self.challenge = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = (attributes.get("class") or "").split()
        if "no-results" in classes or "result--no-result" in classes:
            self.empty = True
        identity = attributes.get("id") or ""
        if identity in {"challenge-form", "img-form"} or "anomaly-modal" in classes:
            self.challenge = True
        if tag == "a":
            self.href = attributes.get("href") if "result__a" in classes else None
            self.title = []

    def handle_data(self, data: str) -> None:
        if self.href is not None:
            self.title.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            title = "".join(self.title).strip()
            if self.href and title:
                self.results.append((title, self.href))
            self.href = None
            self.title = []


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
            raise ToolExecutionError(f"搜索请求失败: HTTP {response.status_code}")

        if response.truncated:
            raise ToolExecutionError("搜索结果页超过响应字节上限")

        text = response.body.decode("utf-8", errors="replace")
        parser = SearchPageParser()
        parser.feed(text)
        parser.close()
        if parser.challenge:
            raise ToolExecutionError("搜索提供方要求验证")
        if not parser.results and parser.empty:
            return f"未找到 '{query}' 的搜索结果"
        if not parser.results:
            raise ToolExecutionError("无法识别搜索结果页")
        return "\n\n".join(f"- {title}\n  {url}" for title, url in parser.results[:max_results])

    return handle
