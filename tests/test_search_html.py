"""DDG result parsing must not manufacture successful empty evidence."""

import socket

import httpx
import pytest

from praxis.config.schemas import ToolsConfig
from praxis.exceptions import ToolError
from praxis.tools.builtins.network.web_search import create_handler
from praxis.tools.policy import ToolPolicy


def search(monkeypatch, html, *, limit=4096):
    monkeypatch.setattr(
        socket, "getaddrinfo",
        lambda host, port, **kw: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))],
    )
    return create_handler(
        ToolPolicy(ToolsConfig(network_allowed=True, network_max_response_bytes=limit)),
        lambda: httpx.MockTransport(lambda req: httpx.Response(200, text=html)),
    )


@pytest.mark.parametrize("anchor", [
    '<a class="other result__a" href="https://example.com/a?a=1&amp;b=2">A <b>result</b></a>',
    "<a href='https://example.com/a?a=1&amp;b=2' class='result__a other'>A <b>result</b></a>",
])
async def test_search_parses_html_attributes_and_nested_text(monkeypatch, anchor):
    result = await search(monkeypatch, anchor)({"query": "test"})
    assert "A result" in result
    assert "https://example.com/a?a=1&b=2" in result


@pytest.mark.parametrize("html", [
    '<a href="https://wrong.example">other</a><a class="result__a">missing</a>',
    '<form id="challenge-form">Please complete the captcha</form>',
    '<html>changed provider markup</html>',
])
async def test_search_rejects_unrecognizable_or_challenged_page(monkeypatch, html):
    with pytest.raises(ToolError):
        await search(monkeypatch, html)({"query": "test"})


async def test_search_empty_requires_provider_empty_marker(monkeypatch):
    result = await search(monkeypatch, '<div class="no-results">No results found</div>')({"query": "test"})
    assert "未找到" in result


async def test_search_truncation_is_not_empty_or_complete_results(monkeypatch):
    html = '<a href="https://example.com" class="result__a">Result</a>' + 'x' * 1024
    with pytest.raises(ToolError):
        await search(monkeypatch, html, limit=128)({"query": "test"})
