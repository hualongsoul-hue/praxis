"""Shared validation and DNS-pinned HTTP requests for Praxis components."""

import asyncio
import ipaddress
import socket
import sys
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

import httpx

HTTP_REDIRECT_STATUS_CODES = frozenset({301, 302, 303, 307, 308})
TransportFactory = Callable[[], httpx.AsyncBaseTransport]


@dataclass(frozen=True)
class ValidatedHttpTarget:
    """Immutable URL metadata and the exact addresses approved for connection."""

    original_url: str
    scheme: str
    hostname: str
    port: int
    host_header: str
    addresses: tuple[str, ...]


@dataclass(frozen=True)
class HttpFetchResult:
    """Bounded response returned by the shared network policy."""

    status_code: int
    headers: dict[str, str]
    body: bytes
    final_url: str
    truncated: bool = False


async def validate_http_url(
    url: str,
    allow_private_networks: bool,
) -> ValidatedHttpTarget:
    """Resolve and freeze every address accepted for one HTTP request hop."""

    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("仅允许 HTTP/HTTPS URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("URL 不允许包含凭据")
    if not parsed.hostname:
        raise ValueError("URL 缺少主机名")
    try:
        explicit_port = parsed.port is not None
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError:
        raise ValueError("URL 端口无效") from None
    try:
        resolved = await asyncio.to_thread(
            socket.getaddrinfo,
            parsed.hostname,
            port,
            type=socket.SOCK_STREAM,
        )
    except OSError:
        raise ValueError("URL 主机名无法解析") from None

    addresses: list[str] = []
    for address in resolved:
        raw_address = str(address[4][0])
        try:
            ip = ipaddress.ip_address(raw_address.split("%", maxsplit=1)[0])
        except ValueError:
            raise ValueError("URL 主机名解析结果无效") from None
        if not allow_private_networks and not ip.is_global:
            raise ValueError("默认禁止私网、回环、链路本地或保留地址")
        normalized = str(ip)
        if normalized not in addresses:
            addresses.append(normalized)
    if not addresses:
        raise ValueError("URL 主机名未解析到可用地址")

    hostname_display = (
        f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    )
    host_header = f"{hostname_display}:{port}" if explicit_port else hostname_display
    return ValidatedHttpTarget(
        original_url=url,
        scheme=parsed.scheme,
        hostname=parsed.hostname,
        port=port,
        host_header=host_header,
        addresses=tuple(addresses),
    )


def build_pinned_http_request(
    client: httpx.AsyncClient,
    target: ValidatedHttpTarget,
    address: str,
) -> httpx.Request:
    """Build a request whose TCP host is pinned while HTTP Host and TLS SNI stay original."""

    original_url = httpx.URL(target.original_url)
    pinned_url = original_url.copy_with(host=address, raw_path=b"/redacted")
    return client.build_request(
        "GET",
        pinned_url,
        headers={"Host": target.host_header},
        extensions={
            "sni_hostname": target.hostname,
            "target": original_url.raw_path,
        },
    )


class TargetOverrideTransport(httpx.AsyncBaseTransport):
    """Use a private wire target while keeping the client-visible URL redacted."""

    def __init__(self, transport: httpx.AsyncBaseTransport) -> None:
        self.transport = transport

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Forward a copied request with the validated path and query restored."""

        wire_target = request.extensions.get("target")
        if not isinstance(wire_target, bytes):
            raise httpx.TransportError("安全 HTTP 请求缺少有效目标")
        wire_request = httpx.Request(
            request.method,
            request.url.copy_with(raw_path=wire_target),
            headers=request.headers,
            stream=request.stream,
            extensions=dict(request.extensions),
        )
        return await self.transport.handle_async_request(wire_request)

    async def aclose(self) -> None:
        """Close the isolated inner connection pool."""

        await self.transport.aclose()


def create_http_transport() -> httpx.AsyncBaseTransport:
    """Create one environment-independent HTTP/1.1 connection pool."""

    return httpx.AsyncHTTPTransport(
        trust_env=False,
        http1=True,
        http2=False,
    )


async def close_http_client(client: httpx.AsyncClient) -> bool:
    """Close a client and report failures without retaining their exceptions."""

    close_failed = False
    try:
        await client.aclose()
    except Exception:
        close_failed = True
    return close_failed


async def close_http_resources(
    response: httpx.Response,
    client: httpx.AsyncClient,
) -> bool:
    """Attempt both response and client cleanup without leaking close failures."""

    close_failed = False
    try:
        await response.aclose()
    except Exception:
        close_failed = True
    if await close_http_client(client):
        close_failed = True
    return close_failed


async def send_pinned_http_request(
    target: ValidatedHttpTarget,
    transport_factory: TransportFactory | None = None,
    *,
    timeout: float = 30.0,
) -> tuple[httpx.Response, httpx.AsyncClient]:
    """Try validated addresses in separate pools and return the live response/client."""

    create_transport = transport_factory or create_http_transport
    for address in target.addresses:
        transport = TargetOverrideTransport(create_transport())
        client = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=False,
            trust_env=False,
            http1=True,
            http2=False,
            transport=transport,
        )
        request = build_pinned_http_request(client, target, address)
        response: httpx.Response | None = None
        try:
            response = await client.send(
                request,
                stream=True,
                follow_redirects=False,
            )
        except (httpx.HTTPError, OSError):
            pass
        if response is None:
            await close_http_client(client)
            continue
        return response, client
    raise ValueError("安全 HTTP 连接失败") from None


class NetworkPolicy:
    """Validate, DNS-pin, redirect-check, and bound every HTTP request hop."""

    def __init__(
        self,
        allow_private_networks: bool = False,
        max_response_bytes: int = 1_000_000,
        max_redirects: int = 5,
    ) -> None:
        if max_response_bytes < 1:
            raise ValueError("HTTP 响应字节上限必须大于零")
        if max_redirects < 0:
            raise ValueError("HTTP 重定向上限不能为负数")
        self.allow_private_networks = allow_private_networks
        self.max_response_bytes = max_response_bytes
        self.max_redirects = max_redirects

    async def validate_url(self, url: str) -> ValidatedHttpTarget:
        """Validate and resolve one URL using this policy."""
        return await validate_http_url(url, self.allow_private_networks)

    async def fetch(
        self,
        url: str,
        transport_factory: TransportFactory | None = None,
        *,
        max_bytes: int | None = None,
        timeout: float = 30.0,
    ) -> HttpFetchResult:
        """Fetch a response with per-hop validation and a hard streamed byte bound."""
        requested_limit = self.max_response_bytes if max_bytes is None else max_bytes
        if requested_limit < 1:
            raise ValueError("HTTP 响应字节上限必须大于零")
        byte_limit = min(requested_limit, self.max_response_bytes)
        target = await self.validate_url(url)

        for redirect_attempt in range(self.max_redirects + 1):
            response, client = await send_pinned_http_request(
                target,
                transport_factory,
                timeout=timeout,
            )
            try:
                if response.status_code in HTTP_REDIRECT_STATUS_CODES:
                    location = response.headers.get("location")
                    if not location:
                        raise ValueError("重定向响应缺少 Location")
                    if redirect_attempt == self.max_redirects:
                        raise ValueError("重定向次数超过上限")
                    target = await self.validate_url(urljoin(target.original_url, location))
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
                    raise ValueError("HTTP 响应读取失败") from None
                return HttpFetchResult(
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    body=bytes(body),
                    final_url=target.original_url,
                    truncated=truncated,
                )
            finally:
                active_error = sys.exc_info()[0] is not None
                close_failed = await close_http_resources(response, client)
                if close_failed and not active_error:
                    raise ValueError("HTTP 响应关闭失败") from None
        raise ValueError("重定向次数超过上限")
