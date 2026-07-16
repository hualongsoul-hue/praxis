"""Shared validation and DNS-pinned HTTP requests for Praxis components."""

import asyncio
import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

HTTP_REDIRECT_STATUS_CODES = frozenset({301, 302, 303, 307, 308})


@dataclass(frozen=True)
class ValidatedHttpTarget:
    """Immutable URL metadata and the exact addresses approved for connection."""

    original_url: str
    scheme: str
    hostname: str
    port: int
    host_header: str
    addresses: tuple[str, ...]


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

    pinned_url = httpx.URL(target.original_url).copy_with(host=address)
    return client.build_request(
        "GET",
        pinned_url,
        headers={"Host": target.host_header},
        extensions={"sni_hostname": target.hostname},
    )


async def send_pinned_http_request(
    client: httpx.AsyncClient,
    target: ValidatedHttpTarget,
) -> httpx.Response:
    """Try only validated addresses and never let HTTPX follow redirects."""

    for address in target.addresses:
        request = build_pinned_http_request(client, target, address)
        try:
            return await client.send(
                request,
                stream=True,
                follow_redirects=False,
            )
        except (httpx.HTTPError, OSError):
            continue
    raise ValueError("安全 HTTP 连接失败") from None
