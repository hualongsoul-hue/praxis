"""Shared validation for HTTP targets reached by Praxis components."""

import asyncio
import ipaddress
import socket
from urllib.parse import urlsplit

HTTP_REDIRECT_STATUS_CODES = frozenset({301, 302, 303, 307, 308})


async def validate_http_url(url: str, allow_private_networks: bool) -> str:
    """Validate an HTTP URL and every resolved address against SSRF policy."""

    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("仅允许 HTTP/HTTPS URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("URL 不允许包含凭据")
    if not parsed.hostname:
        raise ValueError("URL 缺少主机名")
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as exc:
        raise ValueError("URL 端口无效") from exc
    try:
        addresses = await asyncio.to_thread(
            socket.getaddrinfo,
            parsed.hostname,
            port,
            type=socket.SOCK_STREAM,
        )
    except OSError as exc:
        raise ValueError("URL 主机名无法解析") from exc
    if not allow_private_networks:
        for address in addresses:
            ip = ipaddress.ip_address(address[4][0])
            if not ip.is_global:
                raise ValueError("默认禁止私网、回环、链路本地或保留地址")
    return url
