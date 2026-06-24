"""MCP 传输层。

Stdio 传输（子进程 stdin/stdout）和 Streamable HTTP 传输（SSE），
上层统一 MCPClient 接口。可选注入 sampling/elicitation 回调，
使 MCP Server → 客户端的采样/征询请求被路由到 Praxis 管理器。
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamablehttp_client

from praxis.models.mcp import MCPServerConfig, MCPTransportType
from praxis.telemetry.logger import get_logger

log = get_logger("tools.mcp.transport")


@asynccontextmanager
async def create_stdio_transport(
    config: MCPServerConfig,
    sampling_callback: Any = None,
    elicitation_callback: Any = None,
) -> AsyncGenerator[ClientSession, None]:
    """创建 Stdio 传输连接。

    Args:
        config: MCP 服务器配置。
        sampling_callback: 可选 MCP sampling 回调。
        elicitation_callback: 可选 MCP elicitation 回调。

    Yields:
        已初始化的 ClientSession。
    """
    params = StdioServerParameters(
        command=config.command,
        args=config.args,
        env=config.env or None,
    )
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(
            read_stream,
            write_stream,
            sampling_callback=sampling_callback,
            elicitation_callback=elicitation_callback,
        ) as session:
            await session.initialize()
            log.info("Stdio 连接已建立", server=config.name)
            yield session


@asynccontextmanager
async def create_http_transport(
    config: MCPServerConfig,
    sampling_callback: Any = None,
    elicitation_callback: Any = None,
) -> AsyncGenerator[ClientSession, None]:
    """创建 Streamable HTTP 传输连接。

    Args:
        config: MCP 服务器配置。
        sampling_callback: 可选 MCP sampling 回调。
        elicitation_callback: 可选 MCP elicitation 回调。

    Yields:
        已初始化的 ClientSession。
    """
    async with streamablehttp_client(
        url=config.url,
        headers=config.headers or None,
        timeout=config.timeout,
    ) as (read_stream, write_stream, _get_session_id):
        async with ClientSession(
            read_stream,
            write_stream,
            sampling_callback=sampling_callback,
            elicitation_callback=elicitation_callback,
        ) as session:
            await session.initialize()
            log.info("HTTP 连接已建立", server=config.name, url=config.url)
            yield session


@asynccontextmanager
async def create_transport(
    config: MCPServerConfig,
    sampling_callback: Any = None,
    elicitation_callback: Any = None,
) -> AsyncGenerator[ClientSession, None]:
    """根据配置创建 MCP 传输连接（统一入口）。

    Args:
        config: MCP 服务器配置。
        sampling_callback: 可选 MCP sampling 回调。
        elicitation_callback: 可选 MCP elicitation 回调。

    Yields:
        已初始化的 ClientSession。
    """
    if config.transport == MCPTransportType.STDIO:
        async with create_stdio_transport(
            config, sampling_callback, elicitation_callback,
        ) as session:
            yield session
    elif config.transport == MCPTransportType.HTTP:
        async with create_http_transport(
            config, sampling_callback, elicitation_callback,
        ) as session:
            yield session
    else:
        raise ValueError(f"不支持的传输类型: {config.transport}")
