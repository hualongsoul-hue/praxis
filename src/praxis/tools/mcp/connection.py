"""MCP server connection management and supervised reconnection."""

import asyncio
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime

from mcp import ClientSession

from praxis.lifecycle import TaskSupervisor
from praxis.models.mcp import (
    MCPServerCapabilities,
    MCPServerConfig,
    MCPServerStatus,
)
from praxis.telemetry.logger import get_logger
from praxis.tools.mcp.prompts import MCPPromptsBridge
from praxis.tools.mcp.resources import MCPResourcesBridge
from praxis.tools.mcp.roots import RootsManager
from praxis.tools.mcp.tools import MCPToolsBridge

log = get_logger("tools.mcp.connection")

MCPTransportFactory = Callable[[], AbstractAsyncContextManager[ClientSession]]
MCPConnectedCallback = Callable[[], None]


class MCPServerConnection:
    """Live state and probe metadata for one configured MCP server."""

    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config
        self.status: MCPServerStatus = MCPServerStatus.DISCONNECTED
        self.capabilities = MCPServerCapabilities()
        self.session: ClientSession | None = None
        self.reconnect_count = 0
        self.last_error = ""
        self.last_probe_at: datetime | None = None
        self.reconnected_event = asyncio.Event()


class MCPConnectionManager:
    """Own one same-task transport supervisor for every MCP server."""

    def __init__(
        self,
        tools_bridge: MCPToolsBridge,
        resources_bridge: MCPResourcesBridge,
        prompts_bridge: MCPPromptsBridge,
        roots_mgr: RootsManager,
        task_supervisor: TaskSupervisor | None = None,
    ) -> None:
        self.tools_bridge = tools_bridge
        self.resources_bridge = resources_bridge
        self.prompts_bridge = prompts_bridge
        self.roots_mgr = roots_mgr
        self.task_supervisor = task_supervisor
        self.connections: dict[str, MCPServerConnection] = {}
        self.supervision_tasks: dict[str, asyncio.Task[None]] = {}
        self.initial_results: dict[str, asyncio.Future[None]] = {}
        self.closed = False

    def ensure_connection(self, config: MCPServerConfig) -> MCPServerConnection:
        connection = self.connections.get(config.name)
        if connection is None:
            connection = MCPServerConnection(config)
            self.connections[config.name] = connection
        else:
            connection.config = config
        return connection

    async def connect_server(
        self,
        config: MCPServerConfig,
        session: ClientSession,
    ) -> MCPServerCapabilities:
        """Negotiate capabilities and atomically publish a usable session."""
        connection = self.ensure_connection(config)
        connection.status = MCPServerStatus.CONNECTING

        capabilities = session.get_server_capabilities()
        negotiated = MCPServerCapabilities(
            tools=capabilities.tools is not None if capabilities else False,
            resources=capabilities.resources is not None if capabilities else False,
            prompts=capabilities.prompts is not None if capabilities else False,
            sampling=False,
        )
        if negotiated.tools:
            await self.tools_bridge.discover_tools(config.name, session)
        if negotiated.resources:
            self.resources_bridge.register_session(config.name, session)
        if negotiated.prompts:
            self.prompts_bridge.register_session(config.name, session)
        self.roots_mgr.register_session(config.name, session)

        connection.capabilities = negotiated
        connection.session = session
        connection.status = MCPServerStatus.CONNECTED
        connection.last_error = ""
        connection.last_probe_at = datetime.now(UTC)
        log.info(
            "MCP Server 已连接",
            server=config.name,
            capabilities=negotiated.model_dump(),
        )
        return negotiated

    def disconnect_server(self, server_name: str, *, remove: bool = False) -> None:
        """Unpublish a session while retaining its status history by default."""
        self.tools_bridge.disconnect_server(server_name)
        self.resources_bridge.disconnect_server(server_name)
        self.prompts_bridge.disconnect_server(server_name)
        self.roots_mgr.disconnect_server(server_name)

        connection = self.connections.get(server_name)
        if connection is not None:
            connection.session = None
            connection.status = MCPServerStatus.DISCONNECTED
            if remove:
                self.connections.pop(server_name, None)
        log.info("MCP Server 已断开", server=server_name)

    def start_server(
        self,
        config: MCPServerConfig,
        transport_factory: MCPTransportFactory,
        on_connected: MCPConnectedCallback,
    ) -> asyncio.Future[None]:
        """Start supervision and resolve after the initial retry window."""
        if self.closed:
            raise RuntimeError("MCP Connection Manager 已关闭")
        if config.name in self.supervision_tasks:
            raise ValueError(f"MCP Server 已启动: {config.name}")
        initial_result = asyncio.get_running_loop().create_future()
        self.initial_results[config.name] = initial_result
        coroutine = self.supervise_server(
            config,
            transport_factory,
            on_connected,
            initial_result,
        )
        task = (
            self.task_supervisor.create_task(
                coroutine,
                name=f"praxis-mcp-{config.name}",
            )
            if self.task_supervisor is not None
            else asyncio.create_task(coroutine, name=f"praxis-mcp-{config.name}")
        )
        self.supervision_tasks[config.name] = task
        return initial_result

    async def supervise_server(
        self,
        config: MCPServerConfig,
        transport_factory: MCPTransportFactory,
        on_connected: MCPConnectedCallback,
        initial_result: asyncio.Future[None],
    ) -> None:
        """Reconnect forever with bounded probes until the manager is closed."""
        failure_count = 0
        ever_connected = False
        connection = self.ensure_connection(config)
        try:
            while not self.closed:
                connection.status = (
                    MCPServerStatus.RECONNECTING
                    if ever_connected or failure_count
                    else MCPServerStatus.CONNECTING
                )
                connected_this_attempt = False
                try:
                    async with transport_factory() as session:
                        await self.connect_server(config, session)
                        connected_this_attempt = True
                        if ever_connected:
                            connection.reconnect_count += 1
                            connection.reconnected_event.set()
                        ever_connected = True
                        failure_count = 0
                        if not initial_result.done():
                            initial_result.set_result(None)
                        on_connected()
                        try:
                            while not self.closed:
                                await asyncio.sleep(config.health_check_interval)
                                async with asyncio.timeout(config.timeout):
                                    await session.send_ping()
                                connection.last_probe_at = datetime.now(UTC)
                        finally:
                            self.disconnect_server(config.name)
                except asyncio.CancelledError:
                    if not initial_result.done():
                        initial_result.cancel()
                    raise
                except Exception as exc:
                    if connected_this_attempt:
                        self.disconnect_server(config.name)
                    failure_count += 1
                    connection.last_error = type(exc).__name__
                    connection.status = (
                        MCPServerStatus.RECONNECTING
                        if ever_connected
                        else MCPServerStatus.FAILED
                    )
                    log.warning(
                        "MCP Server 连接或探测失败，准备重连",
                        server=config.name,
                        attempt=failure_count,
                        error_type=type(exc).__name__,
                    )
                    if (
                        not initial_result.done()
                        and failure_count > config.reconnect_attempts
                    ):
                        initial_result.set_result(None)

                if self.closed:
                    break
                exponent = min(max(failure_count - 1, 0), 16)
                delay = min(
                    config.reconnect_max_delay,
                    max(0.01, config.reconnect_delay * (2 ** exponent)),
                )
                await asyncio.sleep(delay)
        finally:
            self.disconnect_server(config.name)
            if not initial_result.done():
                initial_result.cancel()

    async def close(self) -> None:
        """Cancel every supervisor and wait for same-task transport cleanup."""
        if self.closed:
            return
        self.closed = True
        tasks = list(self.supervision_tasks.values())
        for task in tasks:
            task.cancel()
        failures: list[BaseException] = []
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            failures = [
                result
                for result in results
                if isinstance(result, BaseException)
                and not isinstance(result, asyncio.CancelledError)
            ]
        for server_name in list(self.connections):
            self.disconnect_server(server_name, remove=True)
        self.supervision_tasks.clear()
        self.initial_results.clear()
        if failures:
            raise BaseExceptionGroup("MCP 传输关闭失败", failures)

    def get_server_status(self, server_name: str) -> MCPServerStatus:
        connection = self.connections.get(server_name)
        if connection is None:
            return MCPServerStatus.DISCONNECTED
        return connection.status

    def list_connected_servers(self) -> list[str]:
        return [
            name
            for name, connection in self.connections.items()
            if connection.status is MCPServerStatus.CONNECTED
        ]

    def get_server_capabilities(
        self,
        server_name: str,
    ) -> MCPServerCapabilities | None:
        connection = self.connections.get(server_name)
        if connection is None:
            return None
        return connection.capabilities
