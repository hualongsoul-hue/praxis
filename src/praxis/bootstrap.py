"""Shared telemetry-aware lifecycle for CLI and example applications."""

from collections.abc import Callable
from types import TracebackType
from typing import Any

from praxis.config.settings import PraxisConfig
from praxis.runtime import PraxisRuntime
from praxis.telemetry import MetricsExporter, configure_cli_telemetry
from praxis.telemetry.tracing import TracingLifecycle

RuntimeFactory = Callable[..., PraxisRuntime]


class PraxisCliApplication:
    """Own CLI logging, metrics export, tracing, and one ``PraxisRuntime``."""

    def __init__(
        self,
        config: PraxisConfig,
        *,
        runtime_factory: RuntimeFactory = PraxisRuntime,
        runtime_options: dict[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.runtime_factory = runtime_factory
        self.runtime_options = dict(runtime_options or {})
        self.runtime: PraxisRuntime | None = None
        self.metrics_exporter: MetricsExporter | None = None
        self.tracing: TracingLifecycle | None = None

    async def __aenter__(self) -> PraxisRuntime:
        self.tracing = configure_cli_telemetry(self.config.telemetry)
        runtime = self.runtime_factory(self.config, **self.runtime_options)
        self.runtime = runtime
        try:
            await runtime.start()
            exporter = MetricsExporter(runtime.metrics, self.config.telemetry)
            exporter.start()
            self.metrics_exporter = exporter
        except BaseException as start_error:
            failures = await self.close_resources()
            for failure in failures:
                start_error.add_note(
                    f"CLI 启动回滚失败: {type(failure).__name__}: {failure}"
                )
            raise
        return runtime

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        failures = await self.close_resources()
        if not failures:
            return
        if exc is not None:
            for failure in failures:
                exc.add_note(f"CLI 关闭失败: {type(failure).__name__}: {failure}")
            return
        raise BaseExceptionGroup("CLI 资源关闭失败", list(failures))

    async def close_resources(self) -> tuple[BaseException, ...]:
        """Close all owned resources in dependency order without short-circuiting."""
        failures: list[BaseException] = []
        if self.runtime is not None:
            try:
                await self.runtime.close()
            except BaseException as error:
                failures.append(error)
            self.runtime = None
        if self.metrics_exporter is not None:
            try:
                self.metrics_exporter.close()
            except BaseException as error:
                failures.append(error)
            self.metrics_exporter = None
        if self.tracing is not None:
            try:
                await self.tracing.close()
            except BaseException as error:
                failures.append(error)
            self.tracing = None
        return tuple(failures)
