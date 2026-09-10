"""工具执行管线。

按序执行：参数验证（Schema 校验）→ 执行 → 结果捕获 → 格式化。
支持并发策略（只读并发、写串行），可配置超时。

注意：沙箱（文件白名单/网络出站/Shell 超时）由各工具处理器在执行时
通过共享的 ToolPolicy 实例自行强制，执行管线本身不做通用路径/网络拦截
（参数无统一的路径语义，无法在此层泛化检查）。
"""

import asyncio
import time
from collections.abc import Mapping
from typing import Any

import jsonschema

from praxis.exceptions import ToolExecutionError, ToolTimeoutError
from praxis.models.tools import ToolResult
from praxis.resources import ResourceController
from praxis.telemetry.tracing import operation_span
from praxis.tools.policy import ToolPolicy
from praxis.tools.registry import ToolRegistry


class ToolExecutor:
    """工具执行管线。"""

    def __init__(
        self,
        registry: ToolRegistry,
        sandbox: ToolPolicy,
        resources: ResourceController | None = None,
    ) -> None:
        self.registry = registry
        self.sandbox = sandbox
        self.resources = resources
        self.write_lock = asyncio.Lock()
        self.read_semaphore = asyncio.Semaphore(sandbox.max_concurrent_readonly)

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        tool_call_id: str = "",
    ) -> ToolResult:
        """执行工具。

        完整管线：参数验证 → 执行 → 结果捕获 → 格式化。
        （沙箱由工具处理器自行强制，见模块文档。）

        Args:
            name: 工具名称。
            arguments: 参数字典。
            tool_call_id: 工具调用 ID（由 LLM 生成）。

        Returns:
            结构化的 ToolResult。
        """
        start = time.perf_counter()

        entry = self.registry.get_entry(name)
        defn = entry.definition
        meta = defn.metadata

        validation_error = validate_arguments(defn.parameters, arguments)
        if validation_error:
            return ToolResult(
                tool_call_id=tool_call_id,
                success=False,
                content="",
                error=f"参数校验失败: {validation_error}",
                execution_time_ms=(time.perf_counter() - start) * 1000,
            )

        timeout = meta.timeout_seconds or self.sandbox.default_timeout

        try:
            if meta.readonly:
                read_limit = (
                    self.resources.read_lease()
                    if self.resources is not None
                    else self.read_semaphore
                )
                async with read_limit:
                    with operation_span("praxis.tool.execute") as span:
                        span.set_attribute("praxis.tool", name)
                        content = await asyncio.wait_for(entry.handler(arguments), timeout=timeout)
            else:
                write_limit = (
                    self.resources.write_lease(
                        self.resources.write_resource_key(name, arguments)
                    )
                    if self.resources is not None
                    else self.write_lock
                )
                async with write_limit:
                    with operation_span("praxis.tool.execute") as span:
                        span.set_attribute("praxis.tool", name)
                        content = await asyncio.wait_for(entry.handler(arguments), timeout=timeout)
        except TimeoutError:
            raise ToolTimeoutError(
                f"工具 '{name}' 执行超时（{timeout}s）",
                details={"tool": name, "timeout": timeout},
            ) from None
        except ToolExecutionError:
            raise
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            return ToolResult(
                tool_call_id=tool_call_id,
                success=False,
                content="",
                error=f"工具执行异常: {type(exc).__name__}: {exc}",
                error_type=f"{type(exc).__module__}.{type(exc).__qualname__}",
                execution_time_ms=elapsed,
            )

        elapsed = (time.perf_counter() - start) * 1000
        return ToolResult(
            tool_call_id=tool_call_id,
            success=True,
            content=content,
            execution_time_ms=elapsed,
        )


def validate_arguments(
    schema: Mapping[str, Any],
    arguments: Mapping[str, Any],
) -> str | None:
    """基于 JSON Schema 校验参数。

    Returns:
        校验错误消息，或 None 表示通过。
    """
    try:
        jsonschema.validate(instance=arguments, schema=schema)
        return None
    except jsonschema.ValidationError as exc:
        return exc.message
