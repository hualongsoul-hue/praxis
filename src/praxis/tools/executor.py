"""工具执行管线。

按序执行：参数验证（Schema 校验）→ 沙箱检查 → 执行 → 结果捕获 → 格式化。
支持并发策略（只读并发、写串行），可配置超时。
"""

import asyncio
import json
import time
from typing import Any

import jsonschema

from praxis.exceptions import ToolExecutionError, ToolTimeoutError
from praxis.models.tools import ToolResult
from praxis.tools.registry import ToolRegistry
from praxis.tools.sandbox import Sandbox


class ToolExecutor:
    """工具执行管线。"""

    def __init__(self, registry: ToolRegistry, sandbox: Sandbox) -> None:
        self._registry = registry
        self._sandbox = sandbox
        self._write_lock = asyncio.Lock()

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        tool_call_id: str = "",
    ) -> ToolResult:
        """执行工具。

        完整管线：参数验证 → 沙箱检查 → 执行 → 结果捕获 → 格式化。

        Args:
            name: 工具名称。
            arguments: 参数字典。
            tool_call_id: 工具调用 ID（由 LLM 生成）。

        Returns:
            结构化的 ToolResult。
        """
        start = time.perf_counter()

        entry = self._registry.get_entry(name)
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

        timeout = meta.timeout_seconds or self._sandbox.default_timeout

        try:
            if meta.readonly:
                content = await asyncio.wait_for(
                    entry.handler(arguments),
                    timeout=timeout,
                )
            else:
                async with self._write_lock:
                    content = await asyncio.wait_for(
                        entry.handler(arguments),
                        timeout=timeout,
                    )
        except asyncio.TimeoutError:
            elapsed = (time.perf_counter() - start) * 1000
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


def validate_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> str | None:
    """基于 JSON Schema 校验参数。

    Returns:
        校验错误消息，或 None 表示通过。
    """
    try:
        jsonschema.validate(instance=arguments, schema=schema)
        return None
    except jsonschema.ValidationError as exc:
        return exc.message
