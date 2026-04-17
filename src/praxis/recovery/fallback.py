"""优雅降级。

维护工具降级映射，当首选工具不可用时自动切换到降级替代方案。
降级事件可观测（日志 + 指标）。
"""

from praxis.telemetry.logger import get_logger
from praxis.telemetry.metrics import emit_metric

log = get_logger("recovery.fallback")


class FallbackRegistry:
    """工具降级映射表。"""

    def __init__(self) -> None:
        self.mappings: dict[str, str] = {}

    def register(self, tool_name: str, fallback_name: str) -> None:
        """注册降级映射。

        Args:
            tool_name: 首选工具名。
            fallback_name: 降级替代工具名。
        """
        self.mappings[tool_name] = fallback_name

    def unregister(self, tool_name: str) -> bool:
        """移除降级映射。返回是否成功。"""
        if tool_name in self.mappings:
            del self.mappings[tool_name]
            return True
        return False

    def get_fallback(self, tool_name: str) -> str | None:
        """获取工具的降级替代方案。

        Args:
            tool_name: 首选工具名。

        Returns:
            降级替代工具名，或 None。
        """
        fallback = self.mappings.get(tool_name)
        if fallback:
            log.info(
                "工具降级触发",
                tool=tool_name,
                fallback=fallback,
            )
            emit_metric(
                "recovery_fallback_triggered",
                1.0,
                {"tool": tool_name, "fallback": fallback},
                "counter",
            )
        return fallback

    def load_mappings(self, mappings: dict[str, str]) -> None:
        """批量加载降级映射（如从配置文件）。"""
        self.mappings.update(mappings)

    def list_mappings(self) -> dict[str, str]:
        """返回所有降级映射的副本。"""
        return dict(self.mappings)
