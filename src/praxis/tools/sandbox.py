"""沙箱执行环境。

文件系统访问限制（白名单路径）、Shell 命令超时限制、网络出站规则。
"""

from pathlib import Path

from praxis.config.schemas import ToolsConfig
from praxis.exceptions import SandboxViolationError


class Sandbox:
    """沙箱安全策略执行器。"""

    def __init__(self, config: ToolsConfig) -> None:
        self._allowed_paths = [
            Path(p).resolve() for p in config.allowed_paths
        ]
        self._shell_timeout = config.shell_timeout
        self._default_timeout = config.default_timeout
        self._network_allowed = config.network_allowed

    @property
    def shell_timeout(self) -> float:
        """Shell 命令超时限制（秒）。"""
        return self._shell_timeout

    @property
    def default_timeout(self) -> float:
        """默认工具执行超时（秒）。"""
        return self._default_timeout

    @property
    def network_allowed(self) -> bool:
        """是否允许网络出站访问。"""
        return self._network_allowed

    def check_path(self, path: str | Path) -> Path:
        """检查文件路径是否在白名单内。

        如果白名单为空（未配置），允许所有路径。

        Args:
            path: 待检查的文件路径。

        Returns:
            解析后的绝对路径。

        Raises:
            SandboxViolationError: 路径不在白名单内。
        """
        resolved = Path(path).resolve()

        if not self._allowed_paths:
            return resolved

        for allowed in self._allowed_paths:
            try:
                resolved.relative_to(allowed)
                return resolved
            except ValueError:
                continue

        raise SandboxViolationError(
            f"路径 '{resolved}' 不在沙箱白名单内",
            details={
                "path": str(resolved),
                "allowed_paths": [str(p) for p in self._allowed_paths],
            },
        )

    def check_network(self) -> None:
        """检查是否允许网络访问。

        Raises:
            SandboxViolationError: 网络访问被禁止。
        """
        if not self._network_allowed:
            raise SandboxViolationError(
                "网络出站访问被沙箱策略禁止",
                details={"network_allowed": False},
            )
