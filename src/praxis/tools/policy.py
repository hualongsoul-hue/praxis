"""文件、网络和 Shell 工具的显式授权策略。"""

import os
from collections.abc import Mapping
from pathlib import Path

from praxis.config.schemas import ToolsConfig
from praxis.exceptions import ToolPolicyViolationError
from praxis.network import ValidatedHttpTarget, validate_http_url

SAFE_ENVIRONMENT = frozenset({
    "COMSPEC",
    "LANG",
    "LC_ALL",
    "PATH",
    "PATHEXT",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "WINDIR",
})


class ToolPolicy:
    """描述进程内工具允许做什么；它不是 OS 级安全沙箱。"""

    def __init__(self, config: ToolsConfig) -> None:
        self.allowed_paths = tuple(Path(item).expanduser().resolve() for item in config.allowed_paths)
        self.configured_shell_timeout = config.shell_timeout
        self.configured_default_timeout = config.default_timeout
        self.shell_enabled = config.shell_enabled
        self.network_access_allowed = config.network_allowed
        self.allow_private_networks = config.allow_private_networks
        self.configured_network_max_response_bytes = config.network_max_response_bytes
        self.environment_allowlist = frozenset(config.shell_environment_allowlist)

    @property
    def shell_timeout(self) -> float:
        return self.configured_shell_timeout

    @property
    def default_timeout(self) -> float:
        return self.configured_default_timeout

    @property
    def network_allowed(self) -> bool:
        return self.network_access_allowed

    @property
    def network_max_response_bytes(self) -> int:
        return self.configured_network_max_response_bytes

    @property
    def default_working_directory(self) -> Path:
        if not self.allowed_paths:
            raise ToolPolicyViolationError("未配置 Shell 授权工作目录")
        return self.allowed_paths[0]

    def check_path(self, path: str | Path) -> Path:
        resolved = Path(path).expanduser().resolve()
        if not self.allowed_paths:
            raise ToolPolicyViolationError(
                "文件访问被拒绝：未配置授权根目录",
                details={"path": str(resolved), "allowed_paths": []},
            )
        for allowed in self.allowed_paths:
            if resolved == allowed or allowed in resolved.parents:
                return resolved
        raise ToolPolicyViolationError(
            f"路径 '{resolved}' 不在工具授权根目录内",
            details={
                "path": str(resolved),
                "allowed_paths": [str(item) for item in self.allowed_paths],
            },
        )

    def check_shell(self) -> None:
        if not self.shell_enabled:
            raise ToolPolicyViolationError("Shell 工具默认禁用，必须显式启用并审批")

    def shell_environment(
        self,
        environ: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        source = os.environ if environ is None else environ
        allowed = SAFE_ENVIRONMENT | self.environment_allowlist
        return {key: value for key, value in source.items() if key.upper() in allowed}

    def check_network(self) -> None:
        if not self.network_access_allowed:
            raise ToolPolicyViolationError(
                "网络出站访问被工具策略禁止",
                details={"network_allowed": False},
            )

    async def check_url(self, url: str) -> ValidatedHttpTarget:
        self.check_network()
        try:
            return await validate_http_url(url, self.allow_private_networks)
        except ValueError as exc:
            raise ToolPolicyViolationError(str(exc)) from None
