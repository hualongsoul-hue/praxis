"""内置 Shell 工具；仅适用于可信命令，非 OS 沙箱。"""

from typing import Any

from praxis.models.tools import ToolDefinition, ToolMetadata
from praxis.tools.policy import ToolPolicy
from praxis.tools.process import ProcessRunner

DEFINITION = ToolDefinition(
    name="run_command",
    description=(
        "在受限工作目录和环境变量下执行可信 Shell 命令。"
        "不可信命令必须放入容器或操作系统沙箱。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string", "minLength": 1},
            "cwd": {"type": "string"},
            "timeout": {"type": "number", "exclusiveMinimum": 0},
        },
        "required": ["command"],
    },
    metadata=ToolMetadata(
        category="shell",
        permission_level="confirm",
        readonly=False,
        idempotent=False,
        timeout_seconds=120.0,
        tags=["shell", "command"],
    ),
)


def create_handler(policy: ToolPolicy, runner: ProcessRunner | None = None):
    process_runner = runner or ProcessRunner()

    async def handle(args: dict[str, Any]) -> str:
        policy.check_shell()
        cwd_arg = args.get("cwd")
        cwd = policy.check_path(cwd_arg) if cwd_arg else policy.default_working_directory
        timeout = min(float(args.get("timeout", policy.shell_timeout)), policy.shell_timeout)
        result = await process_runner.run_shell(
            str(args["command"]),
            cwd=cwd,
            timeout=timeout,
            environment=policy.shell_environment(),
        )
        parts = [f"Exit code: {result.returncode}"]
        if result.stdout:
            parts.append(f"stdout:\n{result.stdout}")
        if result.stderr:
            parts.append(f"stderr:\n{result.stderr}")
        return "\n".join(parts)

    return handle
