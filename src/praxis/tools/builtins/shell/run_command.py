"""内置工具：run_command。

在子进程中执行 Shell 命令，支持超时和工作目录。
"""

import asyncio
from typing import Any

from praxis.models.tools import ToolDefinition, ToolMetadata
from praxis.tools.sandbox import Sandbox

DEFINITION = ToolDefinition(
    name="run_command",
    description="在子进程中执行 Shell 命令。返回 stdout、stderr 和退出码。",
    parameters={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "要执行的 Shell 命令",
            },
            "cwd": {
                "type": "string",
                "description": "工作目录（绝对路径），默认当前目录",
            },
            "timeout": {
                "type": "number",
                "description": "超时秒数，默认使用沙箱配置",
            },
        },
        "required": ["command"],
    },
    metadata=ToolMetadata(
        category="shell",
        permission_level="confirm",
        readonly=False,
        timeout_seconds=120.0,
        tags=["shell", "command"],
    ),
)


def create_handler(sandbox: Sandbox):
    """创建绑定沙箱的处理函数。"""

    async def handle(args: dict[str, Any]) -> str:
        command = args["command"]
        cwd = args.get("cwd")
        timeout = args.get("timeout", sandbox.shell_timeout)

        if cwd:
            sandbox.check_path(cwd)

        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return f"命令超时（{timeout}s）: {command}"

        stdout = stdout_bytes.decode("utf-8", errors="replace").rstrip()
        stderr = stderr_bytes.decode("utf-8", errors="replace").rstrip()

        parts: list[str] = [f"Exit code: {proc.returncode}"]
        if stdout:
            parts.append(f"stdout:\n{stdout}")
        if stderr:
            parts.append(f"stderr:\n{stderr}")
        return "\n".join(parts)

    return handle
