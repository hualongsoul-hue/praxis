"""跨 Windows/Linux 的受控子进程执行器。"""

import asyncio
import os
import signal
import subprocess
from pathlib import Path

from praxis.exceptions import ToolTimeoutError


class ProcessResult:
    def __init__(self, returncode: int, stdout: str, stderr: str) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class ProcessRunner:
    """执行 Shell 命令并在超时时终止整个进程树。"""

    async def run_shell(
        self,
        command: str,
        *,
        cwd: Path,
        timeout: float,
        environment: dict[str, str],
    ) -> ProcessResult:
        if os.name == "nt":
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd),
                env=environment,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
        else:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd),
                env=environment,
                start_new_session=True,
            )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )
        except TimeoutError:
            await self._terminate_tree(process)
            raise ToolTimeoutError(
                f"命令执行超时（{timeout}s）",
                details={"timeout": timeout},
            ) from None
        except asyncio.CancelledError:
            await asyncio.shield(self._terminate_tree(process))
            raise
        return ProcessResult(
            process.returncode or 0,
            stdout_bytes.decode("utf-8", errors="replace").rstrip(),
            stderr_bytes.decode("utf-8", errors="replace").rstrip(),
        )

    @staticmethod
    async def _terminate_tree(process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        if os.name == "nt":
            killer = await asyncio.create_subprocess_exec(
                "taskkill",
                "/PID",
                str(process.pid),
                "/T",
                "/F",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            await killer.wait()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        await process.wait()
