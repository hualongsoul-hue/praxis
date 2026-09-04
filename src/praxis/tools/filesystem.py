"""Bounded asynchronous file operations shared by built-in tools."""

import asyncio
import os
import tempfile
from pathlib import Path

from praxis.exceptions import ToolPolicyViolationError
from praxis.tools.policy import ToolPolicy


def read_file_bytes_sync(path: Path, policy: ToolPolicy) -> bytes:
    """Read one authorized regular file while enforcing a hard byte limit."""
    checked = policy.check_path(path)
    if not checked.exists():
        raise FileNotFoundError(str(checked))
    if not checked.is_file():
        raise IsADirectoryError(str(checked))
    size = checked.stat().st_size
    if size > policy.max_file_bytes:
        raise ToolPolicyViolationError(
            f"文件超过 {policy.max_file_bytes} 字节大小上限"
        )
    data = checked.read_bytes()
    policy.check_path(checked)
    if len(data) > policy.max_file_bytes:
        raise ToolPolicyViolationError(
            f"文件超过 {policy.max_file_bytes} 字节大小上限"
        )
    return data


async def read_file_text(path: Path, policy: ToolPolicy) -> str:
    """Read and decode UTF-8 outside the event-loop thread."""
    data = await asyncio.to_thread(read_file_bytes_sync, path, policy)
    return data.decode("utf-8")


def atomic_write_text_sync(path: Path, content: str, policy: ToolPolicy) -> Path:
    """Write UTF-8 to an authorized path using fsync and atomic replacement."""
    encoded = content.encode("utf-8")
    if len(encoded) > policy.max_file_bytes:
        raise ToolPolicyViolationError(
            f"写入内容超过 {policy.max_file_bytes} 字节大小上限"
        )
    checked = policy.check_path(path)
    checked.parent.mkdir(parents=True, exist_ok=True)
    checked_parent = policy.check_path(checked.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="praxis-write-",
        suffix=".tmp",
        dir=checked_parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if checked.exists():
            os.chmod(temporary_path, checked.stat().st_mode)
        final_path = policy.check_path(path)
        final_parent = policy.check_path(final_path.parent)
        if final_parent != checked_parent:
            raise ToolPolicyViolationError("文件父目录在写入期间发生变化")
        os.replace(temporary_path, final_path)
        return final_path
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass


async def atomic_write_text(path: Path, content: str, policy: ToolPolicy) -> Path:
    """Atomically write text outside the event-loop thread."""
    return await asyncio.to_thread(atomic_write_text_sync, path, content, policy)
