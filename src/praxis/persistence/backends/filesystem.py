"""文件系统存储后端。

简单场景，数据以 JSON 文件存储，人类可读。
目录结构：``{base_path}/{namespace}/{key}.json``。
"""

import asyncio
import base64
import os
import re
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath

from praxis.exceptions import PersistenceError


class FilesystemBackend:
    """文件系统异步存储后端。"""

    def __init__(self, base_path: str) -> None:
        self.base_path = Path(base_path).resolve()

    @staticmethod
    def validate_identifier(value: str, label: str) -> str:
        if not value or "\x00" in value:
            raise PersistenceError(f"{label}不能为空或包含空字节")
        if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
            raise PersistenceError(f"{label}不能是绝对路径")
        if any(part in {".", ".."} for part in re.split(r"[\\/]", value)):
            raise PersistenceError(f"{label}不能包含路径遍历片段")
        return value

    @staticmethod
    def encode_identifier(value: str) -> str:
        return base64.urlsafe_b64encode(value.encode("utf-8")).decode("ascii").rstrip("=")

    @staticmethod
    def decode_identifier(value: str) -> str:
        padding = "=" * (-len(value) % 4)
        try:
            return base64.urlsafe_b64decode(value + padding).decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise PersistenceError("文件存储中存在无效键编码") from exc

    def namespace_directory(self, namespace: str) -> Path:
        validated = self.validate_identifier(namespace, "命名空间")
        return self.base_path / self.encode_identifier(validated)

    def storage_path(self, namespace: str, key: str) -> Path:
        validated_key = self.validate_identifier(key, "键")
        return self.namespace_directory(namespace) / f"{self.encode_identifier(validated_key)}.json"

    async def save(self, namespace: str, key: str, data: bytes) -> None:
        path = self.storage_path(namespace, key)
        await asyncio.to_thread(self.write_bytes, path, data)

    async def save_if_absent(self, namespace: str, key: str, data: bytes) -> bool:
        path = self.storage_path(namespace, key)
        return await asyncio.to_thread(self.write_bytes_if_absent, path, data)

    async def load(self, namespace: str, key: str) -> bytes | None:
        path = self.storage_path(namespace, key)
        return await asyncio.to_thread(self.read_bytes, path)

    async def delete(self, namespace: str, key: str) -> None:
        path = self.storage_path(namespace, key)
        await asyncio.to_thread(self.unlink_path, path)

    async def list_keys(
        self, namespace: str, prefix: str | None = None
    ) -> list[str]:
        ns_dir = self.namespace_directory(namespace)
        return await asyncio.to_thread(list_keys, ns_dir, prefix)

    async def clear_namespace(self, namespace: str) -> int:
        ns_dir = self.namespace_directory(namespace)
        return await asyncio.to_thread(clear_namespace, ns_dir)

    async def close(self) -> None:
        pass

    @staticmethod
    def write_bytes(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)
        finally:
            tmp_path = Path(tmp_name)
            if tmp_path.exists():
                tmp_path.unlink()

    @staticmethod
    def write_bytes_if_absent(path: Path, data: bytes) -> bool:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            return False
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return True

    @staticmethod
    def read_bytes(path: Path) -> bytes | None:
        if not path.exists():
            return None
        return path.read_bytes()

    @staticmethod
    def unlink_path(path: Path) -> None:
        if path.exists():
            path.unlink()


def list_keys(ns_dir: Path, prefix: str | None) -> list[str]:
    if not ns_dir.exists():
        return []
    keys: list[str] = []
    for f in ns_dir.glob("*.json"):
        key = FilesystemBackend.decode_identifier(f.stem)
        if prefix is None or key.startswith(prefix):
            keys.append(key)
    return sorted(keys)


def clear_namespace(ns_dir: Path) -> int:
    if not ns_dir.exists():
        return 0
    count = 0
    for f in ns_dir.glob("*.json"):
        f.unlink()
        count += 1
    return count
