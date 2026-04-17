"""文件系统存储后端。

简单场景，数据以 JSON 文件存储，人类可读。
目录结构：``{base_path}/{namespace}/{key}.json``。
"""

import asyncio
from pathlib import Path


class FilesystemBackend:
    """文件系统异步存储后端。"""

    def __init__(self, base_path: str) -> None:
        self._base = Path(base_path)

    def _path(self, namespace: str, key: str) -> Path:
        safe_key = key.replace("/", "__").replace("\\", "__")
        return self._base / namespace / f"{safe_key}.json"

    async def save(self, namespace: str, key: str, data: bytes) -> None:
        path = self._path(namespace, key)
        await asyncio.to_thread(self._write, path, data)

    async def load(self, namespace: str, key: str) -> bytes | None:
        path = self._path(namespace, key)
        return await asyncio.to_thread(self._read, path)

    async def delete(self, namespace: str, key: str) -> None:
        path = self._path(namespace, key)
        await asyncio.to_thread(self._unlink, path)

    async def list_keys(
        self, namespace: str, prefix: str | None = None
    ) -> list[str]:
        ns_dir = self._base / namespace
        return await asyncio.to_thread(self._list, ns_dir, prefix)

    async def clear_namespace(self, namespace: str) -> int:
        ns_dir = self._base / namespace
        return await asyncio.to_thread(self._clear, ns_dir)

    async def close(self) -> None:
        pass

    @staticmethod
    def _write(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    @staticmethod
    def _read(path: Path) -> bytes | None:
        if not path.exists():
            return None
        return path.read_bytes()

    @staticmethod
    def _unlink(path: Path) -> None:
        if path.exists():
            path.unlink()

    @staticmethod
    def _list(ns_dir: Path, prefix: str | None) -> list[str]:
        if not ns_dir.exists():
            return []
        keys: list[str] = []
        for f in ns_dir.glob("*.json"):
            key = f.stem.replace("__", "/")
            if prefix is None or key.startswith(prefix):
                keys.append(key)
        return sorted(keys)

    @staticmethod
    def _clear(ns_dir: Path) -> int:
        if not ns_dir.exists():
            return 0
        count = 0
        for f in ns_dir.glob("*.json"):
            f.unlink()
            count += 1
        return count
