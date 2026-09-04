"""Praxis 生产 CLI：配置、诊断、交互聊天和版本。"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from praxis import __version__
from praxis.config import load_config
from praxis.config.settings import PraxisConfig

if TYPE_CHECKING:
    from praxis.bootstrap import PraxisCliApplication
    from praxis.runtime import PraxisRuntime

SECRET_CONFIG_KEYS = frozenset({
    "api_key",
    "authorization",
    "credential",
    "password",
    "secret",
})


def redact_config(value: object) -> object:
    if isinstance(value, dict):
        return {
            str(key): (
                "[REDACTED]"
                if str(key).casefold() in SECRET_CONFIG_KEYS
                else redact_config(item)
            )
            for key, item in cast(dict[object, object], value).items()
        }
    if isinstance(value, list):
        return [redact_config(item) for item in cast(list[object], value)]
    return value


def config_path_argument(args: argparse.Namespace) -> str | None:
    value = cast(object, getattr(args, "path", None))
    return value if isinstance(value, str) else None


def load_cli_config(path: str | None) -> PraxisConfig:
    return load_config(Path(path) if path else None)


def create_runtime(config: PraxisConfig) -> PraxisRuntime:
    """Create the production runtime without loading adapters for light commands."""
    from praxis.runtime import PraxisRuntime

    return PraxisRuntime(config)


def create_cli_application(
    config: PraxisConfig,
    *,
    runtime_factory: Callable[..., PraxisRuntime] | None = None,
) -> PraxisCliApplication:
    """Create the CLI lifecycle owner through a lazy import boundary."""
    from praxis.bootstrap import PraxisCliApplication

    return PraxisCliApplication(
        config,
        runtime_factory=runtime_factory or create_runtime,
    )


def cmd_version(arguments: argparse.Namespace) -> int:
    print(f"praxis {__version__}")
    return 0


def cmd_config_validate(args: argparse.Namespace) -> int:
    try:
        load_cli_config(config_path_argument(args))
    except Exception as exc:
        print(f"配置校验失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("配置校验通过。")
    return 0


def cmd_config_show(args: argparse.Namespace) -> int:
    try:
        config = load_cli_config(config_path_argument(args))
    except Exception as exc:
        print(f"配置加载失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    payload = redact_config(config.model_dump(mode="json"))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


async def probe_local_components(config: PraxisConfig) -> bool:
    """Probe storage and embedding without requiring model credentials."""
    from praxis.memory.vector import local_lexical_embed, tei_embed
    from praxis.persistence import create_store

    storage_ready = False
    store = None
    try:
        store = await create_store(config.persistence)
        key = f"doctor-{uuid4().hex}"
        payload = {"status": "ok"}
        async with asyncio.timeout(config.gateway.health_probe_timeout):
            await store.save("cli_doctor", key, payload)
            try:
                if await store.load("cli_doctor", key) != payload:
                    raise RuntimeError("storage round-trip mismatch")
            finally:
                await store.delete("cli_doctor", key)
        storage_ready = True
        print("[READY] storage: 存储读写删除探测通过")
    except Exception as exc:
        print(f"[FAILED] storage: {type(exc).__name__}")
    finally:
        if store is not None:
            try:
                await store.close()
            except Exception as exc:
                storage_ready = False
                print(f"[FAILED] storage_close: {type(exc).__name__}")

    memory = config.memory
    try:
        async with asyncio.timeout(memory.embedding_timeout):
            if memory.embedding_api_base is None:
                vector = await local_lexical_embed(
                    "praxis doctor probe",
                    dimensions=memory.embedding_dimensions or 256,
                )
            else:
                vector = await tei_embed(
                    "praxis doctor probe",
                    api_base=memory.embedding_api_base,
                    api_key=(
                        os.environ.get(memory.embedding_api_key_env, "")
                        if memory.embedding_api_key_env
                        else ""
                    ),
                    model=memory.embedding_model,
                    timeout=memory.embedding_timeout,
                )
        if not vector or not all(math.isfinite(value) for value in vector):
            raise ValueError("embedding vector invalid")
        if memory.embedding_dimensions and len(vector) != memory.embedding_dimensions:
            raise ValueError("embedding dimensions mismatch")
        if memory.embedding_api_base is None:
            print(
                "[DEGRADED] embedding: "
                "本地词法嵌入探测通过；未配置语义嵌入 Provider"
            )
        else:
            print(f"[READY] embedding: 远程 Provider 探测通过，维度 {len(vector)}")
    except Exception as exc:
        print(f"[DEGRADED] embedding: {type(exc).__name__}")
    return storage_ready


async def doctor_command(args: argparse.Namespace) -> int:
    try:
        config = load_cli_config(config_path_argument(args))
    except Exception as exc:
        print(f"[FAILED] config: {type(exc).__name__}: {exc}")
        return 1

    print("[READY] config: 严格配置有效")
    key_name = config.gateway.deployments[0].api_key_env
    credentials_ready = bool(os.environ.get(key_name))
    if credentials_ready:
        print(f"[READY] model_credentials: {key_name} 已设置")
    else:
        print(f"[FAILED] model_credentials: 缺少 {key_name}")
        await probe_local_components(config)
        return 1

    try:
        async with create_cli_application(
            config,
            runtime_factory=create_runtime,
        ) as runtime:
            health = await runtime.health()
    except Exception as exc:
        print(f"[FAILED] runtime: {type(exc).__name__}")
        return 1
    for name, component in health.components.items():
        print(f"[{component.status.value.upper()}] {name}: {component.detail}")
    return 1 if health.status.value == "failed" else 0


def cmd_doctor(args: argparse.Namespace) -> int:
    return asyncio.run(doctor_command(args))


async def chat_command(args: argparse.Namespace) -> int:
    config = load_cli_config(config_path_argument(args))
    async with create_cli_application(
        config,
        runtime_factory=create_runtime,
    ) as runtime:
        async with runtime.session() as session:
            while True:
                try:
                    message = (await asyncio.to_thread(input, "> ")).strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not message:
                    continue
                if message.casefold() in {"exit", "quit", "/exit"}:
                    break
                response = await session.run(message)
                print(response.content)
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    try:
        return asyncio.run(chat_command(args))
    except Exception as exc:
        print(f"聊天启动失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="praxis", description="Praxis Agent SDK/CLI")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("version", help="显示版本").set_defaults(func=cmd_version)

    config_parser = commands.add_parser("config", help="配置管理")
    config_commands = config_parser.add_subparsers(dest="config_command", required=True)
    validate = config_commands.add_parser("validate", help="严格校验配置")
    validate.add_argument("path", nargs="?", default=None)
    validate.set_defaults(func=cmd_config_validate)
    show = config_commands.add_parser("show", help="显示脱敏后的生效配置")
    show.add_argument("path", nargs="?", default=None)
    show.set_defaults(func=cmd_config_show)

    doctor = commands.add_parser("doctor", help="检查配置、凭据和本地依赖")
    doctor.add_argument("path", nargs="?", default=None)
    doctor.set_defaults(func=cmd_doctor)

    chat = commands.add_parser("chat", help="启动交互式 Agent 会话")
    chat.add_argument("path", nargs="?", default=None)
    chat.set_defaults(func=cmd_chat)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    function = cast(Callable[[argparse.Namespace], int], args.func)
    return int(function(args))


if __name__ == "__main__":
    raise SystemExit(main())
