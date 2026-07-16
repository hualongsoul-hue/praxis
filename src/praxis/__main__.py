"""Praxis 生产 CLI：配置、诊断、交互聊天和版本。"""

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

from praxis import PraxisRuntime, __version__, load_config
from praxis.config.settings import PraxisConfig
from praxis.persistence.store import create_store

_SECRET_CONFIG_KEYS = frozenset({
    "api_key",
    "authorization",
    "credential",
    "password",
    "secret",
})


def _redact(value: object) -> object:
    if isinstance(value, dict):
        return {
            str(key): (
                "[REDACTED]"
                if str(key).casefold() in _SECRET_CONFIG_KEYS
                else _redact(item)
            )
            for key, item in cast(dict[object, object], value).items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in cast(list[object], value)]
    return value


def _path_argument(args: argparse.Namespace) -> str | None:
    value = cast(object, getattr(args, "path", None))
    return value if isinstance(value, str) else None


def _load(path: str | None) -> PraxisConfig:
    return load_config(Path(path) if path else None)


def cmd_version(_: argparse.Namespace) -> int:
    print(f"praxis {__version__}")
    return 0


def cmd_config_validate(args: argparse.Namespace) -> int:
    try:
        _load(_path_argument(args))
    except Exception as exc:
        print(f"配置校验失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("配置校验通过。")
    return 0


def cmd_config_show(args: argparse.Namespace) -> int:
    try:
        config = _load(_path_argument(args))
    except Exception as exc:
        print(f"配置加载失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    payload = _redact(config.model_dump(mode="json"))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


async def _doctor(args: argparse.Namespace) -> int:
    try:
        config = _load(_path_argument(args))
    except Exception as exc:
        print(f"[FAILED] config: {type(exc).__name__}: {exc}")
        return 1

    failed = False
    print("[READY] config: 严格配置有效")
    key_name = config.gateway.deployments[0].api_key_env
    if os.environ.get(key_name):
        print(f"[READY] model_credentials: {key_name} 已设置")
    else:
        failed = True
        print(f"[FAILED] model_credentials: 缺少 {key_name}")

    store = None
    try:
        store = await create_store(config.persistence)
        await store.list_keys("_praxis_doctor")
    except Exception as exc:
        failed = True
        print(f"[FAILED] storage: {type(exc).__name__}")
    else:
        print("[READY] storage: 可读写后端已初始化")
    finally:
        if store is not None:
            await store.close()

    if config.memory.embedding_api_base is None:
        print("[DEGRADED] embedding: 使用确定性本地词法检索")
    else:
        print("[READY] embedding: 远程 Provider 已配置")
    return 1 if failed else 0


def cmd_doctor(args: argparse.Namespace) -> int:
    return asyncio.run(_doctor(args))


async def _chat(args: argparse.Namespace) -> int:
    config = _load(_path_argument(args))
    async with PraxisRuntime(config) as runtime:
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
        return asyncio.run(_chat(args))
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
