"""Praxis 命令行入口。

提供运维/CI 友好的子命令：

    python -m praxis version              # 打印版本
    python -m praxis validate [config]    # 校验配置（失败非零退出，适合 CI/部署前置）
    python -m praxis show-config [config]  # 打印生效配置（JSON）
    python -m praxis serve-metrics [config]  # 按配置启动 /metrics 抓取端点并常驻
"""

import argparse
import json
import sys
import time

from praxis import __version__


def _load(config_path: str | None):
    from praxis.config.loader import load_config

    return load_config(config_path)


def cmd_version(_: argparse.Namespace) -> int:
    print(f"praxis {__version__}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    try:
        _load(args.config)
    except Exception as exc:  # 配置/校验错误 → 非零退出
        print(f"配置校验失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("配置校验通过。")
    return 0


def cmd_show_config(args: argparse.Namespace) -> int:
    try:
        config = _load(args.config)
    except Exception as exc:
        print(f"配置加载失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(config.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


def cmd_serve_metrics(args: argparse.Namespace) -> int:
    from praxis.telemetry import configure_telemetry

    try:
        config = _load(args.config)
    except Exception as exc:
        print(f"配置加载失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    tel = config.telemetry.model_copy(update={"metrics_enabled": True, "metrics_export": "prometheus"})
    configure_telemetry(tel)
    print(f"Prometheus /metrics 端点已启动于 :{tel.metrics_port}（Ctrl+C 退出）")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="praxis", description="Praxis AI Agent Harness")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("version", help="打印版本").set_defaults(func=cmd_version)

    p_val = sub.add_parser("validate", help="校验配置文件")
    p_val.add_argument("config", nargs="?", default=None, help="配置文件路径")
    p_val.set_defaults(func=cmd_validate)

    p_show = sub.add_parser("show-config", help="打印生效配置（JSON）")
    p_show.add_argument("config", nargs="?", default=None)
    p_show.set_defaults(func=cmd_show_config)

    p_serve = sub.add_parser("serve-metrics", help="启动 Prometheus /metrics 端点并常驻")
    p_serve.add_argument("config", nargs="?", default=None)
    p_serve.set_defaults(func=cmd_serve_metrics)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
