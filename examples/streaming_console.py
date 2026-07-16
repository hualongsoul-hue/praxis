"""Minimal streaming Praxis SDK example."""

import argparse
import asyncio

from praxis import PraxisRuntime, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stream one Praxis agent turn.")
    parser.add_argument("--config", default="config.yaml", help="Praxis YAML config")
    parser.add_argument("--prompt", default="分步骤分析这个问题", help="User text")
    return parser


async def main(arguments: argparse.Namespace) -> None:
    config = load_config(arguments.config)
    async with PraxisRuntime(config) as runtime:
        async with runtime.session() as session:
            async for event in session.run_stream(str(arguments.prompt)):
                print(event.event_type, event.data)


if __name__ == "__main__":
    asyncio.run(main(build_parser().parse_args()))
