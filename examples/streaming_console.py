"""Minimal streaming Praxis SDK example."""

import asyncio

from praxis import PraxisRuntime, load_config


async def main() -> None:
    config = load_config("config.yaml")
    async with PraxisRuntime(config) as runtime:
        async with runtime.session() as session:
            async for event in session.run_stream("分步骤分析这个问题"):
                print(event.event_type, event.data)


if __name__ == "__main__":
    asyncio.run(main())
