"""Minimal non-streaming Praxis SDK example."""

import asyncio

from praxis import PraxisRuntime, load_config


async def main() -> None:
    config = load_config("config.yaml")
    async with PraxisRuntime(config) as runtime:
        async with runtime.session() as session:
            response = await session.run("你好，请介绍你的能力")
            print(response.content)


if __name__ == "__main__":
    asyncio.run(main())
