"""Minimal non-streaming Praxis SDK example with optional local attachments."""

import argparse
import asyncio
from pathlib import Path

from praxis import (
    AudioInput,
    FileInput,
    ImageInput,
    InputAttachment,
    PraxisRuntime,
    UserInput,
    VideoInput,
    load_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send one Praxis agent turn.")
    parser.add_argument("--config", default="config.yaml", help="Praxis YAML config")
    parser.add_argument("--prompt", default="请介绍你的能力", help="User text")
    parser.add_argument("--image", type=Path, help="Local image path")
    parser.add_argument("--audio", type=Path, help="Local WAV or MP3 path")
    parser.add_argument("--video", type=Path, help="Local MP4, WebM, or MOV path")
    parser.add_argument("--file", type=Path, help="Local document path")
    return parser


def build_input(arguments: argparse.Namespace) -> str | UserInput:
    parts: list[InputAttachment] = []
    if arguments.image is not None:
        parts.append(ImageInput.from_path(arguments.image))
    if arguments.audio is not None:
        parts.append(AudioInput.from_path(arguments.audio))
    if arguments.video is not None:
        parts.append(VideoInput.from_path(arguments.video))
    if arguments.file is not None:
        parts.append(FileInput.from_path(arguments.file))
    if not parts:
        return str(arguments.prompt)
    return UserInput(text=str(arguments.prompt), parts=tuple(parts))


async def main(arguments: argparse.Namespace) -> None:
    config = load_config(arguments.config)
    async with PraxisRuntime(config) as runtime:
        async with runtime.session() as session:
            response = await session.run(build_input(arguments))
            print(response.content)


if __name__ == "__main__":
    asyncio.run(main(build_parser().parse_args()))
