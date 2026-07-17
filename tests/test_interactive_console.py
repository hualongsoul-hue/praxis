"""Behavioral tests for the complete interactive console example."""

import asyncio
from pathlib import Path
from typing import cast

from examples.interactive_console import (
    build_attachment,
    build_user_input,
    format_error,
    render_event,
    split_command,
    stream_turn,
)
from praxis import AgentSession, ImageInput, UserInput
from praxis.exceptions import PraxisError
from praxis.models.orchestrator import AgentEvent


def test_split_command_preserves_windows_and_space_containing_paths() -> None:
    command = split_command('/attach image "C:\\Users\\Agent Data\\chart.png"')

    assert command == ["/attach", "image", "C:\\Users\\Agent Data\\chart.png"]


def test_attachment_is_deferred_to_runtime_input_resolver() -> None:
    attachment = build_attachment("image", "attachments/chart.png")

    assert isinstance(attachment, ImageInput)
    assert Path(attachment.source) == Path("attachments/chart.png")


def test_build_user_input_uses_typed_multimodal_envelope() -> None:
    attachment = build_attachment("image", "attachments/chart.png")
    request = build_user_input("Summarize this", [attachment])

    assert isinstance(request, UserInput)
    assert request.text == "Summarize this"
    assert request.parts == (attachment,)


def test_render_event_prints_only_content_deltas_by_default(capsys) -> None:
    content = AgentEvent(event_type="content_delta", data={"text": "hello"})
    reasoning = AgentEvent(event_type="reasoning_delta", data={"text": "secret reasoning"})

    assert render_event(content, show_reasoning=False) is True
    assert render_event(reasoning, show_reasoning=False) is False
    assert capsys.readouterr().out == "hello"


def test_format_error_redacts_model_key(monkeypatch) -> None:
    key = "sk-" + "A" * 32
    monkeypatch.setenv("PRAXIS_MODEL_API_KEY", key)

    message = format_error(PraxisError(f"provider rejected {key}"))

    assert key not in message
    assert "[REDACTED]" in message


def test_stream_turn_converts_cancellation_to_current_turn_abort(capsys) -> None:
    class CancelledSession:
        def __init__(self) -> None:
            self.aborted = False

        def abort(self) -> None:
            self.aborted = True

        async def run_stream(self, user_input: str):
            raise asyncio.CancelledError
            yield user_input

    session = CancelledSession()
    completed = asyncio.run(stream_turn(cast(AgentSession, session), "hello", False))

    assert completed is False
    assert session.aborted is True
    assert "cancelled" in capsys.readouterr().out
