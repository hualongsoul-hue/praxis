"""Behavioral tests for the complete interactive console example."""

import asyncio
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from examples.interactive_console import (
    EXAMPLE_CONFIG_PATH,
    ConsoleApprovalHandler,
    build_attachment,
    build_parser,
    build_user_input,
    format_error,
    handle_mcp_elicitation,
    interactive_loop,
    main,
    print_attachments,
    print_health,
    print_help,
    read_confirmation,
    render_event,
    review_mcp_sampling,
    split_command,
    stream_turn,
    strip_matching_quotes,
)
from praxis import AgentSession, AudioInput, FileInput, ImageInput, UserInput, VideoInput
from praxis.config.schemas import TelemetryConfig
from praxis.exceptions import ModelValidationError, PraxisError
from praxis.models.mcp import MCPElicitationRequest
from praxis.models.orchestrator import AgentEvent
from praxis.models.runtime import ComponentHealth, HealthStatus, RuntimeHealth, RuntimeState
from praxis.models.session import SessionStatus
from praxis.models.tools import ApprovalRequest
from praxis.telemetry.metrics import MetricsCollector


class ConsoleTestSession:
    def __init__(self, events: list[AgentEvent] | None = None) -> None:
        self.events = events or []
        self.inputs: list[object] = []
        self.aborted = False
        self.session_id = "console-test"
        self.status = SessionStatus.ACTIVE

    async def run_stream(self, user_input: object):
        self.inputs.append(user_input)
        for event in self.events:
            yield event

    def abort(self) -> None:
        self.aborted = True


class ConsoleTestRuntime:
    def __init__(self) -> None:
        self.state = RuntimeState.ACTIVE
        self.metrics = MetricsCollector()
        self.health_calls = 0

    async def health(self) -> RuntimeHealth:
        self.health_calls += 1
        return RuntimeHealth(
            status=HealthStatus.READY,
            runtime_state=RuntimeState.ACTIVE,
            components={
                "model": ComponentHealth(status=HealthStatus.READY, detail="ready"),
            },
        )


def input_sequence(monkeypatch: pytest.MonkeyPatch, values: list[object]) -> None:
    responses = iter(values)

    def read(prompt: str = "") -> str:
        response = next(responses)
        if isinstance(response, BaseException):
            raise response
        return str(response)

    monkeypatch.setattr("builtins.input", read)


def test_split_command_preserves_windows_and_space_containing_paths() -> None:
    command = split_command('/attach image "C:\\Users\\Agent Data\\chart.png"')

    assert command == ["/attach", "image", "C:\\Users\\Agent Data\\chart.png"]


def test_command_and_quote_helpers_cover_short_and_unquoted_values() -> None:
    assert split_command("/health") == ["/health"]
    assert strip_matching_quotes("'file name.txt'") == "file name.txt"
    assert strip_matching_quotes("plain.txt") == "plain.txt"


def test_attachment_is_deferred_to_runtime_input_resolver() -> None:
    attachment = build_attachment("image", "attachments/chart.png")

    assert isinstance(attachment, ImageInput)
    assert Path(attachment.source) == Path("attachments/chart.png")


@pytest.mark.parametrize(
    ("kind", "expected_type"),
    [
        ("image", ImageInput),
        ("audio", AudioInput),
        ("video", VideoInput),
        ("file", FileInput),
    ],
)
def test_all_attachment_kinds_are_constructible(
    kind: str,
    expected_type: type[object],
) -> None:
    assert isinstance(build_attachment(kind, "attachments/input.bin"), expected_type)


def test_attachment_rejects_unknown_kind_and_empty_path() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        build_attachment("archive", "input.zip")
    with pytest.raises(ValueError, match="empty"):
        build_attachment("file", " ")


def test_build_user_input_uses_typed_multimodal_envelope() -> None:
    attachment = build_attachment("image", "attachments/chart.png")
    request = build_user_input("Summarize this", [attachment])

    assert isinstance(request, UserInput)
    assert request.text == "Summarize this"
    assert request.parts == (attachment,)


def test_build_user_input_keeps_text_only_turn_as_string() -> None:
    assert build_user_input("hello", []) == "hello"


def test_render_event_prints_only_content_deltas_by_default(capsys) -> None:
    content = AgentEvent(event_type="content_delta", data={"text": "hello"})
    reasoning = AgentEvent(event_type="reasoning_delta", data={"text": "secret reasoning"})

    assert render_event(content, show_reasoning=False) is True
    assert render_event(reasoning, show_reasoning=False) is False
    assert capsys.readouterr().out == "hello"


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        (AgentEvent(event_type="reasoning_delta", data={"text": "think"}), "[reasoning]"),
        (AgentEvent(event_type="tool_call_start", data={"tool_name": "read"}), "start read"),
        (
            AgentEvent(
                event_type="tool_call_end",
                data={"tool_name": "read", "success": True},
            ),
            "read: ok",
        ),
        (AgentEvent(event_type="tool_retry", data={"tool_name": "read"}), "retry read"),
        (AgentEvent(event_type="verification_result", data={"passed": True}), "True"),
        (AgentEvent(event_type="plan_created", data={"step_count": 3}), "steps=3"),
        (AgentEvent(event_type="turn_start", turn=2), "start 2"),
        (AgentEvent(event_type="llm_request", data={"token_count": 4}), "tokens=4"),
        (
            AgentEvent(
                event_type="llm_response",
                data={
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                    "reasoning_tokens": 3,
                },
            ),
            "completion=2",
        ),
        (AgentEvent(event_type="gav_feedback", data={"feedback": "retry"}), "retry"),
        (AgentEvent(event_type="turn_end", turn=2), "end 2"),
        (AgentEvent(event_type="termination", data={"reason": "natural"}), "natural"),
    ],
)
def test_render_event_covers_public_runtime_events(
    event: AgentEvent,
    expected: str,
    capsys,
) -> None:
    render_event(event, show_reasoning=True)
    assert expected in capsys.readouterr().out


def test_render_event_handles_empty_and_rejects_unknown_events(capsys) -> None:
    assert render_event(
        AgentEvent(event_type="content_delta", data={"text": ""}),
        False,
    ) is False
    with pytest.raises(ModelValidationError):
        AgentEvent(event_type="unknown")
    assert capsys.readouterr().out == ""


def test_format_error_redacts_model_key(monkeypatch) -> None:
    key = "sk-" + "A" * 32
    monkeypatch.setenv("PRAXIS_MODEL_API_KEY", key)

    message = format_error(PraxisError(f"provider rejected {key}"))

    assert key not in message
    assert "[REDACTED]" in message


def test_format_error_handles_empty_message_and_bearer_token(monkeypatch) -> None:
    monkeypatch.delenv("PRAXIS_MODEL_API_KEY", raising=False)
    empty = format_error(Exception())
    bearer = format_error(Exception("Bearer " + "B" * 24))
    assert "no additional details" in empty
    assert "B" * 24 not in bearer


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


@pytest.mark.parametrize("error", [KeyboardInterrupt(), PraxisError("failed"), RuntimeError("bad")])
def test_stream_turn_handles_abort_and_error_paths(error: BaseException, capsys) -> None:
    class FailingSession:
        def __init__(self) -> None:
            self.aborted = False

        def abort(self) -> None:
            self.aborted = True

        async def run_stream(self, user_input: str):
            raise error
            yield user_input

    session = FailingSession()
    completed = asyncio.run(stream_turn(cast(AgentSession, session), "hello", False))
    assert completed is False
    if isinstance(error, KeyboardInterrupt):
        assert session.aborted is True
    output = capsys.readouterr()
    assert output.out or output.err


def test_stream_turn_completes_and_prints_newline(capsys) -> None:
    session = ConsoleTestSession(
        [AgentEvent(event_type="content_delta", data={"text": "done"})]
    )
    completed = asyncio.run(stream_turn(cast(AgentSession, session), "hello", False))
    assert completed is True
    assert capsys.readouterr().out.endswith("\n")


def test_print_helpers_show_health_attachments_and_help(capsys) -> None:
    health = RuntimeHealth(
        status=HealthStatus.DEGRADED,
        runtime_state=RuntimeState.ACTIVE,
        components={
            "embedding": ComponentHealth(
                status=HealthStatus.DEGRADED,
                detail="local",
                required=False,
            )
        },
    )
    print_health(health)
    print_attachments([])
    print_attachments([build_attachment("image", "attachments/chart.png")])
    print_help()
    output = capsys.readouterr().out
    assert "runtime=degraded" in output
    assert "optional" in output
    assert "chart.png" in output
    assert "/health" in output


async def test_confirmation_approval_sampling_and_elicitation(
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    input_sequence(monkeypatch, ["yes", "n", '{"answer": "ok"}'])
    assert await read_confirmation("confirm") is True
    decision = await ConsoleApprovalHandler().request_approval(
        ApprovalRequest(tool_name="write_file", arguments={"path": "x"})
    )
    assert decision.approved is False
    response = await handle_mcp_elicitation(
        MCPElicitationRequest(server_name="srv", message="provide input")
    )
    assert response.accepted is True
    assert response.data == {"answer": "ok"}
    input_sequence(monkeypatch, ["y"])
    assert await review_mcp_sampling([{"role": "user"}], "srv") is True
    assert "mcp elicitation" in capsys.readouterr().out


@pytest.mark.parametrize(
    "answer",
    ["", "not-json", "[]", EOFError(), KeyboardInterrupt()],
)
async def test_elicitation_declines_invalid_or_missing_input(
    answer: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_sequence(monkeypatch, [answer])
    response = await handle_mcp_elicitation(
        MCPElicitationRequest(server_name="srv", message="secret")
    )
    assert response.accepted is False


@pytest.mark.parametrize("error", [EOFError(), KeyboardInterrupt()])
async def test_confirmation_fails_closed(
    error: BaseException,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_sequence(monkeypatch, [error])
    assert await read_confirmation("confirm") is False


async def test_interactive_loop_covers_commands_and_multimodal_turn(
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    runtime = ConsoleTestRuntime()
    session = ConsoleTestSession(
        [
            AgentEvent(event_type="turn_start", turn=1),
            AgentEvent(event_type="content_delta", data={"text": "answer"}),
            AgentEvent(event_type="termination", data={"reason": "natural"}),
        ]
    )
    runtime.metrics.counter("turns")
    input_sequence(
        monkeypatch,
        [
            "",
            "/help",
            "/health",
            "/status",
            "/metrics",
            "/reasoning maybe",
            "/reasoning on",
            '/attach image "attachments/chart.png"',
            "/attachments",
            "describe",
            "/attachments",
            "/clear",
            "/attach broken",
            "/unknown",
            "/quit",
        ],
    )

    await interactive_loop(
        cast(Any, runtime),
        cast(AgentSession, session),
        None,
        False,
    )

    assert runtime.health_calls == 1
    assert len(session.inputs) == 1
    assert isinstance(session.inputs[0], UserInput)
    output = capsys.readouterr().out
    assert "runtime=ready" in output
    assert "session_id=console-test" in output
    assert "turns" in output
    assert "Usage: /reasoning" in output
    assert "Unknown command" in output


@pytest.mark.parametrize("ending", [EOFError(), KeyboardInterrupt()])
async def test_interactive_loop_exits_on_terminal_end(
    ending: BaseException,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_sequence(monkeypatch, [ending])
    runtime = ConsoleTestRuntime()
    session = ConsoleTestSession()
    await interactive_loop(
        cast(Any, runtime),
        cast(AgentSession, session),
        "initial",
        False,
    )
    assert session.inputs == ["initial"]


def test_parser_exposes_runtime_options() -> None:
    arguments = build_parser().parse_args(
        ["--config", "custom.yaml", "--show-reasoning", "--prompt", "hello"]
    )
    assert arguments.config == Path("custom.yaml")
    assert arguments.show_reasoning is True
    assert arguments.prompt == "hello"


def test_parser_defaults_to_example_directory_config() -> None:
    arguments = build_parser().parse_args([])

    assert arguments.config == EXAMPLE_CONFIG_PATH
    assert arguments.config == (
        Path(__file__).parents[1] / "examples" / "config.yaml"
    ).resolve()


async def test_main_wires_all_runtime_handlers_and_telemetry(tmp_path: Path) -> None:
    config = SimpleNamespace(telemetry=TelemetryConfig(metrics_enabled=False))
    runtime = MagicMock()
    runtime.metrics = MetricsCollector(enabled=False)
    runtime.health = AsyncMock(
        return_value=RuntimeHealth(
            status=HealthStatus.READY,
            runtime_state=RuntimeState.ACTIVE,
        )
    )
    runtime.__aenter__ = AsyncMock(return_value=runtime)
    runtime.__aexit__ = AsyncMock(return_value=None)
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    runtime.session.return_value = session

    with (
        patch("examples.interactive_console.load_config", return_value=config),
        patch("examples.interactive_console.configure_cli_telemetry") as telemetry,
        patch("examples.interactive_console.PraxisRuntime", return_value=runtime) as runtime_type,
        patch("examples.interactive_console.MetricsExporter") as exporter,
        patch("examples.interactive_console.interactive_loop", AsyncMock()) as loop,
    ):
        exporter.return_value.__enter__.return_value = exporter.return_value
        await main(Namespace(config=tmp_path / "config.yaml", prompt=None, show_reasoning=False))

    telemetry.assert_called_once_with(config.telemetry)
    runtime_keywords = runtime_type.call_args.kwargs
    assert isinstance(runtime_keywords["approval_handler"], ConsoleApprovalHandler)
    assert runtime_keywords["mcp_elicitation_handler"] is handle_mcp_elicitation
    assert runtime_keywords["mcp_sampling_review_handler"] is review_mcp_sampling
    loop.assert_awaited_once()


async def test_main_converts_startup_failure_to_exit(tmp_path: Path, capsys) -> None:
    with (
        patch(
            "examples.interactive_console.load_config",
            side_effect=PraxisError("invalid"),
        ),
        pytest.raises(SystemExit) as raised,
    ):
        await main(Namespace(config=tmp_path / "missing.yaml", prompt=None, show_reasoning=False))
    assert raised.value.code == 2
    assert "startup error" in capsys.readouterr().err
