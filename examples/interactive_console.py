"""Complete interactive streaming console for the Praxis runtime.

Run from the repository root after creating ``config.yaml`` from
``config.example.yaml`` and exporting ``PRAXIS_MODEL_API_KEY``::

    uv run python examples/interactive_console.py --config config.yaml

The console keeps one runtime and one session alive for the whole process.
Every ordinary message uses ``AgentSession.run_stream`` and renders content
deltas as they arrive.  Attachments are queued with ``/attach`` and consumed
by the next message, so the same example exercises the multimodal input
resolver, guardrails, memory, tools, checkpoints, and output streaming.
"""

import argparse
import asyncio
import json
import os
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from praxis import (
    AgentSession,
    AttachmentInput,
    AudioInput,
    FileInput,
    ImageInput,
    InputAttachment,
    InputValue,
    PraxisRuntime,
    UserInput,
    VideoInput,
    load_config,
)
from praxis.exceptions import PraxisError
from praxis.models.mcp import MCPElicitationRequest, MCPElicitationResponse
from praxis.models.orchestrator import AgentEvent
from praxis.models.runtime import RuntimeHealth
from praxis.models.tools import ApprovalDecision, ApprovalRequest
from praxis.telemetry import MetricsExporter, configure_cli_telemetry

ATTACHMENT_TYPES: dict[str, type[AttachmentInput]] = {
    "image": ImageInput,
    "audio": AudioInput,
    "video": VideoInput,
    "file": FileInput,
}
COMMANDS = "/help, /health, /status, /metrics, /reasoning, /attach, /attachments, /clear, /quit"
SECRET_PATTERN = re.compile(r"(?i)(?:sk-[a-z0-9_-]{16,}|bearer\s+[a-z0-9._-]{16,})")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the interactive console."""

    parser = argparse.ArgumentParser(
        description="Run a complete Praxis Runtime in a realtime streaming console."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to a Praxis YAML configuration (default: config.yaml).",
    )
    parser.add_argument(
        "--show-reasoning",
        action="store_true",
        help="Render reasoning deltas as well as the final answer.",
    )
    parser.add_argument(
        "--prompt",
        help="Send one prompt immediately after startup, then enter interactive mode.",
    )
    return parser


def split_command(line: str) -> list[str]:
    """Split a console command without interpreting Windows path separators.

    The first two fields are command and attachment kind.  The remaining text
    is deliberately kept as one field, allowing paths containing spaces on
    both Windows and Linux without shell-specific escaping rules.
    """

    fields = line.strip().split(maxsplit=2)
    if len(fields) == 3:
        fields[2] = strip_matching_quotes(fields[2].strip())
    return fields


def strip_matching_quotes(value: str) -> str:
    """Remove one matching pair of quotes around a user-supplied path."""

    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def build_attachment(kind: str, path_text: str) -> InputAttachment:
    """Create a deferred, security-checked attachment from console input."""

    attachment_type = ATTACHMENT_TYPES.get(kind.casefold())
    if attachment_type is None:
        supported = ", ".join(sorted(ATTACHMENT_TYPES))
        raise ValueError(f"unsupported attachment kind; use one of: {supported}")
    if not path_text.strip():
        raise ValueError("attachment path cannot be empty")
    return cast(InputAttachment, attachment_type.from_path(Path(path_text.strip())))


def build_user_input(text: str, attachments: Sequence[InputAttachment]) -> InputValue:
    """Return a plain string for text-only turns or a typed multimodal turn."""

    if not attachments:
        return text
    return UserInput(text=text, parts=tuple(attachments))


def format_error(error: BaseException) -> str:
    """Format an error while preventing credentials from reaching the terminal."""

    message = str(error).strip() or "no additional details"
    api_key_value = os.environ.get("PRAXIS_MODEL_API_KEY")
    if api_key_value:
        message = message.replace(api_key_value, "[REDACTED]")
    message = SECRET_PATTERN.sub("[REDACTED]", message)
    return f"{type(error).__name__}: {message[:400]}"


def print_health(health: RuntimeHealth) -> None:
    """Print a concise, secret-free runtime health report."""

    print(f"[health] runtime={health.status.value}")
    for name, component in health.components.items():
        required = "required" if component.required else "optional"
        print(f"[health] {name}={component.status.value} ({required}) {component.detail}")


def print_help() -> None:
    """Print available console commands."""

    print("Commands:")
    print("  /help                         Show this help.")
    print("  /health                       Check Runtime, model, storage, and workers.")
    print("  /status                       Show Runtime and AgentSession state.")
    print("  /metrics                      Export current Runtime metrics to the terminal.")
    print("  /reasoning <on|off>           Toggle reasoning-delta rendering.")
    print("  /attach <kind> <path>         Queue image/audio/video/file for the next turn.")
    print("  /attachments                  List queued attachments.")
    print("  /clear                        Remove queued attachments.")
    print("  /quit                         Close the session and Runtime.")
    print("Press Ctrl+C while a response is streaming to abort that turn.")


async def read_confirmation(prompt: str) -> bool:
    """Read a fail-closed yes/no decision without blocking the event loop."""

    try:
        answer = await asyncio.to_thread(input, prompt)
    except (EOFError, KeyboardInterrupt):
        return False
    return answer.strip().casefold() in {"y", "yes"}


class ConsoleApprovalHandler:
    """Interactive fail-closed ApprovalHandler for guarded tool calls."""

    async def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        argument_names = ", ".join(sorted(request.arguments)) or "none"
        approved = await read_confirmation(
            f"\n[approval] tool={request.tool_name}, argument keys={argument_names}. Allow? [y/N] "
        )
        return ApprovalDecision(
            approved=approved,
            reason="console approval" if approved else "console denial",
        )


async def review_mcp_sampling(
    messages: list[dict[str, Any]],
    server_name: str,
) -> bool:
    """Ask the operator before an MCP Server can request model sampling."""

    return await read_confirmation(
        f"\n[mcp sampling] server={server_name}, messages={len(messages)}. Allow? [y/N] "
    )


async def handle_mcp_elicitation(
    request: MCPElicitationRequest,
) -> MCPElicitationResponse:
    """Collect structured MCP Elicitation data as a JSON object."""

    safe_message = SECRET_PATTERN.sub("[REDACTED]", request.message)[:300]
    print(f"\n[mcp elicitation] server={request.server_name}: {safe_message}")
    try:
        answer = await asyncio.to_thread(
            input,
            "Enter a JSON object to accept, or leave blank to decline: ",
        )
    except (EOFError, KeyboardInterrupt):
        return MCPElicitationResponse(accepted=False)
    if not answer.strip():
        return MCPElicitationResponse(accepted=False)
    try:
        data = json.loads(answer)
    except json.JSONDecodeError:
        print("[mcp elicitation] invalid JSON; request declined")
        return MCPElicitationResponse(accepted=False)
    if not isinstance(data, dict):
        print("[mcp elicitation] response must be a JSON object; request declined")
        return MCPElicitationResponse(accepted=False)
    return MCPElicitationResponse(accepted=True, data=data)


def print_attachments(attachments: Sequence[InputAttachment]) -> None:
    """Print queued attachment metadata without exposing full local paths."""

    if not attachments:
        print("[attachments] none")
        return
    for index, attachment in enumerate(attachments, start=1):
        source = attachment.source
        display_name = Path(source).name if isinstance(source, (str, Path)) else "bytes"
        print(f"[attachments] {index}. {attachment.kind.value}: {display_name}")


def render_event(event: AgentEvent, show_reasoning: bool) -> bool:
    """Render one public AgentEvent and return whether answer text was printed."""

    data = event.data
    if event.event_type == "content_delta":
        text = data.get("text")
        if isinstance(text, str) and text:
            print(text, end="", flush=True)
            return True
        return False
    if event.event_type == "reasoning_delta" and show_reasoning:
        text = data.get("text")
        if isinstance(text, str) and text:
            print(f"\n[reasoning] {text}", end="", flush=True)
        return False
    if event.event_type == "tool_call_start":
        tool_name = data.get("tool_name", "unknown")
        print(f"\n[tool] start {tool_name}", flush=True)
    elif event.event_type == "tool_call_end":
        tool_name = data.get("tool_name", "unknown")
        result = "ok" if data.get("success") else "failed"
        print(f"[tool] {tool_name}: {result}", flush=True)
    elif event.event_type == "tool_retry":
        print(f"[tool] retry {data.get('tool_name', 'unknown')}", flush=True)
    elif event.event_type == "verification_result":
        print(f"[verify] {data.get('passed', data.get('success', 'unknown'))}", flush=True)
    elif event.event_type == "plan_created":
        print(f"[plan] steps={data.get('step_count', 'unknown')}", flush=True)
    elif event.event_type == "turn_start":
        print(f"\n[turn] start {event.turn}", flush=True)
    elif event.event_type == "llm_request":
        print(f"[model] request tokens={data.get('token_count', 'unknown')}", flush=True)
    elif event.event_type == "llm_response":
        print(
            "[model] response "
            f"prompt={data.get('prompt_tokens', 0)} "
            f"completion={data.get('completion_tokens', 0)} "
            f"reasoning={data.get('reasoning_tokens', 0)}",
            flush=True,
        )
    elif event.event_type == "gav_feedback":
        print(f"[verify] feedback={data.get('feedback', 'available')}", flush=True)
    elif event.event_type == "turn_end":
        print(f"[turn] end {event.turn}", flush=True)
    elif event.event_type == "termination":
        print(f"\n[turn] terminated: {data.get('reason', 'unknown')}", flush=True)
    return False


async def stream_turn(
    session: AgentSession,
    user_input: InputValue,
    show_reasoning: bool,
) -> bool:
    """Consume one Runtime stream, preserving cancellation and cleanup semantics."""

    printed_content = False
    try:
        async for event in session.run_stream(user_input):
            printed_content = render_event(event, show_reasoning) or printed_content
    except KeyboardInterrupt:
        session.abort()
        print("\n[turn] aborted", flush=True)
        return False
    except asyncio.CancelledError:
        session.abort()
        print("\n[turn] cancelled", flush=True)
        return False
    except PraxisError as error:
        print(f"\n[error] {format_error(error)}", file=sys.stderr, flush=True)
        return False
    except Exception as error:
        print(f"\n[error] {format_error(error)}", file=sys.stderr, flush=True)
        return False
    if printed_content:
        print(flush=True)
    return True


async def interactive_loop(
    runtime: PraxisRuntime,
    session: AgentSession,
    initial_prompt: str | None,
    show_reasoning: bool,
) -> None:
    """Run the command loop while reusing one Runtime and AgentSession."""

    pending_attachments: list[InputAttachment] = []
    if initial_prompt:
        await stream_turn(session, initial_prompt, show_reasoning)

    print("Praxis interactive console. Type /help for commands.")
    while True:
        try:
            line = await asyncio.to_thread(input, "praxis> ")
        except EOFError:
            print()
            return
        except KeyboardInterrupt:
            print("\nBye.")
            return

        text = line.strip()
        if not text:
            continue
        if not text.startswith("/"):
            user_input = build_user_input(text, pending_attachments)
            if await stream_turn(session, user_input, show_reasoning):
                pending_attachments.clear()
            continue

        fields = split_command(text)
        command = fields[0].casefold()
        if command in {"/quit", "/exit"}:
            return
        if command == "/help":
            print_help()
            continue
        if command == "/health":
            print_health(await runtime.health())
            continue
        if command == "/status":
            print(
                f"[status] runtime={runtime.state.value} "
                f"session={session.status.value} "
                f"session_id={session.session_id or 'unavailable'}"
            )
            continue
        if command == "/metrics":
            metrics = runtime.metrics.export_prometheus().strip()
            print(metrics or "[metrics] no samples")
            continue
        if command == "/reasoning":
            if len(fields) != 2 or fields[1].casefold() not in {"on", "off"}:
                print("Usage: /reasoning <on|off>")
                continue
            show_reasoning = fields[1].casefold() == "on"
            print(f"[reasoning] {'enabled' if show_reasoning else 'disabled'}")
            continue
        if command == "/attachments":
            print_attachments(pending_attachments)
            continue
        if command == "/clear":
            pending_attachments.clear()
            print("[attachments] cleared")
            continue
        if command == "/attach":
            if len(fields) != 3:
                print("Usage: /attach <image|audio|video|file> <path>")
                continue
            try:
                pending_attachments.append(build_attachment(fields[1], fields[2]))
            except (TypeError, ValueError) as error:
                print(f"[attachments] {error}")
            else:
                print_attachments(pending_attachments)
            continue
        print(f"Unknown command. Available commands: {COMMANDS}")


async def main(arguments: argparse.Namespace) -> None:
    """Load configuration and run the complete asynchronous application lifecycle."""

    try:
        config = load_config(arguments.config)
        configure_cli_telemetry(config.telemetry)
        approval_handler = ConsoleApprovalHandler()
        async with PraxisRuntime(
            config,
            approval_handler=approval_handler,
            mcp_elicitation_handler=handle_mcp_elicitation,
            mcp_sampling_review_handler=review_mcp_sampling,
        ) as runtime:
            with MetricsExporter(runtime.metrics, config.telemetry):
                health = await runtime.health()
                print_health(health)
                async with runtime.session() as session:
                    await interactive_loop(
                        runtime,
                        session,
                        arguments.prompt,
                        arguments.show_reasoning,
                    )
    except (PraxisError, OSError, ValueError) as error:
        print(f"[startup error] {format_error(error)}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    asyncio.run(main(build_parser().parse_args()))
