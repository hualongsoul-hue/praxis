"""Standard-library OpenAI-compatible server for real-socket tests."""

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast

import pytest


class ContractHttpServer(ThreadingHTTPServer):
    """Typed HTTP server carrying its public contract state."""

    daemon_threads = False
    block_on_close = True
    contract: "OpenAICompatibleServer"


class OpenAIContractHandler(BaseHTTPRequestHandler):
    """Serve the subset of the OpenAI chat-completions contract used by Praxis."""

    def do_POST(self) -> None:
        contract = cast(ContractHttpServer, self.server).contract
        if self.path != "/v1/chat/completions":
            self.send_payload(404, b'{"error":{"message":"not found"}}')
            return

        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length).decode("utf-8"))
        with contract.requests_lock:
            contract.requests.append(request)
        contract.request_received.set()

        with contract.response_lock:
            response_kind = contract.response_kind
        if response_kind != "stream_block_after_first":
            contract.response_release.wait()
        try:
            if response_kind == "unauthorized":
                self.send_payload(
                    401,
                    b'{"error":{"message":"invalid credential","type":"authentication_error"}}',
                )
            elif response_kind == "malformed":
                self.send_payload(200, b"{not-json")
            elif request.get("stream") is True:
                self.send_stream(block_after_first=response_kind == "stream_block_after_first")
            else:
                self.send_regular(request)
        except OSError:
            contract.client_disconnected.set()

    def send_regular(self, request: dict[str, Any]) -> None:
        if "tool_choice" in request:
            message: dict[str, Any] = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-contract",
                        "type": "function",
                        "function": {
                            "name": "contract_tool",
                            "arguments": '{"value":"forced"}',
                        },
                    }
                ],
            }
            finish_reason = "tool_calls"
        else:
            message = {"role": "assistant", "content": "contract-ok"}
            finish_reason = "stop"
        payload = {
            "id": "chatcmpl-contract",
            "object": "chat.completion",
            "created": 1_700_000_000,
            "model": "contract-model",
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 2,
                "total_tokens": 7,
            },
        }
        self.send_payload(200, json.dumps(payload).encode("utf-8"))

    def send_stream(self, *, block_after_first: bool = False) -> None:
        contract = cast(ContractHttpServer, self.server).contract
        contract.stream_completed.clear()
        frames = [
            {
                "id": "chatcmpl-contract-stream",
                "object": "chat.completion.chunk",
                "created": 1_700_000_000,
                "model": "contract-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "contract-"},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-contract-stream",
                "object": "chat.completion.chunk",
                "created": 1_700_000_000,
                "model": "contract-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
            },
            {
                "id": "chatcmpl-contract-stream",
                "object": "chat.completion.chunk",
                "created": 1_700_000_000,
                "model": "contract-model",
                "choices": [],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 2,
                    "total_tokens": 7,
                },
            },
        ]
        encoded_frames = [
            f"data: {json.dumps(frame)}\n\n".encode() for frame in frames
        ]
        if not block_after_first:
            body = b"".join(encoded_frames) + b"data: [DONE]\n\n"
            self.send_payload(200, body, content_type="text/event-stream")
            contract.stream_completed.set()
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(encoded_frames[0])
        self.wfile.flush()
        contract.stream_frame_sent.set()
        contract.response_release.wait()
        self.wfile.write(b"".join(encoded_frames[1:]) + b"data: [DONE]\n\n")
        self.wfile.flush()
        contract.stream_completed.set()

    def send_payload(
        self,
        status: int,
        body: bytes,
        *,
        content_type: str = "application/json",
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def log_message(self, format_string: str, *args: object) -> None:
        """Suppress the base class's stderr request log."""


class OpenAICompatibleServer:
    """Own a deterministic localhost HTTP server and all observable test state."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.requests_lock = threading.Lock()
        self.response_lock = threading.Lock()
        self.response_kind = "regular"
        self.request_received = threading.Event()
        self.stream_frame_sent = threading.Event()
        self.stream_completed = threading.Event()
        self.response_release = threading.Event()
        self.response_release.set()
        self.client_disconnected = threading.Event()
        self.closed = False
        self.http_server = ContractHttpServer(("127.0.0.1", 0), OpenAIContractHandler)
        self.http_server.contract = self
        host, port = self.http_server.server_address
        self.base_url = f"http://{host}:{port}/v1"
        self.thread = threading.Thread(
            target=self.http_server.serve_forever,
            name=f"openai-contract-{port}",
            daemon=False,
        )
        self.thread.start()

    def choose_response(self, response_kind: str) -> None:
        """Select the response behavior for the next request."""
        with self.response_lock:
            self.response_kind = response_kind

    def block_response(self) -> None:
        """Hold request handlers until the test explicitly releases them."""
        self.request_received.clear()
        self.response_release.clear()

    def block_stream_after_first_frame(self) -> None:
        """Hold a streaming handler after its first flushed SSE frame."""
        with self.response_lock:
            self.response_kind = "stream_block_after_first"
        self.stream_frame_sent.clear()
        self.response_release.clear()

    def release_response(self) -> None:
        """Release every handler waiting to send its response."""
        self.response_release.set()

    def close(self) -> None:
        """Release handlers, stop accepting work, close the socket, and join."""
        if self.closed:
            return
        self.closed = True
        self.release_response()
        self.http_server.shutdown()
        self.http_server.server_close()
        self.thread.join(timeout=5)
        assert self.thread.is_alive() is False


@pytest.fixture
def openai_compatible_server() -> Iterator[OpenAICompatibleServer]:
    """Yield a server whose socket and threads are always reclaimed."""
    server = OpenAICompatibleServer()
    try:
        yield server
    finally:
        server.close()
