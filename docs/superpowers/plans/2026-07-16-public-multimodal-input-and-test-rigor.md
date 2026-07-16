# Public Multimodal Input and Test Rigor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate project-owned private Python symbols, add secure text/image/audio/video/file input to the public SDK, and replace implementation-coupled testing with behavior, property, HTTP-contract, and live-endpoint evidence.

**Architecture:** A frozen `UserInput` envelope and typed attachment classes are resolved by a session-owned `InputResolver` into ephemeral provider content plus a safe text projection. `ModelCapabilities` gates modalities before network calls; the orchestration loop retains binary content only for the current run and always sanitizes history in `finally`. A repository AST policy, property tests, and a real local OpenAI-compatible HTTP server enforce naming and behavior without coupling tests to internal fields.

**Tech Stack:** Python 3.12–3.14, Pydantic v2, httpx, LiteLLM, pytest, pytest-asyncio, Hypothesis, Ruff, Pyright strict, uv.

## Global Constraints

- Do not create a branch or worktree; all commits remain on the current `dev` branch.
- Use Chinese Conventional Commit messages and make multiple task-level commits.
- Do not preserve old APIs, private aliases, old checkpoints, or `supports_vision()` compatibility.
- Project-owned Python in `src/`, `tests/`, `scripts/`, and `examples/` must define and access zero single-leading-underscore variables, methods, or classes; Python-required `__dunder__` protocol names are the only exception.
- Python support remains `>=3.12,<3.15`, with Windows and Linux behavior kept symmetric.
- `AgentSession.run()` and `run_stream()` accept `str | UserInput`; supported attachment kinds are image, audio, video, and file, with bytes, local-path, and HTTP/HTTPS sources.
- API keys are read only from `PRAXIS_MODEL_API_KEY` and never written to files, logs, audit events, checkpoints, test reports, or CLI output.
- The model alias remains `openai/glm-5.1-openai` and API Base remains `http://172.24.23.192:3000/v1`.
- Local paths and remote URLs fail closed by default; binary data is ephemeral and must not survive in memory history, audit, logs, or checkpoints after a run ends, errors, or is cancelled.
- No web framework becomes a runtime dependency; the contract-test HTTP server uses only the Python standard library.
- All implementation work follows Red–Green–Refactor and each task carries its own focused verification before commit.

---

## File Structure

- `src/praxis/models/inputs.py`: public input envelope, attachment sources, resolved metadata, and safe projection models.
- `src/praxis/input_resolver.py`: path, URL, MIME, size, Base64, and provider-content normalization.
- `src/praxis/network.py`: reusable public HTTP target validation shared by media input and Web Fetch.
- `src/praxis/models/messages.py`: typed OpenAI-compatible text, image, audio, video, and file content blocks.
- `src/praxis/config/schemas.py`: `InputConfig` and `ModelCapabilities` strict immutable configuration.
- `src/praxis/runtime.py`, `src/praxis/session/core.py`, `src/praxis/orchestrator/loop.py`, `src/praxis/context/assembler.py`: public input propagation, capability gating, ephemeral history, and cleanup.
- `tests/test_public_api_policy.py`: repository-wide AST naming gate.
- `tests/test_inputs.py`: model, security, resolution, property, and cleanup unit tests.
- `tests/fixtures/openai_compatible_server.py`: standard-library HTTP/SSE contract server.
- `tests/e2e/test_runtime_http_contract.py`: full Runtime → LiteLLM → HTTP → Runtime tests.
- `tests/scenarios/`: mocked component scenarios moved out of the E2E category.
- `tests/integration/test_gateway_live_multimodal.py`: explicit real-endpoint modality probes.

---

### Task 1: Enforce public project-owned Python symbols

**Files:**
- Create: `tests/test_public_api_policy.py`
- Modify: every violating Python file reported under `src/praxis`, `tests`, `scripts`, and `examples`

**Interfaces:**
- Consumes: Python AST and repository root discovered from `Path(__file__).parents[1]`.
- Produces: `collect_private_symbol_violations(root: Path) -> list[str]` and a permanent zero-violation test.

- [ ] **Step 1: Write the failing repository policy test**

```python
import ast
from pathlib import Path

PYTHON_ROOTS = ("src", "tests", "scripts", "examples")


def is_private_name(name: str) -> bool:
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


class PublicSymbolVisitor(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.scopes = ["module"]
        self.violations: list[str] = []

    def report(self, node: ast.AST, kind: str, name: str) -> None:
        self.violations.append(f"{self.path.as_posix()}:{node.lineno}: {kind} {name}")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if is_private_name(node.name):
            self.report(node, "function", node.name)
        self.scopes.append("function")
        self.generic_visit(node)
        self.scopes.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if is_private_name(node.name):
            self.report(node, "function", node.name)
        self.scopes.append("function")
        self.generic_visit(node)
        self.scopes.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if is_private_name(node.name):
            self.report(node, "class", node.name)
        self.scopes.append("class")
        self.generic_visit(node)
        self.scopes.pop()

    def visit_Assign(self, node: ast.Assign) -> None:
        if self.scopes[-1] in {"module", "class"}:
            for target in node.targets:
                if isinstance(target, ast.Name) and is_private_name(target.id):
                    self.report(target, f"{self.scopes[-1]} variable", target.id)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        target = node.target
        if self.scopes[-1] in {"module", "class"} and isinstance(target, ast.Name) and is_private_name(target.id):
            self.report(target, f"{self.scopes[-1]} variable", target.id)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for item in node.names:
            if item.asname is not None and is_private_name(item.asname):
                self.report(item, "import alias", item.asname)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for item in node.names:
            if item.asname is not None and is_private_name(item.asname):
                self.report(item, "import alias", item.asname)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if is_private_name(node.attr):
            self.report(node, "attribute", node.attr)
        self.generic_visit(node)


def collect_private_symbol_violations(root: Path) -> list[str]:
    violations: list[str] = []
    for directory in PYTHON_ROOTS:
        for path in root.joinpath(directory).rglob("*.py"):
            visitor = PublicSymbolVisitor(path.relative_to(root))
            visitor.visit(ast.parse(path.read_text(encoding="utf-8")))
            violations.extend(visitor.violations)
    return sorted(violations)


def test_project_python_uses_only_public_owned_symbols() -> None:
    root = Path(__file__).resolve().parents[1]
    assert collect_private_symbol_violations(root) == []
```

- [ ] **Step 2: Run the policy test and preserve the failing output as the mechanical rename checklist**

Run: `uv run pytest tests/test_public_api_policy.py -q`

Expected: FAIL with violations including Runtime state, Gateway router state, persistence helpers, telemetry helpers, ToolPolicy state, CLI/config helpers, and tests accessing `_router`, `_store`, or `_client`.

- [ ] **Step 3: Rename every reported owned symbol and all call sites without aliases**

Use these public names for the known method and module-global violations:

```text
_on_done -> handle_done
_require_runner -> require_runner
_redact -> redact_config
_path_argument -> config_path_argument
_load -> load_cli_config
_doctor -> doctor_command
_chat -> chat_command
_deep_merge -> merge_config
_environment_config -> environment_config
_load_yaml -> load_yaml_config
_prepare_reservation -> prepare_reservation
_actual_cost -> actual_cost
_settlement_cost -> settlement_cost
_to_litellm_deployment -> to_litellm_deployment
_build_router -> build_litellm_router
_get_client -> get_client
_operation -> run_storage_operation
_apply_continuation -> apply_continuation
_prune_checkpoints -> prune_checkpoints
_discard_completed -> discard_completed
_log -> write_log
_key -> metric_key
_terminate_tree -> terminate_process_tree
_validate_identifier -> validate_identifier
_encode -> encode_identifier
_decode -> decode_identifier
_namespace_dir -> namespace_directory
_path -> storage_path
_write -> write_bytes
_write_if_absent -> write_bytes_if_absent
_read -> read_bytes
_unlink -> unlink_path
_list -> list_keys
_clear -> clear_namespace
_full_key -> full_key
_set_sqlite_pragmas -> set_sqlite_pragmas
_T -> T
_SECRET_CONFIG_KEYS -> SECRET_CONFIG_KEYS
_OBJECT_MAP -> OBJECT_MAP
_SECRET_KEYS -> SECRET_KEYS
_SECRET_VALUE -> SECRET_VALUE
_current_collector -> current_collector
_current_tracer -> current_tracer
_SAFE_ENVIRONMENT -> SAFE_ENVIRONMENT
_REDIRECT_CODES -> REDIRECT_STATUS_CODES
```

Rename instance fields to descriptive public forms, for example
`runtime_state`, `lifecycle_lock`, `sessions`, `runner`, `run_lock`,
`allowed_paths`, `router`, `store`, and `client`. Update tests to assert through public behavior where possible; when state itself is the feature under test, use the renamed public property directly. Do not add deprecated properties or assignments to old names.

- [ ] **Step 4: Run focused policy, lint, and type verification**

Run: `uv run pytest tests/test_public_api_policy.py -q && uv run ruff check . && uv run pyright`

Expected: policy PASS, Ruff PASS, Pyright reports `0 errors, 0 warnings`.

- [ ] **Step 5: Commit the public-symbol refactor**

```powershell
git add src tests scripts examples
git commit -m "refactor(api): 清除项目私有符号与实现耦合"
```

---

### Task 2: Add typed input models and secure resolution

**Files:**
- Create: `src/praxis/models/inputs.py`
- Create: `src/praxis/input_resolver.py`
- Create: `src/praxis/network.py`
- Create: `tests/test_inputs.py`
- Modify: `src/praxis/models/messages.py`
- Modify: `src/praxis/models/__init__.py`
- Modify: `src/praxis/__init__.py`
- Modify: `src/praxis/config/schemas.py`
- Modify: `src/praxis/config/settings.py`
- Modify: `src/praxis/config/__init__.py`
- Modify: `src/praxis/exceptions.py`
- Modify: `src/praxis/tools/policy.py`
- Modify: `src/praxis/tools/builtins/network/web_fetch.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: public `validate_http_url(url: str, allow_private_networks: bool) -> str` and strict `InputConfig`.
- Produces: `InputValue`, `UserInput`, four attachment classes, `ResolvedUserInput`, `InputResolver.resolve(value, capabilities)`, and five provider content blocks.

- [ ] **Step 1: Add Hypothesis and write failing model/config tests**

Add `hypothesis>=6.135,<7` to the development dependency group. Write tests that assert:

```python
def test_user_input_requires_text_or_attachment() -> None:
    with pytest.raises(ValueError):
        UserInput()


def test_input_config_fails_closed() -> None:
    config = InputConfig()
    assert config.allowed_paths == []
    assert config.remote_enabled is False
    assert config.allow_private_networks is False
    assert config.max_attachment_bytes == 20_000_000
    assert config.max_total_bytes == 50_000_000


def test_string_and_text_envelope_resolve_equivalently() -> None:
    resolver = InputResolver(InputConfig())
    first = asyncio.run(resolver.resolve("hello", ModelCapabilities()))
    second = asyncio.run(resolver.resolve(UserInput(text="hello"), ModelCapabilities()))
    assert first == second
```

Run: `uv run pytest tests/test_inputs.py -q`

Expected: FAIL because the public models and resolver do not exist.

- [ ] **Step 2: Implement strict configuration and public input source models**

Implement these exact public fields and constructors in `models/inputs.py`:

```python
class InputKind(StrEnum):
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    FILE = "file"


class InputSourceKind(StrEnum):
    BYTES = "bytes"
    PATH = "path"
    URL = "url"


class AttachmentInput(BaseModel):
    model_config = ConfigDict(frozen=True)
    source_kind: InputSourceKind
    source: bytes | Path | str
    media_type: str | None = None
    filename: str | None = None

    @classmethod
    def from_bytes(cls, data: bytes, *, media_type: str, filename: str | None = None) -> Self:
        return cls(source_kind=InputSourceKind.BYTES, source=bytes(data), media_type=media_type, filename=filename)

    @classmethod
    def from_path(cls, path: str | Path, *, media_type: str | None = None) -> Self:
        return cls(source_kind=InputSourceKind.PATH, source=Path(path), media_type=media_type)

    @classmethod
    def from_url(cls, url: str, *, media_type: str | None = None, filename: str | None = None) -> Self:
        return cls(source_kind=InputSourceKind.URL, source=url, media_type=media_type, filename=filename)


class ImageInput(AttachmentInput):
    kind: Literal[InputKind.IMAGE] = InputKind.IMAGE


class AudioInput(AttachmentInput):
    kind: Literal[InputKind.AUDIO] = InputKind.AUDIO


class VideoInput(AttachmentInput):
    kind: Literal[InputKind.VIDEO] = InputKind.VIDEO


class FileInput(AttachmentInput):
    kind: Literal[InputKind.FILE] = InputKind.FILE


InputAttachment = Annotated[ImageInput | AudioInput | VideoInput | FileInput, Field(discriminator="kind")]


class UserInput(BaseModel):
    model_config = ConfigDict(frozen=True)
    text: str = ""
    parts: tuple[InputAttachment, ...] = ()

    @model_validator(mode="after")
    def validate_content(self) -> "UserInput":
        if not self.text.strip() and not self.parts:
            raise ValueError("用户输入必须包含文本或至少一个附件")
        return self


InputValue = str | UserInput
```

Add these strict configuration models and replace `ModelDeployment.supports_vision` with
`ModelDeployment.capabilities`:

```python
class ModelCapabilities(StrictConfigModel):
    image: bool = False
    audio: bool = False
    video: bool = False
    file: bool = False


class InputConfig(StrictConfigModel):
    allowed_paths: list[str] = Field(default_factory=list)
    remote_enabled: bool = False
    allow_private_networks: bool = False
    max_attachment_bytes: int = Field(default=20_000_000, ge=1, le=100_000_000)
    max_total_bytes: int = Field(default=50_000_000, ge=1, le=500_000_000)
    max_redirects: int = Field(default=5, ge=0, le=20)
    remote_timeout: float = Field(default=30.0, gt=0, le=300.0)
    image_media_types: list[str] = Field(default_factory=lambda: ["image/jpeg", "image/png", "image/gif", "image/webp"])
    audio_media_types: list[str] = Field(default_factory=lambda: ["audio/mpeg", "audio/wav"])
    video_media_types: list[str] = Field(default_factory=lambda: ["video/mp4", "video/webm", "video/quicktime"])
    file_media_types: list[str] = Field(default_factory=lambda: ["application/pdf", "application/json", "application/yaml", "text/csv", "text/markdown", "text/plain"])
```

Add `inputs: InputConfig = Field(default_factory=InputConfig)` to `PraxisConfig` and export both
configuration models. In the existing `ModelDeployment`, delete
`supports_vision: bool = False` and add exactly
`capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)` at the same location;
all other deployment fields stay unchanged.

- [ ] **Step 3: Add typed content blocks and exceptions**

Extend `ContentPart` with:

```python
class AudioData(BaseModel):
    data: str
    format: Literal["wav", "mp3"]


class AudioContent(BaseModel):
    type: Literal["input_audio"] = "input_audio"
    input_audio: AudioData


class VideoUrl(BaseModel):
    url: str


class VideoContent(BaseModel):
    type: Literal["video_url"] = "video_url"
    video_url: VideoUrl


class FileData(BaseModel):
    filename: str
    file_data: str


class FileContent(BaseModel):
    type: Literal["file"] = "file"
    file: FileData


ContentPart = Annotated[
    TextContent | ImageContent | AudioContent | VideoContent | FileContent,
    Field(discriminator="type"),
]


class AttachmentMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: InputKind
    filename: str
    media_type: str
    size_bytes: int = Field(ge=0)


class ResolvedUserInput(BaseModel):
    model_config = ConfigDict(frozen=True)
    content: str | list[ContentPart]
    text_projection: str
    modalities: frozenset[InputKind] = frozenset()
    attachments: tuple[AttachmentMetadata, ...] = ()
```

Add `InputError` and the subclasses `InvalidInputSourceError`, `InputPathError`,
`InputMediaTypeError`, `InputSizeLimitError`, `InputNetworkError`, and
`UnsupportedInputModalityError`. Export all public input/config/content/error types from their package entry points.

- [ ] **Step 4: Write failing security and property tests before the resolver**

Cover these exact behaviors: a path beside but outside the configured temporary root raises
`InputPathError`; a symlink inside the root resolving outside raises `InputPathError`; a URL
containing `user:password@host` raises `InputNetworkError`; a public-looking redirect to
`127.0.0.1` raises `InputNetworkError`; payloads one byte over either configured byte limit raise
`InputSizeLimitError`; `audio/wav` attached as `ImageInput` raises `InputMediaTypeError`; and an
image resolved against `ModelCapabilities(image=False)` raises
`UnsupportedInputModalityError` before the injected HTTP transport records any request.

Use Hypothesis to generate arbitrary byte payloads up to 4096 bytes and safe Unicode filenames. Assert that `resolved.text_projection` never contains `base64.b64encode(payload).decode()` and never contains a `data:` URL.

Run: `uv run pytest tests/test_inputs.py -q`

Expected: FAIL in resolver/security behavior while the model-only tests pass.

- [ ] **Step 5: Implement shared URL validation and `InputResolver`**

Move the public-target DNS rules from `ToolPolicy.check_url()` into
`validate_http_url()`, then call it from both ToolPolicy and InputResolver on every redirect. Implement `InputResolver` public methods `resolve`, `resolve_attachment`, `read_path`, `download_url`, `validate_media_type`, `build_content_part`, and `build_text_projection`.

The resolver must:

```text
1. Normalize str to UserInput(text=value).
2. Check deployment capability before reading bytes or opening HTTP.
3. Resolve paths, enforce allowed roots, reject directories, and enforce size while reading chunks.
4. Download with httpx follow_redirects=False, validate each Location, and close every streamed response.
5. Enforce per-attachment and cumulative bytes.
6. Strip Content-Type parameters and validate against the kind allowlist.
7. Convert image/video/file bytes to data URLs; convert WAV/MP3 audio to raw Base64 plus format.
8. Return attachment metadata and a projection containing only kind, sanitized basename, MIME, and size.
```

Do not persist resolved bytes or include full URLs in exception details.

- [ ] **Step 6: Verify resolver behavior and shared Web Fetch regressions**

Run: `uv run pytest tests/test_inputs.py tests/test_tools.py tests/nfr/test_security.py -q && uv run ruff check . && uv run pyright`

Expected: all selected tests PASS, Ruff PASS, Pyright `0 errors, 0 warnings`.

- [ ] **Step 7: Commit typed secure inputs**

```powershell
git add pyproject.toml uv.lock src tests/test_inputs.py
git commit -m "feat(input): 增加安全的多模态输入模型与解析器"
```

---

### Task 3: Integrate multimodal inputs into Runtime and the unified state machine

**Files:**
- Modify: `src/praxis/protocols.py`
- Modify: `src/praxis/gateway/router.py`
- Modify: `src/praxis/runtime.py`
- Modify: `src/praxis/session/core.py`
- Modify: `src/praxis/models/context.py`
- Modify: `src/praxis/context/assembler.py`
- Modify: `src/praxis/orchestrator/loop.py`
- Modify: `src/praxis/verification/registry.py`
- Modify: `tests/test_runtime.py`
- Modify: `tests/test_session.py`
- Modify: `tests/test_context.py`
- Modify: `tests/test_orchestrator.py`
- Modify: `tests/test_verification.py`
- Modify: `tests/test_gateway.py`

**Interfaces:**
- Consumes: `InputResolver.resolve(InputValue, ModelCapabilities) -> ResolvedUserInput`.
- Produces: `ModelGateway.capabilities(model_name: str | None = None) -> ModelCapabilities`, `AgentSession.run(InputValue)`, and guaranteed history sanitation.

- [ ] **Step 1: Write failing capability and SDK propagation tests**

Add tests that assert:

```python
assert gateway.capabilities() == ModelCapabilities(image=True, audio=False, video=False, file=True)

response = await agent_session.run(UserInput(text="describe", parts=(ImageInput.from_bytes(PNG, media_type="image/png"),)))
assert response.content == "ok"
assert captured_messages[-1]["content"][1]["type"] == "image_url"

events = [event async for event in agent_session.run_stream(the_same_input)]
assert events
assert captured_stream_messages == captured_regular_messages
```

Add success, model-error, and cancellation cases that inspect `assembler.conversation_history` through its public field and assert no `data:` or known Base64 payload remains.

Run: `uv run pytest tests/test_runtime.py tests/test_session.py tests/test_context.py tests/test_orchestrator.py tests/test_gateway.py -q`

Expected: FAIL because public methods still accept strings and the gateway exposes only vision.

- [ ] **Step 2: Replace the vision-only capability interface**

Replace `supports_vision()` in `ModelGateway` and `GatewayRouter` with:

```python
def capabilities(self, model_name: str | None = None) -> ModelCapabilities:
    resolved_name = model_name or self.config.default_model
    return self.resolve_deployment(resolved_name).capabilities
```

Update visual verification to read `.image`. Update all test gateways to implement `capabilities()`. Do not preserve `supports_vision()`.

- [ ] **Step 3: Separate model content from safe text in run context**

Change `TurnContext` to:

```python
class TurnContext(BaseModel):
    user_content: str | list[ContentPart] | None
    user_text: str
    system_prompt_override: str | None = None
    developer_instructions: str = ""
    user_instructions: str = ""
    task_stage: str = "general"
    extra: dict[str, Any] = Field(default_factory=dict)
```

Add `input_history_index: int | None` and `safe_input_projection: str` to `RunContext`. Prompt assembly serializes content parts with `model_dump(mode="json")`; guardrails, memory, semantic search, skill activation, planning, and JIT use `user_text` only.

- [ ] **Step 4: Resolve inputs at Session and propagate through both paths**

Give `Session` an `InputResolver` and `ModelCapabilities`. Change `SessionRunner`, `Session`, `AgentSession`, `OrchestrationLoop.run`, and `run_stream` to accept `InputValue`. Both session methods execute:

```python
resolved = await self.input_resolver.resolve(user_input, self.model_capabilities)
return await self.loop.run(resolved, **kwargs)
```

The streaming method resolves once before iteration and yields the loop stream. `SessionFactory` receives `InputConfig`, obtains capabilities for the selected alias, and injects one resolver into the session.

- [ ] **Step 5: Keep binary history only inside the active run**

On the first assembled turn, append structured user content to conversation history and store its index in `RunContext`. Before the first model request, the current message contains the same structured content; later tool rounds reuse the structured history. Wrap the entire regular and streaming state machine in `try/finally` and call:

```python
def sanitize_run_input(self, context: RunContext) -> None:
    index = context.input_history_index
    if index is None:
        return
    self.assembler.conversation_history[index] = {
        "role": "user",
        "content": context.safe_input_projection,
    }
```

The `finally` block must run after success, tripwire, gateway exception, tool exception, generator close, task cancellation, and maximum-turn termination. Checkpoints continue to read only the now-sanitized `conversation_history`.

- [ ] **Step 6: Verify public runtime behavior and typed boundaries**

Run: `uv run pytest tests/test_gateway.py tests/test_context.py tests/test_orchestrator.py tests/test_session.py tests/test_runtime.py tests/test_verification.py -q && uv run ruff check . && uv run pyright`

Expected: all selected tests PASS, stream and regular captures are equivalent, no binary survives history, Ruff PASS, Pyright `0 errors, 0 warnings`.

- [ ] **Step 7: Commit state-machine integration**

```powershell
git add src tests
git commit -m "feat(runtime): 统一多模态会话与模型能力链路"
```

---

### Task 4: Replace overfit E2E and weak mock assertions

**Files:**
- Create: `tests/fixtures/openai_compatible_server.py`
- Create: `tests/e2e/test_runtime_http_contract.py`
- Create: `tests/scenarios/__init__.py`
- Move: `tests/e2e/test_scenario_*.py` to `tests/scenarios/`
- Move: scenario fixtures from `tests/e2e/conftest.py` to `tests/scenarios/conftest.py`
- Modify: `tests/test_gateway.py`
- Modify: `tests/test_mcp.py`
- Modify: `tests/test_tools.py`
- Modify: `tests/integration/test_component_chain.py`
- Modify: `tests/test_runtime.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: public Runtime, GatewayRouter, input models, and an actual localhost HTTP boundary allowed only inside tests.
- Produces: standard-library contract server with recorded JSON requests, deterministic JSON/SSE responses, and explicit shutdown/join.

- [ ] **Step 1: Reclassify mocked scenarios and write a failing HTTP E2E**

Move every test whose gateway is a `Mock`, `MagicMock`, `AsyncMock`, or scripted in-memory protocol from `tests/e2e` to `tests/scenarios`. Keep the `e2e` marker only for tests that cross a socket boundary.

Write `test_runtime_sends_all_content_parts_over_real_http` that starts the local server, configures GatewayRouter with its `http://127.0.0.1:<port>/v1` URL, runs `PraxisRuntime` and `AgentSession`, then asserts the server-recorded JSON has content types in this exact order:

```python
assert [part["type"] for part in content] == [
    "text", "image_url", "input_audio", "video_url", "file"
]
assert response.content == "contract-ok"
```

Run: `uv run pytest tests/e2e/test_runtime_http_contract.py -q`

Expected: FAIL because the local contract server fixture has not been implemented.

- [ ] **Step 2: Implement the standard-library OpenAI-compatible server**

Use `ThreadingHTTPServer(("127.0.0.1", 0), Handler)` and a non-daemon thread. The handler must:

```text
- accept POST /v1/chat/completions;
- read Content-Length bytes and decode JSON;
- append each request to a thread-safe public requests list;
- return OpenAI-compatible choices/message/usage JSON for regular calls;
- return data: JSON SSE frames followed by data: [DONE] for stream=true;
- return a deterministic forced tool call when tool_choice is present;
- support a blocking response event so cancellation/timeout tests can control timing;
- suppress default request logging without using a private method name;
- expose close() that calls shutdown(), server_close(), and thread.join(timeout=5);
- assert after close that thread.is_alive() is False.
```

Use a context manager fixture so shutdown happens even when assertions fail.

- [ ] **Step 3: Add behavior and metamorphic E2E coverage**

Add real-HTTP tests for regular response, SSE response, forced tool call, stream cancellation, timeout mapping, malformed JSON response, HTTP 401 mapping, and all five content types. Add a metamorphic assertion that regular and streaming requests contain identical user content after removing only `stream` and stream-option fields.

Run: `uv run pytest tests/e2e/test_runtime_http_contract.py -W error -q`

Expected: all E2E tests PASS and no unclosed socket, task, response, or background thread warning appears.

- [ ] **Step 4: Replace weak tests with observable outcomes**

For each previously identified weak case, add these observations in addition to or instead of mock-call assertions:

```text
- Gateway budget tests assert reservation totals and the exact raised exception.
- Callback tests assert meter totals or emitted audit/metric state after success and failure.
- MCP roots tests assert the public roots snapshot sent to the client.
- Redis tests assert stable encoded keys and returned health state through a behavioral fake.
- ToolPolicy allowed-network tests assert the normalized URL and rejection of a changed host.
- Component-chain metric tests assert collector contents.
- Runtime/session tests stop reading private runner/router/store/client fields.
```

Add a test-quality AST report that fails only when a test function contains none of: an `assert`, `pytest.raises`, `pytest.warns`, `pytest.fail`, or an explicit helper call named `assert_*`. It may exempt fixture functions but not test functions.

- [ ] **Step 5: Run scenario, integration, and E2E suites**

Run: `uv run pytest tests/scenarios tests/integration tests/e2e tests/test_gateway.py tests/test_mcp.py tests/test_tools.py tests/test_runtime.py -W error -q`

Expected: all selected tests PASS; mocked scenarios are not marked E2E; every test has an observable oracle.

- [ ] **Step 6: Commit the test architecture hardening**

```powershell
git add pyproject.toml tests
git commit -m "test: 强化多模态契约与行为验证"
```

---

### Task 5: Probe the specified endpoint and document actual capabilities

**Files:**
- Create: `tests/integration/test_gateway_live_multimodal.py`
- Modify: `tests/integration/test_gateway_live.py`
- Modify: `config.example.yaml`
- Modify: `README.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/CONFIGURATION.md`
- Modify: `docs/SECURITY.md`
- Modify: `docs/TROUBLESHOOTING.md`
- Modify: `examples/console.py`
- Modify: `examples/streaming_console.py`

**Interfaces:**
- Consumes: `PRAXIS_MODEL_API_KEY`, `openai/glm-5.1-openai`, `http://172.24.23.192:3000/v1`, and typed input constructors.
- Produces: repeatable live capability evidence plus examples/config that never overstate endpoint support.

- [ ] **Step 1: Write explicit live probes with minimal valid samples**

Generate samples in memory: a valid 1×1 PNG constant, a short PCM WAV created with `wave`, a minimal valid MP4 fixture represented as a Base64 test constant, and a UTF-8 text file payload. For each modality, send the exact provider content block directly through `chat()` with no retry and a 30-second timeout. Record only status/category and never response bodies containing input data.

Each live test must either assert a non-empty model response for a supported modality or assert the stable mapped `GatewayError` category selected after the first real probe. Do not use `xfail` for capability mismatches.

- [ ] **Step 2: Run the real endpoint probes with a transient environment variable**

Run in a child PowerShell process that sets `PRAXIS_MODEL_API_KEY` only for the command and removes it in `finally`:

```powershell
uv run pytest tests/integration/test_gateway_live.py tests/integration/test_gateway_live_multimodal.py -m live_model -W error -q
```

Expected: ordinary, streaming, tool, cancellation, timeout, and error-mapping tests PASS; each modality has an explicit supported or unsupported assertion based on the observed endpoint response. The key does not appear in output.

- [ ] **Step 3: Align configuration and public documentation with observed capabilities**

Set `gateway.deployments[0].capabilities` in `config.example.yaml` to `true` only for modalities proven successful in Step 2; all rejected modalities remain `false`. Document that SDK transport support and endpoint capability are distinct. Add runnable text and multimodal examples using `PRAXIS_MODEL_API_KEY`, allowed input paths, byte limits, local failure behavior, and remote URL opt-in.

- [ ] **Step 4: Verify docs, examples, and secret scanning**

Run: `uv run pytest tests/test_documentation.py tests/test_security_baseline.py tests/integration/test_gateway_live_multimodal.py -m "not live_model" -q && uv run python examples/console.py --help && uv run python examples/streaming_console.py --help`

Expected: documentation and secret checks PASS; both examples import and display help without an API key or network call.

- [ ] **Step 5: Commit endpoint evidence and docs**

```powershell
git add config.example.yaml README.md docs examples tests/integration
git commit -m "docs(input): 记录真实端点多模态能力与安全用法"
```

---

### Task 6: Run production gates and wheel smoke verification

**Files:**
- Modify: `uv.lock`
- Modify: `.github/workflows/ci.yml`
- Modify: `scripts/verify_wheel.py`
- Modify: `CHANGELOG.md`
- Modify: package files only when a gate identifies a concrete defect

**Interfaces:**
- Consumes: completed implementation and all project gates.
- Produces: locked, buildable, installable package with Windows/Linux CI coverage and no private-symbol or multimodal regressions.

- [ ] **Step 1: Update lock and CI coverage**

Run `uv lock`, then ensure CI runs the public-symbol policy, scenario/integration/E2E suites, documentation examples, and the existing Ruff/Pyright/security/build gates on Windows and Linux for Python 3.12, 3.13, and 3.14. Live tests remain explicit and credential-gated.

- [ ] **Step 2: Run the complete release gate**

Run exactly:

```powershell
uv sync --all-extras --frozen
uv run ruff check .
uv run pyright
uv run pytest -W error --strict-config --strict-markers --cov=praxis --cov-branch --cov-fail-under=90
uv run pip-audit
uv build
```

Expected: every command exits 0, Pyright reports `0 errors, 0 warnings`, coverage is at least 90%, pip-audit reports no known vulnerabilities, and both wheel and sdist are built.

- [ ] **Step 3: Install and inspect the wheel in a clean environment**

Create a temporary uv virtual environment outside the repository, install only the built wheel, and run:

```python
import praxis
from praxis import AudioInput, FileInput, ImageInput, UserInput, VideoInput
assert praxis.__version__ == "1.0.0"
assert UserInput(text="ok").text == "ok"
```

Then verify `praxis version`, `praxis config validate`, `praxis doctor`, `py.typed`, built-in skill resources, and clear optional-dependency errors. Expected: all smoke checks PASS without an API key, Redis, MCP, visual packages, or external services.

- [ ] **Step 4: Run a final secret/private-symbol/worktree audit**

Run:

```powershell
uv run pytest tests/test_public_api_policy.py tests/test_security_baseline.py -q
git status --short --branch
git diff --check
```

Expected: policy and secret tests PASS, no unexpected generated or credential files are present, and `git diff --check` prints nothing.

- [ ] **Step 5: Record release changes and commit the gate**

Add the public-symbol rule, multimodal SDK API, input security defaults, test reclassification, real endpoint capability evidence, and gate results to `CHANGELOG.md`, then commit:

```powershell
git add .github CHANGELOG.md scripts uv.lock
git commit -m "chore(release): 完成多模态输入生产门禁"
```
