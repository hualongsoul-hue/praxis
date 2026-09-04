# Praxis Production Correctness Hardening Design

## Status and scope

This specification defines the one-time, backward-incompatible hardening required before
Praxis can truthfully claim production readiness. It covers the defects confirmed by the
2026-09-04 architecture audit: verification, checkpoint safety, lifecycle cleanup, memory
recovery, public protocols, orchestration state, network and filesystem security, process-wide
resource control, health reporting, MCP recovery, event privacy, configuration semantics,
telemetry lifecycle, and non-overfit release tests.

The work stays on the current `dev` branch. It does not add compatibility adapters for old APIs
or checkpoint schemas. Windows and Linux remain first-class targets on Python 3.12, 3.13, and
3.14. Optional Redis, MCP, visual, and OTLP capabilities remain lazy and explicit.

## Design principles

1. Public protocols are executable contracts, not documentation-only facades.
2. Durable state is written at side-effect boundaries, not only at the end of a response.
3. A Runtime owns every shared resource and closes every owned resource even when another close
   operation fails.
4. Streaming and non-streaming are projections of one orchestration state machine.
5. Every path that touches the network or filesystem uses the same policy enforcement layer.
6. Health distinguishes configured, live, degraded, and failed capabilities.
7. Tests assert observable semantics and include negative cases that fail against the audited
   implementation.
8. Configuration snapshots are deeply immutable and relative paths are anchored to the loaded
   configuration file.

## 1. Verification pipeline

`VerifierRegistry.from_config()` will construct the enabled built-in verifiers rather than an
empty registry. Computational verification will be split into explicit request models:

- `LintVerificationRequest`: a bounded workspace path and optional rule selection.
- `TypeVerificationRequest`: a bounded workspace path and configured Pyright command.
- `SuiteVerificationRequest`: an argument-vector command and bounded working directory.
- `SchemaVerificationRequest`: a JSON value and JSON Schema.

Lint, type, and suite verifiers will execute through the existing cross-platform `ProcessRunner`
and `ToolPolicy`. They will never invoke a shell string. Exit code zero produces PASS, a normal
non-zero verification result produces FAIL with bounded stdout/stderr, and infrastructure,
policy, timeout, or cancellation failures produce ERROR. The orchestration policy will treat
ERROR as a verification failure that cannot silently approve output.

The orchestrator will create verifier requests from declared tool metadata and outcomes. A tool
without applicable verification metadata will produce an explicit SKIPPED result rather than an
invented PASS. Visual verification will be opt-in and routed through the same URL and path policy.

## 2. Durable execution and checkpoint schema

Checkpoint schema version 2 will persist a complete `ExecutionSnapshot` containing:

- conversation and cognitive-memory export;
- loop counters, token accounting, termination status, and pending feedback;
- plan identifier, steps, current step, and completion state;
- circuit-breaker and retry state required for deterministic continuation;
- tool-call ledger keyed by stable call ID and argument digest;
- approval decisions and their scope;
- active skill identifiers and resumable subagent references.

The tool-call ledger has PREPARED, STARTED, SUCCEEDED, FAILED, and UNCERTAIN states. Before a
non-idempotent tool runs, Praxis atomically checkpoints PREPARED and then STARTED. After the tool
returns, it atomically records the terminal state before another model turn. Recovery never
automatically repeats SUCCEEDED or UNCERTAIN non-idempotent calls. UNCERTAIN calls require an
approval decision or a tool-specific reconciliation handler.

Every durable transition is serialized through a per-session checkpoint lock. Checkpoint writes
remain checksum-protected and atomic. Schema 1 and malformed data fail with a typed incompatibility
or corruption exception; no implicit migration is provided.

Streaming completion, early consumer disconnect, cancellation, and generator close all finalize
usage accounting and attempt a terminal checkpoint in `finally`. Cancellation can interrupt the
active model stream or tool task; finalization is shielded only for a short configured deadline.

## 3. Runtime lifecycle and ownership

Runtime startup uses an ownership stack. As each owned dependency starts, its async closer is
registered. A later startup failure unwinds all successfully started owned resources in reverse
order. Injected resources have an explicit ownership flag and are not closed unless ownership was
transferred to the Runtime.

Shutdown has the following order:

1. mark the Runtime STOPPING and reject new sessions;
2. signal cancellation to sessions and supervised child tasks;
3. cancel/await supervised child tasks within the shutdown deadline;
4. close sessions concurrently and collect every failure;
5. flush and close audit, telemetry, embeddings, gateway, and persistence independently;
6. mark CLOSED and raise one typed `RuntimeCloseError` containing all cleanup failures.

Session close is a small state machine and becomes CLOSED only after cleanup completes. A failed
close remains retryable. The active run is stored as a task, allowing `cancel()` and `close()` to
perform real cancellation rather than only setting a cooperative flag.

OpenTelemetry setup returns a lifecycle handle that can force-flush and shut down locally created
providers. Host-provided providers are reused but never shut down by Praxis.

## 4. Memory durability and bounded indexing

Memory startup automatically rebuilds or loads the semantic index before the session becomes
ready. The persisted index manifest contains provider identity, embedding dimension, source
generation, and checksum. A mismatch triggers a deterministic rebuild; it never mixes vectors of
different dimensions.

Vector insertion rejects incorrect dimensions. Similarity requires equal non-zero dimensions.
Local lexical retrieval uses a dedicated, smaller configurable dimension with a production-safe
default, while remote providers must report their dimension or supply it in configuration.

The semantic index is Runtime-owned and shared safely by sessions using the same memory namespace.
Capacity enforcement occurs on insertion and startup, independently of Dream. Dream may compact
or consolidate memories, but disabling Dream cannot disable hard capacity limits.

## 5. Public boundary contracts

`ModelGateway` becomes the sole model boundary consumed by orchestration, memory, verification,
and MCP sampling. Its public methods cover:

- normalized completion and stream requests;
- capability and price discovery;
- concurrency/budget reservation as an internal gateway responsibility;
- health probing;
- async lifecycle.

Callers will not reach into LiteLLM router methods. `LiteLLMGateway` implements the protocol and
encapsulates deployment selection, reservations, and error mapping.

`StorageBackend` remains the byte/key-value persistence protocol. `PersistenceStore` composes a
backend but Runtime accepts either the store or a backend and constructs the store once. Ownership
is explicit for both paths. All adapters pass the same backend contract test suite.

No `cast(Any, gateway)` is allowed at the public boundary. Pyright strict must prove that custom
protocol implementations can create and run a Session in a contract test.

## 6. Unified orchestration engine

One async transition engine owns the execution state. It yields typed internal transitions for
input guardrail, prompt assembly, model delta/completion, parse, approval, tool execution,
verification, checkpoint, output guardrail, and termination.

`run()` consumes transitions and returns the final response. `stream()` projects the same
transitions into public events as they occur. Neither method duplicates turn control flow.

Each user request creates a new plan scope and clears the previous plan even when the old plan is
complete, abandoned, or terminated naturally. Every plan and step has a stable identifier saved
in checkpoints. `plan_created` is emitted exactly once before the first `turn_started` event and
is visible to streaming consumers.

## 7. Runtime-wide resource control

The Runtime creates and owns one `ResourceController` for tools and subagents. It provides:

- process-wide subagent concurrency and queue limits;
- process-wide read-tool concurrency;
- keyed write locks for canonical resource identifiers;
- global cancellation and per-session child-task groups;
- bounded result aggregation.

Sessions receive leases from this controller rather than constructing semaphores. Closing a
session cancels only its leases; closing the Runtime cancels all leases. Subagent isolation still
creates independent Session state and passes only the configured tool subset.

## 8. Filesystem, process, and network safety

All blocking filesystem traversal, file reads/writes, regex search, and process waiting run outside
the event-loop thread. File reads enforce the byte limit while streaming instead of after loading
the whole file. Writes and edits use an atomic temporary-file replacement inside the authorized
root.

Path operations open a verified handle where the platform supports it and revalidate the final
canonical target immediately before mutation. Windows junctions/reparse points and Linux symlinks
receive symmetric escape tests. Search has file-count, total-byte, per-file-byte, match-count, and
deadline limits. Regex execution uses a bounded engine or rejects unsafe expressions.

Every HTTP client, browser navigation, redirect, and callback uses one `NetworkPolicy`:

- HTTP/HTTPS only;
- URL credentials rejected;
- private, loopback, link-local, multicast, and unspecified destinations rejected by default;
- every redirect and resolved address revalidated;
- connection pinned to the validated address where applicable;
- response and download byte limits applied while streaming.

The configured private model endpoint is allowed only by the model gateway allow-list and does not
relax autonomous tool networking. Documentation will explicitly state that the configured HTTP
transport requires a trusted encrypted overlay or TLS termination in production.

## 9. Health and MCP recovery

Health results include status, last probe time, latency, reason, and whether the capability is
required. Readiness probes perform bounded live checks:

- model: endpoint/capability probe with recent circuit state;
- storage: namespaced write-read-delete probe;
- embedding: provider probe or deterministic local self-test;
- verification: at least one enabled verifier plus local runner self-test;
- background task: task liveness and last failure;
- MCP: per-session/per-server status, never a union across sessions.

Probe results are cached for a short configurable interval to keep readiness cheap. Required
capability failure makes the Runtime FAILED; optional failure makes it DEGRADED.

Each MCP connection has a supervised lifecycle with exponential backoff and jitter. Unexpected
disconnect changes health immediately, rejects new calls with a typed unavailable error, and
attempts reconnect until Runtime shutdown. Configuration requires unique names and transport-
specific fields. Credentials in headers or environment values use secret references rather than
being displayed by config or health output.

## 10. Typed, privacy-safe events

Public events become a discriminated union with a schema version, monotonically increasing
sequence, runtime/session/run correlation identifiers, timestamp, and typed payload. Unknown event
names cannot be emitted.

Tool events contain tool name, call ID, status, duration, and redacted summaries by default. Full
arguments/results require an explicitly configured trusted event sink and still pass size and
secret redaction limits. Raw exception strings are not exposed to event consumers or model history;
typed error codes and safe messages are used instead. Audit, log, trace, and event redaction share
one immutable policy.

## 11. Configuration and CLI semantics

Configuration models use immutable tuples and immutable mappings for nested collections. Validators
enforce unique model aliases, unique MCP names, required backend/transport fields, compatible
embedding dimensions, positive byte/concurrency limits, and cross-field capability constraints.

`load_config(path)` anchors relative paths to the resolved parent directory of `path`. Programmatic
configuration has an explicit `base_path`; if omitted it uses a documented resolved current
directory snapshot once during construction.

Tool metadata timeout becomes optional. An absent tool timeout uses `tools.default_timeout`; an
explicit tool timeout overrides it.

Both packaged CLI and the interactive example use the same Runtime bootstrap. CLI logging is the
only place allowed to install default handlers. Metrics and locally created tracing providers are
flushed and closed with the Runtime. `config show` remains redacted.

Module-level rule collections are immutable tuples/frozensets. Runtime-specific rule composition
never mutates process-wide state.

## 12. Verification strategy and release gates

Every defect begins with a failing behavioral test. Required test groups include:

- real temporary Python projects proving lint/type/test PASS, FAIL, ERROR, timeout, and policy denial;
- crash injection before and after non-idempotent tool completion;
- startup and shutdown failure injection for every owned resource;
- memory lookup after a new Runtime and dimension mismatch rejection;
- custom `ModelGateway` and `StorageBackend` contract implementations;
- two consecutive planned requests on one Session and event ordering for stream/non-stream;
- SSRF redirect, visual navigation, traversal, symlink/junction race, large-file, ReDoS, and
  event-loop responsiveness tests on Windows/Linux;
- multi-session global concurrency and cancellation tests;
- health false-positive tests and post-start MCP disconnect/reconnect tests;
- event schema, ordering, payload bounds, and secret redaction tests;
- config deep-immutability, duplicate identity, transport requirements, path anchoring, and default
  timeout tests.

Critical tests use real local files, subprocesses, TCP servers, and the stdio MCP fixture. Mocks are
limited to external model calls or deterministic failure injection; assertions target Praxis state
and outputs rather than mock call counts.

The release gate remains:

```powershell
uv lock
uv sync --all-extras --frozen
uv run ruff check .
uv run pyright
uv run pytest -W error --strict-config --strict-markers --cov=praxis --cov-branch --cov-fail-under=90
uv run pip-audit
uv build
```

A clean-environment wheel smoke test verifies SDK execution, CLI, `py.typed`, built-in skill
resources, and optional-dependency errors. Live endpoint tests are a separate credentialed release
job covering non-streaming, streaming, tools, multimodal input, cancellation, timeout, and error
mapping. The API key remains exclusively in `PRAXIS_MODEL_API_KEY` and is never echoed or persisted.

## Migration and completion criteria

Old checkpoints fail explicitly and old gateway/session APIs are removed rather than wrapped.
Completion requires all behavioral tests and release gates to pass on the current branch, no
uncommitted changes, and Chinese Conventional Commit messages grouped by coherent subsystem. No
claim of production readiness is made while a P0 or P1 item in this specification remains
unimplemented.
