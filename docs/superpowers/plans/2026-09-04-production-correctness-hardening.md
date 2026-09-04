# Praxis Production Correctness Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate every confirmed production-readiness defect in verification, recovery, lifecycle, memory, protocols, orchestration, security, resource control, health, MCP, events, configuration, telemetry, and release tests.

**Architecture:** Replace documentation-only abstractions with executable protocols, route streaming and non-streaming through one transition engine, and move shared limits and resource ownership to `PraxisRuntime`. Persist side-effect state at tool boundaries and apply one filesystem/network/privacy policy to every adapter.

**Tech Stack:** Python 3.12–3.14, asyncio, Pydantic v2, LiteLLM, httpx, SQLAlchemy/aiosqlite, MCP, OpenTelemetry, pytest, Hypothesis, Ruff, Pyright strict, uv.

**Spec:** `docs/superpowers/specs/2026-09-04-production-correctness-hardening-design.md`

## Global Constraints

- Work only on the current `dev` branch; do not create a branch or worktree.
- Do not preserve old public APIs or old checkpoint compatibility.
- Do not create module-level names beginning with a single underscore.
- Read the model API key only from `PRAXIS_MODEL_API_KEY`; never persist or display it.
- Keep Windows and Linux behavior symmetric on Python 3.12, 3.13, and 3.14.
- Keep Redis, MCP, visual, and OTLP optional and lazily imported.
- Every production behavior change starts with a failing semantic test.
- Use Chinese Conventional Commit messages.

---

### Task 1: Make computational verification real and fail closed

**Files:**
- Modify: `src/praxis/models/verification.py`
- Modify: `src/praxis/verification/computational.py`
- Modify: `src/praxis/verification/registry.py`
- Modify: `src/praxis/orchestrator/loop.py`
- Modify: `src/praxis/tools/process.py`
- Test: `tests/test_verification.py`
- Test: `tests/integration/test_component_chain.py`

**Interfaces:**
- Produces typed lint, type, suite, and schema requests.
- `VerifierRegistry.from_config()` registers each enabled verifier.
- Process-backed verifiers return PASS only after a real zero exit code, FAIL for verification failures, and ERROR for infrastructure failures.

- [ ] Write tests that create real temporary projects with valid and invalid Python, a failing test command, a missing executable, timeout, and denied working directory.
- [ ] Run `uv run pytest -q tests/test_verification.py tests/integration/test_component_chain.py` and confirm the new cases fail because the registry is empty or verifiers return invented PASS.
- [ ] Add typed request models and execute argument vectors through `ProcessRunner` under `ToolPolicy`.
- [ ] Register enabled verifiers and make ERROR fail closed in orchestration.
- [ ] Run the targeted tests, Ruff, and Pyright.
- [ ] Commit with `fix(verification): 执行真实验证并默认失败关闭`.

### Task 2: Persist side-effect execution state safely

**Files:**
- Modify: `src/praxis/models/session.py`
- Modify: `src/praxis/models/tools.py`
- Modify: `src/praxis/session/checkpoint.py`
- Modify: `src/praxis/session/core.py`
- Modify: `src/praxis/session/resume.py`
- Modify: `src/praxis/orchestrator/tool_coordination.py`
- Modify: `src/praxis/orchestrator/strategy.py`
- Modify: `src/praxis/persistence/checkpoint.py`
- Test: `tests/test_session.py`
- Test: `tests/test_persistence.py`
- Test: `tests/scenarios/test_scenario_session_resume.py`

**Interfaces:**
- Adds `ToolExecutionState` and immutable `ToolExecutionRecord` keyed by call ID and argument digest.
- Checkpoint schema stores plan state, approvals, retry/circuit state, skills, and the execution ledger.
- Non-idempotent SUCCEEDED/UNCERTAIN calls are never automatically repeated.

- [ ] Write crash-injection tests before tool start, during an uncertain write, and after a successful write; assert recovery behavior using a real file counter.
- [ ] Run the targeted tests and confirm duplicate execution or missing checkpoint state.
- [ ] Add the versioned execution snapshot and atomic checkpoint transitions around tool execution.
- [ ] Finalize streaming accounting/checkpointing in `finally`, including generator close and cancellation.
- [ ] Reject legacy checkpoint schema with the existing typed incompatibility error.
- [ ] Run persistence, session, and resume scenario suites.
- [ ] Commit with `fix(recovery): 持久化工具副作用与完整执行状态`.

### Task 3: Make Runtime and Session lifecycle exception-safe

**Files:**
- Modify: `src/praxis/lifecycle.py`
- Modify: `src/praxis/runtime.py`
- Modify: `src/praxis/session/core.py`
- Modify: `src/praxis/models/runtime.py`
- Modify: `src/praxis/exceptions.py`
- Modify: `src/praxis/telemetry/tracing.py`
- Test: `tests/test_runtime.py`
- Test: `tests/test_session.py`
- Test: `tests/test_telemetry.py`
- Test: `tests/nfr/test_reliability.py`

**Interfaces:**
- Adds a public async ownership stack and aggregated `RuntimeCloseError`.
- Session stores its active run task and supports actual cancellation.
- Tracing returns a lifecycle handle with `force_flush()` and `close()` while preserving host ownership.

- [ ] Write parameterized failure-injection tests for each startup and close resource and assert every other owned resource closes exactly once.
- [ ] Write tests proving an in-flight run is cancelled promptly and a failed Session close can be retried.
- [ ] Run targeted tests and confirm current leaks/blocking behavior.
- [ ] Implement reverse-order startup rollback and error-aggregating close with the specified shutdown order.
- [ ] Track/cancel the active Session task and make close-state transitions retryable.
- [ ] Add tracing provider lifecycle ownership and flush tests.
- [ ] Run targeted suites, Ruff, and Pyright.
- [ ] Commit with `fix(runtime): 保证生命周期回滚取消与完整关闭`.

### Task 4: Restore memory across Runtime restarts and bound resource use

**Files:**
- Modify: `src/praxis/config/schemas.py`
- Modify: `src/praxis/memory/core.py`
- Modify: `src/praxis/memory/store.py`
- Modify: `src/praxis/memory/vector.py`
- Modify: `src/praxis/memory/retriever.py`
- Modify: `src/praxis/memory/dream.py`
- Modify: `src/praxis/runtime.py`
- Test: `tests/test_cognitive_memory.py`
- Test: `tests/nfr/test_scalability.py`

**Interfaces:**
- Runtime owns a namespace-aware semantic index.
- Vector insertion and similarity require exact dimensions.
- Capacity enforcement runs on startup and insertion, independent of Dream.

- [ ] Write a two-Runtime persistence test: store memory in Runtime A, close it, start Runtime B, and retrieve the memory semantically without calling a rebuild helper.
- [ ] Write mismatch, zero-dimension, capacity-with-Dream-disabled, and bounded-memory tests.
- [ ] Run the tests and confirm empty restart results and dimension truncation.
- [ ] Load/rebuild the index during startup, persist a provider/dimension/source manifest, and reject mismatches.
- [ ] Reduce the local lexical default dimension and enforce capacity independently of Dream.
- [ ] Share the index through Runtime ownership without sharing Session conversation state.
- [ ] Run memory and scalability suites.
- [ ] Commit with `fix(memory): 恢复持久化索引并限制内存容量`.

### Task 5: Enforce public Gateway and Storage protocols

**Files:**
- Modify: `src/praxis/protocols.py`
- Modify: `src/praxis/gateway/router.py`
- Modify: `src/praxis/gateway/chat.py`
- Modify: `src/praxis/gateway/tasks.py`
- Modify: `src/praxis/orchestrator/loop.py`
- Modify: `src/praxis/session/core.py`
- Modify: `src/praxis/memory/consolidator.py`
- Modify: `src/praxis/memory/extractor.py`
- Modify: `src/praxis/memory/dream.py`
- Modify: `src/praxis/verification/inferential.py`
- Modify: `src/praxis/runtime.py`
- Modify: `src/praxis/persistence/store.py`
- Test: `tests/test_gateway.py`
- Test: `tests/test_runtime.py`
- Test: `tests/test_persistence.py`

**Interfaces:**
- All consumers depend only on `ModelGateway` normalized request methods.
- Runtime accepts either `PersistenceStore` or `StorageBackend` with explicit ownership.
- LiteLLM-specific deployment and budget mechanics stay inside the adapter.

- [ ] Write minimal custom Gateway and StorageBackend contract implementations and run a real Session through them.
- [ ] Run tests and confirm the custom Gateway fails at concrete router calls.
- [ ] Move completion/stream/reservation behavior behind `ModelGateway` and replace concrete type dependencies.
- [ ] Remove public-boundary `cast(Any, ...)` and support backend-to-store composition.
- [ ] Run gateway, runtime, persistence, Pyright, and Ruff.
- [ ] Commit with `refactor(protocols): 落实模型与存储公共契约`.

### Task 6: Unify orchestration and isolate plan state per request

**Files:**
- Create: `src/praxis/orchestrator/engine.py`
- Modify: `src/praxis/orchestrator/loop.py`
- Modify: `src/praxis/orchestrator/events.py`
- Modify: `src/praxis/orchestrator/strategy.py`
- Modify: `src/praxis/session/core.py`
- Test: `tests/test_orchestrator.py`
- Test: `tests/scenarios/test_scenario_single_turn.py`

**Interfaces:**
- `OrchestrationEngine.transitions()` is the only turn-control loop.
- `run()` reduces transitions to a response; `stream()` projects every public event.
- A request-scoped plan is reset and identified independently of previous requests.

- [ ] Write a test that executes two planned requests on one Session and proves the second receives a new plan.
- [ ] Write event-order tests proving `plan_created` precedes `turn_started` for streaming and that stream/non-stream terminal state and usage match.
- [ ] Run tests and confirm stale-plan reuse and missing stream event.
- [ ] Extract the shared transition engine and delete the duplicated stream/non-stream loops.
- [ ] Persist and restore request-scoped plan identifiers through Task 2’s snapshot.
- [ ] Run orchestration and scenario suites.
- [ ] Commit with `refactor(orchestrator): 统一流式状态机与计划生命周期`.

### Task 7: Apply Runtime-wide concurrency and cancellation

**Files:**
- Modify: `src/praxis/lifecycle.py`
- Modify: `src/praxis/runtime.py`
- Modify: `src/praxis/tools/executor.py`
- Modify: `src/praxis/subagent/resource_control.py`
- Modify: `src/praxis/subagent/tools.py`
- Modify: `src/praxis/subagent/spawn.py`
- Test: `tests/test_tools.py`
- Test: `tests/test_subagent.py`
- Test: `tests/nfr/test_scalability.py`

**Interfaces:**
- Runtime owns one public `ResourceController` with subagent/read leases and keyed write locks.
- Session close cancels its child group; Runtime close cancels all child groups.

- [ ] Write multi-session tests using real async barriers and assert total concurrency never exceeds the Runtime limit.
- [ ] Write cancellation and keyed-write serialization tests across two Sessions.
- [ ] Run tests and confirm per-Session limits multiply.
- [ ] Move controller creation to Runtime and inject leases into Session tool/subagent wiring.
- [ ] Implement per-session task groups and Runtime-wide cancellation.
- [ ] Run tools, subagent, and scalability suites.
- [ ] Commit with `fix(concurrency): 统一运行时级资源与取消控制`.

### Task 8: Harden filesystem, search, and all network entry points

**Files:**
- Modify: `src/praxis/network.py`
- Modify: `src/praxis/tools/policy.py`
- Modify: `src/praxis/tools/process.py`
- Modify: `src/praxis/tools/builtins/file_ops/read_file.py`
- Modify: `src/praxis/tools/builtins/file_ops/write_file.py`
- Modify: `src/praxis/tools/builtins/file_ops/edit_file.py`
- Modify: `src/praxis/tools/builtins/search/grep_search.py`
- Modify: `src/praxis/tools/builtins/search/code_search.py`
- Modify: `src/praxis/tools/builtins/search/find_by_name.py`
- Modify: `src/praxis/tools/builtins/network/web_search.py`
- Modify: `src/praxis/verification/visual.py`
- Test: `tests/test_tools.py`
- Test: `tests/test_production_edges.py`
- Test: `tests/nfr/test_security.py`

**Interfaces:**
- One `NetworkPolicy` validates initial URLs, DNS answers, redirects, browser navigation, and byte limits.
- Filesystem tools use bounded worker-thread operations and atomic verified writes.
- Search limits file count, bytes, matches, regex complexity, and deadline.

- [ ] Write SSRF redirect, URL credential, private DNS, visual URL, screenshot traversal, symlink/junction escape, large-file, unsafe-regex, timeout, and event-loop responsiveness tests.
- [ ] Run tests and confirm policy bypasses and blocking behavior.
- [ ] Route web search and visual navigation through `NetworkPolicy` and validate every redirect/request.
- [ ] Implement bounded streaming reads, atomic writes, final path revalidation, and worker-thread search.
- [ ] Add search bounds and reject unsafe regular expressions before traversal.
- [ ] Run security, production-edge, and tool suites on the current platform.
- [ ] Commit with `fix(security): 统一网络文件与搜索安全边界`.

### Task 9: Make health probes truthful and MCP self-healing

**Files:**
- Modify: `src/praxis/models/runtime.py`
- Modify: `src/praxis/models/mcp.py`
- Modify: `src/praxis/config/schemas.py`
- Modify: `src/praxis/gateway/router.py`
- Modify: `src/praxis/runtime.py`
- Modify: `src/praxis/tools/mcp/connection.py`
- Modify: `src/praxis/tools/mcp/wiring.py`
- Modify: `src/praxis/tools/mcp/transport.py`
- Test: `tests/test_runtime.py`
- Test: `tests/test_mcp.py`
- Test: `tests/scenarios/test_scenario_mcp_interaction.py`

**Interfaces:**
- Health components expose last probe time, latency, reason, and required flag.
- Model/storage/embedding/verification probes execute bounded live self-tests.
- Each MCP server has a supervised reconnect loop and independent status.

- [ ] Write tests for configured-but-dead model, read-only storage, empty verifier registry, per-session MCP status, and post-start stdio server death/restart.
- [ ] Run tests and confirm false READY and no reconnect.
- [ ] Add cached live probes and aggregate required/optional status without cross-session union.
- [ ] Validate unique MCP names and required transport fields.
- [ ] Supervise disconnect detection and exponential-backoff reconnect until shutdown.
- [ ] Run runtime, MCP, and scenario suites.
- [ ] Commit with `fix(health): 增加真实探测与MCP断线恢复`.

### Task 10: Publish typed, ordered, redacted events

**Files:**
- Modify: `src/praxis/models/orchestrator.py`
- Modify: `src/praxis/models/telemetry.py`
- Modify: `src/praxis/orchestrator/events.py`
- Modify: `src/praxis/orchestrator/tool_coordination.py`
- Modify: `src/praxis/telemetry/audit.py`
- Modify: `src/praxis/telemetry/logger.py`
- Test: `tests/test_orchestrator.py`
- Test: `tests/test_telemetry.py`
- Test: `tests/test_security_baseline.py`

**Interfaces:**
- Agent events form a discriminated union with schema version, sequence, runtime/session/run IDs, and typed payload.
- One immutable redaction policy serves event, log, audit, and trace adapters.
- Tool payloads expose metadata and bounded redacted summaries by default.

- [ ] Write tests rejecting unknown event types, asserting monotonic sequence/correlation, bounding large values, and removing nested keys/tokens/URL credentials/raw exceptions.
- [ ] Run tests and confirm arbitrary names and sensitive raw payloads are accepted.
- [ ] Add typed event payloads and strict emitter validation.
- [ ] Apply shared recursive redaction and safe error mapping to tool events and model history.
- [ ] Run event, telemetry, and security tests.
- [ ] Commit with `fix(events): 提供类型化有序脱敏事件协议`.

### Task 11: Make configuration deeply immutable and wire CLI telemetry

**Files:**
- Modify: `src/praxis/config/loader.py`
- Modify: `src/praxis/config/schemas.py`
- Modify: `src/praxis/models/mcp.py`
- Modify: `src/praxis/models/tools.py`
- Modify: `src/praxis/guardrails/rules.py`
- Modify: `src/praxis/recovery/classifier.py`
- Modify: `src/praxis/__main__.py`
- Modify: `src/praxis/runtime.py`
- Modify: `examples/interactive_console.py`
- Modify: `config.example.yaml`
- Modify: `examples/config.yaml`
- Test: `tests/test_config.py`
- Test: `tests/test_cli.py`
- Test: `tests/test_interactive_console.py`

**Interfaces:**
- Nested config collections are tuples/frozen mappings and relative paths are config-file-relative.
- Semantic validators reject duplicate aliases/names and incomplete backend/transport configurations.
- CLI and example share one telemetry-aware Runtime bootstrap.

- [ ] Write mutation, duplicate, incomplete transport/backend, path-anchor, default-timeout, CLI metrics, and tracing-shutdown tests.
- [ ] Run tests and confirm mutable nested values, accepted invalid configs, CWD-based paths, and dead timeout/telemetry config.
- [ ] Convert collections and rule tables to immutable forms and add cross-field validators.
- [ ] Anchor paths in the loader and make tool timeout fallback reachable.
- [ ] Share bootstrap/lifecycle code between CLI and the example.
- [ ] Run config, CLI, example, Ruff, and Pyright.
- [ ] Commit with `fix(config): 强化不可变语义与命令行配置生效`.

### Task 12: Close documentation, live testing, and release gates

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `README.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/CONFIGURATION.md`
- Modify: `docs/DEPLOYMENT.md`
- Modify: `docs/EXTENDING.md`
- Modify: `docs/SECURITY.md`
- Modify: `docs/TROUBLESHOOTING.md`
- Modify: `tests/test_documentation.py`
- Modify: `tests/test_test_quality.py`
- Modify: `tests/integration/test_gateway_live.py`
- Modify: `tests/integration/test_gateway_live_multimodal.py`
- Modify: `scripts/verify_wheel.py`

**Interfaces:**
- CI non-secret jobs remain fully offline; a credentialed release job exercises the configured live endpoint without displaying its key.
- Documentation examples are executable and match the final public protocols and event models.

- [ ] Add semantic meta-tests for fake PASS verifiers and implementation-only assertions, plus executable documentation examples.
- [ ] Add a credentialed release workflow covering ordinary, streaming, tool, multimodal, cancellation, timeout, and error mapping behavior.
- [ ] Update architecture/security/deployment/extension/configuration docs and explicitly document trusted transport requirements for the private HTTP model endpoint.
- [ ] Update wheel smoke to exercise a custom protocol adapter, `py.typed`, built-in skill resources, CLI, and optional dependency messages.
- [ ] Run `uv lock` and `uv sync --all-extras --frozen`.
- [ ] Run Ruff, Pyright, the strict full suite with branch coverage, pip-audit, and `uv build`.
- [ ] Install the wheel in a clean temporary environment and run SDK/CLI/resource smoke tests.
- [ ] Run the live endpoint tests with `PRAXIS_MODEL_API_KEY` supplied only through the process environment and report results without exposing the secret.
- [ ] Commit with `chore(release): 完成生产正确性发布门禁`.
