"""Production boundary regression tests for optional and failure-path adapters."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from praxis.config.schemas import GatewayConfig, ModelDeployment, TelemetryConfig, ToolsConfig
from praxis.exceptions import PersistenceError
from praxis.gateway.metering import (
    DEFAULT_MAX_TOKENS,
    check_budget,
    completion_cost,
    estimate_input_cost,
    estimate_output_cost,
    get_max_tokens,
    record_usage,
)
from praxis.gateway.router import GatewayRouter
from praxis.memory.dream import DreamConsolidator
from praxis.memory.vector import VectorStore, cosine_similarity, local_lexical_embed, tei_embed
from praxis.models.memory import MemoryEntry, MemoryScope, MemoryStatus, MemoryType, ScopeType
from praxis.models.responses import (
    FunctionCallDelta,
    ModelResponse,
    ModelResponseChunk,
    ToolCallDelta,
    Usage,
)
from praxis.models.verification import VerificationResult, VerificationStatus, VerificationType
from praxis.orchestrator.parser import OutputParser, StreamAccumulator
from praxis.persistence.backends.redis import RedisBackend
from praxis.recovery.classifier import classify_by_type_name
from praxis.telemetry.tracing import configure_tracing, get_tracer, otlp_processor, start_span
from praxis.tools.builtins.autonomy.update_notes import create_handler as notes_handler
from praxis.tools.builtins.autonomy.update_plan import create_handler as plan_handler
from praxis.tools.builtins.network.web_search import create_handler as search_handler
from praxis.tools.builtins.search.code_search import create_handler as code_search_handler
from praxis.tools.builtins.search.find_by_name import create_handler as find_handler
from praxis.tools.builtins.search.grep_search import create_handler as grep_handler
from praxis.tools.builtins.system.sleep import handle as sleep_handler
from praxis.tools.policy import ToolPolicy
from praxis.verification.visual import VisualVerifier, run_visual


def model_response(content: str) -> ModelResponse:
    return ModelResponse(
        id="response-1",
        content=content,
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        model="default",
        created=0,
    )


class TestStreamAccumulatorEdges:
    def test_accumulates_all_stream_fields_and_tool_calls(self) -> None:
        accumulator = StreamAccumulator()
        usage = Usage(prompt_tokens=3, completion_tokens=4, total_tokens=7)
        first = accumulator.feed(ModelResponseChunk(
            id="stream-1",
            model="default",
            delta_content="answer",
            delta_reasoning_content="reason",
            delta_refusal="refusal",
            delta_tool_calls=[ToolCallDelta(
                index=1,
                id="call-1",
                function=FunctionCallDelta(name="look", arguments='{"q":'),
            )],
            system_fingerprint="fp-1",
        ))
        accumulator.feed(ModelResponseChunk(
            id="stream-1",
            delta_tool_calls=[ToolCallDelta(
                index=1,
                function=FunctionCallDelta(name="up", arguments='"x"}'),
            )],
            usage=usage,
            finish_reason="tool_calls",
        ))

        response = accumulator.build_response()
        assert first.model_dump() == {
            "content": "answer",
            "reasoning": "reason",
            "refusal": "refusal",
        }
        assert response.reasoning_content == "reason"
        assert response.refusal == "refusal"
        assert response.usage == usage
        assert response.finish_reason == "tool_calls"
        assert response.system_fingerprint == "fp-1"
        assert response.tool_calls is not None
        assert response.tool_calls[0].function.name == "lookup"
        assert response.tool_calls[0].function.arguments == '{"q":"x"}'

    def test_empty_accumulator_and_invalid_schema_fall_back_safely(self) -> None:
        response = StreamAccumulator().build_response()
        assert response.content is None
        assert response.tool_calls is None
        assert response.usage.total_tokens == 0
        assert OutputParser.extract_schema_response("not-json", MemoryEntry) is None


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}
        self.keys: list[bytes | str] = []
        self.ping = AsyncMock(return_value=True)
        self.set = AsyncMock(side_effect=self.set_value)
        self.get = AsyncMock(side_effect=self.get_value)
        self.delete = AsyncMock(side_effect=self.delete_value)
        self.aclose = AsyncMock()

    async def set_value(self, key: str, value: bytes, nx: bool = False) -> bool:
        if nx and key in self.values:
            return False
        self.values[key] = value
        return True

    async def get_value(self, key: str) -> bytes | None:
        return self.values.get(key)

    async def delete_value(self, *keys: bytes | str) -> int:
        return len(keys)

    async def scan_iter(self, match: str):
        del match
        for key in self.keys:
            yield key


class TestRedisBackendEdges:
    async def test_create_requires_url_and_pings_client(self) -> None:
        with pytest.raises(PersistenceError, match="redis_url"):
            await RedisBackend.create(None)

        client = FakeRedis()
        with patch("praxis.persistence.backends.redis.aioredis.from_url", return_value=client):
            backend = await RedisBackend.create("redis://example")
        client.ping.assert_awaited_once()
        await backend.close()
        client.aclose.assert_awaited_once()

    async def test_crud_listing_and_batched_namespace_clear(self) -> None:
        client = FakeRedis()
        backend = RedisBackend(client)  # type: ignore[arg-type]
        await backend.save("ns", "one", b"1")
        assert await backend.load("ns", "one") == b"1"
        assert await backend.save_if_absent("ns", "one", b"2") is False
        assert await backend.save_if_absent("ns", "two", b"2") is True
        assert await backend.load("ns", "missing") is None
        await backend.delete("ns", "one")

        client.keys = [b"praxis:ns:b", "praxis:ns:a"]
        assert await backend.list_keys("ns") == ["a", "b"]
        assert await backend.list_keys("ns", prefix="a") == ["a", "b"]

        client.keys = [f"praxis:ns:{index}".encode() for index in range(501)]
        assert await backend.clear_namespace("ns") == 501
        assert client.delete.await_count >= 3


class TestVisualVerifierEdges:
    async def test_evaluate_screenshot_parses_pass_and_invalid_payload(
        self,
        tmp_path: Path,
    ) -> None:
        image = tmp_path / "page.png"
        image.write_bytes(b"png")
        gateway = MagicMock()
        gateway.config.default_model = "default"
        verifier = VisualVerifier(gateway, screenshot_dir=str(tmp_path))

        with patch(
            "praxis.verification.visual.chat",
            AsyncMock(return_value=model_response(
                '{"pass":true,"confidence":0.9,"description":"ok",'
                '"differences":[]}'
            )),
        ) as chat_call:
            result = await verifier.evaluate_screenshot(image, "looks right")
        assert result.status is VerificationStatus.PASS
        assert result.score == 0.9
        assert chat_call.await_args.kwargs["model"] == "default"

        with patch(
            "praxis.verification.visual.chat",
            AsyncMock(return_value=model_response("[]")),
        ):
            invalid = await verifier.evaluate_screenshot(image, "looks right")
        assert invalid.status is VerificationStatus.ERROR

    async def test_verify_reports_capture_evaluation_and_success_paths(
        self,
        tmp_path: Path,
    ) -> None:
        gateway = MagicMock()
        gateway.config.default_model = "default"
        verifier = VisualVerifier(gateway, screenshot_dir=str(tmp_path))
        verifier.capture_screenshot = AsyncMock(side_effect=RuntimeError("browser"))
        capture_error = await verifier.verify("https://example.com", "ok")
        assert capture_error.status is VerificationStatus.ERROR
        assert "截图失败" in capture_error.feedback

        verifier.capture_screenshot = AsyncMock(return_value=tmp_path / "shot.png")
        verifier.evaluate_screenshot = AsyncMock(side_effect=RuntimeError("model"))
        model_error = await verifier.verify("https://example.com", "ok")
        assert model_error.status is VerificationStatus.ERROR
        assert model_error.metadata["screenshot"].endswith("shot.png")

        passed = VerificationResult(
            status=VerificationStatus.PASS,
            verification_type=VerificationType.VISUAL,
            verifier_name="visual",
        )
        verifier.evaluate_screenshot = AsyncMock(return_value=passed)
        success = await verifier.verify(
            "https://example.com",
            "ok",
            screenshot_name="named.png",
        )
        assert success.status is VerificationStatus.PASS
        assert success.metadata["screenshot"].endswith("shot.png")

    async def test_run_visual_convenience_delegates(self) -> None:
        expected = VerificationResult(
            status=VerificationStatus.SKIP,
            verification_type=VerificationType.VISUAL,
            verifier_name="visual",
        )
        with patch.object(VisualVerifier, "verify", AsyncMock(return_value=expected)) as verify:
            result = await run_visual(MagicMock(), "https://example.com", "ok")
        assert result is expected
        verify.assert_awaited_once_with("https://example.com", "ok")


class TestBuiltinToolEdges:
    @staticmethod
    def http_client(response: SimpleNamespace) -> MagicMock:
        client = MagicMock()
        client.get = AsyncMock(return_value=response)
        context = MagicMock()
        context.__aenter__ = AsyncMock(return_value=client)
        context.__aexit__ = AsyncMock(return_value=None)
        return context

    async def test_web_search_success_failure_and_empty_results(self) -> None:
        policy = ToolPolicy(ToolsConfig(network_allowed=True))
        handler = search_handler(policy)
        html = (
            '<a class="result__a">missing href</a>'
            '<a href="https://example.com" class="result__a"><b>Result</b></a>'
        )
        with patch(
            "praxis.tools.builtins.network.web_search.httpx.AsyncClient",
            return_value=self.http_client(SimpleNamespace(status_code=200, text=html)),
        ):
            result = await handler({"query": "praxis", "max_results": 3})
        assert "Result" in result
        assert "https://example.com" in result

        with patch(
            "praxis.tools.builtins.network.web_search.httpx.AsyncClient",
            return_value=self.http_client(SimpleNamespace(status_code=503, text="")),
        ):
            assert "HTTP 503" in await handler({"query": "praxis"})

        with patch(
            "praxis.tools.builtins.network.web_search.httpx.AsyncClient",
            return_value=self.http_client(SimpleNamespace(status_code=200, text="none")),
        ):
            assert "未找到" in await handler({"query": "praxis"})

    async def test_autonomy_handlers_and_sleep_validation(self) -> None:
        store = MagicMock()
        store.save = AsyncMock()
        assert "3 字符" in await plan_handler(store)({"plan": "abc"})
        assert "3 字符" in await notes_handler(store)({"notes": "xyz"})
        assert store.save.await_count == 2
        assert "1 字符" in await plan_handler(None)({"plan": "x"})
        assert "1 字符" in await notes_handler(None)({"notes": "y"})

        assert "必须是数字" in await sleep_handler({"seconds": object()})
        with patch("praxis.tools.builtins.system.sleep.asyncio.sleep", AsyncMock()) as delay:
            assert "0.1 秒" in await sleep_handler({"seconds": -1})
            assert "300.0 秒" in await sleep_handler({"seconds": 999})
        assert delay.await_count == 2

    async def test_search_tools_cover_files_filters_and_invalid_inputs(
        self,
        tmp_path: Path,
    ) -> None:
        source = tmp_path / "sample.py"
        source.write_text("def target():\n    return 'needle'\n", encoding="utf-8")
        ignored = tmp_path / "sample.txt"
        ignored.write_text("needle", encoding="utf-8")
        nested = tmp_path / "deep"
        nested.mkdir()
        (nested / "target.py").write_text("class Target:\n    pass\n", encoding="utf-8")
        policy = ToolPolicy(ToolsConfig(allowed_paths=[str(tmp_path)]))

        code = code_search_handler(policy)
        assert "target" in await code({
            "query": "target",
            "search_path": str(tmp_path),
            "extensions": ["py"],
        })
        assert "target" in await code({"query": "target", "search_path": str(source)})
        assert "未找到" in await code({"query": "absent", "search_path": str(tmp_path)})

        find = find_handler(policy)
        assert "sample.py" in await find({"pattern": "*.py", "search_path": str(tmp_path)})
        assert "deep" not in await find({
            "pattern": "*.py",
            "search_path": str(tmp_path),
            "max_depth": 1,
        })
        assert "不是目录" in await find({"pattern": "*", "search_path": str(source)})
        assert "未找到" in await find({"pattern": "*.rs", "search_path": str(tmp_path)})

        grep = grep_handler(policy)
        assert "needle" in await grep({
            "pattern": "needle",
            "search_path": str(tmp_path),
            "includes": ["*.py"],
        })
        assert "needle" in await grep({"pattern": "needle", "search_path": str(source)})
        assert "无效" in await grep({"pattern": "[", "search_path": str(tmp_path)})
        assert "未找到" in await grep({"pattern": "absent", "search_path": str(tmp_path)})


class TestVectorAndMeteringEdges:
    @staticmethod
    def entry(
        content: str,
        vector: list[float] | None = None,
        *,
        status: MemoryStatus = MemoryStatus.ACTIVE,
    ) -> MemoryEntry:
        return MemoryEntry(
            memory_type=MemoryType.SEMANTIC,
            scope=MemoryScope(scope_type=ScopeType.SESSION, scope_id="s1"),
            content=content,
            embedding=vector,
            status=status,
        )

    async def test_local_and_tei_embeddings_validate_provider_payloads(self) -> None:
        embedded = await local_lexical_embed("Praxis praxis", dimensions=8)
        assert len(embedded) == 8
        assert pytest.approx(sum(value * value for value in embedded)) == 1.0
        assert await local_lexical_embed("", dimensions=4) == [0.0] * 4
        assert cosine_similarity([0.0], [1.0]) == 0.0
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0

        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = [[1, 2.5]]
        client = MagicMock()
        client.post = AsyncMock(return_value=response)
        assert await tei_embed("x", "https://embedding", "token", client=client) == [1.0, 2.5]
        assert client.post.await_args.kwargs["headers"]["Authorization"] == "Bearer token"

        for payload, message in (({}, "异常"), ([], "空嵌入"), (["bad"], "格式无效")):
            response.json.return_value = payload
            with pytest.raises(RuntimeError, match=message):
                await tei_embed("x", "https://embedding", client=client)

        context = MagicMock()
        context.__aenter__ = AsyncMock(return_value=client)
        context.__aexit__ = AsyncMock(return_value=None)
        response.json.return_value = [[3.0]]
        with patch("praxis.memory.vector.httpx.AsyncClient", return_value=context):
            assert await tei_embed("x", "https://embedding") == [3.0]

    async def test_vector_store_lifecycle_search_and_rebuild(self) -> None:
        scoped = MagicMock()
        scoped.save = AsyncMock()
        scoped.update = AsyncMock()
        embed = AsyncMock(side_effect=lambda text: [1.0, 0.0] if "match" in text else [0.0, 1.0])
        vectors = VectorStore(scoped, embed_func=embed, dimensions=2)
        active = self.entry("match")
        inactive = self.entry("match inactive", [1.0, 0.0], status=MemoryStatus.INACTIVE)
        await vectors.add(active)
        await vectors.add(inactive)
        assert active.embedding == [1.0, 0.0]
        assert len(await vectors.search("match", min_score=0.5)) == 1
        assert await vectors.search(
            "match",
            scopes=[MemoryScope(scope_type=ScopeType.USER, scope_id="other")],
        ) == []
        assert await vectors.search("match", memory_type=MemoryType.EPISODIC) == []
        await vectors.remove(active.memory_id)
        vectors.clear()
        assert await vectors.search("anything") == []

        missing = self.entry("match rebuild")
        present = self.entry("already", [0.0, 1.0])
        scoped.list_scope = AsyncMock(return_value=[missing, present])
        count = await vectors.rebuild_index([missing.scope])
        assert count == 2
        scoped.update.assert_awaited_once_with(missing)

        remote = VectorStore(scoped, api_base="https://embedding", dimensions=2)
        with patch("praxis.memory.vector.tei_embed", AsyncMock(return_value=[1.0, 0.0])):
            assert await remote.default_embed("x") == [1.0, 0.0]
        local = VectorStore(scoped, dimensions=4)
        assert len(await local.default_embed("x")) == 4
        remote.client = MagicMock(is_closed=False)
        remote.client.aclose = AsyncMock()
        await remote.aclose()
        assert remote.client is None

    def test_metering_fallbacks_budget_and_metrics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with patch("praxis.gateway.metering.litellm.get_max_tokens", side_effect=ValueError):
            assert get_max_tokens("unknown") == DEFAULT_MAX_TOKENS
        with patch("praxis.gateway.metering.litellm.completion_cost", return_value=1.25):
            assert completion_cost(object()) == 1.25
        with patch("praxis.gateway.metering.litellm.cost_per_token", return_value=(0.2, 0.0)):
            assert estimate_input_cost(2, "model") == 0.2
            assert estimate_output_cost(0, "model") == 0.2
        with patch("praxis.gateway.metering.litellm.cost_per_token", side_effect=ValueError):
            assert estimate_input_cost(2, "model") is None
            assert estimate_output_cost(2, "model") is None

        monkeypatch.setenv("PRAXIS_MODEL_API_KEY", "unit-test-key")
        gateway = GatewayRouter(GatewayConfig(
            deployments=[ModelDeployment(input_cost_per_token=0.1, output_cost_per_token=0.1)],
            max_budget=10,
        ))
        with patch("praxis.gateway.metering.estimate_input_cost", return_value=0.5):
            check_budget(gateway, 5)
        assert gateway.total_tokens == 0
        with patch("praxis.gateway.metering.emit_metric") as emit:
            record_usage("default", 2, 3, 0.5)
        assert emit.call_count == 3


class TestClassifierAndTracingEdges:
    def test_string_error_classification_and_unknown_type(self) -> None:
        timeout = classify_by_type_name(
            "praxis.exceptions.GatewayTimeoutError",
            "timeout",
        )
        assert timeout is not None
        assert timeout.category.value == "transient"
        tool = classify_by_type_name("praxis.exceptions.ToolExecutionError")
        assert tool is not None
        assert tool.category.value == "model_recoverable"
        auth = classify_by_type_name("praxis.exceptions.AuthenticationError")
        assert auth is not None
        assert auth.category.value == "user_fixable"
        assert classify_by_type_name("unknown.Error") is None

    def test_tracing_disabled_otlp_and_span_attributes(self) -> None:
        tracer = configure_tracing(TelemetryConfig(tracing_enabled=False))
        assert tracer is get_tracer()
        exporter = MagicMock()
        with patch(
            "opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter",
            return_value=exporter,
        ) as factory:
            processor = otlp_processor("http://collector")
        assert processor is not None
        factory.assert_called_once_with(endpoint="http://collector")

        span = MagicMock()
        tracer = MagicMock()
        tracer.start_span.return_value = span
        with patch("praxis.telemetry.tracing.get_tracer", return_value=tracer):
            assert start_span("operation", component="runtime", operation="health") is span
        assert span.set_attribute.call_count == 2


class TestDreamConsolidatorEdges:
    @staticmethod
    def consolidator(
        entries: list[MemoryEntry],
        *,
        meta: object = None,
    ) -> tuple[DreamConsolidator, MagicMock, MagicMock, MagicMock]:
        gateway = MagicMock()
        gateway.config.default_model = "default"
        scoped = MagicMock()
        scoped.list_scope = AsyncMock(return_value=entries)
        scoped.update = AsyncMock()
        scoped.save = AsyncMock()
        retention = MagicMock()
        retention.mark_inactive = AsyncMock()
        retention.run_decay_sweep = AsyncMock(return_value=2)
        retention.enforce_capacity = AsyncMock(return_value=1)
        meta_store = MagicMock()
        meta_store.load = AsyncMock(return_value=meta)
        meta_store.save = AsyncMock()
        dream = DreamConsolidator(
            gateway,
            scoped,
            retention,
            meta_store,
            model="default",
            min_sessions=2,
            min_hours_since_last=1,
            max_memories=10,
        )
        return dream, scoped, retention, meta_store

    async def test_should_run_handles_invalid_and_recent_metadata(self) -> None:
        dream, _, _, meta_store = self.consolidator([])
        assert not await dream.should_run(1)
        assert await dream.should_run(2)
        meta_store.load.return_value = {"completed_at": 42}
        assert await dream.should_run(2)
        meta_store.load.return_value = {"completed_at": "invalid"}
        assert await dream.should_run(2)
        meta_store.load.return_value = {
            "completed_at": datetime.now(UTC).isoformat(),
        }
        assert not await dream.should_run(2)
        meta_store.load.return_value = {
            "completed_at": (datetime.now(UTC) - timedelta(hours=2)).isoformat(),
        }
        assert await dream.should_run(2)

    async def test_empty_and_model_failure_runs_are_persisted(self) -> None:
        scope = MemoryScope(scope_type=ScopeType.GLOBAL)
        empty, _, _, empty_meta = self.consolidator([])
        report = await empty.run_dream([scope])
        assert report.summary == "无记忆条目需要整理"
        empty_meta.save.assert_awaited_once()

        entry = TestVectorAndMeteringEdges.entry("old")
        failing, _, _, failure_meta = self.consolidator([entry])
        with patch("praxis.memory.dream.chat", AsyncMock(side_effect=RuntimeError("down"))):
            report = await failing.run_dream([scope])
        assert "LLM 调用失败" in report.summary
        failure_meta.save.assert_awaited_once()

    async def test_full_dream_actions_update_merge_decay_and_capacity(self) -> None:
        scope = MemoryScope(scope_type=ScopeType.GLOBAL)
        first = MemoryEntry(
            memory_type=MemoryType.SEMANTIC,
            scope=scope,
            content="first",
            tags=["tag"],
        )
        second = MemoryEntry(
            memory_type=MemoryType.SEMANTIC,
            scope=scope,
            content="second",
        )
        dream, scoped, retention, meta_store = self.consolidator([first, second])
        payload = {
            "anchored": [
                {"memory_id": first.memory_id, "updated_content": "anchored"},
                {"memory_id": "missing", "updated_content": "ignored"},
            ],
            "conflicts": [{"ids": [first.memory_id, second.memory_id]}],
            "stale": [second.memory_id, "missing"],
            "merge_suggestions": [
                {
                    "ids": [first.memory_id, second.memory_id],
                    "merged_content": "merged",
                },
                {"ids": [first.memory_id], "merged_content": "ignored"},
                {"ids": ["missing", second.memory_id], "merged_content": "ignored"},
            ],
            "summary": "complete",
        }
        with patch(
            "praxis.memory.dream.chat",
            AsyncMock(return_value=model_response(json.dumps(payload))),
        ):
            report = await dream.run_dream([scope])

        assert report.anchored_count == 1
        assert report.stale_marked == 1
        assert report.conflicts_resolved == 1
        assert report.merged_count == 1
        assert report.decayed_count == 2
        assert report.evicted_count == 1
        assert first.content == "anchored"
        scoped.update.assert_awaited_once_with(first)
        scoped.save.assert_awaited_once()
        assert retention.mark_inactive.await_count == 3
        meta_store.save.assert_awaited_once()
        assert DreamConsolidator.parse_response("[]") == {}
        assert DreamConsolidator.string_list("not-list") == []
        assert DreamConsolidator.dict_list("not-list") == []
