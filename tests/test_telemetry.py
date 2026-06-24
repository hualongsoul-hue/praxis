"""S2 遥测系统验证测试。"""

import json

import pytest

from praxis.config.schemas import TelemetryConfig
from praxis.models.telemetry import AuditEvent
from praxis.persistence.store import PersistenceStore, create_store
from praxis.config.schemas import PersistenceConfig
from praxis.telemetry.audit import configure_audit, flush_audit, query_audit, record_audit
from praxis.telemetry.logger import (
    StructuredLogger,
    configure_logging,
    get_logger,
)
from praxis.telemetry.metrics import (
    MetricsCollector,
    emit_metric,
    export_prometheus,
)
from praxis.telemetry.tracing import configure_tracing, start_span


class TestStructuredLogger:
    """Task 3.1: 结构化日志验证。"""

    def test_get_logger_returns_structured(self) -> None:
        logger = get_logger("gateway")
        assert isinstance(logger, StructuredLogger)

    def test_json_format(self, capfd: pytest.CaptureFixture[str]) -> None:
        config = TelemetryConfig(log_level="DEBUG", log_format="json")
        configure_logging(config)
        logger = get_logger("gateway")
        logger.info("test message", request_id="abc123")
        captured = capfd.readouterr()
        data = json.loads(captured.err.strip())
        assert data["level"] == "INFO"
        assert data["component"] == "gateway"
        assert data["message"] == "test message"
        assert data["request_id"] == "abc123"
        assert "timestamp" in data

    def test_bind_context(self, capfd: pytest.CaptureFixture[str]) -> None:
        config = TelemetryConfig(log_level="DEBUG", log_format="json")
        configure_logging(config)
        logger = get_logger("orchestrator").bind(session_id="sess-1", turn=3)
        logger.info("turn started")
        captured = capfd.readouterr()
        data = json.loads(captured.err.strip())
        assert data["session_id"] == "sess-1"
        assert data["turn"] == 3

    def test_per_component_level(self, capfd: pytest.CaptureFixture[str]) -> None:
        config = TelemetryConfig(
            log_level="WARNING",
            log_format="json",
            log_levels={"gateway": "DEBUG"},
        )
        configure_logging(config)
        gw_logger = get_logger("gateway")
        orch_logger = get_logger("orchestrator")
        gw_logger.debug("gw debug visible")
        orch_logger.debug("orch debug hidden")
        captured = capfd.readouterr()
        assert "gw debug visible" in captured.err
        assert "orch debug hidden" not in captured.err

    def test_text_format(self, capfd: pytest.CaptureFixture[str]) -> None:
        config = TelemetryConfig(log_level="INFO", log_format="text")
        configure_logging(config)
        logger = get_logger("tools")
        logger.info("text output")
        captured = capfd.readouterr()
        assert "tools" in captured.err
        assert "text output" in captured.err


class TestMetrics:
    """Task 3.2: 指标采集与导出验证。"""

    def test_counter(self) -> None:
        collector = MetricsCollector()
        collector.counter("requests_total", 1, {"component": "gateway"})
        collector.counter("requests_total", 1, {"component": "gateway"})
        text = collector.export_prometheus()
        assert "# TYPE requests_total counter" in text
        assert 'requests_total{component="gateway"} 2.0' in text

    def test_gauge(self) -> None:
        collector = MetricsCollector()
        collector.gauge("active_sessions", 5)
        collector.gauge("active_sessions", 3)
        text = collector.export_prometheus()
        assert "# TYPE active_sessions gauge" in text
        assert "active_sessions 3.0" in text

    def test_histogram(self) -> None:
        collector = MetricsCollector()
        collector.histogram("llm_latency_ms", 150.0, {"model": "gpt-4"})
        collector.histogram("llm_latency_ms", 250.0, {"model": "gpt-4"})
        text = collector.export_prometheus()
        assert "# TYPE llm_latency_ms histogram" in text
        assert 'llm_latency_ms_count{model="gpt-4"} 2' in text
        assert 'llm_latency_ms_sum{model="gpt-4"} 400.0' in text

    def test_emit_metric_function(self) -> None:
        emit_metric("test_counter", 1.0, metric_type="counter")
        emit_metric("test_gauge", 42.0, metric_type="gauge")
        emit_metric("test_hist", 100.0, metric_type="histogram")
        text = export_prometheus()
        assert "test_counter" in text
        assert "test_gauge" in text
        assert "test_hist" in text

    def test_export_to_file(self, tmp_path: pytest.TempPathFactory) -> None:
        collector = MetricsCollector()
        collector.counter("file_test", 1)
        path = tmp_path / "metrics.prom"  # type: ignore[operator]
        collector.export_to_file(path)
        content = path.read_text(encoding="utf-8")  # type: ignore[union-attr]
        assert "file_test" in content


class TestTracing:
    """Task 3.3: 分布式追踪验证。"""

    def test_start_span_basic(self) -> None:
        config = TelemetryConfig(tracing_enabled=True, tracing_export="console")
        configure_tracing(config)
        span = start_span("test-operation", component="gateway", operation="chat")
        assert span is not None
        assert span.is_recording()
        span.end()

    def test_parent_child_span(self) -> None:
        config = TelemetryConfig(tracing_enabled=True, tracing_export="console")
        configure_tracing(config)
        parent = start_span("parent-op", component="orchestrator")
        child = start_span("child-op", parent=parent, component="gateway")
        child_ctx = child.get_span_context()
        parent_ctx = parent.get_span_context()
        assert child_ctx.trace_id == parent_ctx.trace_id
        assert child_ctx.span_id != parent_ctx.span_id
        child.end()
        parent.end()


class TestAudit:
    """Task 3.4: 审计日志验证。"""

    @pytest.fixture
    async def audit_store(self, tmp_path: pytest.TempPathFactory) -> PersistenceStore:
        config = PersistenceConfig(
            backend="sqlite",
            sqlite_path=str(tmp_path / "audit_test.db"),  # type: ignore[operator]
        )
        store = await create_store(config)
        configure_audit(store)
        yield store  # type: ignore[misc]
        await store.close()

    async def test_record_and_query(self, audit_store: PersistenceStore) -> None:
        event = AuditEvent(
            event_type="tool_call",
            component="tools",
            session_id="sess-1",
            details={
                "tool_name": "read_file",
                "arguments": "/tmp/test.txt",
                "result": "success",
                "permission": "auto_approve",
            },
        )
        await record_audit(event)
        await flush_audit()
        events = await query_audit()
        assert len(events) >= 1
        found = [e for e in events if e.event_id == event.event_id]
        assert len(found) == 1
        assert found[0].event_type == "tool_call"
        assert found[0].details["tool_name"] == "read_file"

    async def test_audit_without_store(self) -> None:
        import praxis.telemetry.audit as audit_mod
        saved = audit_mod.audit_store
        audit_mod.audit_store = None
        event = AuditEvent(
            event_type="llm_call",
            component="gateway",
            details={"model": "gpt-4"},
        )
        await record_audit(event)
        audit_mod.audit_store = saved


class TestTelemetryProductionHardening:
    """生产化加固：直方图分桶/分位、审计开关、统一初始化、基数。"""

    def test_histogram_buckets_and_quantile(self) -> None:
        from praxis.telemetry.metrics import Histogram

        h = Histogram(buckets=(10.0, 100.0, 1000.0))
        for v in (5, 5, 50, 500, 5000):
            h.record(v)
        assert h.count == 5
        # 分位：p50 应落在含中位数的桶
        assert h.quantile(0.5) in (10.0, 100.0)
        lines = h.bucket_lines("lat", '{svc="x"}')
        assert any('le="+Inf"' in ln for ln in lines)
        # +Inf 桶应等于总计数
        assert any(ln.endswith(" 5") and 'le="+Inf"' in ln for ln in lines)

    def test_export_includes_buckets(self) -> None:
        from praxis.telemetry.metrics import MetricsCollector

        c = MetricsCollector()
        c.histogram("op_ms", 12.0)
        out = c.export_prometheus()
        assert "op_ms_bucket" in out
        assert "op_ms_count" in out and "op_ms_sum" in out

    async def test_audit_disabled_writes_nothing(self) -> None:
        from praxis.config.schemas import PersistenceConfig
        from praxis.persistence.store import create_store
        from praxis.telemetry.audit import (
            configure_audit, flush_audit, query_audit, record_audit,
        )

        store = await create_store(PersistenceConfig())
        try:
            configure_audit(store, enabled=False)
            await record_audit(AuditEvent(
                event_type="tool_call", component="c", action="a", details={},
            ))
            await flush_audit()
            assert await query_audit() == []
        finally:
            configure_audit(store, enabled=True)
            await store.close()

    def test_configure_telemetry_applies(self) -> None:
        from praxis.telemetry import configure_telemetry
        from praxis.telemetry.metrics import get_collector

        configure_telemetry(TelemetryConfig(
            metrics_enabled=True, metrics_export="file", tracing_enabled=False,
        ))
        # 指标采集器可用
        get_collector().counter("probe", 1.0)
        assert "probe" in get_collector().export_prometheus()


class TestMetricsConcurrency:
    """指标导出与并发 emit 不应崩溃（HTTP 端点线程场景）。"""

    def test_export_during_concurrent_emit(self) -> None:
        import threading
        from praxis.telemetry.metrics import MetricsCollector

        c = MetricsCollector()
        errors: list[Exception] = []

        def emitter() -> None:
            # 插入有界但持续新增的 key，制造导出迭代期间的字典扩容
            for i in range(3000):
                c.counter(f"m_{i}", 1.0)
                c.histogram("lat", float(i % 100))

        def exporter() -> None:
            try:
                while emit_thread.is_alive():
                    c.export_prometheus()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        emit_thread = threading.Thread(target=emitter)
        emit_thread.start()
        exporter()
        emit_thread.join()
        # 修复前：export 在锁外迭代 → "dictionary changed size during iteration"
        assert errors == []
