"""指标采集与导出。

支持 Counter（单调递增）、Gauge（瞬时值）、Histogram（分布统计）三种类型。
可导出为 Prometheus 文本格式或写入本地文件。
"""

import threading
import time
from pathlib import Path
from typing import Any, Literal

from praxis.config.schemas import TelemetryConfig

MetricKey = tuple[str, tuple[tuple[str, str], ...]]


class Counter:
    """单调递增计数器。"""

    __slots__ = ("_value", "_lock")

    def __init__(self) -> None:
        self._value = 0.0
        self._lock = threading.Lock()

    def add(self, value: float = 1.0) -> None:
        with self._lock:
            self._value += value

    @property
    def value(self) -> float:
        return self._value


class Gauge:
    """可任意设置的瞬时值。"""

    __slots__ = ("_value", "_lock")

    def __init__(self) -> None:
        self._value = 0.0
        self._lock = threading.Lock()

    def set(self, value: float) -> None:
        with self._lock:
            self._value = float(value)

    @property
    def value(self) -> float:
        return self._value


# Prometheus histogram 桶上界（秒/毫秒通用的对数刻度，覆盖亚毫秒到分钟级）
DEFAULT_BUCKETS: tuple[float, ...] = (
    1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000, 60000,
)


class Histogram:
    """分布统计（计数 + 累计和 + 分桶），支持 Prometheus 直方图与分位估算。"""

    __slots__ = ("_count", "_sum", "_buckets", "_bounds", "_lock")

    def __init__(self, buckets: tuple[float, ...] = DEFAULT_BUCKETS) -> None:
        self._count = 0
        self._sum = 0.0
        self._bounds = buckets
        self._buckets = [0] * len(buckets)  # 每个 ≤ 上界的累计计数（非累积，导出时累加）
        self._lock = threading.Lock()

    def record(self, value: float) -> None:
        with self._lock:
            self._count += 1
            self._sum += value
            for i, bound in enumerate(self._bounds):
                if value <= bound:
                    self._buckets[i] += 1
                    break

    @property
    def count(self) -> int:
        return self._count

    @property
    def sum(self) -> float:
        return self._sum

    def bucket_lines(self, name: str, labels: str) -> list[str]:
        """生成 Prometheus 累积桶行（含 +Inf）。"""
        lines: list[str] = []
        cumulative = 0
        # labels 形如 {k="v"} 或 ""；需在花括号内追加 le 标签
        inner = labels[1:-1] if labels else ""
        for i, bound in enumerate(self._bounds):
            cumulative += self._buckets[i]
            le = f'le="{bound}"'
            tag = "{" + (f"{inner}," if inner else "") + le + "}"
            lines.append(f"{name}_bucket{tag} {cumulative}")
        inf_tag = "{" + (f"{inner}," if inner else "") + 'le="+Inf"}'
        lines.append(f"{name}_bucket{inf_tag} {self._count}")
        return lines

    def quantile(self, q: float) -> float:
        """基于桶的近似分位数（返回命中桶的上界）。"""
        if self._count == 0:
            return 0.0
        target = q * self._count
        cumulative = 0
        for i, bound in enumerate(self._bounds):
            cumulative += self._buckets[i]
            if cumulative >= target:
                return bound
        return float("inf")


def format_labels(tags: tuple[tuple[str, str], ...]) -> str:
    if not tags:
        return ""
    pairs = ",".join(f'{k}="{v}"' for k, v in tags)
    return f"{{{pairs}}}"


class MetricsCollector:
    """线程安全的指标采集器。"""

    def __init__(self) -> None:
        self._counters: dict[MetricKey, Counter] = {}
        self._gauges: dict[MetricKey, Gauge] = {}
        self._histograms: dict[MetricKey, Histogram] = {}
        self._lock = threading.Lock()

    def _key(self, name: str, tags: dict[str, str] | None) -> MetricKey:
        return (name, tuple(sorted((tags or {}).items())))

    def counter(
        self, name: str, value: float = 1.0, tags: dict[str, str] | None = None
    ) -> None:
        key = self._key(name, tags)
        with self._lock:
            if key not in self._counters:
                self._counters[key] = Counter()
        self._counters[key].add(value)

    def gauge(
        self, name: str, value: float, tags: dict[str, str] | None = None
    ) -> None:
        key = self._key(name, tags)
        with self._lock:
            if key not in self._gauges:
                self._gauges[key] = Gauge()
        self._gauges[key].set(value)

    def histogram(
        self, name: str, value: float, tags: dict[str, str] | None = None
    ) -> None:
        key = self._key(name, tags)
        with self._lock:
            if key not in self._histograms:
                self._histograms[key] = Histogram()
        self._histograms[key].record(value)

    def export_prometheus(self) -> str:
        """导出 Prometheus 文本格式。

        在锁内对三类指标取快照，避免与并发 emit（如指标 HTTP 端点线程
        与主线程同时访问）发生 "dictionary changed size during iteration"。
        """
        with self._lock:
            counters = list(self._counters.items())
            gauges = list(self._gauges.items())
            histograms = list(self._histograms.items())

        lines: list[str] = []
        seen: set[str] = set()

        for (name, tags), c in sorted(counters):
            if name not in seen:
                lines.append(f"# TYPE {name} counter")
                seen.add(name)
            lines.append(f"{name}{format_labels(tags)} {c.value}")

        for (name, tags), g in sorted(gauges):
            if name not in seen:
                lines.append(f"# TYPE {name} gauge")
                seen.add(name)
            lines.append(f"{name}{format_labels(tags)} {g.value}")

        for (name, tags), h in sorted(histograms):
            if name not in seen:
                lines.append(f"# TYPE {name} histogram")
                seen.add(name)
            labels = format_labels(tags)
            lines.extend(h.bucket_lines(name, labels))
            lines.append(f"{name}_count{labels} {h.count}")
            lines.append(f"{name}_sum{labels} {h.sum}")

        return "\n".join(lines) + "\n" if lines else ""

    def export_to_file(self, path: str | Path) -> None:
        """导出指标到文件。"""
        Path(path).write_text(self.export_prometheus(), encoding="utf-8")


collector: MetricsCollector | None = None
_metrics_server: Any = None
_file_exporter_started: bool = False


def _start_file_exporter(path: str, interval: float = 15.0) -> None:
    """后台线程：周期性把指标快照写入文件（metrics_export=file）。"""
    global _file_exporter_started
    if _file_exporter_started:
        return

    def _loop() -> None:
        while True:
            time.sleep(interval)
            try:
                get_collector().export_to_file(path)
            except Exception:  # 写文件失败不应使线程退出
                pass

    thread = threading.Thread(target=_loop, name="praxis-metrics-file", daemon=True)
    thread.start()
    _file_exporter_started = True


def _start_metrics_server(port: int) -> None:
    """在后台线程启动一个仅暴露 /metrics 的 Prometheus 抓取端点。"""
    global _metrics_server
    if _metrics_server is not None:
        return
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path.rstrip("/") in ("/metrics", ""):
                body = export_prometheus().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *args: Any) -> None:  # 静默 HTTP 访问日志
            return

    server = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
    thread = threading.Thread(target=server.serve_forever, name="praxis-metrics", daemon=True)
    thread.start()
    _metrics_server = server


def configure_metrics(config: TelemetryConfig) -> None:
    """初始化指标采集器；按 metrics_export 选择导出方式。

    幂等：已存在采集器时保留之（避免 create_agent_session 每会话重复调用
    时清空累计指标）。
    """
    global collector
    if not config.metrics_enabled:
        collector = None
        return
    if collector is None:
        collector = MetricsCollector()
    if config.metrics_export == "prometheus":
        try:
            _start_metrics_server(config.metrics_port)
        except Exception:  # 端口占用等不应阻断主流程
            pass
    elif config.metrics_export == "file" and config.metrics_file:
        try:
            _start_file_exporter(config.metrics_file)
        except Exception:
            pass


def get_collector() -> MetricsCollector:
    global collector
    if collector is None:
        collector = MetricsCollector()
    return collector


def emit_metric(
    name: str,
    value: float,
    tags: dict[str, str] | None = None,
    metric_type: Literal["counter", "gauge", "histogram"] = "counter",
) -> None:
    """发射一个指标数据点。

    Args:
        name: 指标名称。
        value: 指标值。Counter 为增量，Gauge 为绝对值，Histogram 为观测值。
        tags: 标签键值对。
        metric_type: 指标类型。
    """
    active = get_collector()
    if metric_type == "counter":
        active.counter(name, value, tags)
    elif metric_type == "gauge":
        active.gauge(name, value, tags)
    elif metric_type == "histogram":
        active.histogram(name, value, tags)


def export_prometheus() -> str:
    """导出当前所有指标为 Prometheus 文本格式。"""
    return get_collector().export_prometheus()
