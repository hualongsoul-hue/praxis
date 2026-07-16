"""指标采集与导出。

支持 Counter（单调递增）、Gauge（瞬时值）、Histogram（分布统计）三种类型。
可导出为 Prometheus 文本格式或写入本地文件。
"""

import threading
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Literal

from praxis.config.schemas import TelemetryConfig

MetricKey = tuple[str, tuple[tuple[str, str], ...]]


class Counter:
    """单调递增计数器。"""

    __slots__ = ("_lock", "_value")

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

    __slots__ = ("_lock", "_value")

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

    __slots__ = ("_bounds", "_buckets", "_count", "_lock", "_sum")

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


_current_collector: ContextVar[MetricsCollector | None] = ContextVar(
    "praxis_metrics_collector",
    default=None,
)


def configure_metrics(config: TelemetryConfig) -> None:
    """为当前 CLI 上下文绑定采集器；不启动永久线程或 HTTP 服务。"""
    _current_collector.set(MetricsCollector())


@contextmanager
def use_metrics(collector: MetricsCollector) -> Generator[None]:
    """在当前异步上下文中使用 Runtime 实例拥有的采集器。"""
    token = _current_collector.set(collector)
    try:
        yield
    finally:
        _current_collector.reset(token)


def get_collector() -> MetricsCollector:
    collector = _current_collector.get()
    if collector is None:
        collector = MetricsCollector()
        _current_collector.set(collector)
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
