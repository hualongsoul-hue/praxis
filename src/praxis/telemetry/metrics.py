"""指标采集与导出。

支持 Counter（单调递增）、Gauge（瞬时值）、Histogram（分布统计）三种类型。
可导出为 Prometheus 文本格式或写入本地文件。
"""

import threading
from pathlib import Path
from typing import Literal

from praxis.config.subsystems import TelemetryConfig

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


class Histogram:
    """分布统计（计数 + 累计和）。"""

    __slots__ = ("_count", "_sum", "_lock")

    def __init__(self) -> None:
        self._count = 0
        self._sum = 0.0
        self._lock = threading.Lock()

    def record(self, value: float) -> None:
        with self._lock:
            self._count += 1
            self._sum += value

    @property
    def count(self) -> int:
        return self._count

    @property
    def sum(self) -> float:
        return self._sum


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
        """导出 Prometheus 文本格式。"""
        lines: list[str] = []
        seen: set[str] = set()

        for (name, tags), c in sorted(self._counters.items()):
            if name not in seen:
                lines.append(f"# TYPE {name} counter")
                seen.add(name)
            lines.append(f"{name}{format_labels(tags)} {c.value}")

        for (name, tags), g in sorted(self._gauges.items()):
            if name not in seen:
                lines.append(f"# TYPE {name} gauge")
                seen.add(name)
            lines.append(f"{name}{format_labels(tags)} {g.value}")

        for (name, tags), h in sorted(self._histograms.items()):
            if name not in seen:
                lines.append(f"# TYPE {name} histogram")
                seen.add(name)
            labels = format_labels(tags)
            lines.append(f"{name}_count{labels} {h.count}")
            lines.append(f"{name}_sum{labels} {h.sum}")

        return "\n".join(lines) + "\n" if lines else ""

    def export_to_file(self, path: str | Path) -> None:
        """导出指标到文件。"""
        Path(path).write_text(self.export_prometheus(), encoding="utf-8")


collector: MetricsCollector | None = None


def configure_metrics(config: TelemetryConfig) -> None:
    """初始化指标采集器。"""
    global collector
    if config.metrics_enabled:
        collector = MetricsCollector()
    else:
        collector = None


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
