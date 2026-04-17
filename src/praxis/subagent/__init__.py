"""praxis.subagent — 子代理协调（S13）：子代理创建、上下文隔离、结果聚合。"""

from praxis.subagent.aggregation import ResultAggregator
from praxis.subagent.fork import ForkManager
from praxis.subagent.handoff import HandoffManager
from praxis.subagent.isolation import IsolatedContext
from praxis.subagent.resource_control import ResourceController
from praxis.subagent.spawn import SubagentSpawner

__all__ = [
    "ForkManager",
    "HandoffManager",
    "IsolatedContext",
    "ResourceController",
    "ResultAggregator",
    "SubagentSpawner",
]
