"""praxis.memory — 记忆系统（S6）。

对外暴露统一门面 CognitiveMemory，集成四类认知记忆、模型辅助管线、
档案模式、Scratchpad、Dream 调度和后台自治 Worker。公共对象按需导入，
因此使用本地向量或存储组件不会隐式加载模型适配器。
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from praxis.memory.consolidator import ConsolidationResult, MemoryConsolidator
    from praxis.memory.core import CognitiveMemory
    from praxis.memory.dream import DreamConsolidator, DreamReport, DreamScheduler
    from praxis.memory.extractor import MemoryExtractor
    from praxis.memory.profile import ProfileManager
    from praxis.memory.project_loader import ProjectMemoryLoader
    from praxis.memory.retention import RetentionManager
    from praxis.memory.retriever import MemoryRetriever
    from praxis.memory.scratchpad import Scratchpad
    from praxis.memory.store import ProfileStore, ScopedMemoryStore, entry_from_dict
    from praxis.memory.vector import VectorStore, cosine_similarity, tei_embed
    from praxis.memory.worker import BackgroundWorker

MEMORY_EXPORTS = {
    "BackgroundWorker": ("praxis.memory.worker", "BackgroundWorker"),
    "CognitiveMemory": ("praxis.memory.core", "CognitiveMemory"),
    "ConsolidationResult": ("praxis.memory.consolidator", "ConsolidationResult"),
    "DreamConsolidator": ("praxis.memory.dream", "DreamConsolidator"),
    "DreamReport": ("praxis.memory.dream", "DreamReport"),
    "DreamScheduler": ("praxis.memory.dream", "DreamScheduler"),
    "MemoryConsolidator": ("praxis.memory.consolidator", "MemoryConsolidator"),
    "MemoryExtractor": ("praxis.memory.extractor", "MemoryExtractor"),
    "MemoryRetriever": ("praxis.memory.retriever", "MemoryRetriever"),
    "ProfileManager": ("praxis.memory.profile", "ProfileManager"),
    "ProfileStore": ("praxis.memory.store", "ProfileStore"),
    "ProjectMemoryLoader": ("praxis.memory.project_loader", "ProjectMemoryLoader"),
    "RetentionManager": ("praxis.memory.retention", "RetentionManager"),
    "ScopedMemoryStore": ("praxis.memory.store", "ScopedMemoryStore"),
    "Scratchpad": ("praxis.memory.scratchpad", "Scratchpad"),
    "VectorStore": ("praxis.memory.vector", "VectorStore"),
    "cosine_similarity": ("praxis.memory.vector", "cosine_similarity"),
    "entry_from_dict": ("praxis.memory.store", "entry_from_dict"),
    "tei_embed": ("praxis.memory.vector", "tei_embed"),
}


def __getattr__(name: str) -> object:
    """Resolve a public memory facade object on first use."""
    target = MEMORY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    return getattr(import_module(module_name), attribute_name)

__all__ = [
    "BackgroundWorker",
    "CognitiveMemory",
    "ConsolidationResult",
    "DreamConsolidator",
    "DreamReport",
    "DreamScheduler",
    "MemoryConsolidator",
    "MemoryExtractor",
    "MemoryRetriever",
    "ProfileManager",
    "ProfileStore",
    "ProjectMemoryLoader",
    "RetentionManager",
    "ScopedMemoryStore",
    "Scratchpad",
    "VectorStore",
    "cosine_similarity",
    "entry_from_dict",
    "tei_embed",
]
