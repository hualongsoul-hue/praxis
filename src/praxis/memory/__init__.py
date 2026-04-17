"""praxis.memory — 记忆系统（S6）：认知记忆管理、模型辅助提取与整合。"""

from praxis.memory.background import BackgroundProcessor
from praxis.memory.consolidation import MemoryConsolidator
from praxis.memory.dream import DreamConsolidator, DreamReport
from praxis.memory.extraction import MemoryExtractor
from praxis.memory.lifecycle import LifecycleManager
from praxis.memory.pipeline import MemoryPipeline
from praxis.memory.retrieval import MemoryRetriever
from praxis.memory.scope import ScopedMemoryStore
from praxis.memory.scratchpad import Scratchpad
from praxis.memory.vector_store import VectorStore

__all__ = [
    "BackgroundProcessor",
    "DreamConsolidator",
    "DreamReport",
    "LifecycleManager",
    "MemoryConsolidator",
    "MemoryExtractor",
    "MemoryPipeline",
    "MemoryRetriever",
    "ScopedMemoryStore",
    "Scratchpad",
    "VectorStore",
]
