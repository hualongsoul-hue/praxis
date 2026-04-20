"""praxis.memory — 记忆系统（S6）。

对外暴露统一门面 CognitiveMemory，集成四类认知记忆、模型辅助管线、
档案模式、Scratchpad、Dream 调度和后台自治 Worker。
"""

from praxis.memory.consolidator import ConsolidationResult, MemoryConsolidator
from praxis.memory.dream import DreamConsolidator, DreamReport, DreamScheduler
from praxis.memory.extractor import MemoryExtractor
from praxis.memory.profile import ProfileManager
from praxis.memory.project_loader import ProjectMemoryLoader
from praxis.memory.retention import RetentionManager
from praxis.memory.retriever import MemoryRetriever
from praxis.memory.scratchpad import Scratchpad
from praxis.memory.store import ProfileStore, ScopedMemoryStore, entry_from_dict
from praxis.memory.core import CognitiveMemory
from praxis.memory.vector import VectorStore, cosine_similarity, tei_embed
from praxis.memory.worker import BackgroundWorker

__all__ = [
    "BackgroundWorker",
    "ConsolidationResult",
    "DreamConsolidator",
    "DreamReport",
    "DreamScheduler",
    "MemoryConsolidator",
    "MemoryExtractor",
    "MemoryRetriever",
    "CognitiveMemory",
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
