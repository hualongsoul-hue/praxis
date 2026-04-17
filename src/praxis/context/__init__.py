"""praxis.context — 上下文引擎（S7）：Prompt 组装、上下文压缩、即时检索。"""

from praxis.context.assembler import PromptAssembler
from praxis.context.compaction import ContextCompactor
from praxis.context.jit_retrieval import ContentLoader, JITRetriever
from praxis.context.masking import ObservationMasker
from praxis.context.tool_injection import ToolInjector

__all__ = [
    "ContentLoader",
    "ContextCompactor",
    "JITRetriever",
    "ObservationMasker",
    "PromptAssembler",
    "ToolInjector",
]
