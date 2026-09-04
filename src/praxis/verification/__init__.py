"""praxis.verification — 验证引擎（S10）：计算型/推理型/视觉验证。

公共对象按需导入，计算验证和注册表不会隐式加载模型或视觉适配器。
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from praxis.verification.computational import (
        LintVerifier,
        SchemaVerifier,
        SuiteTestVerifier,
        TypeCheckVerifier,
        Verifier,
        run_computational,
    )
    from praxis.verification.gav import (
        ControlQuadrant,
        GAVController,
        GAVPhase,
        GAVVerifyRequest,
        GAVVerifyResponse,
    )
    from praxis.verification.inferential import InferentialVerifier, run_inferential
    from praxis.verification.registry import VerifierRegistry
    from praxis.verification.visual import VisualVerifier, run_visual

VERIFICATION_EXPORTS = {
    "ControlQuadrant": ("praxis.verification.gav", "ControlQuadrant"),
    "GAVController": ("praxis.verification.gav", "GAVController"),
    "GAVPhase": ("praxis.verification.gav", "GAVPhase"),
    "GAVVerifyRequest": ("praxis.verification.gav", "GAVVerifyRequest"),
    "GAVVerifyResponse": ("praxis.verification.gav", "GAVVerifyResponse"),
    "InferentialVerifier": (
        "praxis.verification.inferential",
        "InferentialVerifier",
    ),
    "LintVerifier": ("praxis.verification.computational", "LintVerifier"),
    "SchemaVerifier": ("praxis.verification.computational", "SchemaVerifier"),
    "SuiteTestVerifier": (
        "praxis.verification.computational",
        "SuiteTestVerifier",
    ),
    "TypeCheckVerifier": (
        "praxis.verification.computational",
        "TypeCheckVerifier",
    ),
    "Verifier": ("praxis.verification.computational", "Verifier"),
    "VerifierRegistry": ("praxis.verification.registry", "VerifierRegistry"),
    "VisualVerifier": ("praxis.verification.visual", "VisualVerifier"),
    "run_computational": (
        "praxis.verification.computational",
        "run_computational",
    ),
    "run_inferential": ("praxis.verification.inferential", "run_inferential"),
    "run_visual": ("praxis.verification.visual", "run_visual"),
}


def __getattr__(name: str) -> object:
    """Resolve one public verifier implementation on demand."""
    target = VERIFICATION_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    return getattr(import_module(module_name), attribute_name)

__all__ = [
    "ControlQuadrant",
    "GAVController",
    "GAVPhase",
    "GAVVerifyRequest",
    "GAVVerifyResponse",
    "InferentialVerifier",
    "LintVerifier",
    "SchemaVerifier",
    "SuiteTestVerifier",
    "TypeCheckVerifier",
    "Verifier",
    "VerifierRegistry",
    "VisualVerifier",
    "run_computational",
    "run_inferential",
    "run_visual",
]
