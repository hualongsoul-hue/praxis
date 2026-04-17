"""praxis.verification — 验证引擎（S10）：计算型/推理型/视觉验证。"""

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
