"""GAV（Gather-Act-Verify）循环支持。

验证引擎作为 Verify 阶段执行者，验证失败时返回结构化结果供 S11 注入上下文。
前馈/反馈控制矩阵覆盖四象限。
"""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from praxis.models.verification import (
    VerificationResult,
    VerificationType,
)
from praxis.telemetry.logger import get_logger
from praxis.telemetry.metrics import emit_metric

log = get_logger("verification.gav")


class ControlQuadrant(StrEnum):
    """前馈/反馈控制矩阵象限。"""

    FEEDFORWARD_COMPUTATIONAL = "feedforward_computational"
    FEEDFORWARD_INFERENTIAL = "feedforward_inferential"
    FEEDBACK_COMPUTATIONAL = "feedback_computational"
    FEEDBACK_INFERENTIAL = "feedback_inferential"


class GAVPhase(StrEnum):
    """GAV 循环阶段。"""

    GATHER = "gather"
    ACT = "act"
    VERIFY = "verify"


class GAVVerifyRequest(BaseModel):
    """Verify 阶段请求。"""

    results: list[VerificationResult]
    quadrant: ControlQuadrant = ControlQuadrant.FEEDBACK_COMPUTATIONAL
    metadata: dict[str, Any] = Field(default_factory=dict)


class GAVVerifyResponse(BaseModel):
    """Verify 阶段响应——供 S11 消费。"""

    passed: bool
    results: list[VerificationResult]
    context_injection: str = ""
    retry_hint: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class GAVController:
    """GAV 循环控制器。

    不控制循环流程（由 S11 控制），只提供 Verify 阶段的验证编排
    和结果格式化。
    """

    def evaluate(self, request: GAVVerifyRequest) -> GAVVerifyResponse:
        """评估 Verify 阶段结果。

        Args:
            request: Verify 请求，包含多个验证结果。

        Returns:
            格式化的 Verify 响应，供 S11 决定是否重新进入 Gather。
        """
        all_passed = all(r.passed for r in request.results)

        context_lines: list[str] = []
        retry_hints: list[str] = []

        for result in request.results:
            if not result.passed:
                context_lines.append(
                    f"[{result.verifier_name}] {result.status.value}: {result.feedback}"
                )
                for failure in result.failures:
                    loc = ""
                    if failure.file:
                        loc = failure.file
                        if failure.line is not None:
                            loc += f":{failure.line}"
                    detail = f"  - {loc}: {failure.message}" if loc else f"  - {failure.message}"
                    context_lines.append(detail)

                if result.feedback:
                    retry_hints.append(result.feedback)

        context_injection = "\n".join(context_lines) if context_lines else ""
        retry_hint = "; ".join(retry_hints) if retry_hints else ""

        emit_metric(
            "gav_verify",
            1.0,
            {
                "quadrant": request.quadrant.value,
                "passed": str(all_passed).lower(),
            },
            "counter",
        )
        log.info(
            "GAV Verify 评估",
            passed=all_passed,
            quadrant=request.quadrant.value,
            result_count=len(request.results),
        )

        return GAVVerifyResponse(
            passed=all_passed,
            results=request.results,
            context_injection=context_injection,
            retry_hint=retry_hint,
            metadata=request.metadata,
        )

    @staticmethod
    def select_quadrant(
        is_feedforward: bool,
        verification_type: VerificationType,
    ) -> ControlQuadrant:
        """根据时机和验证类型选择控制象限。

        Args:
            is_feedforward: 是否为前馈（行动前）。
            verification_type: 验证类型。

        Returns:
            控制象限。
        """
        if is_feedforward:
            if verification_type == VerificationType.COMPUTATIONAL:
                return ControlQuadrant.FEEDFORWARD_COMPUTATIONAL
            return ControlQuadrant.FEEDFORWARD_INFERENTIAL
        else:
            if verification_type == VerificationType.COMPUTATIONAL:
                return ControlQuadrant.FEEDBACK_COMPUTATIONAL
            return ControlQuadrant.FEEDBACK_INFERENTIAL

    @staticmethod
    def format_for_context(response: GAVVerifyResponse) -> str:
        """将 Verify 响应格式化为可注入上下文的文本。

        Args:
            response: GAV Verify 响应。

        Returns:
            格式化文本。
        """
        if response.passed:
            return "验证通过，无需修改。"

        lines = ["## 验证失败详情\n"]
        if response.context_injection:
            lines.append(response.context_injection)
        if response.retry_hint:
            lines.append(f"\n修复建议：{response.retry_hint}")
        return "\n".join(lines)
