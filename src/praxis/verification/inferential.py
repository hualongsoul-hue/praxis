"""推理型验证（Inferential Verification / LLM-as-Judge）。

通过 S4 judge 接口独立评估，评估代理与执行代理分离（不同上下文），
自定义评估标准，返回数值评分 + 判定 + 文字反馈。
"""

import time
from typing import Any

from praxis.gateway.router import GatewayRouter
from praxis.gateway.tasks import judge
from praxis.models.verification import (
    VerificationResult,
    VerificationStatus,
    VerificationType,
)
from praxis.telemetry.logger import get_logger
from praxis.telemetry.metrics import emit_metric

log = get_logger("verification.inferential")


class InferentialVerifier:
    """推理型验证器——通过 LLM 独立评估。

    使用独立上下文调用 S4 judge 接口，
    避免执行 Agent 的自我偏见。
    """

    def __init__(
        self,
        gateway: GatewayRouter,
        model: str | None = None,
        pass_threshold: float = 0.7,
    ) -> None:
        self.gateway = gateway
        self.model = model
        self.pass_threshold = pass_threshold

    async def verify(
        self,
        criteria: str,
        content: str,
        dimensions: list[str] | None = None,
    ) -> VerificationResult:
        """执行推理型验证。

        Args:
            criteria: 评估标准描述。
            content: 待评估内容。
            dimensions: 可选评估维度列表（正确性、完整性、代码质量等）。

        Returns:
            包含评分和反馈的验证结果。
        """
        start = time.perf_counter()

        full_criteria = criteria
        if dimensions:
            dims_text = "、".join(dimensions)
            full_criteria = f"{criteria}\n评估维度：{dims_text}"

        try:
            judge_result = await judge(
                self.gateway,
                criteria=full_criteria,
                content=content,
                model=self.model,
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            log.error("推理型验证失败", error=str(exc))
            return VerificationResult(
                status=VerificationStatus.ERROR,
                verification_type=VerificationType.INFERENTIAL,
                verifier_name="inferential",
                feedback=f"LLM 评估失败: {exc}",
                duration_ms=elapsed,
            )

        elapsed = (time.perf_counter() - start) * 1000

        if judge_result.verdict and judge_result.confidence >= self.pass_threshold:
            status = VerificationStatus.PASS
        else:
            status = VerificationStatus.FAIL

        emit_metric(
            "verification_inferential",
            1.0,
            {"status": status.value},
            "counter",
        )
        log.info(
            "推理型验证完成",
            status=status.value,
            confidence=judge_result.confidence,
        )

        return VerificationResult(
            status=status,
            verification_type=VerificationType.INFERENTIAL,
            verifier_name="inferential",
            score=judge_result.confidence,
            feedback=judge_result.reasoning,
            metadata={
                "verdict": judge_result.verdict,
                "raw_response": judge_result.raw_response,
            },
            duration_ms=elapsed,
        )


async def run_inferential(
    gateway: GatewayRouter,
    criteria: str,
    content: str,
    model: str | None = None,
    pass_threshold: float = 0.7,
    dimensions: list[str] | None = None,
) -> VerificationResult:
    """便捷函数：执行推理型验证。

    Args:
        gateway: S4 网关路由器。
        criteria: 评估标准。
        content: 待评估内容。
        model: 模型别名。
        pass_threshold: 通过阈值。
        dimensions: 评估维度列表。

    Returns:
        验证结果。
    """
    verifier = InferentialVerifier(gateway, model=model, pass_threshold=pass_threshold)
    return await verifier.verify(criteria, content, dimensions=dimensions)
