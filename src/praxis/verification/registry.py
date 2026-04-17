"""验证器注册与质量左移。

register_verifier 接口，支持自定义验证器扩展。
质量左移策略（集成前/集成后/持续监控/运行时反馈）配置。
"""

from typing import Any

from praxis.models.verification import (
    QualityPhase,
    VerificationResult,
    VerificationType,
)
from praxis.verification.computational import Verifier, run_computational
from praxis.telemetry.logger import get_logger

log = get_logger("verification.registry")


class VerifierEntry:
    """注册表条目。"""

    def __init__(
        self,
        verifier: Verifier,
        verification_type: VerificationType,
        phases: list[QualityPhase],
    ) -> None:
        self.verifier = verifier
        self.verification_type = verification_type
        self.phases = phases


class VerifierRegistry:
    """验证器注册表。

    管理所有已注册的验证器，按类型和质量阶段组织。
    """

    def __init__(self) -> None:
        self.entries: dict[str, VerifierEntry] = {}

    def register(
        self,
        verifier: Verifier,
        verification_type: VerificationType = VerificationType.COMPUTATIONAL,
        phases: list[QualityPhase] | None = None,
    ) -> None:
        """注册验证器。

        Args:
            verifier: 验证器实例（实现 Verifier Protocol）。
            verification_type: 验证类型。
            phases: 适用的质量阶段列表，默认全部。
        """
        if phases is None:
            phases = list(QualityPhase)

        self.entries[verifier.name] = VerifierEntry(
            verifier=verifier,
            verification_type=verification_type,
            phases=phases,
        )
        log.info("验证器已注册", verifier_name=verifier.name, verifier_type=verification_type.value)

    def unregister(self, name: str) -> None:
        """注销验证器。"""
        if name in self.entries:
            del self.entries[name]
            log.info("验证器已注销", verifier_name=name)

    def get(self, name: str) -> VerifierEntry | None:
        """按名称获取验证器。"""
        return self.entries.get(name)

    def list_verifiers(
        self,
        verification_type: VerificationType | None = None,
        phase: QualityPhase | None = None,
    ) -> list[VerifierEntry]:
        """列出验证器。

        Args:
            verification_type: 可选类型过滤。
            phase: 可选质量阶段过滤。

        Returns:
            匹配的验证器条目列表。
        """
        results: list[VerifierEntry] = []
        for entry in self.entries.values():
            if verification_type and entry.verification_type != verification_type:
                continue
            if phase and phase not in entry.phases:
                continue
            results.append(entry)
        return results

    async def run_computational(
        self,
        target: dict[str, Any],
        phase: QualityPhase | None = None,
    ) -> list[VerificationResult]:
        """执行指定阶段的所有计算型验证器。

        Args:
            target: 验证目标。
            phase: 可选质量阶段过滤。

        Returns:
            验证结果列表。
        """
        entries = self.list_verifiers(
            verification_type=VerificationType.COMPUTATIONAL,
            phase=phase,
        )
        verifiers = [e.verifier for e in entries]
        return await run_computational(verifiers, target)

    def get_phase_config(self, phase: QualityPhase) -> list[str]:
        """获取指定阶段的验证器名称列表。

        Args:
            phase: 质量阶段。

        Returns:
            验证器名称列表。
        """
        entries = self.list_verifiers(phase=phase)
        return [e.verifier.name for e in entries]
