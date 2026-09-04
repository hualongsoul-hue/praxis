"""验证器注册与质量左移。

register_verifier 接口，支持自定义验证器扩展。
质量左移策略（集成前/集成后/持续监控/运行时反馈）配置。
"""

from importlib.util import find_spec
from typing import Any

from praxis.config.schemas import VerificationConfig
from praxis.models.verification import (
    QualityPhase,
    VerificationResult,
    VerificationStatus,
    VerificationType,
)
from praxis.protocols import ModelGateway
from praxis.telemetry.logger import get_logger
from praxis.tools.policy import ToolPolicy
from praxis.tools.process import ProcessRunner
from praxis.verification.computational import (
    LintVerifier,
    SchemaVerifier,
    SuiteTestVerifier,
    TypeCheckVerifier,
    Verifier,
    run_computational,
)
from praxis.verification.inferential import run_inferential

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

    def __init__(
        self,
        gateway: ModelGateway | None = None,
        computational_enabled: bool = True,
        inferential_enabled: bool = True,
        visual_enabled: bool = True,
    ) -> None:
        self.entries: dict[str, VerifierEntry] = {}
        # 推理型（LLM judge）与视觉型验证需要 S4 网关；可选注入以启用。
        self.gateway = gateway
        self.computational_enabled = computational_enabled
        self.inferential_enabled = inferential_enabled
        self.visual_enabled = visual_enabled

    @classmethod
    def from_config(
        cls,
        config: VerificationConfig,
        gateway: ModelGateway | None = None,
        policy: ToolPolicy | None = None,
        runner: ProcessRunner | None = None,
    ) -> "VerifierRegistry":
        """按 S10 配置构建注册表（消费三类验证的启停开关）。"""
        registry = cls(
            gateway=gateway,
            computational_enabled=config.computational_enabled,
            inferential_enabled=config.inferential_enabled,
            visual_enabled=config.visual_enabled,
        )
        if config.computational_enabled:
            registry.register(LintVerifier(runner=runner, policy=policy))
            registry.register(TypeCheckVerifier(runner=runner, policy=policy))
            registry.register(SchemaVerifier())
            registry.register(SuiteTestVerifier(runner=runner, policy=policy))
        return registry

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
        if not self.computational_enabled:
            return []
        entries = self.list_verifiers(
            verification_type=VerificationType.COMPUTATIONAL,
            phase=phase,
        )
        verifiers = [e.verifier for e in entries]
        return await run_computational(verifiers, target)

    async def run_inferential(
        self,
        criteria: str,
        content: str,
        dimensions: list[str] | None = None,
        pass_threshold: float = 0.7,
    ) -> VerificationResult:
        """执行推理型验证（独立 LLM judge）。

        需要注册表持有 S4 网关；未配置时返回 SKIP 结果。
        """
        if not self.inferential_enabled:
            return VerificationResult(
                status=VerificationStatus.SKIP,
                verification_type=VerificationType.INFERENTIAL,
                verifier_name="inferential",
                feedback="推理型验证已禁用",
            )
        if self.gateway is None:
            log.warning("推理型验证已跳过：未配置网关")
            return VerificationResult(
                status=VerificationStatus.SKIP,
                verification_type=VerificationType.INFERENTIAL,
                verifier_name="inferential",
                feedback="未配置网关，推理型验证不可用",
            )
        return await run_inferential(
            self.gateway,
            criteria=criteria,
            content=content,
            pass_threshold=pass_threshold,
            dimensions=dimensions,
        )

    async def run_visual(
        self,
        url: str,
        expectations: str,
    ) -> VerificationResult:
        """执行视觉型验证（截图 + 多模态 LLM）。

        需要注册表持有 S4 网关；未配置时返回 SKIP 结果。
        """
        if not self.visual_enabled:
            return VerificationResult(
                status=VerificationStatus.SKIP,
                verification_type=VerificationType.VISUAL,
                verifier_name="visual",
                feedback="视觉型验证已禁用",
            )
        if self.gateway is None:
            log.warning("视觉型验证已跳过：未配置网关")
            return VerificationResult(
                status=VerificationStatus.SKIP,
                verification_type=VerificationType.VISUAL,
                verifier_name="visual",
                feedback="未配置网关，视觉型验证不可用",
            )
        if not self.gateway.capabilities().image:
            return VerificationResult(
                status=VerificationStatus.SKIP,
                verification_type=VerificationType.VISUAL,
                verifier_name="visual",
                feedback="默认模型未声明视觉能力，视觉型验证不可用",
            )
        if find_spec("playwright") is None:
            raise RuntimeError("视觉验证不可用，请安装 praxis[visual]")

        from praxis.verification.visual import run_visual

        return await run_visual(
            self.gateway,
            url=url,
            expectations=expectations,
        )

    def get_phase_config(self, phase: QualityPhase) -> list[str]:
        """获取指定阶段的验证器名称列表。

        Args:
            phase: 质量阶段。

        Returns:
            验证器名称列表。
        """
        entries = self.list_verifiers(phase=phase)
        return [e.verifier.name for e in entries]
