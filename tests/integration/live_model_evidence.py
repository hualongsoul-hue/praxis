"""Safe, deterministic classification for repeated live-model evidence."""

from dataclasses import dataclass
from enum import StrEnum


class CapabilityClassification(StrEnum):
    """Endpoint capability state derived only from healthy repeated windows."""

    UNCLASSIFIED = "unclassified"
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"


CLASSIFYING_UNSUPPORTED_SIGNATURES = frozenset(
    {("GatewayError", "BadRequestError")}
)


@dataclass(frozen=True, slots=True)
class ModalityObservation:
    """One modality result without response or request payloads."""

    success: bool
    error_category: str | None = None
    original_type: str | None = None

    @classmethod
    def succeeded(cls) -> "ModalityObservation":
        return cls(success=True)

    @classmethod
    def failed(cls, error_category: str, original_type: str) -> "ModalityObservation":
        if not error_category or not original_type:
            raise ValueError("error observations require safe categories")
        return cls(
            success=False,
            error_category=error_category,
            original_type=original_type,
        )


@dataclass(frozen=True, slots=True)
class HealthWindowEvidence:
    """Before/modality/after results for one independent endpoint window."""

    before_healthy: bool
    modality: ModalityObservation | None
    after_healthy: bool
    control_category: str | None = None


@dataclass(frozen=True, slots=True)
class CapabilityEvidence:
    """A capability classification and safe diagnostic categories."""

    classification: CapabilityClassification
    window_count: int
    error_category: str | None = None
    original_type: str | None = None
    control_categories: tuple[str, ...] = ()


def classify_capability_windows(
    windows: list[HealthWindowEvidence],
    *,
    required_windows: int = 3,
) -> CapabilityEvidence:
    """Classify only three or more healthy windows with identical outcomes."""

    if required_windows < 3:
        raise ValueError("at least three independent health windows are required")
    control_categories = tuple(
        dict.fromkeys(
            window.control_category
            for window in windows
            if window.control_category is not None
        )
    )
    unclassified = CapabilityEvidence(
        classification=CapabilityClassification.UNCLASSIFIED,
        window_count=len(windows),
        control_categories=control_categories,
    )
    if len(windows) < required_windows:
        return unclassified
    if any(
        not window.before_healthy
        or not window.after_healthy
        or window.modality is None
        for window in windows
    ):
        return unclassified

    observations = [window.modality for window in windows]
    if all(observation is not None and observation.success for observation in observations):
        return CapabilityEvidence(
            classification=CapabilityClassification.SUPPORTED,
            window_count=len(windows),
        )

    failures = [
        observation
        for observation in observations
        if observation is not None and not observation.success
    ]
    if len(failures) != len(windows):
        return unclassified
    signatures = {
        (observation.error_category, observation.original_type)
        for observation in failures
    }
    if len(signatures) != 1:
        return unclassified
    error_category, original_type = signatures.pop()
    if error_category is None or original_type is None:
        return unclassified
    if (error_category, original_type) not in CLASSIFYING_UNSUPPORTED_SIGNATURES:
        return unclassified
    return CapabilityEvidence(
        classification=CapabilityClassification.UNSUPPORTED,
        window_count=len(windows),
        error_category=error_category,
        original_type=original_type,
    )


def format_safe_evidence(modality: str, evidence: CapabilityEvidence) -> str:
    """Project only modality, counts, and mapped categories into test output."""

    fields = [
        f"modality={modality}",
        f"classification={evidence.classification.value}",
        f"windows={evidence.window_count}",
    ]
    if evidence.error_category is not None:
        fields.append(f"error_category={evidence.error_category}")
    if evidence.original_type is not None:
        fields.append(f"original_type={evidence.original_type}")
    if evidence.control_categories:
        fields.append(f"control_categories={','.join(evidence.control_categories)}")
    return ";".join(fields)
