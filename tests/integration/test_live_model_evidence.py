"""Deterministic classification contracts for live-model evidence."""

from types import SimpleNamespace

from tests.integration.live_model_evidence import (
    CapabilityClassification,
    HealthWindowEvidence,
    ModalityObservation,
    classify_capability_windows,
    format_safe_evidence,
    response_has_model_output,
)


def healthy_success_window() -> HealthWindowEvidence:
    return HealthWindowEvidence(
        before_healthy=True,
        modality=ModalityObservation.succeeded(),
        after_healthy=True,
    )


def healthy_error_window(
    category: str = "GatewayError",
    original_type: str = "BadRequestError",
) -> HealthWindowEvidence:
    return HealthWindowEvidence(
        before_healthy=True,
        modality=ModalityObservation.failed(category, original_type),
        after_healthy=True,
    )


def test_fewer_than_three_windows_remain_unclassified() -> None:
    evidence = classify_capability_windows(
        [healthy_success_window(), healthy_success_window()]
    )

    assert evidence.classification is CapabilityClassification.UNCLASSIFIED
    assert evidence.window_count == 2


def test_three_independent_successes_classify_supported() -> None:
    evidence = classify_capability_windows(
        [healthy_success_window(), healthy_success_window(), healthy_success_window()]
    )

    assert evidence.classification is CapabilityClassification.SUPPORTED
    assert evidence.error_category is None
    assert evidence.original_type is None


def test_three_identical_errors_classify_unsupported() -> None:
    evidence = classify_capability_windows(
        [healthy_error_window(), healthy_error_window(), healthy_error_window()]
    )

    assert evidence.classification is CapabilityClassification.UNSUPPORTED
    assert evidence.error_category == "GatewayError"
    assert evidence.original_type == "BadRequestError"


def test_mixed_success_and_error_remains_unclassified() -> None:
    evidence = classify_capability_windows(
        [healthy_success_window(), healthy_error_window(), healthy_success_window()]
    )

    assert evidence.classification is CapabilityClassification.UNCLASSIFIED


def test_mixed_error_categories_remain_unclassified() -> None:
    evidence = classify_capability_windows(
        [
            healthy_error_window(),
            healthy_error_window("ProviderUnavailableError", "ServiceUnavailableError"),
            healthy_error_window(),
        ]
    )

    assert evidence.classification is CapabilityClassification.UNCLASSIFIED


def test_repeated_transient_provider_errors_remain_unclassified() -> None:
    transient = healthy_error_window(
        "ProviderUnavailableError",
        "ServiceUnavailableError",
    )
    evidence = classify_capability_windows([transient, transient, transient])

    assert evidence.classification is CapabilityClassification.UNCLASSIFIED


def test_unhealthy_control_invalidates_all_modality_results() -> None:
    unhealthy = HealthWindowEvidence(
        before_healthy=False,
        modality=None,
        after_healthy=False,
        control_category="ProviderUnavailableError",
    )
    evidence = classify_capability_windows(
        [healthy_error_window(), unhealthy, healthy_error_window()]
    )

    assert evidence.classification is CapabilityClassification.UNCLASSIFIED
    assert evidence.control_categories == ("ProviderUnavailableError",)


def test_safe_evidence_projection_contains_categories_but_no_payload() -> None:
    unhealthy = HealthWindowEvidence(
        before_healthy=False,
        modality=None,
        after_healthy=False,
        control_category="ProviderUnavailableError",
    )
    evidence = classify_capability_windows([unhealthy, unhealthy, unhealthy])
    projection = format_safe_evidence("image", evidence)

    assert "classification=unclassified" in projection
    assert "ProviderUnavailableError" in projection
    assert "base64" not in projection.lower()
    assert "data:" not in projection.lower()


def test_reasoning_only_model_response_is_model_output() -> None:
    assert response_has_model_output(SimpleNamespace(content="", reasoning_content="step"))


def test_empty_model_response_has_no_output() -> None:
    assert not response_has_model_output(SimpleNamespace(content="", reasoning_content=""))
