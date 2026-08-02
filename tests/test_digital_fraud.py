import pytest

from src.digital_fraud import (
    assess_digital_event,
)


def test_critical_digital_event() -> None:
    event = {
        "new_device": 1,
        "unusual_ip": 1,
        "geolocation_mismatch": 1,
        "failed_mfa_attempts": 6,
        "recent_password_reset": 1,
        "new_beneficiary": 1,
        "amount_spike_ratio": 7.5,
        "rapid_transaction_count_10m": 6,
    }

    result = assess_digital_event(
        event
    )

    assert result.device_score == 15.0
    assert result.network_score == 25.0
    assert (
        result.authentication_score
        == 25.0
    )
    assert (
        result.transaction_score
        == 35.0
    )
    assert (
        result.digital_risk_score
        == 100.0
    )
    assert (
        result.assessment_coverage
        == 100.0
    )
    assert result.alert_level == "CRITICO"

    assert (
        result.recommended_action
        == "BLOQUEO_TEMPORAL_Y_REVISION"
    )

    assert (
        "NEW_DEVICE"
        in result.triggered_signals
    )

    assert (
        "MFA_FAILURES_HIGH"
        in result.triggered_signals
    )


def test_normal_digital_event() -> None:
    event = {
        "new_device": 0,
        "unusual_ip": 0,
        "geolocation_mismatch": 0,
        "failed_mfa_attempts": 0,
        "recent_password_reset": 0,
        "new_beneficiary": 0,
        "amount_spike_ratio": 1.0,
        "rapid_transaction_count_10m": 1,
    }

    result = assess_digital_event(
        event
    )

    assert (
        result.digital_risk_score
        == 0.0
    )
    assert (
        result.assessment_coverage
        == 100.0
    )
    assert result.alert_level == "BAJO"

    assert (
        result.recommended_action
        == "MONITOREO_NORMAL"
    )

    assert result.triggered_signals == ()


def test_incomplete_digital_event() -> None:
    event = {
        "new_device": 1,
        "unusual_ip": None,
        "geolocation_mismatch": None,
        "failed_mfa_attempts": None,
        "recent_password_reset": 1,
        "new_beneficiary": None,
        "amount_spike_ratio": 3.0,
        "rapid_transaction_count_10m": None,
    }

    result = assess_digital_event(
        event
    )

    assert (
        result.digital_risk_score
        == 32.5
    )

    assert (
        result.assessment_coverage
        == 40.0
    )

    assert (
        result.alert_level
        == "INCOMPLETO"
    )

    assert (
        result.recommended_action
        == "REQUIERE_MAS_ANTECEDENTES"
    )


def test_invalid_binary_signal() -> None:
    event = {
        "new_device": 3,
        "unusual_ip": 0,
        "geolocation_mismatch": 0,
        "failed_mfa_attempts": 0,
        "recent_password_reset": 0,
        "new_beneficiary": 0,
        "amount_spike_ratio": 1.0,
        "rapid_transaction_count_10m": 1,
    }

    with pytest.raises(
        ValueError,
        match="new_device",
    ):
        assess_digital_event(
            event
        )


def test_negative_mfa_attempts_are_rejected() -> None:
    event = {
        "new_device": 0,
        "unusual_ip": 0,
        "geolocation_mismatch": 0,
        "failed_mfa_attempts": -1,
        "recent_password_reset": 0,
        "new_beneficiary": 0,
        "amount_spike_ratio": 1.0,
        "rapid_transaction_count_10m": 1,
    }

    with pytest.raises(
        ValueError,
        match="failed_mfa_attempts",
    ):
        assess_digital_event(
            event
        )