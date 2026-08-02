import pandas as pd
import pytest

from src.digital_batch import (
    process_digital_events,
    validate_digital_events,
)


def build_digital_test_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": [
                "EV-001",
                "EV-002",
            ],
            "case_id": [
                "CL-001",
                "CL-002",
            ],
            "customer_id": [
                "CUS-001",
                "CUS-002",
            ],
            "event_timestamp": [
                "2026-07-01T03:14:00",
                "2026-07-01T10:30:00",
            ],
            "channel": [
                "WEB",
                "MOBILE",
            ],
            "region": [
                "Metropolitana",
                "Valparaíso",
            ],
            "new_device": [
                1,
                0,
            ],
            "unusual_ip": [
                1,
                0,
            ],
            "geolocation_mismatch": [
                1,
                0,
            ],
            "failed_mfa_attempts": [
                6,
                0,
            ],
            "recent_password_reset": [
                1,
                0,
            ],
            "new_beneficiary": [
                1,
                0,
            ],
            "amount_spike_ratio": [
                7.0,
                1.0,
            ],
            "rapid_transaction_count_10m": [
                6,
                1,
            ],
        }
    )


def test_process_digital_events() -> None:
    df = build_digital_test_dataframe()

    result = process_digital_events(df)

    assert len(result) == 2

    critical_event = result.loc[
        result["event_id"] == "EV-001"
    ].iloc[0]

    normal_event = result.loc[
        result["event_id"] == "EV-002"
    ].iloc[0]

    assert (
        critical_event[
            "digital_risk_score"
        ]
        == 100.0
    )

    assert (
        critical_event[
            "digital_alert_level"
        ]
        == "CRITICO"
    )

    assert (
        normal_event[
            "digital_risk_score"
        ]
        == 0.0
    )

    assert (
        normal_event[
            "digital_alert_level"
        ]
        == "BAJO"
    )


def test_duplicate_event_is_rejected() -> None:
    df = build_digital_test_dataframe()

    df.loc[1, "event_id"] = "EV-001"

    with pytest.raises(
        ValueError,
        match="event_id duplicados",
    ):
        validate_digital_events(df)


def test_invalid_timestamp_is_rejected() -> None:
    df = build_digital_test_dataframe()

    df.loc[
        0,
        "event_timestamp",
    ] = "fecha-invalida"

    with pytest.raises(
        ValueError,
        match="fechas inválidas",
    ):
        validate_digital_events(df)