import pandas as pd
import pytest

from src.case_consolidation import (
    aggregate_digital_events_by_case,
    consolidate_case_and_digital_data,
    highest_risk_alert,
    validate_general_cases,
)


def build_general_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": [
                "CL-001",
                "CL-002",
            ],
            "sector": [
                "Fintech",
                "Municipalidad",
            ],
            "region": [
                "Metropolitana",
                "Valparaíso",
            ],
            "fraud_type_name": [
                "Toma de cuenta",
                "Corrupción",
            ],
            "fraud_category": [
                "FRAUDE_DIGITAL",
                "CORRUPCION",
            ],
            "risk_score": [
                60.0,
                50.0,
            ],
            "assessment_coverage": [
                100.0,
                100.0,
            ],
            "alert_level": [
                "ALTO",
                "ALTO",
            ],
            "decision_status": [
                "REVISION_REFORZADA",
                "REVISION_REFORZADA",
            ],
            "estimated_loss_clp": [
                10_000_000,
                5_000_000,
            ],
            "detection_delay_days": [
                30,
                60,
            ],
            "detection_delay_level": [
                "OPORTUNA",
                "TARDIA",
            ],
        }
    )


def build_digital_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": [
                "EV-001",
                "EV-002",
            ],
            "case_id": [
                "CL-001",
                "CL-001",
            ],
            "event_timestamp": [
                "2026-07-01T03:14:00",
                "2026-07-01T03:18:00",
            ],
            "digital_risk_score": [
                100.0,
                75.0,
            ],
            "digital_assessment_coverage": [
                100.0,
                100.0,
            ],
            "digital_alert_level": [
                "CRITICO",
                "ALTO",
            ],
            "digital_recommended_action": [
                "BLOQUEO_TEMPORAL_Y_REVISION",
                "DESAFIO_MFA_Y_REVISION",
            ],
            "digital_triggered_signals": [
                (
                    "NEW_DEVICE | "
                    "UNUSUAL_IP"
                ),
                (
                    "NEW_DEVICE | "
                    "NEW_BENEFICIARY"
                ),
            ],
        }
    )


def test_highest_risk_alert() -> None:
    result = highest_risk_alert(
        [
            "BAJO",
            "ALTO",
            "CRITICO",
        ]
    )

    assert result == "CRITICO"


def test_aggregate_digital_events() -> None:
    df = build_digital_dataframe()

    result = (
        aggregate_digital_events_by_case(
            df
        )
    )

    row = result.iloc[0]

    assert row["case_id"] == "CL-001"
    assert row["digital_event_count"] == 2

    assert (
        row["digital_risk_score_max"]
        == 100.0
    )

    assert (
        row["digital_risk_score_mean"]
        == pytest.approx(87.5)
    )

    assert (
        row["digital_case_alert_level"]
        == "CRITICO"
    )

    assert (
        row["digital_critical_event_count"]
        == 1
    )

    assert (
        row["digital_high_event_count"]
        == 1
    )

    assert (
        "NEW_BENEFICIARY"
        in row["digital_triggered_signals"]
    )


def test_consolidate_case_and_digital_data() -> None:
    general_df = build_general_dataframe()
    digital_df = build_digital_dataframe()

    result = (
        consolidate_case_and_digital_data(
            general_df,
            digital_df,
        )
    )

    cl_001 = result.loc[
        result["case_id"] == "CL-001"
    ].iloc[0]

    assert cl_001["digital_event_count"] == 2

    assert (
        cl_001[
            "consolidated_alert_level"
        ]
        == "CRITICO"
    )

    assert (
        cl_001[
            "consolidated_recommended_action"
        ]
        == "REVISION_PRIORITARIA"
    )


def test_case_without_digital_events() -> None:
    general_df = build_general_dataframe()
    digital_df = build_digital_dataframe()

    result = (
        consolidate_case_and_digital_data(
            general_df,
            digital_df,
        )
    )

    cl_002 = result.loc[
        result["case_id"] == "CL-002"
    ].iloc[0]

    assert cl_002["digital_event_count"] == 0

    assert not bool(
        cl_002["has_digital_events"]
    )

    assert (
        cl_002[
            "consolidated_alert_level"
        ]
        == "ALTO"
    )

    assert (
        cl_002[
            "consolidated_data_status"
        ]
        == "SIN_EVENTOS_DIGITALES"
    )


def test_duplicate_general_case_is_rejected() -> None:
    general_df = build_general_dataframe()

    general_df.loc[
        1,
        "case_id",
    ] = "CL-001"

    with pytest.raises(
        ValueError,
        match="case_id duplicados",
    ):
        validate_general_cases(
            general_df
        )