import pandas as pd
import pytest

from src.model_feedback import (
    build_feedback_dataset,
    calculate_backtesting_metrics,
    classify_prediction_result,
)


def build_assessments_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "assessment_id": [
                "ASM-001",
                "ASM-002",
                "ASM-003",
                "ASM-004",
            ],
            "case_id": [
                "CL-001",
                "CL-002",
                "CL-003",
                "CL-004",
            ],
            "engine_type": [
                "RULE_ENGINE",
                "RULE_ENGINE",
                "RULE_ENGINE",
                "RULE_ENGINE",
            ],
            "engine_version": [
                "1.0.0",
                "1.0.0",
                "1.0.0",
                "1.0.0",
            ],
            "policy_id": [
                "POLICY-001",
                "POLICY-001",
                "POLICY-001",
                "POLICY-001",
            ],
            "policy_version": [
                "1.0.0",
                "1.0.0",
                "1.0.0",
                "1.0.0",
            ],
            "evaluated_at": [
                "2026-08-01T10:00:00-04:00",
                "2026-08-01T10:01:00-04:00",
                "2026-08-01T10:02:00-04:00",
                "2026-08-01T10:03:00-04:00",
            ],
            "risk_score": [
                90.0,
                70.0,
                20.0,
                10.0,
            ],
            "assessment_coverage": [
                100.0,
                100.0,
                100.0,
                100.0,
            ],
            "calculated_alert_level": [
                "CRITICO",
                "ALTO",
                "BAJO",
                "BAJO",
            ],
            "data_quality_status": [
                "COMPLETO",
                "COMPLETO",
                "COMPLETO",
                "COMPLETO",
            ],
        }
    )


def build_alerts_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "alert_id": [
                "ALT-001",
                "ALT-002",
            ],
            "assessment_id": [
                "ASM-001",
                "ASM-002",
            ],
            "case_id": [
                "CL-001",
                "CL-002",
            ],
            "alert_status": [
                "ABIERTA",
                "ABIERTA",
            ],
            "alert_priority": [
                "CRITICA",
                "ALTA",
            ],
        }
    )


def build_outcomes_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": [
                "CL-001",
                "CL-002",
                "CL-003",
                "CL-004",
            ],
            "investigation_outcome": [
                "CONFIRMADO",
                "DESCARTADO",
                "DESCARTADO",
                "CONFIRMADO",
            ],
            "resolved_at": [
                "2026-08-01T18:00:00-04:00",
                "2026-08-01T18:10:00-04:00",
                "2026-08-01T18:20:00-04:00",
                "2026-08-01T18:30:00-04:00",
            ],
            "reviewer_id": [
                "SUP-001",
                "SUP-001",
                "SUP-002",
                "SUP-002",
            ],
            "resolution_summary": [
                "Fraude confirmado.",
                "Alerta descartada.",
                "Actividad legítima.",
                "Fraude no detectado inicialmente.",
            ],
        }
    )


def test_prediction_result_classification() -> None:
    assert classify_prediction_result(
        True,
        True,
    ) == "TRUE_POSITIVE"

    assert classify_prediction_result(
        True,
        False,
    ) == "FALSE_POSITIVE"

    assert classify_prediction_result(
        False,
        False,
    ) == "TRUE_NEGATIVE"

    assert classify_prediction_result(
        False,
        True,
    ) == "FALSE_NEGATIVE"


def test_build_feedback_dataset() -> None:
    result = build_feedback_dataset(
        build_assessments_dataframe(),
        build_alerts_dataframe(),
        build_outcomes_dataframe(),
    )

    assert len(result) == 4

    assert set(
        result["prediction_result"]
    ) == {
        "TRUE_POSITIVE",
        "FALSE_POSITIVE",
        "TRUE_NEGATIVE",
        "FALSE_NEGATIVE",
    }


def test_feedback_labels() -> None:
    result = build_feedback_dataset(
        build_assessments_dataframe(),
        build_alerts_dataframe(),
        build_outcomes_dataframe(),
    )

    confirmed = result.loc[
        result["investigation_outcome"]
        == "CONFIRMADO"
    ]

    discarded = result.loc[
        result["investigation_outcome"]
        == "DESCARTADO"
    ]

    assert (
        confirmed["outcome_label"]
        .eq(1)
        .all()
    )

    assert (
        discarded["outcome_label"]
        .eq(0)
        .all()
    )


def test_calculate_backtesting_metrics() -> None:
    feedback_df = build_feedback_dataset(
        build_assessments_dataframe(),
        build_alerts_dataframe(),
        build_outcomes_dataframe(),
    )

    metrics = calculate_backtesting_metrics(
        feedback_df
    )

    assert metrics[
        "true_positive"
    ] == 1

    assert metrics[
        "false_positive"
    ] == 1

    assert metrics[
        "true_negative"
    ] == 1

    assert metrics[
        "false_negative"
    ] == 1

    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["specificity"] == 0.5
    assert metrics["accuracy"] == 0.5


def test_invalid_outcome_is_rejected() -> None:
    outcomes_df = (
        build_outcomes_dataframe()
    )

    outcomes_df.loc[
        0,
        "investigation_outcome",
    ] = "PENDIENTE"

    with pytest.raises(
        ValueError,
        match=(
            "Resultado de investigación "
            "desconocido"
        ),
    ):
        build_feedback_dataset(
            build_assessments_dataframe(),
            build_alerts_dataframe(),
            outcomes_df,
        )


def test_outcome_without_assessment_is_rejected() -> None:
    outcomes_df = (
        build_outcomes_dataframe()
    )

    outcomes_df.loc[
        3,
        "case_id",
    ] = "CL-999"

    with pytest.raises(
        ValueError,
        match=(
            "resultados sin una evaluación"
        ),
    ):
        build_feedback_dataset(
            build_assessments_dataframe(),
            build_alerts_dataframe(),
            outcomes_df,
        )