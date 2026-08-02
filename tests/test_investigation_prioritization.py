import pandas as pd
import pytest

from src.investigation_prioritization import (
    calculate_coverage_points,
    calculate_loss_points,
    calculate_recurrence_points,
    classify_investigation_priority,
    prioritize_investigations,
)


def build_alerts_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "alert_id": [
                "ALT-001",
                "ALT-002",
                "ALT-003",
                "ALT-004",
            ],
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
            "created_at": [
                "2026-08-01T10:00:00-04:00",
                "2026-08-01T10:01:00-04:00",
                "2026-08-01T10:02:00-04:00",
                "2026-08-01T10:03:00-04:00",
            ],
            "alert_status": [
                "ABIERTA",
                "ABIERTA",
                "ABIERTA",
                "ABIERTA",
            ],
            "risk_level": [
                "CRITICO",
                "ALTO",
                "MEDIO",
                "ALTO",
            ],
            "estimated_loss_clp": [
                1_000_000,
                60_000_000,
                3_000_000,
                15_000_000,
            ],
        }
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
            "assessment_coverage": [
                100.0,
                100.0,
                80.0,
                60.0,
            ],
            "data_quality_status": [
                "COMPLETO",
                "COMPLETO",
                "COMPLETO",
                "REQUIERE_REVISION_DE_DATOS",
            ],
        }
    )


def build_consolidated_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": [
                "CL-001",
                "CL-002",
                "CL-003",
                "CL-004",
            ],
            "digital_event_count": [
                1,
                5,
                2,
                0,
            ],
        }
    )


def test_component_points() -> None:
    assert calculate_loss_points(
        50_000_000
    ) == 25

    assert calculate_loss_points(
        10_000_000
    ) == 20

    assert calculate_recurrence_points(
        4
    ) == 15

    assert calculate_recurrence_points(
        2
    ) == 10

    assert calculate_coverage_points(
        95
    ) == 10

    assert calculate_coverage_points(
        75
    ) == 7


def test_priority_classification() -> None:
    assert (
        classify_investigation_priority(
            75
        )
        == "CRITICA"
    )

    assert (
        classify_investigation_priority(
            55
        )
        == "ALTA"
    )

    assert (
        classify_investigation_priority(
            35
        )
        == "MEDIA"
    )

    assert (
        classify_investigation_priority(
            20
        )
        == "BAJA"
    )


def test_prioritization_orders_by_total_priority() -> None:
    result = prioritize_investigations(
        build_alerts_dataframe(),
        build_assessments_dataframe(),
        build_consolidated_dataframe(),
        investigation_capacity=4,
    )

    assert (
        result.iloc[0]["case_id"]
        == "CL-002"
    )

    assert (
        result.iloc[0][
            "priority_score"
        ]
        == 80
    )

    assert (
        result.iloc[0][
            "investigation_priority"
        ]
        == "CRITICA"
    )


def test_capacity_selects_only_top_alerts() -> None:
    result = prioritize_investigations(
        build_alerts_dataframe(),
        build_assessments_dataframe(),
        build_consolidated_dataframe(),
        investigation_capacity=2,
    )

    selected = result.loc[
        result["queue_status"]
        == "SELECCIONADA"
    ]

    waiting = result.loc[
        result["queue_status"]
        == "EN_ESPERA"
    ]

    assert len(selected) == 2
    assert len(waiting) == 2

    assert selected[
        "queue_position"
    ].tolist() == [
        1,
        2,
    ]


def test_zero_capacity_keeps_all_alerts_waiting() -> None:
    result = prioritize_investigations(
        build_alerts_dataframe(),
        build_assessments_dataframe(),
        build_consolidated_dataframe(),
        investigation_capacity=0,
    )

    assert (
        result["queue_status"]
        .eq("EN_ESPERA")
        .all()
    )


def test_missing_assessment_relation_is_rejected() -> None:
    assessments_df = (
        build_assessments_dataframe()
    )

    assessments_df = assessments_df.loc[
        assessments_df["assessment_id"]
        != "ASM-004"
    ]

    with pytest.raises(
        ValueError,
        match=(
            "No fue posible relacionar"
        ),
    ):
        prioritize_investigations(
            build_alerts_dataframe(),
            assessments_df,
            build_consolidated_dataframe(),
            investigation_capacity=2,
        )


def test_invalid_capacity_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="investigation_capacity",
    ):
        prioritize_investigations(
            build_alerts_dataframe(),
            build_assessments_dataframe(),
            build_consolidated_dataframe(),
            investigation_capacity=-1,
        )