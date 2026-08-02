import pandas as pd
import pytest

from src.professional_pipeline import (
    AssessmentRecord,
    create_alert_record,
    process_professional_pipeline,
    select_alert,
)


TEST_TIMESTAMP = (
    "2026-08-01T20:00:00-04:00"
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
            "risk_score": [
                90.0,
                70.0,
                40.0,
                10.0,
            ],
            "assessment_coverage": [
                100.0,
                100.0,
                100.0,
                100.0,
            ],
            "alert_level": [
                "CRITICO",
                "ALTO",
                "MEDIO",
                "BAJO",
            ],
            "decision_status": [
                "REVISION_PRIORITARIA",
                "REVISION_REFORZADA",
                "REVISION_ESTANDAR",
                "MONITOREO_NORMAL",
            ],
            "consolidated_alert_level": [
                "CRITICO",
                "ALTO",
                "MEDIO",
                "BAJO",
            ],
            "consolidated_data_status": [
                "COMPLETO",
                "COMPLETO",
                "COMPLETO",
                "COMPLETO",
            ],
            "consolidated_recommended_action": [
                "REVISION_PRIORITARIA",
                "REVISION_REFORZADA",
                "REVISION_ESTANDAR",
                "MONITOREO_NORMAL",
            ],
            "estimated_loss_clp": [
                50_000_000,
                20_000_000,
                5_000_000,
                100_000,
            ],
        }
    )


def sequential_factory(
    prefix: str,
):
    counter = 0

    def factory() -> str:
        nonlocal counter
        counter += 1

        return (
            f"{prefix}-{counter:06d}"
        )

    return factory


def test_pipeline_creates_one_assessment_per_case() -> None:
    df = build_consolidated_dataframe()

    result = process_professional_pipeline(
        df,
        timestamp=TEST_TIMESTAMP,
        assessment_id_factory=(
            sequential_factory("ASM")
        ),
        alert_id_factory=(
            sequential_factory("ALT")
        ),
    )

    assert len(
        result.assessments
    ) == 4

    assert set(
        result.assessments["case_id"]
    ) == {
        "CL-001",
        "CL-002",
        "CL-003",
        "CL-004",
    }

    assert (
        result.assessments[
            "engine_version"
        ]
        .eq("1.0.0")
        .all()
    )

    assert (
        result.assessments[
            "policy_version"
        ]
        .eq("1.0.0")
        .all()
    )


def test_low_risk_assessment_does_not_create_alert() -> None:
    df = build_consolidated_dataframe()

    result = process_professional_pipeline(
        df,
        timestamp=TEST_TIMESTAMP,
        assessment_id_factory=(
            sequential_factory("ASM")
        ),
        alert_id_factory=(
            sequential_factory("ALT")
        ),
    )

    assert len(
        result.alerts
    ) == 3

    assert "CL-004" not in set(
        result.alerts["case_id"]
    )


def test_critical_assessment_creates_critical_alert() -> None:
    df = build_consolidated_dataframe()

    result = process_professional_pipeline(
        df,
        timestamp=TEST_TIMESTAMP,
        assessment_id_factory=(
            sequential_factory("ASM")
        ),
        alert_id_factory=(
            sequential_factory("ALT")
        ),
    )

    alert = result.alerts.loc[
        result.alerts["case_id"]
        == "CL-001"
    ].iloc[0]

    assert (
        alert["alert_priority"]
        == "CRITICA"
    )

    assert (
        alert["risk_level"]
        == "CRITICO"
    )

    assert (
        alert["alert_status"]
        == "ABIERTA"
    )


def test_data_quality_problem_creates_alert() -> None:
    assessment = AssessmentRecord(
        assessment_id="ASM-000001",
        case_id="CL-100",
        engine_type=(
            "CONSOLIDATED_RULE_ENGINE"
        ),
        engine_version="1.0.0",
        policy_id=(
            "CONSOLIDATED_RISK_POLICY"
        ),
        policy_version="1.0.0",
        input_schema_version="1.0.0",
        evaluated_at=TEST_TIMESTAMP,
        risk_score=20.0,
        assessment_coverage=50.0,
        calculated_alert_level="BAJO",
        recommended_action=(
            "REQUIERE_MAS_ANTECEDENTES"
        ),
        data_quality_status=(
            "REQUIERE_REVISION_DE_DATOS"
        ),
    )

    decision = select_alert(
        assessment
    )

    assert (
        decision.should_create_alert
    )

    assert (
        decision.alert_priority
        == "ALTA"
    )


def test_cannot_create_alert_from_negative_decision() -> None:
    assessment = AssessmentRecord(
        assessment_id="ASM-000001",
        case_id="CL-100",
        engine_type=(
            "CONSOLIDATED_RULE_ENGINE"
        ),
        engine_version="1.0.0",
        policy_id=(
            "CONSOLIDATED_RISK_POLICY"
        ),
        policy_version="1.0.0",
        input_schema_version="1.0.0",
        evaluated_at=TEST_TIMESTAMP,
        risk_score=10.0,
        assessment_coverage=100.0,
        calculated_alert_level="BAJO",
        recommended_action=(
            "MONITOREO_NORMAL"
        ),
        data_quality_status="COMPLETO",
    )

    decision = select_alert(
        assessment
    )

    assert not (
        decision.should_create_alert
    )

    with pytest.raises(
        ValueError,
        match=(
            "No se puede crear una alerta"
        ),
    ):
        create_alert_record(
            assessment,
            decision,
            alert_id="ALT-000001",
            created_at=TEST_TIMESTAMP,
            estimated_loss_clp=100_000,
        )


def test_assessment_and_alert_are_traceable() -> None:
    df = build_consolidated_dataframe()

    result = process_professional_pipeline(
        df,
        timestamp=TEST_TIMESTAMP,
        assessment_id_factory=(
            sequential_factory("ASM")
        ),
        alert_id_factory=(
            sequential_factory("ALT")
        ),
    )

    alert = result.alerts.loc[
        result.alerts["case_id"]
        == "CL-001"
    ].iloc[0]

    assessment_id = alert[
        "assessment_id"
    ]

    matching_assessments = (
        result.assessments.loc[
            result.assessments[
                "assessment_id"
            ]
            == assessment_id
        ]
    )

    assert len(
        matching_assessments
    ) == 1