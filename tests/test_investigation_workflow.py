import pandas as pd
import pytest

from src.investigation_workflow import (
    assign_case,
    change_investigation_status,
    create_empty_audit_log,
    initialize_case_management,
)


TEST_TIMESTAMP = (
    "2026-08-01T10:00:00-04:00"
)


def build_consolidated_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": [
                "CL-001",
                "CL-002",
            ],
            "consolidated_alert_level": [
                "CRITICO",
                "ALTO",
            ],
            "consolidated_recommended_action": [
                "REVISION_PRIORITARIA",
                "REVISION_REFORZADA",
            ],
            "estimated_loss_clp": [
                10_000_000,
                5_000_000,
            ],
        }
    )


def test_initialize_case_management() -> None:
    consolidated_df = (
        build_consolidated_dataframe()
    )

    result = initialize_case_management(
        consolidated_df,
        timestamp=TEST_TIMESTAMP,
    )

    assert len(result) == 2

    assert set(
        result["investigation_status"]
    ) == {
        "NUEVO",
    }

    assert (
        result["assigned_analyst_id"]
        .eq("")
        .all()
    )

    assert (
        result[
            "investigation_created_at"
        ]
        .eq(TEST_TIMESTAMP)
        .all()
    )


def test_assign_case() -> None:
    cases_df = initialize_case_management(
        build_consolidated_dataframe(),
        timestamp=TEST_TIMESTAMP,
    )

    audit_log = create_empty_audit_log()

    result = assign_case(
        cases_df,
        audit_log,
        case_id="CL-001",
        analyst_id="ANA-001",
        supervisor_id="SUP-001",
        actor_id="SUP-001",
        comment="Asignación de prueba.",
        timestamp=TEST_TIMESTAMP,
    )

    case = result.cases.loc[
        result.cases["case_id"]
        == "CL-001"
    ].iloc[0]

    assert (
        case["investigation_status"]
        == "ASIGNADO"
    )

    assert (
        case["assigned_analyst_id"]
        == "ANA-001"
    )

    assert len(
        result.audit_log
    ) == 2

    assert set(
        result.audit_log[
            "action_type"
        ]
    ) == {
        "ASSIGN_CASE",
        "CHANGE_STATUS",
    }


def test_start_investigation() -> None:
    cases_df = initialize_case_management(
        build_consolidated_dataframe(),
        timestamp=TEST_TIMESTAMP,
    )

    audit_log = create_empty_audit_log()

    assignment = assign_case(
        cases_df,
        audit_log,
        case_id="CL-001",
        analyst_id="ANA-001",
        actor_id="SUP-001",
        timestamp=TEST_TIMESTAMP,
    )

    result = change_investigation_status(
        assignment.cases,
        assignment.audit_log,
        case_id="CL-001",
        new_status="EN_INVESTIGACION",
        actor_id="ANA-001",
        comment="Inicio de análisis.",
        timestamp=TEST_TIMESTAMP,
    )

    case = result.cases.loc[
        result.cases["case_id"]
        == "CL-001"
    ].iloc[0]

    assert (
        case["investigation_status"]
        == "EN_INVESTIGACION"
    )


def test_invalid_transition_is_rejected() -> None:
    cases_df = initialize_case_management(
        build_consolidated_dataframe(),
        timestamp=TEST_TIMESTAMP,
    )

    audit_log = create_empty_audit_log()

    with pytest.raises(
        ValueError,
        match="Transición de estado no permitida",
    ):
        change_investigation_status(
            cases_df,
            audit_log,
            case_id="CL-001",
            new_status="CONFIRMADO",
            actor_id="ANA-001",
            comment="Transición inválida.",
            resolution=(
                "Intento de confirmación "
                "sin investigación."
            ),
            timestamp=TEST_TIMESTAMP,
        )


def test_confirmation_requires_resolution() -> None:
    cases_df = initialize_case_management(
        build_consolidated_dataframe(),
        timestamp=TEST_TIMESTAMP,
    )

    audit_log = create_empty_audit_log()

    assignment = assign_case(
        cases_df,
        audit_log,
        case_id="CL-001",
        analyst_id="ANA-001",
        actor_id="SUP-001",
        timestamp=TEST_TIMESTAMP,
    )

    investigation = (
        change_investigation_status(
            assignment.cases,
            assignment.audit_log,
            case_id="CL-001",
            new_status="EN_INVESTIGACION",
            actor_id="ANA-001",
            comment="Inicio de investigación.",
            timestamp=TEST_TIMESTAMP,
        )
    )

    with pytest.raises(
        ValueError,
        match="requieren una resolución",
    ):
        change_investigation_status(
            investigation.cases,
            investigation.audit_log,
            case_id="CL-001",
            new_status="CONFIRMADO",
            actor_id="ANA-001",
            comment=(
                "La evidencia fue revisada."
            ),
            resolution="",
            timestamp=TEST_TIMESTAMP,
        )


def test_confirm_and_close_case() -> None:
    cases_df = initialize_case_management(
        build_consolidated_dataframe(),
        timestamp=TEST_TIMESTAMP,
    )

    audit_log = create_empty_audit_log()

    assignment = assign_case(
        cases_df,
        audit_log,
        case_id="CL-001",
        analyst_id="ANA-001",
        actor_id="SUP-001",
        timestamp=TEST_TIMESTAMP,
    )

    investigation = (
        change_investigation_status(
            assignment.cases,
            assignment.audit_log,
            case_id="CL-001",
            new_status="EN_INVESTIGACION",
            actor_id="ANA-001",
            comment="Inicio de investigación.",
            timestamp=TEST_TIMESTAMP,
        )
    )

    confirmation = (
        change_investigation_status(
            investigation.cases,
            investigation.audit_log,
            case_id="CL-001",
            new_status="CONFIRMADO",
            actor_id="ANA-001",
            comment=(
                "Evidencia documental "
                "y transaccional consistente."
            ),
            resolution=(
                "Caso confirmado internamente "
                "y enviado a supervisión."
            ),
            timestamp=TEST_TIMESTAMP,
        )
    )

    closure = change_investigation_status(
        confirmation.cases,
        confirmation.audit_log,
        case_id="CL-001",
        new_status="CERRADO",
        actor_id="SUP-001",
        comment=(
            "Proceso administrativo cerrado."
        ),
        timestamp=TEST_TIMESTAMP,
    )

    case = closure.cases.loc[
        closure.cases["case_id"]
        == "CL-001"
    ].iloc[0]

    assert (
        case["investigation_status"]
        == "CERRADO"
    )

    assert (
        case["investigation_closed_at"]
        == TEST_TIMESTAMP
    )

    assert (
        "confirmado internamente"
        in case["resolution"].lower()
    )

    assert len(
        closure.audit_log
    ) == 6