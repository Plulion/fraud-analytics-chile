import pandas as pd
import pytest

from src.case_closure import (
    close_case_with_supervisor_approval,
    create_empty_feedback_registry,
    create_empty_resolution_registry,
)


def build_cases_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": [
                "CL-001",
                "CL-002",
            ],
            "investigation_status": [
                "EN_INVESTIGACION",
                "CERRADO",
            ],
            "assigned_analyst_id": [
                "ANA-001",
                "ANA-002",
            ],
            "supervisor_id": [
                "SUP-001",
                "SUP-002",
            ],
            "investigation_created_at": [
                "2026-08-01T20:00:00-04:00",
                "2026-08-01T20:00:00-04:00",
            ],
            "investigation_updated_at": [
                "2026-08-01T21:00:00-04:00",
                "2026-08-01T21:00:00-04:00",
            ],
            "investigation_closed_at": [
                "",
                "2026-08-01T22:00:00-04:00",
            ],
            "resolution": [
                "",
                "DESCARTADO",
            ],
        }
    )


def build_audit_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "audit_id": [
                "AUD-000001",
            ],
            "case_id": [
                "CL-001",
            ],
            "event_timestamp": [
                "2026-08-01T20:00:00-04:00",
            ],
            "actor_id": [
                "SUP-001",
            ],
            "action_type": [
                "ASSIGN_CASE",
            ],
            "field_name": [
                "assigned_analyst_id",
            ],
            "previous_value": [
                "",
            ],
            "new_value": [
                "ANA-001",
            ],
            "comment": [
                "Asignación inicial.",
            ],
        }
    )


def close_confirmed_case():
    return close_case_with_supervisor_approval(
        build_cases_dataframe(),
        create_empty_resolution_registry(),
        build_audit_dataframe(),
        create_empty_feedback_registry(),
        case_id="CL-001",
        proposed_outcome="CONFIRMADO",
        resolution_summary=(
            "La investigación confirma el caso."
        ),
        resolution_rationale=(
            "Existen antecedentes suficientes "
            "para confirmar el resultado."
        ),
        proposed_by="ANA-001",
        supervisor_id="SUP-001",
        confirmed_loss_clp=10_000_000,
        recovered_amount_clp=2_000_000,
        proposed_at=(
            "2026-08-01T22:00:00-04:00"
        ),
        approved_at=(
            "2026-08-01T22:30:00-04:00"
        ),
    )


def test_confirmed_case_is_closed() -> None:
    result = close_confirmed_case()

    case = result.cases.loc[
        result.cases["case_id"]
        == "CL-001"
    ].iloc[0]

    assert (
        case["investigation_status"]
        == "CERRADO"
    )

    assert (
        case["resolution"]
        == "CONFIRMADO"
    )

    assert (
        case["investigation_closed_at"]
        == "2026-08-01T22:30:00-04:00"
    )


def test_resolution_is_approved() -> None:
    result = close_confirmed_case()

    resolution = (
        result.resolutions.iloc[0]
    )

    assert (
        resolution["resolution_id"]
        == "RES-000001"
    )

    assert (
        resolution["approval_status"]
        == "APROBADA"
    )

    assert (
        resolution["proposed_by"]
        == "ANA-001"
    )

    assert (
        resolution["supervisor_id"]
        == "SUP-001"
    )


def test_confirmed_case_creates_positive_feedback() -> None:
    result = close_confirmed_case()

    feedback = result.feedback.iloc[0]

    assert (
        feedback["investigation_outcome"]
        == "CONFIRMADO"
    )

    assert (
        feedback["outcome_label"]
        == 1
    )


def test_discarded_case_creates_negative_feedback() -> None:
    result = close_case_with_supervisor_approval(
        build_cases_dataframe(),
        create_empty_resolution_registry(),
        build_audit_dataframe(),
        create_empty_feedback_registry(),
        case_id="CL-001",
        proposed_outcome="DESCARTADO",
        resolution_summary=(
            "La alerta fue descartada."
        ),
        resolution_rationale=(
            "Los antecedentes revisados explican "
            "razonablemente la actividad."
        ),
        proposed_by="ANA-001",
        supervisor_id="SUP-001",
        confirmed_loss_clp=0,
        recovered_amount_clp=0,
        approved_at=(
            "2026-08-01T22:30:00-04:00"
        ),
    )

    feedback = result.feedback.iloc[0]

    assert (
        feedback["outcome_label"]
        == 0
    )

    assert (
        feedback["investigation_outcome"]
        == "DESCARTADO"
    )


def test_closed_case_cannot_be_closed_again() -> None:
    with pytest.raises(
        ValueError,
        match="ya está cerrado",
    ):
        close_case_with_supervisor_approval(
            build_cases_dataframe(),
            create_empty_resolution_registry(),
            build_audit_dataframe(),
            create_empty_feedback_registry(),
            case_id="CL-002",
            proposed_outcome="DESCARTADO",
            resolution_summary=(
                "Intento de segundo cierre."
            ),
            resolution_rationale=(
                "El caso ya estaba cerrado."
            ),
            proposed_by="ANA-002",
            supervisor_id="SUP-002",
            confirmed_loss_clp=0,
            recovered_amount_clp=0,
        )


def test_wrong_analyst_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="analista asignado",
    ):
        close_case_with_supervisor_approval(
            build_cases_dataframe(),
            create_empty_resolution_registry(),
            build_audit_dataframe(),
            create_empty_feedback_registry(),
            case_id="CL-001",
            proposed_outcome="CONFIRMADO",
            resolution_summary=(
                "Resultado propuesto."
            ),
            resolution_rationale=(
                "Antecedentes suficientes."
            ),
            proposed_by="ANA-999",
            supervisor_id="SUP-001",
            confirmed_loss_clp=10,
            recovered_amount_clp=0,
        )


def test_wrong_supervisor_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="no coincide",
    ):
        close_case_with_supervisor_approval(
            build_cases_dataframe(),
            create_empty_resolution_registry(),
            build_audit_dataframe(),
            create_empty_feedback_registry(),
            case_id="CL-001",
            proposed_outcome="CONFIRMADO",
            resolution_summary=(
                "Resultado propuesto."
            ),
            resolution_rationale=(
                "Antecedentes suficientes."
            ),
            proposed_by="ANA-001",
            supervisor_id="SUP-999",
            confirmed_loss_clp=10,
            recovered_amount_clp=0,
        )


def test_recovery_cannot_exceed_confirmed_loss() -> None:
    with pytest.raises(
        ValueError,
        match="no puede ser mayor",
    ):
        close_case_with_supervisor_approval(
            build_cases_dataframe(),
            create_empty_resolution_registry(),
            build_audit_dataframe(),
            create_empty_feedback_registry(),
            case_id="CL-001",
            proposed_outcome="CONFIRMADO",
            resolution_summary=(
                "Resultado propuesto."
            ),
            resolution_rationale=(
                "Antecedentes suficientes."
            ),
            proposed_by="ANA-001",
            supervisor_id="SUP-001",
            confirmed_loss_clp=100,
            recovered_amount_clp=200,
        )


def test_discarded_case_cannot_have_confirmed_loss() -> None:
    with pytest.raises(
        ValueError,
        match="debe tener",
    ):
        close_case_with_supervisor_approval(
            build_cases_dataframe(),
            create_empty_resolution_registry(),
            build_audit_dataframe(),
            create_empty_feedback_registry(),
            case_id="CL-001",
            proposed_outcome="DESCARTADO",
            resolution_summary=(
                "Resultado descartado."
            ),
            resolution_rationale=(
                "La operación tiene explicación."
            ),
            proposed_by="ANA-001",
            supervisor_id="SUP-001",
            confirmed_loss_clp=500,
            recovered_amount_clp=0,
        )


def test_audit_events_are_created() -> None:
    result = close_confirmed_case()

    new_events = result.audit.tail(
        4
    )

    actions = set(
        new_events["action_type"]
    )

    assert (
        "PROPOSE_RESOLUTION"
        in actions
    )

    assert (
        "APPROVE_RESOLUTION"
        in actions
    )

    assert (
        "CHANGE_STATUS"
        in actions
    )