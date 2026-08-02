import pandas as pd
import pytest

from src.case_reopening import (
    reopen_closed_case,
)


def build_cases_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": [
                "CL-001",
                "CL-002",
            ],
            "investigation_status": [
                "CERRADO",
                "EN_INVESTIGACION",
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
                "2026-08-01T22:30:00-04:00",
                "2026-08-01T21:00:00-04:00",
            ],
            "investigation_closed_at": [
                "2026-08-01T22:30:00-04:00",
                "",
            ],
            "resolution": [
                "CONFIRMADO",
                "",
            ],
        }
    )


def build_resolutions_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "resolution_id": [
                "RES-000001",
            ],
            "case_id": [
                "CL-001",
            ],
            "resolution_version": [
                1,
            ],
            "proposed_outcome": [
                "CONFIRMADO",
            ],
            "resolution_summary": [
                "Caso confirmado.",
            ],
            "resolution_rationale": [
                "Antecedentes suficientes.",
            ],
            "proposed_by": [
                "ANA-001",
            ],
            "proposed_at": [
                "2026-08-01T22:00:00-04:00",
            ],
            "supervisor_id": [
                "SUP-001",
            ],
            "approval_status": [
                "APROBADA",
            ],
            "approved_at": [
                "2026-08-01T22:30:00-04:00",
            ],
            "confirmed_loss_clp": [
                10_000_000.0,
            ],
            "recovered_amount_clp": [
                2_000_000.0,
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
                "2026-08-01T22:30:00-04:00",
            ],
            "actor_id": [
                "SUP-001",
            ],
            "action_type": [
                "CHANGE_STATUS",
            ],
            "field_name": [
                "investigation_status",
            ],
            "previous_value": [
                "CONFIRMADO",
            ],
            "new_value": [
                "CERRADO",
            ],
            "comment": [
                "Cierre aprobado.",
            ],
        }
    )


def build_feedback_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": [
                "CL-001",
            ],
            "resolution_id": [
                "RES-000001",
            ],
            "investigation_outcome": [
                "CONFIRMADO",
            ],
            "outcome_label": [
                1,
            ],
            "closed_at": [
                "2026-08-01T22:30:00-04:00",
            ],
            "supervisor_id": [
                "SUP-001",
            ],
            "confirmed_loss_clp": [
                10_000_000.0,
            ],
            "recovered_amount_clp": [
                2_000_000.0,
            ],
            "feedback_source": [
                "SUPERVISOR_APPROVED_RESOLUTION",
            ],
        }
    )


def reopen_demo_case():
    return reopen_closed_case(
        build_cases_dataframe(),
        build_resolutions_dataframe(),
        build_audit_dataframe(),
        build_feedback_dataframe(),
        case_id="CL-001",
        supervisor_id="SUP-001",
        reopening_reason=(
            "Se recibió nueva evidencia que puede "
            "modificar la conclusión anterior."
        ),
        target_status="EN_INVESTIGACION",
        reopened_at=(
            "2026-08-02T09:00:00-04:00"
        ),
    )


def test_closed_case_is_reopened() -> None:
    result = reopen_demo_case()

    case = result.cases.loc[
        result.cases["case_id"]
        .eq("CL-001")
    ].iloc[0]

    assert (
        case["investigation_status"]
        == "EN_INVESTIGACION"
    )

    assert (
        case["investigation_closed_at"]
        == ""
    )

    assert (
        case["resolution"]
        == ""
    )


def test_previous_resolution_is_superseded() -> None:
    result = reopen_demo_case()

    resolution = (
        result.resolutions.iloc[0]
    )

    assert (
        resolution[
            "resolution_lifecycle_status"
        ]
        == "SUPERSEDIDA_POR_REAPERTURA"
    )

    assert (
        resolution["superseded_by"]
        == "SUP-001"
    )


def test_feedback_is_invalidated() -> None:
    result = reopen_demo_case()

    feedback = result.feedback.iloc[0]

    assert (
        feedback["feedback_status"]
        == "INVALIDADO"
    )

    assert (
        feedback["invalidated_by"]
        == "SUP-001"
    )


def test_three_audit_events_are_created() -> None:
    result = reopen_demo_case()

    new_events = result.audit.tail(
        3
    )

    assert set(
        new_events["action_type"]
    ) == {
        "REOPEN_CASE",
        "INVALIDATE_FEEDBACK",
        "CHANGE_STATUS",
    }


def test_open_case_cannot_be_reopened() -> None:
    with pytest.raises(
        ValueError,
        match="CERRADO",
    ):
        reopen_closed_case(
            build_cases_dataframe(),
            build_resolutions_dataframe(),
            build_audit_dataframe(),
            build_feedback_dataframe(),
            case_id="CL-002",
            supervisor_id="SUP-002",
            reopening_reason=(
                "Existe nueva evidencia suficiente "
                "para revisar el caso."
            ),
        )


def test_wrong_supervisor_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="supervisor asignado",
    ):
        reopen_closed_case(
            build_cases_dataframe(),
            build_resolutions_dataframe(),
            build_audit_dataframe(),
            build_feedback_dataframe(),
            case_id="CL-001",
            supervisor_id="SUP-999",
            reopening_reason=(
                "Existe nueva evidencia suficiente "
                "para revisar el caso."
            ),
        )


def test_short_reason_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="20 caracteres",
    ):
        reopen_closed_case(
            build_cases_dataframe(),
            build_resolutions_dataframe(),
            build_audit_dataframe(),
            build_feedback_dataframe(),
            case_id="CL-001",
            supervisor_id="SUP-001",
            reopening_reason="Nueva evidencia.",
        )


def test_invalid_target_status_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="Estado de reapertura inválido",
    ):
        reopen_closed_case(
            build_cases_dataframe(),
            build_resolutions_dataframe(),
            build_audit_dataframe(),
            build_feedback_dataframe(),
            case_id="CL-001",
            supervisor_id="SUP-001",
            reopening_reason=(
                "Existe nueva evidencia suficiente "
                "para revisar el caso."
            ),
            target_status="CERRADO",
        )


def test_missing_active_feedback_is_rejected() -> None:
    feedback = build_feedback_dataframe()
    feedback["feedback_status"] = (
        "INVALIDADO"
    )

    with pytest.raises(
        ValueError,
        match="feedback activo",
    ):
        reopen_closed_case(
            build_cases_dataframe(),
            build_resolutions_dataframe(),
            build_audit_dataframe(),
            feedback,
            case_id="CL-001",
            supervisor_id="SUP-001",
            reopening_reason=(
                "Existe nueva evidencia suficiente "
                "para revisar el caso."
            ),
        )


def test_invalid_timestamp_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="reopened_at es inválido",
    ):
        reopen_closed_case(
            build_cases_dataframe(),
            build_resolutions_dataframe(),
            build_audit_dataframe(),
            build_feedback_dataframe(),
            case_id="CL-001",
            supervisor_id="SUP-001",
            reopening_reason=(
                "Existe nueva evidencia suficiente "
                "para revisar el caso."
            ),
            reopened_at="fecha-invalida",
        )