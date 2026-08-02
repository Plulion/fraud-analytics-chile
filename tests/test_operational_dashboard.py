import pandas as pd
import pytest

from src.operational_dashboard import (
    build_analyst_workload,
    build_operational_dashboard,
    calculate_operational_kpis,
    determine_sla_status,
)


REFERENCE_TIMESTAMP = (
    "2026-08-02T12:00:00-04:00"
)


def build_cases_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": [
                "CL-001",
                "CL-002",
                "CL-003",
                "CL-004",
            ],
            "investigation_status": [
                "EN_INVESTIGACION",
                "ASIGNADO",
                "CERRADO",
                "NUEVO",
            ],
            "assigned_analyst_id": [
                "ANA-001",
                "ANA-001",
                "ANA-002",
                "",
            ],
            "supervisor_id": [
                "SUP-001",
                "SUP-001",
                "SUP-002",
                "",
            ],
            "investigation_created_at": [
                "2026-08-01T08:00:00-04:00",
                "2026-08-01T08:00:00-04:00",
                "2026-08-01T08:00:00-04:00",
                "2026-08-01T08:00:00-04:00",
            ],
            "investigation_updated_at": [
                "2026-08-01T10:00:00-04:00",
                "2026-08-01T09:00:00-04:00",
                "2026-08-01T18:00:00-04:00",
                "2026-08-01T08:00:00-04:00",
            ],
            "investigation_closed_at": [
                "",
                "",
                "2026-08-01T18:00:00-04:00",
                "",
            ],
        }
    )


def build_audit_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": [
                "CL-001",
                "CL-001",
                "CL-002",
                "CL-003",
                "CL-003",
            ],
            "event_timestamp": [
                "2026-08-01T08:30:00-04:00",
                "2026-08-01T09:00:00-04:00",
                "2026-08-01T09:00:00-04:00",
                "2026-08-01T08:15:00-04:00",
                "2026-08-01T09:00:00-04:00",
            ],
            "actor_id": [
                "SUP-001",
                "ANA-001",
                "SUP-001",
                "SUP-002",
                "ANA-002",
            ],
            "action_type": [
                "ASSIGNED",
                "INVESTIGATION_STARTED",
                "ASSIGNED",
                "ASSIGNED",
                "INVESTIGATION_STARTED",
            ],
        }
    )


def build_priorities_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": [
                "CL-001",
                "CL-002",
                "CL-003",
                "CL-004",
            ],
            "investigation_priority": [
                "CRITICA",
                "ALTA",
                "MEDIA",
                "BAJA",
            ],
            "priority_score": [
                80,
                65,
                45,
                20,
            ],
            "queue_position": [
                1,
                2,
                3,
                4,
            ],
            "queue_status": [
                "SELECCIONADA",
                "SELECCIONADA",
                "EN_ESPERA",
                "EN_ESPERA",
            ],
        }
    )


def build_dashboard() -> pd.DataFrame:
    return build_operational_dashboard(
        build_cases_dataframe(),
        build_audit_dataframe(),
        build_priorities_dataframe(),
        reference_timestamp=(
            REFERENCE_TIMESTAMP
        ),
    )


def test_determine_sla_status() -> None:
    assert determine_sla_status(
        hours_to_start=1.0,
        case_age_hours=10.0,
        sla_hours=2.0,
        investigation_started=True,
        case_closed=False,
    ) == "CUMPLIDO"

    assert determine_sla_status(
        hours_to_start=4.0,
        case_age_hours=10.0,
        sla_hours=2.0,
        investigation_started=True,
        case_closed=False,
    ) == "INCUMPLIDO"

    assert determine_sla_status(
        hours_to_start=None,
        case_age_hours=10.0,
        sla_hours=8.0,
        investigation_started=False,
        case_closed=False,
    ) == "VENCIDO"

    assert determine_sla_status(
        hours_to_start=None,
        case_age_hours=4.0,
        sla_hours=8.0,
        investigation_started=False,
        case_closed=False,
    ) == "EN_PLAZO"


def test_build_operational_dashboard() -> None:
    result = build_dashboard()

    assert len(result) == 4

    assert set(
        result["case_id"]
    ) == {
        "CL-001",
        "CL-002",
        "CL-003",
        "CL-004",
    }


def test_critical_case_meets_start_sla() -> None:
    result = build_dashboard()

    case = result.loc[
        result["case_id"]
        == "CL-001"
    ].iloc[0]

    assert (
        case[
            "investigation_priority"
        ]
        == "CRITICA"
    )

    assert (
        case["hours_to_start"]
        == 1.0
    )

    assert (
        case["sla_target_hours"]
        == 2.0
    )

    assert (
        case["sla_status"]
        == "CUMPLIDO"
    )


def test_assigned_case_without_start_is_overdue() -> None:
    result = build_dashboard()

    case = result.loc[
        result["case_id"]
        == "CL-002"
    ].iloc[0]

    assert (
        case["investigation_status"]
        == "ASIGNADO"
    )

    assert pd.isna(
        case["hours_to_start"]
    )

    assert (
        case["sla_status"]
        == "VENCIDO"
    )


def test_closed_case_has_close_time() -> None:
    result = build_dashboard()

    case = result.loc[
        result["case_id"]
        == "CL-003"
    ].iloc[0]

    assert bool(
        case["is_closed"]
    )

    assert (
        case["hours_to_close"]
        == 10.0
    )


def test_unassigned_case_is_normalized() -> None:
    result = build_dashboard()

    case = result.loc[
        result["case_id"]
        == "CL-004"
    ].iloc[0]

    assert (
        case["assigned_analyst_id"]
        == "SIN_ASIGNAR"
    )


def test_build_analyst_workload() -> None:
    dashboard_df = build_dashboard()

    workload_df = build_analyst_workload(
        dashboard_df
    )

    analyst = workload_df.loc[
        workload_df[
            "assigned_analyst_id"
        ]
        == "ANA-001"
    ].iloc[0]

    assert (
        analyst["total_cases"]
        == 2
    )

    assert (
        analyst["open_cases"]
        == 2
    )

    assert (
        analyst["overdue_cases"]
        == 1
    )


def test_calculate_operational_kpis() -> None:
    dashboard_df = build_dashboard()

    kpis = calculate_operational_kpis(
        dashboard_df
    )

    assert kpis[
        "total_cases"
    ] == 4

    assert kpis[
        "open_cases"
    ] == 3

    assert kpis[
        "closed_cases"
    ] == 1

    assert kpis[
        "unassigned_cases"
    ] == 1

    assert kpis[
        "overdue_or_breached_cases"
    ] == 1


def test_unknown_priority_is_rejected() -> None:
    priorities_df = (
        build_priorities_dataframe()
    )

    priorities_df.loc[
        0,
        "investigation_priority",
    ] = "URGENTISIMA"

    with pytest.raises(
        ValueError,
        match="Prioridad desconocida",
    ):
        build_operational_dashboard(
            build_cases_dataframe(),
            build_audit_dataframe(),
            priorities_df,
            reference_timestamp=(
                REFERENCE_TIMESTAMP
            ),
        )


def test_duplicate_case_is_rejected() -> None:
    cases_df = build_cases_dataframe()

    duplicated_row = (
        cases_df.iloc[
            [0]
        ].copy()
    )

    cases_df = pd.concat(
        [
            cases_df,
            duplicated_row,
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="case_id duplicados",
    ):
        build_operational_dashboard(
            cases_df,
            build_audit_dataframe(),
            build_priorities_dataframe(),
            reference_timestamp=(
                REFERENCE_TIMESTAMP
            ),
        )