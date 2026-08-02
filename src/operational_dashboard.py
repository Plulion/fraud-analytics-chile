from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd


CHILE_TIME_ZONE = ZoneInfo(
    "America/Santiago"
)


REQUIRED_CASE_COLUMNS = {
    "case_id",
    "investigation_status",
    "assigned_analyst_id",
    "supervisor_id",
    "investigation_created_at",
    "investigation_updated_at",
    "investigation_closed_at",
}


REQUIRED_AUDIT_COLUMNS = {
    "case_id",
    "event_timestamp",
    "actor_id",
    "action_type",
}


REQUIRED_PRIORITY_COLUMNS = {
    "case_id",
    "investigation_priority",
    "priority_score",
    "queue_position",
    "queue_status",
}


OPEN_STATUSES = {
    "NUEVO",
    "ASIGNADO",
    "EN_INVESTIGACION",
    "REQUIERE_ANTECEDENTES",
    "ESCALADO",
    "CONFIRMADO",
    "DESCARTADO",
}


CLOSED_STATUSES = {
    "CERRADO",
}


PRIORITY_SLA_HOURS = {
    "CRITICA": 2.0,
    "ALTA": 8.0,
    "MEDIA": 24.0,
    "BAJA": 72.0,
}


def normalize_text(
    value: Any,
) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def normalize_upper_text(
    value: Any,
) -> str:
    return normalize_text(
        value
    ).upper()


def parse_datetime_series(
    series: pd.Series,
) -> pd.Series:
    """
    Convierte fechas ISO 8601 en timestamps.

    utc=True permite manejar correctamente fechas
    con distintas zonas horarias.
    """

    return pd.to_datetime(
        series,
        errors="coerce",
        utc=True,
    )


def parse_single_datetime(
    value: str | datetime | pd.Timestamp,
) -> pd.Timestamp:
    parsed = pd.to_datetime(
        value,
        errors="coerce",
        utc=True,
    )

    if pd.isna(parsed):
        raise ValueError(
            f"Fecha inválida: {value!r}"
        )

    return parsed


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: set[str],
    dataset_name: str,
) -> None:
    missing_columns = (
        required_columns.difference(
            df.columns
        )
    )

    if missing_columns:
        raise ValueError(
            f"{dataset_name} no contiene las "
            "columnas requeridas: "
            f"{sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError(
            f"{dataset_name} está vacío."
        )


def validate_unique_case_ids(
    cases_df: pd.DataFrame,
) -> None:
    if cases_df[
        "case_id"
    ].isna().any():
        raise ValueError(
            "Existen casos sin case_id."
        )

    normalized_case_ids = (
        cases_df["case_id"]
        .astype(str)
        .str.strip()
    )

    if normalized_case_ids.eq(
        ""
    ).any():
        raise ValueError(
            "Existen case_id vacíos."
        )

    if normalized_case_ids.duplicated().any():
        duplicated = sorted(
            normalized_case_ids.loc[
                normalized_case_ids.duplicated(
                    keep=False
                )
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Existen case_id duplicados: "
            f"{duplicated}"
        )


def validate_dashboard_inputs(
    cases_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    priorities_df: pd.DataFrame,
) -> None:
    validate_required_columns(
        cases_df,
        REQUIRED_CASE_COLUMNS,
        "La tabla de casos",
    )

    validate_required_columns(
        audit_df,
        REQUIRED_AUDIT_COLUMNS,
        "La tabla de auditoría",
    )

    validate_required_columns(
        priorities_df,
        REQUIRED_PRIORITY_COLUMNS,
        "La tabla de prioridades",
    )

    validate_unique_case_ids(
        cases_df
    )

    priority_case_ids = (
        priorities_df["case_id"]
        .astype(str)
        .str.strip()
    )

    if priority_case_ids.duplicated().any():
        duplicated = sorted(
            priority_case_ids.loc[
                priority_case_ids.duplicated(
                    keep=False
                )
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "La tabla de prioridades contiene "
            "casos duplicados: "
            f"{duplicated}"
        )

    numeric_priority_score = pd.to_numeric(
        priorities_df[
            "priority_score"
        ],
        errors="coerce",
    )

    if numeric_priority_score.isna().any():
        raise ValueError(
            "priority_score contiene valores "
            "no numéricos."
        )

    numeric_queue_position = pd.to_numeric(
        priorities_df[
            "queue_position"
        ],
        errors="coerce",
    )

    if numeric_queue_position.isna().any():
        raise ValueError(
            "queue_position contiene valores "
            "no numéricos."
        )

    for priority in priorities_df[
        "investigation_priority"
    ]:
        normalized = normalize_upper_text(
            priority
        )

        if normalized not in (
            PRIORITY_SLA_HOURS
        ):
            raise ValueError(
                "Prioridad desconocida: "
                f"{priority!r}"
            )


def find_first_audit_timestamp(
    audit_df: pd.DataFrame,
    *,
    case_id: str,
    action_types: set[str],
) -> pd.Timestamp | pd.NaT:
    normalized_actions = {
        normalize_upper_text(
            action
        )
        for action in action_types
    }

    case_events = audit_df.loc[
        audit_df["case_id"]
        .astype(str)
        .str.strip()
        == normalize_text(
            case_id
        )
    ].copy()

    if case_events.empty:
        return pd.NaT

    case_events[
        "_action_normalized"
    ] = (
        case_events[
            "action_type"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    matching_events = case_events.loc[
        case_events[
            "_action_normalized"
        ].isin(
            normalized_actions
        )
    ].copy()

    if matching_events.empty:
        return pd.NaT

    timestamps = parse_datetime_series(
        matching_events[
            "event_timestamp"
        ]
    ).dropna()

    if timestamps.empty:
        return pd.NaT

    return timestamps.min()



def find_investigation_start_timestamp(
    audit_df: pd.DataFrame,
    *,
    case_id: str,
    current_status: str,
) -> tuple[pd.Timestamp | pd.NaT, str]:
    """
    Busca el primer evento que realmente represente el inicio
    de la investigación.

    Compatibilidad:
    - acciones explícitas de inicio;
    - CHANGE_STATUS con una columna de estado destino;
    - CHANGE_STATUS cuyo detalle menciona EN_INVESTIGACION;
    - inferencia controlada cuando existe un único cambio de estado
      para un caso cuyo estado actual es EN_INVESTIGACION.
    """

    explicit_timestamp = find_first_audit_timestamp(
        audit_df,
        case_id=case_id,
        action_types={
            "INVESTIGATION_STARTED",
            "STATUS_CHANGED_TO_EN_INVESTIGACION",
            "EN_INVESTIGACION",
        },
    )

    if not pd.isna(explicit_timestamp):
        return explicit_timestamp, "EVENTO_EXPLICITO"

    case_events = audit_df.loc[
        audit_df["case_id"]
        .fillna("")
        .astype(str)
        .str.strip()
        == normalize_text(case_id)
    ].copy()

    if case_events.empty:
        return pd.NaT, "SIN_EVENTO_IDENTIFICABLE"

    case_events["_action_normalized"] = (
        case_events["action_type"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    status_events = case_events.loc[
        case_events["_action_normalized"]
        == "CHANGE_STATUS"
    ].copy()

    if status_events.empty:
        return pd.NaT, "SIN_EVENTO_IDENTIFICABLE"

    destination_columns = [
        "new_status",
        "to_status",
        "target_status",
        "status_after",
        "current_status",
        "investigation_status",
        "new_value",
    ]

    for column in destination_columns:
        if column not in status_events.columns:
            continue

        destination_mask = (
            status_events[column]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
            .eq("EN_INVESTIGACION")
        )

        matching = status_events.loc[destination_mask]

        if not matching.empty:
            timestamps = parse_datetime_series(
                matching["event_timestamp"]
            ).dropna()

            if not timestamps.empty:
                return (
                    timestamps.min(),
                    f"CHANGE_STATUS:{column}",
                )

    descriptive_columns = [
        "details",
        "detail",
        "comment",
        "description",
        "metadata",
        "message",
    ]

    for column in descriptive_columns:
        if column not in status_events.columns:
            continue

        detail_mask = (
            status_events[column]
            .fillna("")
            .astype(str)
            .str.upper()
            .str.contains(
                r"EN[_ ]INVESTIGACION",
                regex=True,
                na=False,
            )
        )

        matching = status_events.loc[detail_mask]

        if not matching.empty:
            timestamps = parse_datetime_series(
                matching["event_timestamp"]
            ).dropna()

            if not timestamps.empty:
                return (
                    timestamps.min(),
                    f"CHANGE_STATUS:{column}",
                )

    normalized_current_status = normalize_upper_text(
        current_status
    )

    # Fallback conservador para el esquema actual del proyecto:
    # solo se infiere cuando hay un único CHANGE_STATUS del caso
    # y el caso actualmente está en investigación.
    if (
        normalized_current_status == "EN_INVESTIGACION"
        and len(status_events) == 1
    ):
        timestamps = parse_datetime_series(
            status_events["event_timestamp"]
        ).dropna()

        if not timestamps.empty:
            return (
                timestamps.min(),
                "CHANGE_STATUS_INFERIDO_UNICO",
            )

    return pd.NaT, "SIN_EVENTO_IDENTIFICABLE"

def hours_between(
    start: pd.Timestamp | pd.NaT,
    end: pd.Timestamp | pd.NaT,
) -> float | None:
    """
    Calcula la diferencia entre dos marcas de tiempo.

    Una diferencia negativa se considera una inconsistencia
    de datos y no se corrige silenciosamente.
    """

    if pd.isna(start) or pd.isna(end):
        return None

    difference = (
        end - start
    )

    hours = (
        difference.total_seconds()
        / 3600
    )

    if hours < 0:
        raise ValueError(
            "Se detectó una diferencia temporal "
            "negativa. La fecha final es anterior "
            "a la fecha inicial. "
            f"Inicio: {start.isoformat()}. "
            f"Fin: {end.isoformat()}."
        )

    return round(
        hours,
        2,
    )


def determine_sla_status(
    *,
    hours_to_start: float | None,
    case_age_hours: float,
    sla_hours: float,
    investigation_started: bool,
    case_closed: bool,
) -> str:
    """
    Determina el cumplimiento del SLA de inicio.

    El SLA mide cuánto demoró el caso en comenzar
    a investigarse desde su creación.
    """

    if investigation_started:
        if hours_to_start is None:
            return "SIN_DATOS"

        if hours_to_start <= sla_hours:
            return "CUMPLIDO"

        return "INCUMPLIDO"

    if case_closed:
        return "SIN_DATOS"

    if case_age_hours > sla_hours:
        return "VENCIDO"

    return "EN_PLAZO"


def build_operational_dashboard(
    cases_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    priorities_df: pd.DataFrame,
    *,
    reference_timestamp: str | datetime | pd.Timestamp,
) -> pd.DataFrame:
    """
    Construye una vista operativa de los casos.

    La función no modifica las tablas originales.
    """

    validate_dashboard_inputs(
        cases_df,
        audit_df,
        priorities_df,
    )

    reference_time = parse_single_datetime(
        reference_timestamp
    )

    cases = cases_df.copy()
    priorities = priorities_df.copy()

    cases["case_id"] = (
        cases["case_id"]
        .astype(str)
        .str.strip()
    )

    priorities["case_id"] = (
        priorities["case_id"]
        .astype(str)
        .str.strip()
    )

    dashboard_df = cases.merge(
        priorities[
            [
                "case_id",
                "investigation_priority",
                "priority_score",
                "queue_position",
                "queue_status",
            ]
        ],
        on="case_id",
        how="left",
        validate="one_to_one",
    )

    dashboard_df[
        "investigation_priority"
    ] = (
        dashboard_df[
            "investigation_priority"
        ]
        .fillna("BAJA")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    dashboard_df[
        "priority_score"
    ] = pd.to_numeric(
        dashboard_df[
            "priority_score"
        ],
        errors="coerce",
    ).fillna(0)

    dashboard_df[
        "queue_position"
    ] = pd.to_numeric(
        dashboard_df[
            "queue_position"
        ],
        errors="coerce",
    ).fillna(0).astype(int)

    dashboard_df[
        "queue_status"
    ] = (
        dashboard_df[
            "queue_status"
        ]
        .fillna("NO_PRIORIZADA")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    created_timestamps = (
        parse_datetime_series(
            dashboard_df[
                "investigation_created_at"
            ]
        )
    )

    updated_timestamps = (
        parse_datetime_series(
            dashboard_df[
                "investigation_updated_at"
            ]
        )
    )

    closed_timestamps = (
        parse_datetime_series(
            dashboard_df[
                "investigation_closed_at"
            ]
        )
    )

    output_rows: list[
        dict[str, Any]
    ] = []

    for index, row in (
        dashboard_df.iterrows()
    ):
        case_id = normalize_text(
            row["case_id"]
        )

        status = normalize_upper_text(
            row[
                "investigation_status"
            ]
        )

        priority = normalize_upper_text(
            row[
                "investigation_priority"
            ]
        )

        created_at = (
            created_timestamps.loc[
                index
            ]
        )

        updated_at = (
            updated_timestamps.loc[
                index
            ]
        )

        closed_at = (
            closed_timestamps.loc[
                index
            ]
        )

        if pd.isna(created_at):
            raise ValueError(
                "Fecha de creación inválida "
                f"para el caso {case_id}."
            )

        assigned_at = (
            find_first_audit_timestamp(
                audit_df,
                case_id=case_id,
                action_types={
                    "ASSIGNED",
                    "CASE_ASSIGNED",
                    "ASSIGN_CASE",
                    "ASIGNADO",
                },
            )
        )

        (
            investigation_started_at,
            investigation_start_source,
        ) = find_investigation_start_timestamp(
            audit_df,
            case_id=case_id,
            current_status=status,
        )

        case_closed = (
            status in CLOSED_STATUSES
            or not pd.isna(
                closed_at
            )
        )

        operational_end = (
            closed_at
            if case_closed
            and not pd.isna(
                closed_at
            )
            else reference_time
        )

        case_age_hours = (
            hours_between(
                created_at,
                operational_end,
            )
        )

        if case_age_hours is None:
            case_age_hours = 0.0

        hours_to_assignment = (
            hours_between(
                created_at,
                assigned_at,
            )
        )

        hours_to_start = (
            hours_between(
                created_at,
                investigation_started_at,
            )
        )

        hours_to_close = (
            hours_between(
                created_at,
                closed_at,
            )
        )

        sla_hours = (
            PRIORITY_SLA_HOURS[
                priority
            ]
        )

        sla_status = determine_sla_status(
            hours_to_start=hours_to_start,
            case_age_hours=case_age_hours,
            sla_hours=sla_hours,
            investigation_started=(
                not pd.isna(
                    investigation_started_at
                )
            ),
            case_closed=case_closed,
        )

        assigned_analyst = (
            normalize_text(
                row[
                    "assigned_analyst_id"
                ]
            )
            or "SIN_ASIGNAR"
        )

        supervisor = (
            normalize_text(
                row[
                    "supervisor_id"
                ]
            )
            or "SIN_SUPERVISOR"
        )

        output_rows.append(
            {
                "case_id": case_id,
                "investigation_status": (
                    status
                ),
                "assigned_analyst_id": (
                    assigned_analyst
                ),
                "supervisor_id": supervisor,
                "investigation_priority": (
                    priority
                ),
                "priority_score": float(
                    row[
                        "priority_score"
                    ]
                ),
                "queue_position": int(
                    row[
                        "queue_position"
                    ]
                ),
                "queue_status": (
                    normalize_upper_text(
                        row[
                            "queue_status"
                        ]
                    )
                ),
                "investigation_created_at": (
                    created_at.isoformat()
                ),
                "investigation_updated_at": (
                    updated_at.isoformat()
                    if not pd.isna(
                        updated_at
                    )
                    else ""
                ),
                "assigned_at": (
                    assigned_at.isoformat()
                    if not pd.isna(
                        assigned_at
                    )
                    else ""
                ),
                "investigation_started_at": (
                    investigation_started_at
                    .isoformat()
                    if not pd.isna(
                        investigation_started_at
                    )
                    else ""
                ),
                "investigation_start_source": (
                    investigation_start_source
                ),
                "investigation_closed_at": (
                    closed_at.isoformat()
                    if not pd.isna(
                        closed_at
                    )
                    else ""
                ),
                "hours_to_assignment": (
                    hours_to_assignment
                ),
                "hours_to_start": (
                    hours_to_start
                ),
                "hours_to_close": (
                    hours_to_close
                ),
                "case_age_hours": (
                    case_age_hours
                ),
                "sla_target_hours": (
                    sla_hours
                ),
                "sla_status": (
                    sla_status
                ),
                "is_open": (
                    status in OPEN_STATUSES
                    and not case_closed
                ),
                "is_closed": (
                    case_closed
                ),
            }
        )

    result_df = pd.DataFrame(
        output_rows
    )

    priority_rank = {
        "CRITICA": 4,
        "ALTA": 3,
        "MEDIA": 2,
        "BAJA": 1,
    }

    sla_rank = {
        "VENCIDO": 5,
        "INCUMPLIDO": 4,
        "SIN_DATOS": 3,
        "EN_PLAZO": 2,
        "CUMPLIDO": 1,
    }

    result_df[
        "_priority_rank"
    ] = (
        result_df[
            "investigation_priority"
        ]
        .map(
            priority_rank
        )
        .fillna(0)
    )

    result_df[
        "_sla_rank"
    ] = (
        result_df[
            "sla_status"
        ]
        .map(
            sla_rank
        )
        .fillna(0)
    )

    result_df = result_df.sort_values(
        by=[
            "is_open",
            "_sla_rank",
            "_priority_rank",
            "priority_score",
            "case_age_hours",
            "case_id",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            False,
            True,
        ],
    ).drop(
        columns=[
            "_priority_rank",
            "_sla_rank",
        ]
    ).reset_index(
        drop=True
    )

    return result_df


def build_analyst_workload(
    dashboard_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Resume la carga operacional por analista.
    """

    required_columns = {
        "assigned_analyst_id",
        "investigation_status",
        "is_open",
        "is_closed",
        "sla_status",
        "case_age_hours",
        "hours_to_close",
    }

    validate_required_columns(
        dashboard_df,
        required_columns,
        "El panel operativo",
    )

    rows: list[
        dict[str, Any]
    ] = []

    for analyst_id, group in (
        dashboard_df.groupby(
            "assigned_analyst_id",
            dropna=False,
        )
    ):
        normalized_analyst = (
            normalize_text(
                analyst_id
            )
            or "SIN_ASIGNAR"
        )

        status_counts = (
            group[
                "investigation_status"
            ]
            .value_counts()
            .to_dict()
        )

        open_cases = int(
            group[
                "is_open"
            ].sum()
        )

        closed_cases = int(
            group[
                "is_closed"
            ].sum()
        )

        overdue_cases = int(
            group[
                "sla_status"
            ]
            .isin(
                {
                    "VENCIDO",
                    "INCUMPLIDO",
                }
            )
            .sum()
        )

        average_open_age = (
            group.loc[
                group["is_open"],
                "case_age_hours",
            ]
            .dropna()
            .mean()
        )

        average_close_time = (
            group.loc[
                group["is_closed"],
                "hours_to_close",
            ]
            .dropna()
            .mean()
        )

        rows.append(
            {
                "assigned_analyst_id": (
                    normalized_analyst
                ),
                "total_cases": int(
                    len(group)
                ),
                "open_cases": open_cases,
                "closed_cases": closed_cases,
                "new_cases": int(
                    status_counts.get(
                        "NUEVO",
                        0,
                    )
                ),
                "assigned_cases": int(
                    status_counts.get(
                        "ASIGNADO",
                        0,
                    )
                ),
                "in_investigation_cases": int(
                    status_counts.get(
                        "EN_INVESTIGACION",
                        0,
                    )
                ),
                "requires_information_cases": int(
                    status_counts.get(
                        "REQUIERE_ANTECEDENTES",
                        0,
                    )
                ),
                "escalated_cases": int(
                    status_counts.get(
                        "ESCALADO",
                        0,
                    )
                ),
                "overdue_cases": (
                    overdue_cases
                ),
                "average_open_age_hours": (
                    round(
                        float(
                            average_open_age
                        ),
                        2,
                    )
                    if not pd.isna(
                        average_open_age
                    )
                    else 0.0
                ),
                "average_close_time_hours": (
                    round(
                        float(
                            average_close_time
                        ),
                        2,
                    )
                    if not pd.isna(
                        average_close_time
                    )
                    else 0.0
                ),
            }
        )

    workload_df = pd.DataFrame(
        rows
    )

    workload_df = workload_df.sort_values(
        by=[
            "open_cases",
            "overdue_cases",
            "total_cases",
            "assigned_analyst_id",
        ],
        ascending=[
            False,
            False,
            False,
            True,
        ],
    ).reset_index(
        drop=True
    )

    return workload_df


def safe_divide(
    numerator: int | float,
    denominator: int | float,
) -> float:
    if denominator == 0:
        return 0.0

    return float(
        numerator / denominator
    )


def calculate_operational_kpis(
    dashboard_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Calcula KPI globales del proceso investigativo.
    """

    required_columns = {
        "case_id",
        "assigned_analyst_id",
        "is_open",
        "is_closed",
        "sla_status",
        "hours_to_assignment",
        "hours_to_start",
        "hours_to_close",
        "case_age_hours",
    }

    validate_required_columns(
        dashboard_df,
        required_columns,
        "El panel operativo",
    )

    total_cases = int(
        len(dashboard_df)
    )

    open_cases = int(
        dashboard_df[
            "is_open"
        ].sum()
    )

    closed_cases = int(
        dashboard_df[
            "is_closed"
        ].sum()
    )

    unassigned_cases = int(
        dashboard_df[
            "assigned_analyst_id"
        ]
        .eq("SIN_ASIGNAR")
        .sum()
    )

    overdue_cases = int(
        dashboard_df[
            "sla_status"
        ]
        .isin(
            {
                "VENCIDO",
                "INCUMPLIDO",
            }
        )
        .sum()
    )

    measurable_sla_cases = (
        dashboard_df[
            "sla_status"
        ].isin(
            {
                "CUMPLIDO",
                "INCUMPLIDO",
            }
        )
    )

    completed_within_sla = int(
        dashboard_df[
            "sla_status"
        ]
        .eq("CUMPLIDO")
        .sum()
    )

    measurable_sla_count = int(
        measurable_sla_cases.sum()
    )

    average_assignment_hours = (
        dashboard_df[
            "hours_to_assignment"
        ]
        .dropna()
        .mean()
    )

    average_start_hours = (
        dashboard_df[
            "hours_to_start"
        ]
        .dropna()
        .mean()
    )

    average_close_hours = (
        dashboard_df[
            "hours_to_close"
        ]
        .dropna()
        .mean()
    )

    average_open_age_hours = (
        dashboard_df.loc[
            dashboard_df[
                "is_open"
            ],
            "case_age_hours",
        ]
        .dropna()
        .mean()
    )

    return {
        "total_cases": total_cases,
        "open_cases": open_cases,
        "closed_cases": closed_cases,
        "unassigned_cases": (
            unassigned_cases
        ),
        "overdue_or_breached_cases": (
            overdue_cases
        ),
        "closure_rate": round(
            safe_divide(
                closed_cases,
                total_cases,
            ),
            4,
        ),
        "sla_compliance_rate": round(
            safe_divide(
                completed_within_sla,
                measurable_sla_count,
            ),
            4,
        ),
        "measurable_sla_cases": (
            measurable_sla_count
        ),
        "average_hours_to_assignment": (
            round(
                float(
                    average_assignment_hours
                ),
                2,
            )
            if not pd.isna(
                average_assignment_hours
            )
            else 0.0
        ),
        "average_hours_to_start": (
            round(
                float(
                    average_start_hours
                ),
                2,
            )
            if not pd.isna(
                average_start_hours
            )
            else 0.0
        ),
        "average_hours_to_close": (
            round(
                float(
                    average_close_hours
                ),
                2,
            )
            if not pd.isna(
                average_close_hours
            )
            else 0.0
        ),
        "average_open_case_age_hours": (
            round(
                float(
                    average_open_age_hours
                ),
                2,
            )
            if not pd.isna(
                average_open_age_hours
            )
            else 0.0
        ),
        "analyst_count": int(
            dashboard_df.loc[
                dashboard_df[
                    "assigned_analyst_id"
                ]
                != "SIN_ASIGNAR",
                "assigned_analyst_id",
            ].nunique()
        ),
    }