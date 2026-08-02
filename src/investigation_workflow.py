from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd


CHILE_TIME_ZONE = ZoneInfo(
    "America/Santiago"
)


INVESTIGATION_STATES = {
    "NUEVO",
    "ASIGNADO",
    "EN_INVESTIGACION",
    "REQUIERE_ANTECEDENTES",
    "ESCALADO",
    "CONFIRMADO",
    "DESCARTADO",
    "CERRADO",
}


FINAL_RESULT_STATES = {
    "CONFIRMADO",
    "DESCARTADO",
}


ALLOWED_TRANSITIONS = {
    "NUEVO": {
        "ASIGNADO",
    },
    "ASIGNADO": {
        "EN_INVESTIGACION",
        "ESCALADO",
    },
    "EN_INVESTIGACION": {
        "REQUIERE_ANTECEDENTES",
        "ESCALADO",
        "CONFIRMADO",
        "DESCARTADO",
    },
    "REQUIERE_ANTECEDENTES": {
        "EN_INVESTIGACION",
        "ESCALADO",
    },
    "ESCALADO": {
        "EN_INVESTIGACION",
        "CONFIRMADO",
        "DESCARTADO",
    },
    "CONFIRMADO": {
        "CERRADO",
    },
    "DESCARTADO": {
        "CERRADO",
    },
    "CERRADO": set(),
}


REQUIRED_CASE_COLUMNS = {
    "case_id",
    "consolidated_alert_level",
    "consolidated_recommended_action",
    "estimated_loss_clp",
}


@dataclass(frozen=True)
class WorkflowResult:
    """
    Contiene las tablas actualizadas después
    de una operación de gestión.
    """

    cases: pd.DataFrame
    audit_log: pd.DataFrame


def current_chile_timestamp() -> str:
    """
    Devuelve una fecha ISO 8601 utilizando
    la zona horaria de Chile.
    """

    return datetime.now(
        CHILE_TIME_ZONE
    ).isoformat(
        timespec="seconds"
    )


def normalize_required_text(
    value: Any,
    field_name: str,
) -> str:
    """
    Normaliza un texto obligatorio.
    """

    if value is None:
        raise ValueError(
            f"El campo {field_name!r} "
            "no puede ser vacío."
        )

    try:
        if pd.isna(value):
            raise ValueError(
                f"El campo {field_name!r} "
                "no puede ser vacío."
            )
    except (TypeError, ValueError):
        pass

    normalized = str(value).strip()

    if not normalized:
        raise ValueError(
            f"El campo {field_name!r} "
            "no puede ser vacío."
        )

    return normalized


def normalize_state(
    state: Any,
) -> str:
    """
    Convierte un estado a mayúsculas y verifica
    que pertenezca al catálogo.
    """

    normalized = normalize_required_text(
        state,
        "investigation_status",
    ).upper()

    if normalized not in INVESTIGATION_STATES:
        raise ValueError(
            "Estado de investigación desconocido: "
            f"{state!r}"
        )

    return normalized


def validate_case_management_data(
    df: pd.DataFrame,
) -> None:
    """
    Valida la tabla de gestión de casos.
    """

    required_columns = {
        "case_id",
        "investigation_status",
        "assigned_analyst_id",
        "supervisor_id",
        "investigation_created_at",
        "investigation_updated_at",
        "investigation_closed_at",
        "resolution",
    }

    missing_columns = (
        required_columns.difference(
            df.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "La tabla de gestión no contiene "
            "las columnas requeridas: "
            f"{sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError(
            "La tabla de gestión de casos "
            "está vacía."
        )

    if df["case_id"].isna().any():
        raise ValueError(
            "Existen casos sin case_id."
        )

    if df["case_id"].duplicated().any():
        duplicated_cases = (
            df.loc[
                df["case_id"].duplicated(
                    keep=False
                ),
                "case_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            "Existen case_id duplicados: "
            f"{sorted(duplicated_cases)}"
        )

    for state in df[
        "investigation_status"
    ]:
        normalize_state(state)


def initialize_case_management(
    consolidated_df: pd.DataFrame,
    *,
    timestamp: str | None = None,
) -> pd.DataFrame:
    """
    Crea una tabla de gestión a partir de la vista
    consolidada.

    Todos los casos comienzan en estado NUEVO.
    """

    missing_columns = (
        REQUIRED_CASE_COLUMNS.difference(
            consolidated_df.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "La vista consolidada no contiene "
            "las columnas requeridas: "
            f"{sorted(missing_columns)}"
        )

    if consolidated_df.empty:
        raise ValueError(
            "La vista consolidada está vacía."
        )

    if consolidated_df[
        "case_id"
    ].duplicated().any():
        raise ValueError(
            "La vista consolidada contiene "
            "case_id duplicados."
        )

    created_at = (
        timestamp
        if timestamp is not None
        else current_chile_timestamp()
    )

    result_df = consolidated_df.copy()

    result_df[
        "investigation_status"
    ] = "NUEVO"

    result_df[
        "assigned_analyst_id"
    ] = ""

    result_df[
        "supervisor_id"
    ] = ""

    result_df[
        "investigation_created_at"
    ] = created_at

    result_df[
        "investigation_updated_at"
    ] = created_at

    result_df[
        "investigation_closed_at"
    ] = ""

    result_df[
        "resolution"
    ] = ""

    return result_df


def create_empty_audit_log() -> pd.DataFrame:
    """
    Crea una bitácora sin registros.
    """

    return pd.DataFrame(
        columns=[
            "audit_id",
            "case_id",
            "event_timestamp",
            "actor_id",
            "action_type",
            "field_name",
            "previous_value",
            "new_value",
            "comment",
        ]
    )


def next_audit_id(
    audit_log: pd.DataFrame,
) -> str:
    """
    Genera identificadores:

        AUD-000001
        AUD-000002
        ...
    """

    if audit_log.empty:
        return "AUD-000001"

    numeric_parts: list[int] = []

    for value in audit_log["audit_id"]:
        text = str(value).strip()

        if not text.startswith(
            "AUD-"
        ):
            raise ValueError(
                "Identificador de auditoría "
                f"inválido: {value!r}"
            )

        try:
            numeric_parts.append(
                int(
                    text.replace(
                        "AUD-",
                        "",
                        1,
                    )
                )
            )
        except ValueError as exc:
            raise ValueError(
                "Identificador de auditoría "
                f"inválido: {value!r}"
            ) from exc

    next_number = max(
        numeric_parts
    ) + 1

    return f"AUD-{next_number:06d}"


def append_audit_event(
    audit_log: pd.DataFrame,
    *,
    case_id: str,
    actor_id: str,
    action_type: str,
    field_name: str,
    previous_value: Any,
    new_value: Any,
    comment: str,
    timestamp: str,
) -> pd.DataFrame:
    """
    Agrega una fila a la bitácora.

    La función no modifica registros anteriores.
    """

    new_row = pd.DataFrame(
        [
            {
                "audit_id": next_audit_id(
                    audit_log
                ),
                "case_id": case_id,
                "event_timestamp": timestamp,
                "actor_id": actor_id,
                "action_type": action_type,
                "field_name": field_name,
                "previous_value": (
                    ""
                    if previous_value is None
                    else str(previous_value)
                ),
                "new_value": (
                    ""
                    if new_value is None
                    else str(new_value)
                ),
                "comment": comment,
            }
        ]
    )

    return pd.concat(
        [
            audit_log,
            new_row,
        ],
        ignore_index=True,
    )


def find_case_index(
    cases_df: pd.DataFrame,
    case_id: str,
) -> int:
    """
    Localiza una única fila mediante case_id.
    """

    normalized_case_id = (
        normalize_required_text(
            case_id,
            "case_id",
        )
    )

    matches = cases_df.index[
        cases_df["case_id"].astype(str)
        == normalized_case_id
    ].tolist()

    if not matches:
        raise ValueError(
            "No existe el caso: "
            f"{normalized_case_id}"
        )

    if len(matches) > 1:
        raise ValueError(
            "El case_id aparece más de una vez: "
            f"{normalized_case_id}"
        )

    return int(matches[0])


def assign_case(
    cases_df: pd.DataFrame,
    audit_log: pd.DataFrame,
    *,
    case_id: str,
    analyst_id: str,
    actor_id: str,
    supervisor_id: str = "",
    comment: str = "",
    timestamp: str | None = None,
) -> WorkflowResult:
    """
    Asigna un caso nuevo a un analista y cambia
    su estado de NUEVO a ASIGNADO.
    """

    validate_case_management_data(
        cases_df
    )

    index = find_case_index(
        cases_df,
        case_id,
    )

    analyst = normalize_required_text(
        analyst_id,
        "analyst_id",
    )

    actor = normalize_required_text(
        actor_id,
        "actor_id",
    )

    event_timestamp = (
        timestamp
        if timestamp is not None
        else current_chile_timestamp()
    )

    current_status = normalize_state(
        cases_df.loc[
            index,
            "investigation_status",
        ]
    )

    if current_status != "NUEVO":
        raise ValueError(
            "Solo se pueden asignar casos "
            "en estado NUEVO. "
            f"Estado actual: {current_status}"
        )

    result_df = cases_df.copy()

    previous_analyst = result_df.loc[
        index,
        "assigned_analyst_id",
    ]

    result_df.loc[
        index,
        "assigned_analyst_id",
    ] = analyst

    result_df.loc[
        index,
        "supervisor_id",
    ] = str(
        supervisor_id
    ).strip()

    result_df.loc[
        index,
        "investigation_status",
    ] = "ASIGNADO"

    result_df.loc[
        index,
        "investigation_updated_at",
    ] = event_timestamp

    updated_audit_log = append_audit_event(
        audit_log,
        case_id=case_id,
        actor_id=actor,
        action_type="ASSIGN_CASE",
        field_name="assigned_analyst_id",
        previous_value=previous_analyst,
        new_value=analyst,
        comment=comment,
        timestamp=event_timestamp,
    )

    updated_audit_log = append_audit_event(
        updated_audit_log,
        case_id=case_id,
        actor_id=actor,
        action_type="CHANGE_STATUS",
        field_name="investigation_status",
        previous_value="NUEVO",
        new_value="ASIGNADO",
        comment=comment,
        timestamp=event_timestamp,
    )

    return WorkflowResult(
        cases=result_df,
        audit_log=updated_audit_log,
    )


def change_investigation_status(
    cases_df: pd.DataFrame,
    audit_log: pd.DataFrame,
    *,
    case_id: str,
    new_status: str,
    actor_id: str,
    comment: str,
    resolution: str = "",
    timestamp: str | None = None,
) -> WorkflowResult:
    """
    Cambia el estado de una investigación y registra
    la operación en la bitácora.
    """

    validate_case_management_data(
        cases_df
    )

    index = find_case_index(
        cases_df,
        case_id,
    )

    actor = normalize_required_text(
        actor_id,
        "actor_id",
    )

    explanation = (
        normalize_required_text(
            comment,
            "comment",
        )
    )

    target_status = normalize_state(
        new_status
    )

    current_status = normalize_state(
        cases_df.loc[
            index,
            "investigation_status",
        ]
    )

    permitted_states = (
        ALLOWED_TRANSITIONS[
            current_status
        ]
    )

    if target_status not in permitted_states:
        raise ValueError(
            "Transición de estado no permitida: "
            f"{current_status} → {target_status}"
        )

    assigned_analyst = str(
        cases_df.loc[
            index,
            "assigned_analyst_id",
        ]
    ).strip()

    if (
        target_status
        == "EN_INVESTIGACION"
        and not assigned_analyst
    ):
        raise ValueError(
            "El caso debe tener un analista "
            "antes de iniciar la investigación."
        )

    normalized_resolution = str(
        resolution
    ).strip()

    if (
        target_status
        in FINAL_RESULT_STATES
        and not normalized_resolution
    ):
        raise ValueError(
            "Los estados CONFIRMADO y DESCARTADO "
            "requieren una resolución."
        )

    event_timestamp = (
        timestamp
        if timestamp is not None
        else current_chile_timestamp()
    )

    result_df = cases_df.copy()

    previous_resolution = result_df.loc[
        index,
        "resolution",
    ]

    result_df.loc[
        index,
        "investigation_status",
    ] = target_status

    result_df.loc[
        index,
        "investigation_updated_at",
    ] = event_timestamp

    if target_status in FINAL_RESULT_STATES:
        result_df.loc[
            index,
            "resolution",
        ] = normalized_resolution

    if target_status == "CERRADO":
        result_df.loc[
            index,
            "investigation_closed_at",
        ] = event_timestamp

    updated_audit_log = append_audit_event(
        audit_log,
        case_id=case_id,
        actor_id=actor,
        action_type="CHANGE_STATUS",
        field_name="investigation_status",
        previous_value=current_status,
        new_value=target_status,
        comment=explanation,
        timestamp=event_timestamp,
    )

    if target_status in FINAL_RESULT_STATES:
        updated_audit_log = append_audit_event(
            updated_audit_log,
            case_id=case_id,
            actor_id=actor,
            action_type="SET_RESOLUTION",
            field_name="resolution",
            previous_value=previous_resolution,
            new_value=normalized_resolution,
            comment=explanation,
            timestamp=event_timestamp,
        )

    return WorkflowResult(
        cases=result_df,
        audit_log=updated_audit_log,
    )