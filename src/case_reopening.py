"""
Controlled reopening of previously closed fraud investigation cases.

This module reactivates a closed case under supervisor authority while
preserving the historical resolution and invalidating the supervised feedback
that was derived from it.

Business rules
--------------
- Only a case in ``CERRADO`` status may be reopened.
- Only the assigned supervisor may authorize reopening.
- A documented reason of at least 20 characters is required.
- The target status must return the case to active investigation.
- The latest approved resolution is preserved and marked as superseded.
- The feedback linked to that resolution is preserved and marked invalid.
- Reopening writes append-only audit events.
- Invalidated feedback must not be used for later model training.

Interpretation
--------------
Reopening does not erase the previous investigation outcome. It records that
the prior conclusion is no longer active because the investigation has resumed.
This preserves lineage, auditability, and model-training integrity.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd


CHILE_TIME_ZONE = ZoneInfo(
    "America/Santiago"
)


# Reopening must return the case to an active investigative state.
# Final outcome and closed states are intentionally excluded.
ALLOWED_REOPEN_STATUSES = {
    "EN_INVESTIGACION",
    "REQUIERE_ANTECEDENTES",
}


REQUIRED_CASE_COLUMNS = {
    "case_id",
    "investigation_status",
    "assigned_analyst_id",
    "supervisor_id",
    "investigation_created_at",
    "investigation_updated_at",
    "investigation_closed_at",
    "resolution",
}


REQUIRED_RESOLUTION_COLUMNS = {
    "resolution_id",
    "case_id",
    "proposed_outcome",
    "supervisor_id",
    "approval_status",
    "approved_at",
}


REQUIRED_AUDIT_COLUMNS = {
    "audit_id",
    "case_id",
    "event_timestamp",
    "actor_id",
    "action_type",
    "field_name",
    "previous_value",
    "new_value",
    "comment",
}


REQUIRED_FEEDBACK_COLUMNS = {
    "case_id",
    "resolution_id",
    "investigation_outcome",
    "outcome_label",
    "closed_at",
    "supervisor_id",
    "feedback_source",
}


# Lifecycle fields preserve the approved resolution as historical evidence
# while clearly marking that it is no longer the active conclusion.
RESOLUTION_LIFECYCLE_COLUMNS = {
    "resolution_lifecycle_status": "VIGENTE",
    "superseded_at": "",
    "superseded_by": "",
    "superseded_reason": "",
}


# Feedback lifecycle fields prevent a superseded investigation outcome
# from contaminating supervised model-training data.
FEEDBACK_LIFECYCLE_COLUMNS = {
    "feedback_status": "ACTIVO",
    "invalidated_at": "",
    "invalidated_by": "",
    "invalidation_reason": "",
}


@dataclass(frozen=True)
class CaseReopeningResult:
    """
    Immutable container with every registry affected by case reopening.

    Attributes:
        cases:
            Updated case registry with the case returned to investigation.
        resolutions:
            Resolution history with the previous approved resolution marked
            as superseded.
        audit:
            Append-only audit history including reopening, invalidation, and
            status-change events.
        feedback:
            Feedback registry with the previous label marked invalid.
    """

    cases: pd.DataFrame
    resolutions: pd.DataFrame
    audit: pd.DataFrame
    feedback: pd.DataFrame


def current_chile_timestamp() -> str:
    """
    Return the current Chilean local time as an ISO 8601 string.

    The timezone offset is preserved for later normalization and audit review.
    """
    return datetime.now(
        CHILE_TIME_ZONE
    ).isoformat(
        timespec="seconds"
    )


def normalize_text(
    value: Any,
) -> str:
    """
    Convert a scalar value into normalized text.

    Missing values become an empty string. Other values are converted to text
    and stripped of surrounding whitespace.
    """
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
    """Normalize a scalar value and convert it to uppercase."""
    return normalize_text(
        value
    ).upper()


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: set[str],
    dataset_name: str,
) -> None:
    """
    Validate the minimum schema required by a reopening operation.

    Raises:
        ValueError: If one or more required columns are missing.
    """
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


def ensure_text_columns(
    df: pd.DataFrame,
    defaults: dict[str, str],
) -> pd.DataFrame:
    """
    Add lifecycle columns when missing and normalize them as text.

    This helper keeps backward compatibility with registries produced before
    lifecycle fields were introduced.

    Returns:
        A copied DataFrame with all requested columns initialized.
    """
    result = df.copy()

    for column, default_value in defaults.items():
        if column not in result.columns:
            result[column] = default_value

        result[column] = (
            result[column]
            .astype("object")
        )

        empty_mask = (
            result[column].isna()
            | result[column]
            .astype(str)
            .str.strip()
            .eq("")
        )

        result.loc[
            empty_mask,
            column,
        ] = default_value

    return result


def next_sequential_id(
    df: pd.DataFrame,
    *,
    column: str,
    prefix: str,
) -> str:
    """
    Generate the next sequential identifier for a registry.

    Values that do not match the expected ``PREFIX-######`` pattern are
    ignored when determining the next sequence number.
    """
    if column not in df.columns:
        raise ValueError(
            f"No existe la columna {column!r}."
        )

    maximum = 0
    expected_prefix = f"{prefix}-"

    for value in df[column]:
        text = normalize_text(
            value
        )

        if not text.startswith(
            expected_prefix
        ):
            continue

        numeric_part = text[
            len(expected_prefix):
        ]

        if numeric_part.isdigit():
            maximum = max(
                maximum,
                int(numeric_part),
            )

    return (
        f"{prefix}-{maximum + 1:06d}"
    )


def append_audit_event(
    audit_df: pd.DataFrame,
    *,
    case_id: str,
    actor_id: str,
    action_type: str,
    field_name: str,
    previous_value: Any,
    new_value: Any,
    comment: str,
    event_timestamp: str,
) -> pd.DataFrame:
    """
    Append one immutable business event to the audit registry.

    The function returns a new DataFrame and leaves the caller-owned audit
    history unchanged.
    """
    validate_required_columns(
        audit_df,
        REQUIRED_AUDIT_COLUMNS,
        "La tabla de auditoría",
    )

    normalized_actor = normalize_text(
        actor_id
    )

    if not normalized_actor:
        raise ValueError(
            "actor_id es obligatorio."
        )

    row = {
        "audit_id": next_sequential_id(
            audit_df,
            column="audit_id",
            prefix="AUD",
        ),
        "case_id": normalize_text(
            case_id
        ),
        "event_timestamp": normalize_text(
            event_timestamp
        ),
        "actor_id": normalized_actor,
        "action_type": normalize_upper_text(
            action_type
        ),
        "field_name": normalize_text(
            field_name
        ),
        "previous_value": normalize_text(
            previous_value
        ),
        "new_value": normalize_text(
            new_value
        ),
        "comment": normalize_text(
            comment
        ),
    }

    row_df = pd.DataFrame(
        [row],
        columns=list(
            audit_df.columns
        ),
    )

    if audit_df.empty:
        return row_df.reset_index(
            drop=True
        )

    return pd.concat(
        [
            audit_df,
            row_df,
        ],
        ignore_index=True,
    )


def find_single_case(
    cases_df: pd.DataFrame,
    case_id: str,
) -> tuple[Any, pd.Series]:
    """
    Locate exactly one case in the case registry.

    Returns:
        A tuple containing the DataFrame index and matching row.

    Raises:
        ValueError: If the case is missing or duplicated.
    """
    normalized_case_id = normalize_text(
        case_id
    )

    matches = cases_df.loc[
        cases_df["case_id"]
        .fillna("")
        .astype(str)
        .str.strip()
        == normalized_case_id
    ]

    if matches.empty:
        raise ValueError(
            f"No existe el caso {normalized_case_id!r}."
        )

    if len(matches) > 1:
        raise ValueError(
            "El caso aparece más de una vez: "
            f"{normalized_case_id}"
        )

    index = matches.index[0]

    return (
        index,
        matches.iloc[0],
    )


def find_latest_approved_resolution(
    resolutions_df: pd.DataFrame,
    *,
    case_id: str,
) -> tuple[Any, pd.Series]:
    """
    Locate the most recent approved resolution for a case.

    The approval timestamp is used when valid. If all timestamps are invalid,
    the last matching row is selected as a conservative compatibility fallback.

    Returns:
        A tuple containing the resolution-row index and resolution data.

    Raises:
        ValueError: If no approved resolution exists.
    """
    normalized_case_id = normalize_text(
        case_id
    )

    matches = resolutions_df.loc[
        (
            resolutions_df["case_id"]
            .fillna("")
            .astype(str)
            .str.strip()
            == normalized_case_id
        )
        & (
            resolutions_df["approval_status"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
            == "APROBADA"
        )
    ].copy()

    if matches.empty:
        raise ValueError(
            "El caso no posee una resolución "
            "aprobada que pueda invalidarse."
        )

    parsed_approved_at = pd.to_datetime(
        matches["approved_at"],
        errors="coerce",
        format="mixed",
        utc=True,
    )

    if parsed_approved_at.isna().all():
        index = matches.index[-1]
    else:
        index = parsed_approved_at.idxmax()

    return (
        index,
        resolutions_df.loc[index],
    )


def validate_reopening_request(
    *,
    case_row: pd.Series,
    supervisor_id: str,
    target_status: str,
    reopening_reason: str,
) -> str:
    """
    Validate authority, case state, target state, and reopening rationale.

    Returns:
        The normalized target investigation status.

    Raises:
        ValueError:
            If the case is not closed, the actor is not the assigned
            supervisor, the target status is not allowed, or the reason is too
            short.
    """
    current_status = normalize_upper_text(
        case_row[
            "investigation_status"
        ]
    )

    if current_status != "CERRADO":
        raise ValueError(
            "Solo se puede reabrir un caso "
            "que esté CERRADO."
        )

    case_supervisor = normalize_text(
        case_row[
            "supervisor_id"
        ]
    )

    normalized_supervisor = normalize_text(
        supervisor_id
    )

    if not normalized_supervisor:
        raise ValueError(
            "supervisor_id es obligatorio."
        )

    # Only the supervisor already assigned to the case may reverse the
    # closure decision and reactivate the investigation.
    if normalized_supervisor != case_supervisor:
        raise ValueError(
            "La reapertura debe ser autorizada "
            "por el supervisor asignado al caso."
        )

    normalized_target = normalize_upper_text(
        target_status
    )

    if normalized_target not in (
        ALLOWED_REOPEN_STATUSES
    ):
        raise ValueError(
            "Estado de reapertura inválido: "
            f"{target_status!r}"
        )

    normalized_reason = normalize_text(
        reopening_reason
    )

    # A meaningful reason is required because reopening changes the active
    # investigation outcome and invalidates downstream supervised feedback.
    if len(normalized_reason) < 20:
        raise ValueError(
            "reopening_reason debe contener al "
            "menos 20 caracteres."
        )

    return normalized_target


def reopen_closed_case(
    cases_df: pd.DataFrame,
    resolutions_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    feedback_df: pd.DataFrame,
    *,
    case_id: str,
    supervisor_id: str,
    reopening_reason: str,
    target_status: str = "EN_INVESTIGACION",
    reopened_at: str | None = None,
) -> CaseReopeningResult:
    """
    Reopen a closed case and invalidate its prior supervised feedback.

    The operation validates all rules before returning updated copies of the
    case, resolution, audit, and feedback registries.

    Args:
        cases_df:
            Case registry containing one row per case.
        resolutions_df:
            Structured resolution history.
        audit_df:
            Append-only audit history.
        feedback_df:
            Supervised feedback registry.
        case_id:
            Closed case to reopen.
        supervisor_id:
            Assigned supervisor authorizing the reopening.
        reopening_reason:
            Documented reason for reopening.
        target_status:
            Active investigation status to assign.
        reopened_at:
            Optional deterministic timestamp for tests or replay.

    Returns:
        ``CaseReopeningResult`` containing all updated registries.

    Raises:
        ValueError:
            If schemas, authority, status, timestamp, resolution lifecycle, or
            feedback lifecycle requirements are violated.
    """
    validate_required_columns(
        cases_df,
        REQUIRED_CASE_COLUMNS,
        "La tabla de casos",
    )

    validate_required_columns(
        resolutions_df,
        REQUIRED_RESOLUTION_COLUMNS,
        "La tabla de resoluciones",
    )

    validate_required_columns(
        audit_df,
        REQUIRED_AUDIT_COLUMNS,
        "La tabla de auditoría",
    )

    validate_required_columns(
        feedback_df,
        REQUIRED_FEEDBACK_COLUMNS,
        "La tabla de feedback",
    )

    normalized_case_id = normalize_text(
        case_id
    )

    case_index, case_row = find_single_case(
        cases_df,
        normalized_case_id,
    )

    normalized_target = validate_reopening_request(
        case_row=case_row,
        supervisor_id=supervisor_id,
        target_status=target_status,
        reopening_reason=reopening_reason,
    )

    resolution_index, resolution_row = (
        find_latest_approved_resolution(
            resolutions_df,
            case_id=normalized_case_id,
        )
    )

    resolution_id = normalize_text(
        resolution_row[
            "resolution_id"
        ]
    )

    timestamp = (
        normalize_text(
            reopened_at
        )
        if reopened_at is not None
        else current_chile_timestamp()
    )

    if not timestamp:
        raise ValueError(
            "reopened_at no puede estar vacío."
        )

    parsed_timestamp = pd.to_datetime(
        timestamp,
        errors="coerce",
        format="mixed",
        utc=True,
    )

    if pd.isna(
        parsed_timestamp
    ):
        raise ValueError(
            f"reopened_at es inválido: {timestamp!r}"
        )

    # Work on copies so a failed validation never partially mutates the
    # caller's registries.
    updated_cases = cases_df.copy()

    for column in [
        "investigation_status",
        "investigation_updated_at",
        "investigation_closed_at",
        "resolution",
    ]:
        updated_cases[column] = (
            updated_cases[column]
            .astype("object")
        )

    previous_resolution = normalize_upper_text(
        case_row[
            "resolution"
        ]
    )

    updated_cases.loc[
        case_index,
        "investigation_status",
    ] = normalized_target

    updated_cases.loc[
        case_index,
        "investigation_updated_at",
    ] = timestamp

    updated_cases.loc[
        case_index,
        "investigation_closed_at",
    ] = ""

    updated_cases.loc[
        case_index,
        "resolution",
    ] = ""

    updated_resolutions = ensure_text_columns(
        resolutions_df,
        RESOLUTION_LIFECYCLE_COLUMNS,
    )

    current_lifecycle_status = normalize_upper_text(
        updated_resolutions.loc[
            resolution_index,
            "resolution_lifecycle_status",
        ]
    )

    if current_lifecycle_status not in {
        "",
        "VIGENTE",
    }:
        raise ValueError(
            "La resolución aprobada ya no se "
            "encuentra vigente."
        )

    # Preserve the approved resolution as historical evidence while marking
    # that it is no longer the active conclusion after reopening.
    updated_resolutions.loc[
        resolution_index,
        "resolution_lifecycle_status",
    ] = "SUPERSEDIDA_POR_REAPERTURA"

    updated_resolutions.loc[
        resolution_index,
        "superseded_at",
    ] = timestamp

    updated_resolutions.loc[
        resolution_index,
        "superseded_by",
    ] = normalize_text(
        supervisor_id
    )

    updated_resolutions.loc[
        resolution_index,
        "superseded_reason",
    ] = normalize_text(
        reopening_reason
    )

    updated_feedback = ensure_text_columns(
        feedback_df,
        FEEDBACK_LIFECYCLE_COLUMNS,
    )

    feedback_mask = (
        updated_feedback["case_id"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq(
            normalized_case_id
        )
        & updated_feedback["resolution_id"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq(
            resolution_id
        )
        & updated_feedback["feedback_status"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
        .eq("ACTIVO")
    )

    if not feedback_mask.any():
        raise ValueError(
            "No existe feedback activo asociado "
            "a la resolución aprobada."
        )

    # Keep the original label for lineage, but mark it inactive so training
    # dataset builders can exclude it deterministically.
    updated_feedback.loc[
        feedback_mask,
        "feedback_status",
    ] = "INVALIDADO"

    updated_feedback.loc[
        feedback_mask,
        "invalidated_at",
    ] = timestamp

    updated_feedback.loc[
        feedback_mask,
        "invalidated_by",
    ] = normalize_text(
        supervisor_id
    )

    updated_feedback.loc[
        feedback_mask,
        "invalidation_reason",
    ] = normalize_text(
        reopening_reason
    )

    updated_audit = append_audit_event(
        audit_df,
        case_id=normalized_case_id,
        actor_id=supervisor_id,
        action_type="REOPEN_CASE",
        field_name="resolution",
        previous_value=previous_resolution,
        new_value="",
        comment=(
            "Se autoriza la reapertura. "
            f"Motivo: {normalize_text(reopening_reason)}"
        ),
        event_timestamp=timestamp,
    )

    updated_audit = append_audit_event(
        updated_audit,
        case_id=normalized_case_id,
        actor_id=supervisor_id,
        action_type="INVALIDATE_FEEDBACK",
        field_name="feedback_status",
        previous_value="ACTIVO",
        new_value="INVALIDADO",
        comment=(
            "Se invalida la etiqueta asociada a "
            f"la resolución {resolution_id}."
        ),
        event_timestamp=timestamp,
    )

    updated_audit = append_audit_event(
        updated_audit,
        case_id=normalized_case_id,
        actor_id=supervisor_id,
        action_type="CHANGE_STATUS",
        field_name="investigation_status",
        previous_value="CERRADO",
        new_value=normalized_target,
        comment=(
            "Cambio de estado producido por una "
            "reapertura autorizada."
        ),
        event_timestamp=timestamp,
    )

    return CaseReopeningResult(
        cases=updated_cases,
        resolutions=updated_resolutions,
        audit=updated_audit,
        feedback=updated_feedback,
    )