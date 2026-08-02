"""
Supervised case closure and feedback generation.

This module implements the controlled transition from an investigated case to
a formally closed case. It enforces segregation of duties between the assigned
analyst and the assigned supervisor, preserves a structured resolution record,
writes append-only audit events, and generates supervised feedback for later
machine-learning use.

Business rules
--------------
- Only supported final outcomes may be approved.
- The assigned analyst proposes the resolution.
- The assigned supervisor approves it.
- Analyst and supervisor must be different people.
- A discarded case cannot contain a confirmed loss.
- Recovered value cannot exceed confirmed loss.
- A case cannot receive a second approved resolution.
- Closure generates an immutable historical resolution, audit events, and one
  supervised label.

Interpretation
--------------
The generated label is a supervised investigation outcome, not an independent
proof of fraud. Its quality depends on the investigation, evidence, approval,
and later lifecycle controls such as reopening and feedback invalidation.
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


# Final outcomes accepted by the supervised closure workflow.
# CONFIRMADO maps to the positive label and DESCARTADO to the negative label.
ALLOWED_FINAL_OUTCOMES = {
    "CONFIRMADO",
    "DESCARTADO",
}


# Cases may be resolved only from active investigative or provisional
# outcome states. CERRADO is intentionally excluded to prevent duplicate
# approved resolutions.
ALLOWED_PRE_CLOSURE_STATUSES = {
    "EN_INVESTIGACION",
    "REQUIERE_ANTECEDENTES",
    "ESCALADO",
    "CONFIRMADO",
    "DESCARTADO",
}


RESOLUTION_COLUMNS = [
    "resolution_id",
    "case_id",
    "resolution_version",
    "proposed_outcome",
    "resolution_summary",
    "resolution_rationale",
    "proposed_by",
    "proposed_at",
    "supervisor_id",
    "approval_status",
    "approved_at",
    "confirmed_loss_clp",
    "recovered_amount_clp",
]


FEEDBACK_COLUMNS = [
    "case_id",
    "resolution_id",
    "investigation_outcome",
    "outcome_label",
    "closed_at",
    "supervisor_id",
    "confirmed_loss_clp",
    "recovered_amount_clp",
    "feedback_source",
]


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


@dataclass(frozen=True)
class CaseClosureResult:
    """
    Immutable container with every table affected by formal case closure.

    Attributes:
        cases:
            Updated case registry.
        resolutions:
            Resolution registry including the new approved version.
        audit:
            Append-only audit registry with proposal, approval, and status
            transition events.
        feedback:
            Supervised feedback registry including the new active label.
    """

    cases: pd.DataFrame
    resolutions: pd.DataFrame
    audit: pd.DataFrame
    feedback: pd.DataFrame


def current_chile_timestamp() -> str:
    """
    Return the current Chilean local time as an ISO 8601 string.

    The timezone offset is preserved so later UTC normalization remains
    deterministic and auditable.
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


def create_empty_resolution_registry() -> pd.DataFrame:
    """
    Create an empty resolution registry with the canonical schema.

    Returns:
        A DataFrame containing ``RESOLUTION_COLUMNS`` in stable order.
    """
    return pd.DataFrame(
        columns=RESOLUTION_COLUMNS
    )


def create_empty_feedback_registry() -> pd.DataFrame:
    """
    Create an empty supervised-feedback registry with the canonical schema.

    Returns:
        A DataFrame containing ``FEEDBACK_COLUMNS`` in stable order.
    """
    return pd.DataFrame(
        columns=FEEDBACK_COLUMNS
    )


def append_typed_row(
    df: pd.DataFrame,
    row: dict[str, Any],
    *,
    columns: list[str],
) -> pd.DataFrame:
    """
    Append one row while preserving a stable column structure.

    Constructing the row with the target schema prevents pandas warnings when
    the original registry is empty or contains all-null columns.

    Args:
        df:
            Existing registry.
        row:
            Row values to append.
        columns:
            Canonical column order for the registry.

    Returns:
        A new DataFrame containing the appended row.
    """

    row_df = pd.DataFrame(
        [row],
        columns=columns,
    )

    if df.empty:
        return row_df.reset_index(
            drop=True
        )

    return pd.concat(
        [
            df,
            row_df,
        ],
        ignore_index=True,
    )


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: set[str],
    dataset_name: str,
) -> None:
    """
    Validate the minimum schema required by a closure operation.

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


def next_sequential_id(
    df: pd.DataFrame,
    *,
    column: str,
    prefix: str,
) -> str:
    """
    Generate the next sequential identifier for a registry.

    Existing values that do not match the expected ``PREFIX-######`` pattern
    are ignored rather than treated as valid sequence members.
    """
    if df.empty:
        return f"{prefix}-000001"

    if column not in df.columns:
        raise ValueError(
            f"No existe la columna {column!r}."
        )

    maximum = 0

    for value in df[column]:
        text = normalize_text(
            value
        )

        expected_prefix = (
            f"{prefix}-"
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


def parse_nonnegative_amount(
    value: Any,
    field_name: str,
) -> float:
    """
    Parse a monetary value and require it to be nonnegative.

    Raises:
        ValueError: If the value is non-numeric or negative.
    """
    numeric_value = pd.to_numeric(
        value,
        errors="coerce",
    )

    if pd.isna(
        numeric_value
    ):
        raise ValueError(
            f"{field_name} debe ser numérico."
        )

    numeric_value = float(
        numeric_value
    )

    if numeric_value < 0:
        raise ValueError(
            f"{field_name} no puede ser negativo."
        )

    return numeric_value


def find_case_row(
    cases_df: pd.DataFrame,
    case_id: str,
) -> tuple[int, pd.Series]:
    """
    Locate exactly one case in the case registry.

    Returns:
        A tuple with the DataFrame index and the matching case row.

    Raises:
        ValueError: If the case is missing or appears more than once.
    """
    validate_required_columns(
        cases_df,
        REQUIRED_CASE_COLUMNS,
        "La tabla de casos",
    )

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
            "No existe el caso: "
            f"{normalized_case_id}"
        )

    if len(matches) > 1:
        raise ValueError(
            "El caso aparece más de una vez: "
            f"{normalized_case_id}"
        )

    index = int(
        matches.index[0]
    )

    return (
        index,
        matches.iloc[0],
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

    The function allocates a sequential audit identifier and returns a new
    DataFrame, leaving the caller-owned audit table unchanged.
    """
    validate_required_columns(
        audit_df,
        REQUIRED_AUDIT_COLUMNS,
        "La tabla de auditoría",
    )

    normalized_actor_id = normalize_text(
        actor_id
    )

    if not normalized_actor_id:
        raise ValueError(
            "actor_id es obligatorio."
        )

    audit_id = next_sequential_id(
        audit_df,
        column="audit_id",
        prefix="AUD",
    )

    row = {
        "audit_id": audit_id,
        "case_id": normalize_text(
            case_id
        ),
        "event_timestamp": normalize_text(
            event_timestamp
        ),
        "actor_id": normalized_actor_id,
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


def validate_resolution_request(
    *,
    case_row: pd.Series,
    proposed_outcome: str,
    proposed_by: str,
    supervisor_id: str,
    resolution_summary: str,
    resolution_rationale: str,
    confirmed_loss_clp: Any,
    recovered_amount_clp: Any,
) -> tuple[str, float, float]:
    """
    Validate authority, outcome, rationale, and monetary closure rules.

    Returns:
        A tuple containing normalized outcome, confirmed loss, and recovered
        amount.

    Raises:
        ValueError:
            If the case state is not closable, the outcome is unsupported,
            the proposer or supervisor is unauthorized, segregation of duties
            is violated, required narrative fields are empty, or monetary
            values are inconsistent.
    """
    current_status = normalize_upper_text(
        case_row[
            "investigation_status"
        ]
    )

    if current_status == "CERRADO":
        raise ValueError(
            "El caso ya está cerrado."
        )

    if current_status not in (
        ALLOWED_PRE_CLOSURE_STATUSES
    ):
        raise ValueError(
            "El caso no se encuentra en un estado "
            "apto para resolución: "
            f"{current_status}"
        )

    normalized_outcome = normalize_upper_text(
        proposed_outcome
    )

    if normalized_outcome not in (
        ALLOWED_FINAL_OUTCOMES
    ):
        raise ValueError(
            "Resultado final inválido: "
            f"{proposed_outcome!r}"
        )

    normalized_proposer = normalize_text(
        proposed_by
    )

    assigned_analyst = normalize_text(
        case_row[
            "assigned_analyst_id"
        ]
    )

    if not normalized_proposer:
        raise ValueError(
            "proposed_by es obligatorio."
        )

    if not assigned_analyst:
        raise ValueError(
            "El caso no tiene analista asignado."
        )

    if normalized_proposer != assigned_analyst:
        raise ValueError(
            "La propuesta debe ser realizada por "
            "el analista asignado al caso."
        )

    normalized_supervisor = normalize_text(
        supervisor_id
    )

    case_supervisor = normalize_text(
        case_row[
            "supervisor_id"
        ]
    )

    if not normalized_supervisor:
        raise ValueError(
            "supervisor_id es obligatorio."
        )

    if not case_supervisor:
        raise ValueError(
            "El caso no tiene supervisor asignado."
        )

    if normalized_supervisor != case_supervisor:
        raise ValueError(
            "El supervisor no coincide con el "
            "supervisor asignado al caso."
        )

    # Segregation of duties: the investigator who proposes the outcome
    # cannot be the same person who approves and closes the case.
    if normalized_supervisor == normalized_proposer:
        raise ValueError(
            "El analista y el supervisor deben "
            "ser personas distintas."
        )

    if not normalize_text(
        resolution_summary
    ):
        raise ValueError(
            "resolution_summary es obligatorio."
        )

    if not normalize_text(
        resolution_rationale
    ):
        raise ValueError(
            "resolution_rationale es obligatorio."
        )

    confirmed_loss = parse_nonnegative_amount(
        confirmed_loss_clp,
        "confirmed_loss_clp",
    )

    recovered_amount = parse_nonnegative_amount(
        recovered_amount_clp,
        "recovered_amount_clp",
    )

    # Recovery cannot exceed the loss confirmed by the approved
    # investigation outcome.
    if recovered_amount > confirmed_loss:
        raise ValueError(
            "recovered_amount_clp no puede ser "
            "mayor que confirmed_loss_clp."
        )

    # A discarded investigation represents no confirmed fraud loss in the
    # supervised label. Nonzero loss would contradict that outcome.
    if (
        normalized_outcome == "DESCARTADO"
        and confirmed_loss != 0
    ):
        raise ValueError(
            "Un caso descartado debe tener "
            "confirmed_loss_clp igual a 0."
        )

    return (
        normalized_outcome,
        confirmed_loss,
        recovered_amount,
    )


def close_case_with_supervisor_approval(
    cases_df: pd.DataFrame,
    resolutions_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    feedback_df: pd.DataFrame,
    *,
    case_id: str,
    proposed_outcome: str,
    resolution_summary: str,
    resolution_rationale: str,
    proposed_by: str,
    supervisor_id: str,
    confirmed_loss_clp: Any,
    recovered_amount_clp: Any,
    proposed_at: str | None = None,
    approved_at: str | None = None,
) -> CaseClosureResult:
    """
    Close a case after analyst proposal and supervisor approval.

    The operation is transactional at the function level: it validates all
    business rules before producing updated copies of the case, resolution,
    audit, and feedback registries.

    Args:
        cases_df:
            Case registry containing one row per case.
        resolutions_df:
            Structured resolution history.
        audit_df:
            Append-only case audit history.
        feedback_df:
            Supervised feedback registry.
        case_id:
            Case to close.
        proposed_outcome:
            ``CONFIRMADO`` or ``DESCARTADO``.
        resolution_summary:
            Concise description of the final conclusion.
        resolution_rationale:
            Investigative reasoning supporting the conclusion.
        proposed_by:
            Assigned analyst identifier.
        supervisor_id:
            Assigned supervisor identifier.
        confirmed_loss_clp:
            Confirmed loss in Chilean pesos.
        recovered_amount_clp:
            Recovered amount in Chilean pesos.
        proposed_at:
            Optional deterministic proposal timestamp for tests or replay.
        approved_at:
            Optional deterministic approval timestamp for tests or replay.

    Returns:
        ``CaseClosureResult`` with all updated registries.

    Raises:
        ValueError:
            If schemas, authority, status, outcome, narrative, monetary, or
            duplicate-resolution rules are violated.
    """
    validate_required_columns(
        resolutions_df,
        set(
            RESOLUTION_COLUMNS
        ),
        "El registro de resoluciones",
    )

    validate_required_columns(
        feedback_df,
        set(
            FEEDBACK_COLUMNS
        ),
        "El registro de feedback",
    )

    case_index, case_row = find_case_row(
        cases_df,
        case_id,
    )

    (
        normalized_outcome,
        confirmed_loss,
        recovered_amount,
    ) = validate_resolution_request(
        case_row=case_row,
        proposed_outcome=proposed_outcome,
        proposed_by=proposed_by,
        supervisor_id=supervisor_id,
        resolution_summary=(
            resolution_summary
        ),
        resolution_rationale=(
            resolution_rationale
        ),
        confirmed_loss_clp=(
            confirmed_loss_clp
        ),
        recovered_amount_clp=(
            recovered_amount_clp
        ),
    )

    normalized_case_id = normalize_text(
        case_id
    )

    previous_resolutions = (
        resolutions_df.loc[
            resolutions_df["case_id"]
            .fillna("")
            .astype(str)
            .str.strip()
            == normalized_case_id
        ]
    )

    # Historical proposals may exist, but only one approved resolution is
    # allowed for a case until a later reopening workflow supersedes it.
    if not previous_resolutions.empty:
        approved_resolutions = (
            previous_resolutions[
                "approval_status"
            ]
            .fillna("")
            .astype(str)
            .str.upper()
            .eq("APROBADA")
        )

        if approved_resolutions.any():
            raise ValueError(
                "El caso ya posee una resolución "
                "aprobada."
            )

    resolution_version = (
        len(
            previous_resolutions
        )
        + 1
    )

    resolution_id = next_sequential_id(
        resolutions_df,
        column="resolution_id",
        prefix="RES",
    )

    proposal_timestamp = (
        proposed_at
        if proposed_at is not None
        else current_chile_timestamp()
    )

    approval_timestamp = (
        approved_at
        if approved_at is not None
        else current_chile_timestamp()
    )

    resolution_row = {
        "resolution_id": resolution_id,
        "case_id": normalized_case_id,
        "resolution_version": (
            resolution_version
        ),
        "proposed_outcome": (
            normalized_outcome
        ),
        "resolution_summary": (
            normalize_text(
                resolution_summary
            )
        ),
        "resolution_rationale": (
            normalize_text(
                resolution_rationale
            )
        ),
        "proposed_by": normalize_text(
            proposed_by
        ),
        "proposed_at": (
            proposal_timestamp
        ),
        "supervisor_id": normalize_text(
            supervisor_id
        ),
        "approval_status": "APROBADA",
        "approved_at": (
            approval_timestamp
        ),
        "confirmed_loss_clp": (
            confirmed_loss
        ),
        "recovered_amount_clp": (
            recovered_amount
        ),
    }

    updated_resolutions = append_typed_row(
        resolutions_df,
        resolution_row,
        columns=RESOLUTION_COLUMNS,
    )

    # Work on copies so validation failures never partially mutate caller
    # data and successful closure returns an explicit new state.
    updated_cases = cases_df.copy()

    text_columns = [
        "investigation_status",
        "investigation_updated_at",
        "investigation_closed_at",
        "resolution",
    ]

    for column in text_columns:
        updated_cases[column] = (
            updated_cases[column]
            .astype("object")
        )

    previous_status = normalize_upper_text(
        case_row[
            "investigation_status"
        ]
    )

    updated_cases.loc[
        case_index,
        "investigation_status",
    ] = "CERRADO"

    updated_cases.loc[
        case_index,
        "investigation_updated_at",
    ] = approval_timestamp

    updated_cases.loc[
        case_index,
        "investigation_closed_at",
    ] = approval_timestamp

    updated_cases.loc[
        case_index,
        "resolution",
    ] = normalized_outcome

    updated_audit = append_audit_event(
        audit_df,
        case_id=normalized_case_id,
        actor_id=proposed_by,
        action_type=(
            "PROPOSE_RESOLUTION"
        ),
        field_name="resolution",
        previous_value=(
            case_row[
                "resolution"
            ]
        ),
        new_value=normalized_outcome,
        comment=(
            "El analista propone una resolución "
            "estructurada para revisión."
        ),
        event_timestamp=(
            proposal_timestamp
        ),
    )

    updated_audit = append_audit_event(
        updated_audit,
        case_id=normalized_case_id,
        actor_id=supervisor_id,
        action_type=(
            "APPROVE_RESOLUTION"
        ),
        field_name="resolution",
        previous_value="PENDIENTE",
        new_value=normalized_outcome,
        comment=(
            "El supervisor aprueba la resolución "
            f"{resolution_id}."
        ),
        event_timestamp=(
            approval_timestamp
        ),
    )

    if previous_status != normalized_outcome:
        updated_audit = append_audit_event(
            updated_audit,
            case_id=normalized_case_id,
            actor_id=supervisor_id,
            action_type="CHANGE_STATUS",
            field_name=(
                "investigation_status"
            ),
            previous_value=previous_status,
            new_value=normalized_outcome,
            comment=(
                "Se registra el resultado final "
                "aprobado."
            ),
            event_timestamp=(
                approval_timestamp
            ),
        )

    updated_audit = append_audit_event(
        updated_audit,
        case_id=normalized_case_id,
        actor_id=supervisor_id,
        action_type="CHANGE_STATUS",
        field_name=(
            "investigation_status"
        ),
        previous_value=(
            normalized_outcome
        ),
        new_value="CERRADO",
        comment=(
            "Cierre formal posterior a la "
            "aprobación de la resolución."
        ),
        event_timestamp=(
            approval_timestamp
        ),
    )

    # The supervised target is derived from the approved investigation
    # outcome, never from the original analytical risk score.
    outcome_label = (
        1
        if normalized_outcome
        == "CONFIRMADO"
        else 0
    )

    feedback_row = {
        "case_id": normalized_case_id,
        "resolution_id": resolution_id,
        "investigation_outcome": (
            normalized_outcome
        ),
        "outcome_label": outcome_label,
        "closed_at": approval_timestamp,
        "supervisor_id": normalize_text(
            supervisor_id
        ),
        "confirmed_loss_clp": (
            confirmed_loss
        ),
        "recovered_amount_clp": (
            recovered_amount
        ),
        "feedback_source": (
            "SUPERVISOR_APPROVED_RESOLUTION"
        ),
    }

    updated_feedback = append_typed_row(
        feedback_df,
        feedback_row,
        columns=FEEDBACK_COLUMNS,
    )

    return CaseClosureResult(
        cases=updated_cases,
        resolutions=updated_resolutions,
        audit=updated_audit,
        feedback=updated_feedback,
    )