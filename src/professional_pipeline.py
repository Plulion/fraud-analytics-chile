"""
Professional assessment and alert-generation pipeline.

This module converts one consolidated case-level risk view into two governed
outputs:

- one reproducible analytical assessment per case;
- zero or one alert per case, according to the active selection policy.

Business interpretation
-----------------------
An assessment is an analytical record, not a finding of guilt, fraud, or legal
responsibility. An alert is an operational selection decision indicating that a
case should receive attention under the current policy.

Governance principles
---------------------
- Every assessment records engine, policy, and schema versions.
- Every alert records the selection-policy version that created it.
- Alert creation is separated from risk-score calculation.
- Data-quality concerns may independently trigger operational review.
- Identifiers and timestamps can be injected for deterministic testing.
- Alerts are prioritized for workflow visibility, not as proof of fraud.

Current limitations
-------------------
The current policy is rule-based and educational. It does not yet implement
capacity constraints, cost-sensitive optimization, threshold calibration,
fairness testing, drift monitoring, or production approval workflows.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Callable, Iterable
from uuid import uuid4
from zoneinfo import ZoneInfo

import pandas as pd


CHILE_TIME_ZONE = ZoneInfo(
    "America/Santiago"
)


REQUIRED_CONSOLIDATED_COLUMNS = {
    "case_id",
    "risk_score",
    "assessment_coverage",
    "alert_level",
    "decision_status",
    "consolidated_alert_level",
    "consolidated_data_status",
    "consolidated_recommended_action",
    "estimated_loss_clp",
}


# Alert levels accepted from the consolidated analytical layer.
VALID_ALERT_LEVELS = {
    "BAJO",
    "MEDIO",
    "ALTO",
    "CRITICO",
    "INCOMPLETO",
    "SIN_DATOS",
}


# Operational queue priorities created by the alert-selection policy.
VALID_ALERT_PRIORITIES = {
    "BAJA",
    "MEDIA",
    "ALTA",
    "CRITICA",
}


@dataclass(frozen=True)
class AssessmentRecord:
    """
    Reproducible output of one analytical engine execution.

    The record preserves the case, engine, policy, schema, timestamp, score,
    coverage, recommendation, and data-quality context required for later
    audit and backtesting.

    It does not represent guilt or a judicial, administrative, or final
    investigative decision.
    """

    assessment_id: str
    case_id: str
    engine_type: str
    engine_version: str
    policy_id: str
    policy_version: str
    input_schema_version: str
    evaluated_at: str
    risk_score: float
    assessment_coverage: float
    calculated_alert_level: str
    recommended_action: str
    data_quality_status: str


@dataclass(frozen=True)
class AlertRecord:
    """
    Operational alert created when an assessment satisfies selection policy.

    The record preserves the originating assessment and the exact policy
    version responsible for alert creation.
    """

    alert_id: str
    assessment_id: str
    case_id: str
    selection_policy_id: str
    selection_policy_version: str
    created_at: str
    alert_status: str
    alert_priority: str
    alert_reason: str
    risk_level: str
    estimated_loss_clp: float


@dataclass(frozen=True)
class AlertSelectionDecision:
    """
    Internal result returned by the alert-selection policy.

    ``should_create_alert`` separates policy evaluation from persistence of the
    final alert record.
    """

    should_create_alert: bool
    alert_priority: str | None
    alert_reason: str


@dataclass(frozen=True)
class ProfessionalPipelineResult:
    """
    Immutable container with assessments and alerts produced by one run.

    Attributes:
        assessments:
            One versioned assessment per input case.
        alerts:
            Only the cases selected by the active alert policy.
    """

    assessments: pd.DataFrame
    alerts: pd.DataFrame


def current_chile_timestamp() -> str:
    """
    Return the current Chilean local time as an ISO 8601 string.

    The timezone offset is retained for auditability and later UTC
    normalization.
    """

    return datetime.now(
        CHILE_TIME_ZONE
    ).isoformat(
        timespec="seconds"
    )


def generate_assessment_id() -> str:
    """Generate a globally unique technical assessment identifier."""

    return (
        "ASM-"
        f"{uuid4().hex.upper()}"
    )


def generate_alert_id() -> str:
    """Generate a globally unique technical alert identifier."""

    return (
        "ALT-"
        f"{uuid4().hex.upper()}"
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


def validate_consolidated_input(
    df: pd.DataFrame,
) -> None:
    """
    Validate the consolidated case view before assessment generation.

    Validation covers required columns, non-empty input, unique case IDs,
    numeric and nonnegative analytical values, and supported alert levels.

    Raises:
        ValueError:
            If the consolidated view is empty, malformed, duplicated, or
            contains unsupported or invalid values.
    """

    missing_columns = (
        REQUIRED_CONSOLIDATED_COLUMNS
        .difference(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "La vista consolidada no contiene "
            "las columnas requeridas: "
            f"{sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError(
            "La vista consolidada está vacía."
        )

    if df["case_id"].isna().any():
        raise ValueError(
            "Existen registros sin case_id."
        )

    empty_case_ids = (
        df["case_id"]
        .astype(str)
        .str.strip()
        .eq("")
    )

    if empty_case_ids.any():
        raise ValueError(
            "Existen registros con case_id vacío."
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

    numeric_columns = [
        "risk_score",
        "assessment_coverage",
        "estimated_loss_clp",
    ]

    for column in numeric_columns:
        numeric_values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if numeric_values.isna().any():
            affected_cases = (
                df.loc[
                    numeric_values.isna(),
                    "case_id",
                ]
                .astype(str)
                .tolist()
            )

            raise ValueError(
                f"La columna {column!r} contiene "
                "valores no numéricos. "
                f"Casos afectados: "
                f"{affected_cases}"
            )

        if (
            numeric_values < 0
        ).any():
            affected_cases = (
                df.loc[
                    numeric_values < 0,
                    "case_id",
                ]
                .astype(str)
                .tolist()
            )

            raise ValueError(
                f"La columna {column!r} contiene "
                "valores negativos. "
                f"Casos afectados: "
                f"{affected_cases}"
            )

    for value in df[
        "consolidated_alert_level"
    ]:
        normalized = normalize_upper_text(
            value
        )

        if normalized not in (
            VALID_ALERT_LEVELS
        ):
            raise ValueError(
                "Nivel consolidado desconocido: "
                f"{value!r}"
            )


def create_assessment_record(
    row: pd.Series,
    *,
    assessment_id: str,
    evaluated_at: str,
    engine_type: str = (
        "CONSOLIDATED_RULE_ENGINE"
    ),
    engine_version: str = "1.0.0",
    policy_id: str = (
        "CONSOLIDATED_RISK_POLICY"
    ),
    policy_version: str = "1.0.0",
    input_schema_version: str = "1.0.0",
) -> AssessmentRecord:
    """
    Transform one consolidated case row into a versioned assessment.

    Args:
        row:
            One validated consolidated case record.
        assessment_id:
            Unique assessment identifier.
        evaluated_at:
            Timestamp shared by the pipeline run.
        engine_type:
            Analytical engine family.
        engine_version:
            Executed engine version.
        policy_id:
            Risk-policy identifier.
        policy_version:
            Risk-policy version.
        input_schema_version:
            Version of the input contract.

    Returns:
        A frozen ``AssessmentRecord`` suitable for audit and backtesting.
    """

    case_id = normalize_text(
        row["case_id"]
    )

    alert_level = normalize_upper_text(
        row[
            "consolidated_alert_level"
        ]
    )

    if alert_level not in (
        VALID_ALERT_LEVELS
    ):
        raise ValueError(
            "Nivel consolidado desconocido: "
            f"{alert_level!r}"
        )

    return AssessmentRecord(
        assessment_id=assessment_id,
        case_id=case_id,
        engine_type=engine_type,
        engine_version=engine_version,
        policy_id=policy_id,
        policy_version=policy_version,
        input_schema_version=(
            input_schema_version
        ),
        evaluated_at=evaluated_at,
        risk_score=float(
            row["risk_score"]
        ),
        assessment_coverage=float(
            row["assessment_coverage"]
        ),
        calculated_alert_level=(
            alert_level
        ),
        recommended_action=(
            normalize_upper_text(
                row[
                    "consolidated_recommended_action"
                ]
            )
        ),
        data_quality_status=(
            normalize_upper_text(
                row[
                    "consolidated_data_status"
                ]
            )
        ),
    )


def select_alert(
    assessment: AssessmentRecord,
) -> AlertSelectionDecision:
    """
    Decide whether an assessment should create an operational alert.

    The initial policy considers consolidated risk level and data-quality
    status. Estimated loss is preserved in the final alert and used for queue
    ordering, but it does not yet alter the selection decision itself.

    Returns:
        An ``AlertSelectionDecision`` with creation flag, priority, and reason.
    """

    risk_level = (
        assessment.calculated_alert_level
    )

    data_status = (
        assessment.data_quality_status
    )

    # Data-quality issues trigger human review independently of the numeric
    # risk level because incomplete or inconsistent inputs can make the
    # analytical conclusion unreliable.
    if data_status == (
        "REQUIERE_REVISION_DE_DATOS"
    ):
        return AlertSelectionDecision(
            should_create_alert=True,
            alert_priority="ALTA",
            alert_reason=(
                "La evaluación requiere revisar "
                "la calidad o integridad de los datos."
            ),
        )

    if risk_level == "CRITICO":
        return AlertSelectionDecision(
            should_create_alert=True,
            alert_priority="CRITICA",
            alert_reason=(
                "La evaluación alcanzó nivel "
                "de riesgo CRITICO."
            ),
        )

    if risk_level == "ALTO":
        return AlertSelectionDecision(
            should_create_alert=True,
            alert_priority="ALTA",
            alert_reason=(
                "La evaluación alcanzó nivel "
                "de riesgo ALTO."
            ),
        )

    if risk_level == "MEDIO":
        return AlertSelectionDecision(
            should_create_alert=True,
            alert_priority="MEDIA",
            alert_reason=(
                "La evaluación alcanzó nivel "
                "de riesgo MEDIO."
            ),
        )

    if risk_level == "INCOMPLETO":
        return AlertSelectionDecision(
            should_create_alert=True,
            alert_priority="MEDIA",
            alert_reason=(
                "La evaluación quedó incompleta "
                "y necesita antecedentes."
            ),
        )

    return AlertSelectionDecision(
        should_create_alert=False,
        alert_priority=None,
        alert_reason=(
            "La evaluación no cumple la política "
            "actual de selección de alertas."
        ),
    )


def create_alert_record(
    assessment: AssessmentRecord,
    decision: AlertSelectionDecision,
    *,
    alert_id: str,
    created_at: str,
    estimated_loss_clp: float,
    selection_policy_id: str = (
        "ALERT_SELECTION_POLICY"
    ),
    selection_policy_version: str = "1.0.0",
) -> AlertRecord:
    """
    Create a versioned alert from a positive selection decision.

    Raises:
        ValueError:
            If policy did not select the assessment or returned an unsupported
            alert priority.
    """

    if not decision.should_create_alert:
        raise ValueError(
            "No se puede crear una alerta "
            "para una decisión negativa."
        )

    if decision.alert_priority not in (
        VALID_ALERT_PRIORITIES
    ):
        raise ValueError(
            "Prioridad de alerta inválida: "
            f"{decision.alert_priority!r}"
        )

    return AlertRecord(
        alert_id=alert_id,
        assessment_id=(
            assessment.assessment_id
        ),
        case_id=assessment.case_id,
        selection_policy_id=(
            selection_policy_id
        ),
        selection_policy_version=(
            selection_policy_version
        ),
        created_at=created_at,
        alert_status="ABIERTA",
        alert_priority=(
            decision.alert_priority
        ),
        alert_reason=(
            decision.alert_reason
        ),
        risk_level=(
            assessment.calculated_alert_level
        ),
        estimated_loss_clp=float(
            estimated_loss_clp
        ),
    )


def records_to_dataframe(
    records: Iterable[Any],
    *,
    columns: list[str],
) -> pd.DataFrame:
    """
    Convert dataclass records into a DataFrame with stable schema.

    Supplying explicit columns preserves the expected output contract even
    when the record collection is empty.
    """

    rows = [
        asdict(record)
        for record in records
    ]

    return pd.DataFrame(
        rows,
        columns=columns,
    )


def process_professional_pipeline(
    consolidated_df: pd.DataFrame,
    *,
    timestamp: str | None = None,
    assessment_id_factory: Callable[
        [],
        str,
    ] = generate_assessment_id,
    alert_id_factory: Callable[
        [],
        str,
    ] = generate_alert_id,
) -> ProfessionalPipelineResult:
    """
    Run the professional assessment and alert-selection pipeline.

    One assessment is created for every validated case. An alert is created
    only when the active selection policy returns a positive decision.

    Args:
        consolidated_df:
            One-row-per-case consolidated analytical view.
        timestamp:
            Optional deterministic timestamp for tests or replay.
        assessment_id_factory:
            Injectable assessment-ID generator.
        alert_id_factory:
            Injectable alert-ID generator.

    Returns:
        ``ProfessionalPipelineResult`` containing stable assessment and alert
        DataFrames.

    Raises:
        ValueError:
            If the consolidated input or any generated alert decision is
            invalid.
    """

    validate_consolidated_input(
        consolidated_df
    )

    processing_timestamp = (
        timestamp
        if timestamp is not None
        else current_chile_timestamp()
    )

    assessments: list[
        AssessmentRecord
    ] = []

    alerts: list[
        AlertRecord
    ] = []

    # Assessment creation is unconditional for valid input rows. Alert
    # creation is a separate policy decision, preserving all analytical
    # outcomes for later audit and backtesting.
    for _, row in consolidated_df.iterrows():
        assessment = (
            create_assessment_record(
                row,
                assessment_id=(
                    assessment_id_factory()
                ),
                evaluated_at=(
                    processing_timestamp
                ),
            )
        )

        assessments.append(
            assessment
        )

        decision = select_alert(
            assessment
        )

        if decision.should_create_alert:
            alert = create_alert_record(
                assessment,
                decision,
                alert_id=(
                    alert_id_factory()
                ),
                created_at=(
                    processing_timestamp
                ),
                estimated_loss_clp=float(
                    row[
                        "estimated_loss_clp"
                    ]
                ),
            )

            alerts.append(
                alert
            )

    assessment_columns = [
        field_name
        for field_name in (
            AssessmentRecord
            .__dataclass_fields__
            .keys()
        )
    ]

    alert_columns = [
        field_name
        for field_name in (
            AlertRecord
            .__dataclass_fields__
            .keys()
        )
    ]

    assessments_df = records_to_dataframe(
        assessments,
        columns=assessment_columns,
    )

    alerts_df = records_to_dataframe(
        alerts,
        columns=alert_columns,
    )

    priority_rank = {
        "CRITICA": 4,
        "ALTA": 3,
        "MEDIA": 2,
        "BAJA": 1,
    }

    # Queue ordering gives operational visibility to higher priority and
    # larger estimated loss. It does not change the underlying assessment.
    if not alerts_df.empty:
        alerts_df[
            "_priority_rank"
        ] = (
            alerts_df[
                "alert_priority"
            ]
            .map(priority_rank)
            .fillna(0)
        )

        alerts_df = alerts_df.sort_values(
            by=[
                "_priority_rank",
                "estimated_loss_clp",
            ],
            ascending=[
                False,
                False,
            ],
        ).drop(
            columns=[
                "_priority_rank"
            ]
        ).reset_index(
            drop=True
        )

    return ProfessionalPipelineResult(
        assessments=assessments_df,
        alerts=alerts_df,
    )