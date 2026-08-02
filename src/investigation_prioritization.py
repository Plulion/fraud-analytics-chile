from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


REQUIRED_ALERT_COLUMNS = {
    "alert_id",
    "assessment_id",
    "case_id",
    "created_at",
    "alert_status",
    "risk_level",
    "estimated_loss_clp",
}


REQUIRED_ASSESSMENT_COLUMNS = {
    "assessment_id",
    "case_id",
    "assessment_coverage",
    "data_quality_status",
}


REQUIRED_CONSOLIDATED_COLUMNS = {
    "case_id",
    "digital_event_count",
}


VALID_RISK_LEVELS = {
    "CRITICO",
    "ALTO",
    "MEDIO",
    "BAJO",
    "INCOMPLETO",
    "SIN_DATOS",
}


RISK_POINTS = {
    "CRITICO": 40,
    "ALTO": 30,
    "MEDIO": 20,
    "INCOMPLETO": 15,
    "BAJO": 5,
    "SIN_DATOS": 0,
}


PRIORITY_RANK = {
    "CRITICA": 4,
    "ALTA": 3,
    "MEDIA": 2,
    "BAJA": 1,
}


@dataclass(frozen=True)
class InvestigationPriorityRecord:
    alert_id: str
    assessment_id: str
    case_id: str
    risk_level: str
    estimated_loss_clp: float
    assessment_coverage: float
    digital_event_count: int
    data_quality_status: str
    risk_points: int
    loss_points: int
    recurrence_points: int
    coverage_points: int
    priority_score: int
    investigation_priority: str
    priority_reasons: str
    queue_position: int
    queue_status: str
    capacity_policy_id: str
    capacity_policy_version: str


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


def validate_unique_non_empty_identifier(
    df: pd.DataFrame,
    column: str,
    dataset_name: str,
) -> None:
    if df[column].isna().any():
        raise ValueError(
            f"{dataset_name} contiene valores "
            f"vacíos en {column!r}."
        )

    empty_values = (
        df[column]
        .astype(str)
        .str.strip()
        .eq("")
    )

    if empty_values.any():
        raise ValueError(
            f"{dataset_name} contiene textos "
            f"vacíos en {column!r}."
        )

    if df[column].duplicated().any():
        duplicated_values = (
            df.loc[
                df[column].duplicated(
                    keep=False
                ),
                column,
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            f"{dataset_name} contiene "
            f"{column} duplicados: "
            f"{sorted(duplicated_values)}"
        )


def validate_prioritization_inputs(
    alerts_df: pd.DataFrame,
    assessments_df: pd.DataFrame,
    consolidated_df: pd.DataFrame,
) -> None:
    missing_alert_columns = (
        REQUIRED_ALERT_COLUMNS.difference(
            alerts_df.columns
        )
    )

    if missing_alert_columns:
        raise ValueError(
            "Faltan columnas en alertas: "
            f"{sorted(missing_alert_columns)}"
        )

    missing_assessment_columns = (
        REQUIRED_ASSESSMENT_COLUMNS.difference(
            assessments_df.columns
        )
    )

    if missing_assessment_columns:
        raise ValueError(
            "Faltan columnas en evaluaciones: "
            f"{sorted(missing_assessment_columns)}"
        )

    missing_consolidated_columns = (
        REQUIRED_CONSOLIDATED_COLUMNS.difference(
            consolidated_df.columns
        )
    )

    if missing_consolidated_columns:
        raise ValueError(
            "Faltan columnas en la vista consolidada: "
            f"{sorted(missing_consolidated_columns)}"
        )

    if alerts_df.empty:
        raise ValueError(
            "La tabla de alertas está vacía."
        )

    if assessments_df.empty:
        raise ValueError(
            "La tabla de evaluaciones está vacía."
        )

    if consolidated_df.empty:
        raise ValueError(
            "La vista consolidada está vacía."
        )

    validate_unique_non_empty_identifier(
        alerts_df,
        "alert_id",
        "La tabla de alertas",
    )

    validate_unique_non_empty_identifier(
        assessments_df,
        "assessment_id",
        "La tabla de evaluaciones",
    )

    validate_unique_non_empty_identifier(
        consolidated_df,
        "case_id",
        "La vista consolidada",
    )

    numeric_alert_loss = pd.to_numeric(
        alerts_df["estimated_loss_clp"],
        errors="coerce",
    )

    if numeric_alert_loss.isna().any():
        raise ValueError(
            "estimated_loss_clp contiene "
            "valores no numéricos."
        )

    if (
        numeric_alert_loss < 0
    ).any():
        raise ValueError(
            "estimated_loss_clp no puede "
            "contener valores negativos."
        )

    coverage = pd.to_numeric(
        assessments_df[
            "assessment_coverage"
        ],
        errors="coerce",
    )

    if coverage.isna().any():
        raise ValueError(
            "assessment_coverage contiene "
            "valores no numéricos."
        )

    if (
        (coverage < 0)
        | (coverage > 100)
    ).any():
        raise ValueError(
            "assessment_coverage debe estar "
            "entre 0 y 100."
        )

    event_counts = pd.to_numeric(
        consolidated_df[
            "digital_event_count"
        ],
        errors="coerce",
    )

    if event_counts.isna().any():
        raise ValueError(
            "digital_event_count contiene "
            "valores no numéricos."
        )

    if (
        event_counts < 0
    ).any():
        raise ValueError(
            "digital_event_count no puede "
            "ser negativo."
        )

    for risk_level in alerts_df[
        "risk_level"
    ]:
        normalized = normalize_upper_text(
            risk_level
        )

        if normalized not in VALID_RISK_LEVELS:
            raise ValueError(
                "Nivel de riesgo desconocido: "
                f"{risk_level!r}"
            )


def calculate_loss_points(
    estimated_loss_clp: float,
) -> int:
    if estimated_loss_clp >= 50_000_000:
        return 25

    if estimated_loss_clp >= 10_000_000:
        return 20

    if estimated_loss_clp >= 1_000_000:
        return 10

    return 5


def calculate_recurrence_points(
    digital_event_count: int,
) -> int:
    if digital_event_count >= 4:
        return 15

    if digital_event_count >= 2:
        return 10

    if digital_event_count == 1:
        return 5

    return 0


def calculate_coverage_points(
    assessment_coverage: float,
) -> int:
    if assessment_coverage >= 90:
        return 10

    if assessment_coverage >= 70:
        return 7

    return 3


def classify_investigation_priority(
    priority_score: int,
) -> str:
    if priority_score >= 75:
        return "CRITICA"

    if priority_score >= 55:
        return "ALTA"

    if priority_score >= 35:
        return "MEDIA"

    return "BAJA"


def build_priority_reasons(
    *,
    risk_level: str,
    estimated_loss_clp: float,
    digital_event_count: int,
    assessment_coverage: float,
    data_quality_status: str,
) -> str:
    reasons = [
        f"RISK_LEVEL_{risk_level}",
    ]

    if estimated_loss_clp >= 50_000_000:
        reasons.append(
            "VERY_HIGH_ESTIMATED_LOSS"
        )
    elif estimated_loss_clp >= 10_000_000:
        reasons.append(
            "HIGH_ESTIMATED_LOSS"
        )
    elif estimated_loss_clp >= 1_000_000:
        reasons.append(
            "MODERATE_ESTIMATED_LOSS"
        )
    else:
        reasons.append(
            "LOW_ESTIMATED_LOSS"
        )

    if digital_event_count >= 4:
        reasons.append(
            "REPEATED_DIGITAL_ACTIVITY_HIGH"
        )
    elif digital_event_count >= 2:
        reasons.append(
            "REPEATED_DIGITAL_ACTIVITY_MEDIUM"
        )
    elif digital_event_count == 1:
        reasons.append(
            "SINGLE_DIGITAL_EVENT"
        )
    else:
        reasons.append(
            "NO_DIGITAL_EVENTS"
        )

    if assessment_coverage < 70:
        reasons.append(
            "LOW_ASSESSMENT_COVERAGE"
        )
    elif assessment_coverage < 90:
        reasons.append(
            "MEDIUM_ASSESSMENT_COVERAGE"
        )
    else:
        reasons.append(
            "HIGH_ASSESSMENT_COVERAGE"
        )

    if data_quality_status == (
        "REQUIERE_REVISION_DE_DATOS"
    ):
        reasons.append(
            "DATA_QUALITY_REVIEW_REQUIRED"
        )

    return " | ".join(
        reasons
    )


def prioritize_investigations(
    alerts_df: pd.DataFrame,
    assessments_df: pd.DataFrame,
    consolidated_df: pd.DataFrame,
    *,
    investigation_capacity: int,
    capacity_policy_id: str = (
        "INVESTIGATION_CAPACITY_POLICY"
    ),
    capacity_policy_version: str = "1.0.0",
) -> pd.DataFrame:
    """
    Construye una cola priorizada y selecciona
    las primeras alertas según la capacidad.

    No elimina ni descarta alertas.
    """

    validate_prioritization_inputs(
        alerts_df,
        assessments_df,
        consolidated_df,
    )

    if (
        not isinstance(
            investigation_capacity,
            int,
        )
        or isinstance(
            investigation_capacity,
            bool,
        )
        or investigation_capacity < 0
    ):
        raise ValueError(
            "investigation_capacity debe ser "
            "un entero mayor o igual a cero."
        )

    alerts = alerts_df.copy()
    assessments = assessments_df.copy()
    consolidated = consolidated_df.copy()

    alerts["assessment_id"] = (
        alerts["assessment_id"]
        .astype(str)
    )

    alerts["case_id"] = (
        alerts["case_id"]
        .astype(str)
    )

    assessments["assessment_id"] = (
        assessments["assessment_id"]
        .astype(str)
    )

    assessments["case_id"] = (
        assessments["case_id"]
        .astype(str)
    )

    consolidated["case_id"] = (
        consolidated["case_id"]
        .astype(str)
    )

    joined_df = alerts.merge(
        assessments[
            [
                "assessment_id",
                "case_id",
                "assessment_coverage",
                "data_quality_status",
            ]
        ],
        on=[
            "assessment_id",
            "case_id",
        ],
        how="left",
        validate="one_to_one",
    )

    joined_df = joined_df.merge(
        consolidated[
            [
                "case_id",
                "digital_event_count",
            ]
        ],
        on="case_id",
        how="left",
        validate="many_to_one",
    )

    required_joined_values = [
        "assessment_coverage",
        "data_quality_status",
        "digital_event_count",
    ]

    for column in required_joined_values:
        if joined_df[column].isna().any():
            affected_alerts = (
                joined_df.loc[
                    joined_df[column].isna(),
                    "alert_id",
                ]
                .astype(str)
                .tolist()
            )

            raise ValueError(
                "No fue posible relacionar "
                f"{column!r} para las alertas: "
                f"{affected_alerts}"
            )

    records: list[
        InvestigationPriorityRecord
    ] = []

    temporary_rows: list[
        dict[str, Any]
    ] = []

    for _, row in joined_df.iterrows():
        risk_level = normalize_upper_text(
            row["risk_level"]
        )

        estimated_loss = float(
            row["estimated_loss_clp"]
        )

        coverage = float(
            row["assessment_coverage"]
        )

        event_count = int(
            row["digital_event_count"]
        )

        data_quality_status = (
            normalize_upper_text(
                row["data_quality_status"]
            )
        )

        risk_points = RISK_POINTS[
            risk_level
        ]

        loss_points = (
            calculate_loss_points(
                estimated_loss
            )
        )

        recurrence_points = (
            calculate_recurrence_points(
                event_count
            )
        )

        coverage_points = (
            calculate_coverage_points(
                coverage
            )
        )

        priority_score = (
            risk_points
            + loss_points
            + recurrence_points
            + coverage_points
        )

        priority = (
            classify_investigation_priority(
                priority_score
            )
        )

        reasons = build_priority_reasons(
            risk_level=risk_level,
            estimated_loss_clp=estimated_loss,
            digital_event_count=event_count,
            assessment_coverage=coverage,
            data_quality_status=(
                data_quality_status
            ),
        )

        temporary_rows.append(
            {
                "alert_id": str(
                    row["alert_id"]
                ),
                "assessment_id": str(
                    row["assessment_id"]
                ),
                "case_id": str(
                    row["case_id"]
                ),
                "risk_level": risk_level,
                "estimated_loss_clp": (
                    estimated_loss
                ),
                "assessment_coverage": (
                    coverage
                ),
                "digital_event_count": (
                    event_count
                ),
                "data_quality_status": (
                    data_quality_status
                ),
                "risk_points": risk_points,
                "loss_points": loss_points,
                "recurrence_points": (
                    recurrence_points
                ),
                "coverage_points": (
                    coverage_points
                ),
                "priority_score": (
                    priority_score
                ),
                "investigation_priority": (
                    priority
                ),
                "priority_reasons": reasons,
                "_priority_rank": (
                    PRIORITY_RANK[
                        priority
                    ]
                ),
                "_created_at": str(
                    row["created_at"]
                ),
            }
        )

    queue_df = pd.DataFrame(
        temporary_rows
    )

    queue_df = queue_df.sort_values(
        by=[
            "_priority_rank",
            "priority_score",
            "estimated_loss_clp",
            "assessment_coverage",
            "_created_at",
            "alert_id",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            True,
            True,
        ],
    ).reset_index(
        drop=True
    )

    for position, row in queue_df.iterrows():
        queue_position = position + 1

        queue_status = (
            "SELECCIONADA"
            if queue_position
            <= investigation_capacity
            else "EN_ESPERA"
        )

        records.append(
            InvestigationPriorityRecord(
                alert_id=str(
                    row["alert_id"]
                ),
                assessment_id=str(
                    row["assessment_id"]
                ),
                case_id=str(
                    row["case_id"]
                ),
                risk_level=str(
                    row["risk_level"]
                ),
                estimated_loss_clp=float(
                    row[
                        "estimated_loss_clp"
                    ]
                ),
                assessment_coverage=float(
                    row[
                        "assessment_coverage"
                    ]
                ),
                digital_event_count=int(
                    row[
                        "digital_event_count"
                    ]
                ),
                data_quality_status=str(
                    row[
                        "data_quality_status"
                    ]
                ),
                risk_points=int(
                    row["risk_points"]
                ),
                loss_points=int(
                    row["loss_points"]
                ),
                recurrence_points=int(
                    row[
                        "recurrence_points"
                    ]
                ),
                coverage_points=int(
                    row["coverage_points"]
                ),
                priority_score=int(
                    row["priority_score"]
                ),
                investigation_priority=str(
                    row[
                        "investigation_priority"
                    ]
                ),
                priority_reasons=str(
                    row["priority_reasons"]
                ),
                queue_position=(
                    queue_position
                ),
                queue_status=queue_status,
                capacity_policy_id=(
                    capacity_policy_id
                ),
                capacity_policy_version=(
                    capacity_policy_version
                ),
            )
        )

    return pd.DataFrame(
        [
            asdict(record)
            for record in records
        ],
        columns=list(
            InvestigationPriorityRecord
            .__dataclass_fields__
            .keys()
        ),
    )