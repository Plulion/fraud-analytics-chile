from __future__ import annotations

from typing import Any

import pandas as pd


REQUIRED_ASSESSMENT_COLUMNS = {
    "assessment_id",
    "case_id",
    "engine_type",
    "engine_version",
    "policy_id",
    "policy_version",
    "evaluated_at",
    "risk_score",
    "assessment_coverage",
    "calculated_alert_level",
    "data_quality_status",
}


REQUIRED_ALERT_COLUMNS = {
    "alert_id",
    "assessment_id",
    "case_id",
    "alert_status",
    "alert_priority",
}


REQUIRED_OUTCOME_COLUMNS = {
    "case_id",
    "investigation_outcome",
    "resolved_at",
    "reviewer_id",
    "resolution_summary",
}


VALID_INVESTIGATION_OUTCOMES = {
    "CONFIRMADO",
    "DESCARTADO",
}


def normalize_text(
    value: Any,
) -> str:
    """
    Convierte un valor en texto sin espacios
    al comienzo ni al final.
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
    return normalize_text(
        value
    ).upper()


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: set[str],
    dataset_name: str,
) -> None:
    """
    Comprueba que un DataFrame contenga
    las columnas necesarias.
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

    if df.empty:
        raise ValueError(
            f"{dataset_name} está vacío."
        )


def validate_unique_identifier(
    df: pd.DataFrame,
    column: str,
    dataset_name: str,
) -> None:
    """
    Comprueba que un identificador sea obligatorio
    y no esté repetido.
    """

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


def validate_feedback_inputs(
    assessments_df: pd.DataFrame,
    alerts_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
) -> None:
    """
    Valida las tres fuentes necesarias para
    construir el conjunto de retroalimentación.
    """

    validate_required_columns(
        assessments_df,
        REQUIRED_ASSESSMENT_COLUMNS,
        "La tabla de evaluaciones",
    )

    validate_required_columns(
        alerts_df,
        REQUIRED_ALERT_COLUMNS,
        "La tabla de alertas",
    )

    validate_required_columns(
        outcomes_df,
        REQUIRED_OUTCOME_COLUMNS,
        "La tabla de resultados",
    )

    validate_unique_identifier(
        assessments_df,
        "assessment_id",
        "La tabla de evaluaciones",
    )

    validate_unique_identifier(
        outcomes_df,
        "case_id",
        "La tabla de resultados",
    )

    if not alerts_df.empty:
        validate_unique_identifier(
            alerts_df,
            "alert_id",
            "La tabla de alertas",
        )

    assessment_case_counts = (
        assessments_df["case_id"]
        .astype(str)
        .value_counts()
    )

    duplicated_case_assessments = (
        assessment_case_counts[
            assessment_case_counts > 1
        ]
    )

    if not duplicated_case_assessments.empty:
        raise ValueError(
            "Esta versión del backtesting requiere "
            "una evaluación por caso. "
            "Casos con múltiples evaluaciones: "
            f"{duplicated_case_assessments.index.tolist()}"
        )

    for value in outcomes_df[
        "investigation_outcome"
    ]:
        normalized = normalize_upper_text(
            value
        )

        if normalized not in (
            VALID_INVESTIGATION_OUTCOMES
        ):
            raise ValueError(
                "Resultado de investigación "
                "desconocido: "
                f"{value!r}"
            )

    parsed_dates = pd.to_datetime(
        outcomes_df["resolved_at"],
        errors="coerce",
    )

    if parsed_dates.isna().any():
        affected_cases = (
            outcomes_df.loc[
                parsed_dates.isna(),
                "case_id",
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "Existen fechas de resolución "
            "inválidas. Casos afectados: "
            f"{affected_cases}"
        )

    empty_reviewers = (
        outcomes_df["reviewer_id"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
    )

    if empty_reviewers.any():
        affected_cases = (
            outcomes_df.loc[
                empty_reviewers,
                "case_id",
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "Toda resolución debe identificar "
            "al revisor. Casos afectados: "
            f"{affected_cases}"
        )

    empty_summaries = (
        outcomes_df["resolution_summary"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
    )

    if empty_summaries.any():
        affected_cases = (
            outcomes_df.loc[
                empty_summaries,
                "case_id",
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "Toda resolución debe contener "
            "un resumen. Casos afectados: "
            f"{affected_cases}"
        )


def classify_prediction_result(
    predicted_positive: bool,
    actual_positive: bool,
) -> str:
    """
    Clasifica una predicción mediante la matriz
    de confusión.
    """

    if predicted_positive and actual_positive:
        return "TRUE_POSITIVE"

    if predicted_positive and not actual_positive:
        return "FALSE_POSITIVE"

    if (
        not predicted_positive
        and not actual_positive
    ):
        return "TRUE_NEGATIVE"

    return "FALSE_NEGATIVE"


def build_feedback_dataset(
    assessments_df: pd.DataFrame,
    alerts_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Relaciona evaluaciones, alertas y resultados
    de investigaciones.

    Una alerta presente representa una predicción
    positiva bajo la política de selección actual.
    """

    validate_feedback_inputs(
        assessments_df,
        alerts_df,
        outcomes_df,
    )

    assessments = assessments_df.copy()
    alerts = alerts_df.copy()
    outcomes = outcomes_df.copy()

    assessments["assessment_id"] = (
        assessments["assessment_id"]
        .astype(str)
    )

    assessments["case_id"] = (
        assessments["case_id"]
        .astype(str)
    )

    alerts["assessment_id"] = (
        alerts["assessment_id"]
        .astype(str)
    )

    alerts["case_id"] = (
        alerts["case_id"]
        .astype(str)
    )

    outcomes["case_id"] = (
        outcomes["case_id"]
        .astype(str)
    )

    outcomes[
        "investigation_outcome"
    ] = (
        outcomes[
            "investigation_outcome"
        ]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    alert_reference = alerts[
        [
            "alert_id",
            "assessment_id",
            "case_id",
            "alert_status",
            "alert_priority",
        ]
    ].copy()

    feedback_df = assessments.merge(
        alert_reference,
        on=[
            "assessment_id",
            "case_id",
        ],
        how="left",
        validate="one_to_one",
    )

    feedback_df = feedback_df.merge(
        outcomes,
        on="case_id",
        how="inner",
        validate="one_to_one",
    )

    if len(feedback_df) != len(
        outcomes
    ):
        assessment_case_ids = set(
            assessments["case_id"]
        )

        outcome_case_ids = set(
            outcomes["case_id"]
        )

        missing_assessments = sorted(
            outcome_case_ids
            - assessment_case_ids
        )

        raise ValueError(
            "Existen resultados sin una "
            "evaluación relacionada. Casos: "
            f"{missing_assessments}"
        )

    feedback_df[
        "predicted_positive"
    ] = feedback_df[
        "alert_id"
    ].notna()

    feedback_df[
        "actual_positive"
    ] = (
        feedback_df[
            "investigation_outcome"
        ]
        == "CONFIRMADO"
    )

    feedback_df[
        "prediction_result"
    ] = [
        classify_prediction_result(
            bool(predicted),
            bool(actual),
        )
        for predicted, actual
        in zip(
            feedback_df[
                "predicted_positive"
            ],
            feedback_df[
                "actual_positive"
            ],
        )
    ]

    feedback_df[
        "outcome_label"
    ] = (
        feedback_df[
            "actual_positive"
        ]
        .astype(int)
    )

    feedback_df[
        "alert_created"
    ] = (
        feedback_df[
            "predicted_positive"
        ]
        .map(
            {
                True: "SI",
                False: "NO",
            }
        )
    )

    feedback_df[
        "alert_id"
    ] = feedback_df[
        "alert_id"
    ].fillna("")

    feedback_df[
        "alert_status"
    ] = feedback_df[
        "alert_status"
    ].fillna("SIN_ALERTA")

    feedback_df[
        "alert_priority"
    ] = feedback_df[
        "alert_priority"
    ].fillna("SIN_ALERTA")

    result_columns = [
        "assessment_id",
        "case_id",
        "engine_type",
        "engine_version",
        "policy_id",
        "policy_version",
        "evaluated_at",
        "risk_score",
        "assessment_coverage",
        "calculated_alert_level",
        "data_quality_status",
        "alert_id",
        "alert_created",
        "alert_status",
        "alert_priority",
        "predicted_positive",
        "investigation_outcome",
        "actual_positive",
        "outcome_label",
        "prediction_result",
        "resolved_at",
        "reviewer_id",
        "resolution_summary",
    ]

    feedback_df = feedback_df[
        result_columns
    ].sort_values(
        by=[
            "prediction_result",
            "case_id",
        ],
        ascending=[
            True,
            True,
        ],
    ).reset_index(
        drop=True
    )

    return feedback_df


def safe_divide(
    numerator: int | float,
    denominator: int | float,
) -> float:
    """
    Evita divisiones por cero.
    """

    if denominator == 0:
        return 0.0

    return float(
        numerator / denominator
    )


def calculate_backtesting_metrics(
    feedback_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Calcula métricas iniciales de clasificación.

    Estas métricas describen el comportamiento de
    la política sobre los casos investigados.
    """

    required_columns = {
        "predicted_positive",
        "actual_positive",
        "prediction_result",
    }

    validate_required_columns(
        feedback_df,
        required_columns,
        "El conjunto de retroalimentación",
    )

    result_counts = (
        feedback_df[
            "prediction_result"
        ]
        .value_counts()
        .to_dict()
    )

    true_positive = int(
        result_counts.get(
            "TRUE_POSITIVE",
            0,
        )
    )

    false_positive = int(
        result_counts.get(
            "FALSE_POSITIVE",
            0,
        )
    )

    true_negative = int(
        result_counts.get(
            "TRUE_NEGATIVE",
            0,
        )
    )

    false_negative = int(
        result_counts.get(
            "FALSE_NEGATIVE",
            0,
        )
    )

    total = int(
        len(feedback_df)
    )

    precision = safe_divide(
        true_positive,
        true_positive + false_positive,
    )

    recall = safe_divide(
        true_positive,
        true_positive + false_negative,
    )

    specificity = safe_divide(
        true_negative,
        true_negative + false_positive,
    )

    accuracy = safe_divide(
        true_positive + true_negative,
        total,
    )

    false_positive_rate = safe_divide(
        false_positive,
        false_positive + true_negative,
    )

    false_negative_rate = safe_divide(
        false_negative,
        false_negative + true_positive,
    )

    metrics: dict[str, Any] = {
        "total_resolved_cases": total,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "precision": round(
            precision,
            4,
        ),
        "recall": round(
            recall,
            4,
        ),
        "specificity": round(
            specificity,
            4,
        ),
        "accuracy": round(
            accuracy,
            4,
        ),
        "false_positive_rate": round(
            false_positive_rate,
            4,
        ),
        "false_negative_rate": round(
            false_negative_rate,
            4,
        ),
        "engine_versions": sorted(
            feedback_df[
                "engine_version"
            ]
            .astype(str)
            .unique()
            .tolist()
        ),
        "policy_versions": sorted(
            feedback_df[
                "policy_version"
            ]
            .astype(str)
            .unique()
            .tolist()
        ),
    }

    return metrics