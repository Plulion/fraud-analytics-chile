from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd


DEFAULT_FEEDBACK_STATUS = "ACTIVO"


REQUIRED_FEATURE_COLUMNS = {
    "case_id",
    "feature_snapshot_at",
}


REQUIRED_FEEDBACK_COLUMNS = {
    "case_id",
    "resolution_id",
    "investigation_outcome",
    "outcome_label",
    "closed_at",
    "feedback_source",
}


ALLOWED_OUTCOMES = {
    "CONFIRMADO": 1,
    "DESCARTADO": 0,
}


@dataclass(frozen=True)
class TrainingDatasetResult:
    dataset: pd.DataFrame
    exclusions: pd.DataFrame
    report: dict[str, Any]


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


def parse_datetime_series(
    series: pd.Series,
) -> pd.Series:
    return pd.to_datetime(
        series,
        errors="coerce",
        format="mixed",
        utc=True,
    )


def ensure_feedback_status(
    feedback_df: pd.DataFrame,
) -> pd.DataFrame:
    result = feedback_df.copy()

    if "feedback_status" not in result.columns:
        result["feedback_status"] = (
            DEFAULT_FEEDBACK_STATUS
        )

    result["feedback_status"] = (
        result["feedback_status"]
        .fillna(DEFAULT_FEEDBACK_STATUS)
        .astype(str)
        .str.strip()
        .str.upper()
    )

    result.loc[
        result["feedback_status"].eq(""),
        "feedback_status",
    ] = DEFAULT_FEEDBACK_STATUS

    return result


def validate_binary_labels(
    feedback_df: pd.DataFrame,
) -> None:
    numeric_labels = pd.to_numeric(
        feedback_df["outcome_label"],
        errors="coerce",
    )

    invalid_mask = (
        numeric_labels.isna()
        | ~numeric_labels.isin(
            [0, 1]
        )
    )

    if invalid_mask.any():
        affected = (
            feedback_df.loc[
                invalid_mask,
                "case_id",
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "outcome_label solo puede contener "
            "0 o 1. Casos afectados: "
            f"{affected}"
        )

    normalized_outcomes = (
        feedback_df[
            "investigation_outcome"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    expected_labels = normalized_outcomes.map(
        ALLOWED_OUTCOMES
    )

    unknown_outcomes = (
        expected_labels.isna()
    )

    if unknown_outcomes.any():
        affected = (
            feedback_df.loc[
                unknown_outcomes,
                "case_id",
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "Existen resultados investigativos "
            "desconocidos. Casos afectados: "
            f"{affected}"
        )

    inconsistent_mask = (
        numeric_labels.astype(int)
        != expected_labels.astype(int)
    )

    if inconsistent_mask.any():
        affected = (
            feedback_df.loc[
                inconsistent_mask,
                "case_id",
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "La etiqueta no coincide con el "
            "resultado investigativo. Casos "
            f"afectados: {affected}"
        )


def validate_feature_columns(
    features_df: pd.DataFrame,
    feature_columns: Iterable[str],
) -> list[str]:
    columns = [
        normalize_text(column)
        for column in feature_columns
    ]

    if not columns:
        raise ValueError(
            "Debe indicar al menos una variable "
            "predictora."
        )

    if any(
        not column
        for column in columns
    ):
        raise ValueError(
            "Los nombres de variables predictoras "
            "no pueden estar vacíos."
        )

    duplicated = sorted(
        {
            column
            for column in columns
            if columns.count(column) > 1
        }
    )

    if duplicated:
        raise ValueError(
            "Existen variables predictoras "
            f"duplicadas: {duplicated}"
        )

    missing = sorted(
        set(columns).difference(
            features_df.columns
        )
    )

    if missing:
        raise ValueError(
            "La tabla de variables no contiene: "
            f"{missing}"
        )

    forbidden_columns = {
        "outcome_label",
        "investigation_outcome",
        "resolution",
        "resolution_id",
        "approval_status",
        "feedback_status",
        "closed_at",
        "investigation_closed_at",
        "confirmed_loss_clp",
        "recovered_amount_clp",
    }

    forbidden_selected = sorted(
        set(columns).intersection(
            forbidden_columns
        )
    )

    if forbidden_selected:
        raise ValueError(
            "Se detectaron variables con fuga de "
            "información: "
            f"{forbidden_selected}"
        )

    return columns


def build_exclusion_row(
    *,
    case_id: Any,
    resolution_id: Any,
    reason_code: str,
    reason_detail: str,
) -> dict[str, str]:
    return {
        "case_id": normalize_text(
            case_id
        ),
        "resolution_id": normalize_text(
            resolution_id
        ),
        "exclusion_reason_code": (
            normalize_upper_text(
                reason_code
            )
        ),
        "exclusion_reason_detail": (
            normalize_text(
                reason_detail
            )
        ),
    }


def resolve_feedback_lifecycle(
    feedback_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Consolida múltiples versiones del mismo feedback.

    Si para una resolución existen registros ACTIVO e
    INVALIDADO, prevalece INVALIDADO. Esto evita usar
    por accidente una etiqueta que fue anulada después.
    """

    feedback = ensure_feedback_status(
        feedback_df
    )

    status_rank = {
        "INVALIDADO": 3,
        "REVOCADO": 3,
        "SUPERSEDIDO": 3,
        "ACTIVO": 1,
    }

    feedback["_status_rank"] = (
        feedback["feedback_status"]
        .map(status_rank)
        .fillna(2)
        .astype(int)
    )

    invalidated_at = (
        feedback["invalidated_at"]
        if "invalidated_at" in feedback.columns
        else pd.Series(
            "",
            index=feedback.index,
        )
    )

    closed_at = feedback["closed_at"]

    feedback["_lifecycle_timestamp"] = (
        parse_datetime_series(
            invalidated_at.where(
                invalidated_at
                .fillna("")
                .astype(str)
                .str.strip()
                .ne(""),
                closed_at,
            )
        )
    )

    feedback = feedback.sort_values(
        by=[
            "case_id",
            "resolution_id",
            "_status_rank",
            "_lifecycle_timestamp",
        ],
        ascending=[
            True,
            True,
            False,
            False,
        ],
        na_position="last",
    )

    feedback = feedback.drop_duplicates(
        subset=[
            "case_id",
            "resolution_id",
        ],
        keep="first",
    )

    return feedback.drop(
        columns=[
            "_status_rank",
            "_lifecycle_timestamp",
        ]
    ).reset_index(
        drop=True
    )


def build_training_dataset(
    features_df: pd.DataFrame,
    feedback_df: pd.DataFrame,
    *,
    feature_columns: Iterable[str],
) -> TrainingDatasetResult:
    """
    Construye un dataset supervisado con etiquetas
    investigativas activas y variables disponibles
    antes o al momento del cierre.

    Cada caso aporta como máximo una etiqueta activa.
    """

    validate_required_columns(
        features_df,
        REQUIRED_FEATURE_COLUMNS,
        "La tabla de variables",
    )

    validate_required_columns(
        feedback_df,
        REQUIRED_FEEDBACK_COLUMNS,
        "La tabla de feedback",
    )

    selected_features = (
        validate_feature_columns(
            features_df,
            feature_columns,
        )
    )

    features = features_df.copy()

    features["case_id"] = (
        features["case_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    if features["case_id"].eq("").any():
        raise ValueError(
            "La tabla de variables contiene "
            "case_id vacíos."
        )

    if features["case_id"].duplicated().any():
        duplicated = sorted(
            features.loc[
                features["case_id"]
                .duplicated(
                    keep=False
                ),
                "case_id",
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "La tabla de variables contiene "
            f"case_id duplicados: {duplicated}"
        )

    feature_timestamps = (
        parse_datetime_series(
            features[
                "feature_snapshot_at"
            ]
        )
    )

    invalid_feature_timestamp = (
        feature_timestamps.isna()
    )

    if invalid_feature_timestamp.any():
        affected = (
            features.loc[
                invalid_feature_timestamp,
                "case_id",
            ]
            .tolist()
        )

        raise ValueError(
            "Existen feature_snapshot_at "
            "inválidos. Casos afectados: "
            f"{affected}"
        )

    feedback = resolve_feedback_lifecycle(
        feedback_df
    )

    feedback["case_id"] = (
        feedback["case_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    feedback["resolution_id"] = (
        feedback["resolution_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    validate_binary_labels(
        feedback
    )

    exclusions: list[
        dict[str, str]
    ] = []

    candidate_rows: list[
        dict[str, Any]
    ] = []

    for _, feedback_row in feedback.iterrows():
        case_id = normalize_text(
            feedback_row[
                "case_id"
            ]
        )

        resolution_id = normalize_text(
            feedback_row[
                "resolution_id"
            ]
        )

        feedback_status = normalize_upper_text(
            feedback_row[
                "feedback_status"
            ]
        )

        if feedback_status != "ACTIVO":
            exclusions.append(
                build_exclusion_row(
                    case_id=case_id,
                    resolution_id=resolution_id,
                    reason_code=(
                        "FEEDBACK_NO_ACTIVO"
                    ),
                    reason_detail=(
                        "La etiqueta fue invalidada, "
                        "revocada o reemplazada."
                    ),
                )
            )
            continue

        feature_match = features.loc[
            features["case_id"].eq(
                case_id
            )
        ]

        if feature_match.empty:
            exclusions.append(
                build_exclusion_row(
                    case_id=case_id,
                    resolution_id=resolution_id,
                    reason_code=(
                        "SIN_VARIABLES"
                    ),
                    reason_detail=(
                        "No existe un snapshot de "
                        "variables para el caso."
                    ),
                )
            )
            continue

        feature_row = feature_match.iloc[0]

        closed_at = pd.to_datetime(
            feedback_row[
                "closed_at"
            ],
            errors="coerce",
            format="mixed",
            utc=True,
        )

        if pd.isna(
            closed_at
        ):
            exclusions.append(
                build_exclusion_row(
                    case_id=case_id,
                    resolution_id=resolution_id,
                    reason_code=(
                        "FECHA_CIERRE_INVALIDA"
                    ),
                    reason_detail=(
                        "closed_at no es una fecha "
                        "válida."
                    ),
                )
            )
            continue

        feature_snapshot_at = (
            pd.to_datetime(
                feature_row[
                    "feature_snapshot_at"
                ],
                errors="coerce",
                format="mixed",
                utc=True,
            )
        )

        if feature_snapshot_at > closed_at:
            exclusions.append(
                build_exclusion_row(
                    case_id=case_id,
                    resolution_id=resolution_id,
                    reason_code=(
                        "FUGA_TEMPORAL"
                    ),
                    reason_detail=(
                        "El snapshot de variables es "
                        "posterior al cierre del caso."
                    ),
                )
            )
            continue

        candidate = {
            "case_id": case_id,
            "resolution_id": resolution_id,
            "feature_snapshot_at": (
                feature_snapshot_at.isoformat()
            ),
            "closed_at": (
                closed_at.isoformat()
            ),
        }

        for column in selected_features:
            candidate[column] = (
                feature_row[column]
            )

        candidate[
            "investigation_outcome"
        ] = normalize_upper_text(
            feedback_row[
                "investigation_outcome"
            ]
        )

        candidate["outcome_label"] = int(
            feedback_row[
                "outcome_label"
            ]
        )

        candidate["feedback_source"] = (
            normalize_text(
                feedback_row[
                    "feedback_source"
                ]
            )
        )

        candidate_rows.append(
            candidate
        )

    candidate_df = pd.DataFrame(
        candidate_rows
    )

    if not candidate_df.empty:
        active_case_counts = (
            candidate_df["case_id"]
            .value_counts()
        )

        duplicated_active_cases = set(
            active_case_counts.loc[
                active_case_counts > 1
            ].index.tolist()
        )

        if duplicated_active_cases:
            retained_rows: list[
                dict[str, Any]
            ] = []

            for _, row in candidate_df.iterrows():
                if (
                    row["case_id"]
                    in duplicated_active_cases
                ):
                    exclusions.append(
                        build_exclusion_row(
                            case_id=row[
                                "case_id"
                            ],
                            resolution_id=row[
                                "resolution_id"
                            ],
                            reason_code=(
                                "MULTIPLES_ETIQUETAS_ACTIVAS"
                            ),
                            reason_detail=(
                                "El caso posee más de "
                                "una etiqueta activa."
                            ),
                        )
                    )
                else:
                    retained_rows.append(
                        row.to_dict()
                    )

            candidate_df = pd.DataFrame(
                retained_rows
            )

    output_columns = [
        "case_id",
        "resolution_id",
        "feature_snapshot_at",
        "closed_at",
        *selected_features,
        "investigation_outcome",
        "outcome_label",
        "feedback_source",
    ]

    dataset = candidate_df.reindex(
        columns=output_columns
    )

    if not dataset.empty:
        dataset = dataset.sort_values(
            by=[
                "closed_at",
                "case_id",
            ]
        ).reset_index(
            drop=True
        )

    exclusions_df = pd.DataFrame(
        exclusions,
        columns=[
            "case_id",
            "resolution_id",
            "exclusion_reason_code",
            "exclusion_reason_detail",
        ],
    )

    total_feedback_records = int(
        len(feedback_df)
    )

    resolved_feedback_records = int(
        len(feedback)
    )

    included_records = int(
        len(dataset)
    )

    excluded_records = int(
        len(exclusions_df)
    )

    positive_labels = (
        int(
            dataset[
                "outcome_label"
            ].eq(1).sum()
        )
        if not dataset.empty
        else 0
    )

    negative_labels = (
        int(
            dataset[
                "outcome_label"
            ].eq(0).sum()
        )
        if not dataset.empty
        else 0
    )

    exclusion_counts = (
        exclusions_df[
            "exclusion_reason_code"
        ]
        .value_counts()
        .to_dict()
        if not exclusions_df.empty
        else {}
    )

    report = {
        "total_feedback_records_received": (
            total_feedback_records
        ),
        "feedback_records_after_lifecycle_resolution": (
            resolved_feedback_records
        ),
        "included_training_records": (
            included_records
        ),
        "excluded_records": (
            excluded_records
        ),
        "positive_labels": (
            positive_labels
        ),
        "negative_labels": (
            negative_labels
        ),
        "positive_rate": (
            round(
                positive_labels
                / included_records,
                4,
            )
            if included_records > 0
            else 0.0
        ),
        "selected_feature_columns": (
            selected_features
        ),
        "exclusion_counts": (
            {
                str(key): int(value)
                for key, value
                in exclusion_counts.items()
            }
        ),
    }

    return TrainingDatasetResult(
        dataset=dataset,
        exclusions=exclusions_df,
        report=report,
    )