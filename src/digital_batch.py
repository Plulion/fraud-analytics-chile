from __future__ import annotations

from typing import Any

import pandas as pd

from src.digital_fraud import (
    assess_digital_event,
)


REQUIRED_DIGITAL_COLUMNS = {
    "event_id",
    "case_id",
    "customer_id",
    "event_timestamp",
    "channel",
    "region",
    "new_device",
    "unusual_ip",
    "geolocation_mismatch",
    "failed_mfa_attempts",
    "recent_password_reset",
    "new_beneficiary",
    "amount_spike_ratio",
    "rapid_transaction_count_10m",
}


BINARY_DIGITAL_COLUMNS = [
    "new_device",
    "unusual_ip",
    "geolocation_mismatch",
    "recent_password_reset",
    "new_beneficiary",
]


NUMERIC_DIGITAL_COLUMNS = [
    "failed_mfa_attempts",
    "amount_spike_ratio",
    "rapid_transaction_count_10m",
]


def parse_event_timestamps(
    series: pd.Series,
    *,
    errors: str,
) -> pd.Series:
    """
    Convierte event_timestamp usando formatos ISO 8601
    o formatos mixtos compatibles con pandas.

    format="mixed" evita que pandas intente inferir un
    único formato para toda la serie y elimina la
    advertencia asociada a esa inferencia.
    """

    return pd.to_datetime(
        series,
        errors=errors,
        format="mixed",
    )


def validate_digital_events(
    df: pd.DataFrame,
) -> None:
    """
    Comprueba que el archivo de eventos digitales
    tenga una estructura válida.
    """

    missing_columns = (
        REQUIRED_DIGITAL_COLUMNS.difference(
            df.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "El archivo de eventos digitales no "
            "contiene las columnas requeridas: "
            f"{sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError(
            "El archivo de eventos digitales "
            "no contiene registros."
        )

    identifier_columns = [
        "event_id",
        "case_id",
        "customer_id",
    ]

    for column in identifier_columns:
        if df[column].isna().any():
            raise ValueError(
                f"La columna {column!r} "
                "contiene valores vacíos."
            )

        empty_values = (
            df[column]
            .astype(str)
            .str.strip()
            .eq("")
        )

        if empty_values.any():
            raise ValueError(
                f"La columna {column!r} "
                "contiene textos vacíos."
            )

    if df["event_id"].duplicated().any():
        duplicated_events = (
            df.loc[
                df["event_id"].duplicated(
                    keep=False
                ),
                "event_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            "Existen event_id duplicados: "
            f"{sorted(duplicated_events)}"
        )

    parsed_timestamps = (
        parse_event_timestamps(
            df["event_timestamp"],
            errors="coerce",
        )
    )

    if parsed_timestamps.isna().any():
        invalid_events = (
            df.loc[
                parsed_timestamps.isna(),
                "event_id",
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "Existen fechas inválidas en "
            "'event_timestamp'. "
            f"Eventos afectados: {invalid_events}"
        )

    for column in BINARY_DIGITAL_COLUMNS:
        for value in df[column]:
            if pd.isna(value):
                continue

            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"La columna {column!r} "
                    "solo puede contener 0, 1 "
                    "o un valor vacío."
                ) from exc

            if numeric_value not in {
                0.0,
                1.0,
            }:
                raise ValueError(
                    f"La columna {column!r} "
                    "solo puede contener 0, 1 "
                    "o un valor vacío."
                )

    for column in NUMERIC_DIGITAL_COLUMNS:
        numeric_values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        invalid_values = (
            df[column].notna()
            & numeric_values.isna()
        )

        if invalid_values.any():
            affected_events = (
                df.loc[
                    invalid_values,
                    "event_id",
                ]
                .astype(str)
                .tolist()
            )

            raise ValueError(
                f"La columna {column!r} debe "
                "contener valores numéricos. "
                f"Eventos afectados: "
                f"{affected_events}"
            )

        negative_values = (
            numeric_values.notna()
            & (numeric_values < 0)
        )

        if negative_values.any():
            affected_events = (
                df.loc[
                    negative_values,
                    "event_id",
                ]
                .astype(str)
                .tolist()
            )

            raise ValueError(
                f"La columna {column!r} no "
                "puede contener valores negativos. "
                f"Eventos afectados: "
                f"{affected_events}"
            )


def process_digital_events(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ejecuta el motor especializado para cada
    evento del DataFrame.
    """

    validate_digital_events(
        df
    )

    result_df = df.copy()

    result_df["event_timestamp"] = (
        parse_event_timestamps(
            result_df["event_timestamp"],
            errors="raise",
        )
    )

    records: list[dict[str, Any]] = (
        result_df.to_dict(
            orient="records"
        )
    )

    assessments = [
        assess_digital_event(
            record
        )
        for record in records
    ]

    result_df["digital_device_score"] = [
        assessment.device_score
        for assessment in assessments
    ]

    result_df["digital_network_score"] = [
        assessment.network_score
        for assessment in assessments
    ]

    result_df[
        "digital_authentication_score"
    ] = [
        assessment.authentication_score
        for assessment in assessments
    ]

    result_df[
        "digital_transaction_score"
    ] = [
        assessment.transaction_score
        for assessment in assessments
    ]

    result_df["digital_risk_score"] = [
        assessment.digital_risk_score
        for assessment in assessments
    ]

    result_df[
        "digital_assessment_coverage"
    ] = [
        assessment.assessment_coverage
        for assessment in assessments
    ]

    result_df["digital_alert_level"] = [
        assessment.alert_level
        for assessment in assessments
    ]

    result_df[
        "digital_recommended_action"
    ] = [
        assessment.recommended_action
        for assessment in assessments
    ]

    result_df[
        "digital_triggered_signals"
    ] = [
        " | ".join(
            assessment.triggered_signals
        )
        for assessment in assessments
    ]

    result_df[
        "digital_triggered_signal_count"
    ] = [
        len(
            assessment.triggered_signals
        )
        for assessment in assessments
    ]

    result_df = result_df.sort_values(
        by=[
            "digital_risk_score",
            "event_timestamp",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(
        drop=True
    )

    return result_df