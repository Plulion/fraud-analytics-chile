"""
Case-level consolidation of general and digital fraud assessments.

This module combines one general case assessment with zero or more digital
event assessments. Digital events are first aggregated to one row per case,
then merged with the general assessment to produce a consolidated operational
view.

Business interpretation
-----------------------
- General and digital engines remain separate sources.
- Alert levels are combined by selecting the highest observed risk category.
- Scores are not averaged across engines.
- ``INCOMPLETO`` represents missing or insufficient information, not a risk
  level above ``CRITICO``.
- Missing digital events are preserved explicitly as ``SIN_EVENTOS``.
- Consolidated actions are recommendations only. They do not block activity,
  confirm fraud, or replace human investigation.

Governance principles
---------------------
- General cases must be unique by ``case_id``.
- Digital events must be unique by ``event_id``.
- Digital timestamps and numeric values are validated before aggregation.
- Event counts, maximum and mean scores, coverage, signals, and actions remain
  available for later audit and prioritization.
- The merge preserves all general cases, including those without digital data.

Current limitations
-------------------
The consolidation logic is educational and rule-based. Production use would
require formal source lineage, schema contracts, late-arriving-event handling,
deduplication policies, calibration, and monitored data-quality thresholds.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd


GENERAL_REQUIRED_COLUMNS = {
    "case_id",
    "sector",
    "region",
    "fraud_type_name",
    "fraud_category",
    "risk_score",
    "assessment_coverage",
    "alert_level",
    "decision_status",
    "estimated_loss_clp",
    "detection_delay_days",
    "detection_delay_level",
}


DIGITAL_REQUIRED_COLUMNS = {
    "event_id",
    "case_id",
    "event_timestamp",
    "digital_risk_score",
    "digital_assessment_coverage",
    "digital_alert_level",
    "digital_recommended_action",
    "digital_triggered_signals",
}


# Ordering used only for categorical alert consolidation.
RISK_ALERT_RANK = {
    "BAJO": 1,
    "MEDIO": 2,
    "ALTO": 3,
    "CRITICO": 4,
}


def _normalize_text(
    value: Any,
) -> str:
    """
    Convert a scalar value into normalized uppercase text.

    Missing values become an empty string. Other values are converted to text,
    stripped of surrounding whitespace, and uppercased.
    """

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip().upper()


def highest_risk_alert(
    values: Iterable[Any],
) -> str:
    """
    Return the highest valid risk alert from an iterable.

    ``INCOMPLETO`` represents insufficient information rather than a higher
    risk category. It is returned only when no valid risk level is available.

    Returns:
        One of ``CRITICO``, ``ALTO``, ``MEDIO``, ``BAJO``, ``INCOMPLETO``, or
        ``SIN_DATOS``.

    Raises:
        ValueError: If an unknown non-empty alert value is encountered.
    """

    risk_alerts: list[str] = []
    found_incomplete = False

    for value in values:
        normalized = _normalize_text(value)

        if not normalized:
            continue

        if normalized == "INCOMPLETO":
            found_incomplete = True
            continue

        if normalized in {
            "SIN_DATOS",
            "SIN_EVENTOS",
            "SIN_EVENTOS_DIGITALES",
        }:
            continue

        if normalized not in RISK_ALERT_RANK:
            raise ValueError(
                "Nivel de alerta desconocido: "
                f"{value!r}"
            )

        risk_alerts.append(normalized)

    if risk_alerts:
        return max(
            risk_alerts,
            key=lambda alert: (
                RISK_ALERT_RANK[alert]
            ),
        )

    if found_incomplete:
        return "INCOMPLETO"

    return "SIN_DATOS"


def _join_unique_values(
    values: Iterable[Any],
    *,
    split_pipe: bool = False,
) -> str:
    """
    Join unique non-empty values into a pipe-delimited string.

    When ``split_pipe`` is true, values already containing pipe-delimited items
    are split before deduplication.
    """

    unique_values: set[str] = set()

    for value in values:
        if value is None:
            continue

        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass

        text = str(value).strip()

        if not text:
            continue

        parts = (
            text.split("|")
            if split_pipe
            else [text]
        )

        for part in parts:
            normalized_part = part.strip()

            if normalized_part:
                unique_values.add(
                    normalized_part
                )

    return " | ".join(
        sorted(unique_values)
    )


def validate_general_cases(
    df: pd.DataFrame,
) -> None:
    """
    Validate the one-row-per-case general assessment table.

    Raises:
        ValueError:
            If required columns are missing, the table is empty, or case
            identifiers are null or duplicated.
    """

    missing_columns = (
        GENERAL_REQUIRED_COLUMNS.difference(
            df.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "El archivo general no contiene "
            "las columnas requeridas: "
            f"{sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError(
            "El archivo de casos generales "
            "está vacío."
        )

    if df["case_id"].isna().any():
        raise ValueError(
            "Existen casos generales sin case_id."
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
            "El archivo general contiene "
            "case_id duplicados: "
            f"{sorted(duplicated_cases)}"
        )


def validate_digital_assessments(
    df: pd.DataFrame,
) -> None:
    """
    Validate the event-level digital assessment table.

    Validation covers required columns, unique event identifiers, case
    identifiers, valid timestamps, and numeric values between 0 and 100.

    Raises:
        ValueError:
            If the digital source is empty, malformed, duplicated, or contains
            invalid dates or numeric values.
    """

    missing_columns = (
        DIGITAL_REQUIRED_COLUMNS.difference(
            df.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "El archivo digital no contiene "
            "las columnas requeridas: "
            f"{sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError(
            "El archivo de evaluaciones digitales "
            "está vacío."
        )

    if df["event_id"].isna().any():
        raise ValueError(
            "Existen eventos sin event_id."
        )

    if df["case_id"].isna().any():
        raise ValueError(
            "Existen eventos digitales sin case_id."
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
            "El archivo digital contiene "
            "event_id duplicados: "
            f"{sorted(duplicated_events)}"
        )

    parsed_timestamps = pd.to_datetime(
        df["event_timestamp"],
        errors="coerce",
    )

    if parsed_timestamps.isna().any():
        affected_events = (
            df.loc[
                parsed_timestamps.isna(),
                "event_id",
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "Existen fechas digitales inválidas. "
            f"Eventos afectados: {affected_events}"
        )

    numeric_columns = [
        "digital_risk_score",
        "digital_assessment_coverage",
    ]

    for column in numeric_columns:
        numeric_values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if numeric_values.isna().any():
            affected_events = (
                df.loc[
                    numeric_values.isna(),
                    "event_id",
                ]
                .astype(str)
                .tolist()
            )

            raise ValueError(
                f"La columna {column!r} contiene "
                "valores no numéricos. "
                f"Eventos afectados: "
                f"{affected_events}"
            )

        outside_range = (
            (numeric_values < 0)
            | (numeric_values > 100)
        )

        if outside_range.any():
            affected_events = (
                df.loc[
                    outside_range,
                    "event_id",
                ]
                .astype(str)
                .tolist()
            )

            raise ValueError(
                f"La columna {column!r} debe estar "
                "entre 0 y 100. "
                f"Eventos afectados: "
                f"{affected_events}"
            )


def aggregate_digital_events_by_case(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate multiple digital events into one row per case.

    The output preserves event volume, maximum and mean risk, mean coverage,
    alert counts, highest case-level alert, latest timestamp, recommended
    actions, and triggered signals.
    """

    validate_digital_assessments(df)

    working_df = df.copy()

    working_df["event_timestamp"] = (
        pd.to_datetime(
            working_df["event_timestamp"],
            errors="raise",
        )
    )

    working_df["digital_risk_score"] = (
        pd.to_numeric(
            working_df[
                "digital_risk_score"
            ],
            errors="raise",
        )
    )

    working_df[
        "digital_assessment_coverage"
    ] = pd.to_numeric(
        working_df[
            "digital_assessment_coverage"
        ],
        errors="raise",
    )

    rows: list[dict[str, Any]] = []

    # Preserve transparent aggregate components instead of collapsing
    # digital activity into a single opaque score.
    for case_id, group in working_df.groupby(
        "case_id",
        sort=True,
    ):
        alert_values = (
            group["digital_alert_level"]
        )

        latest_timestamp = (
            group["event_timestamp"].max()
        )

        row = {
            "case_id": str(case_id),
            "digital_event_count": int(
                len(group)
            ),
            "digital_risk_score_max": round(
                float(
                    group[
                        "digital_risk_score"
                    ].max()
                ),
                2,
            ),
            "digital_risk_score_mean": round(
                float(
                    group[
                        "digital_risk_score"
                    ].mean()
                ),
                2,
            ),
            "digital_assessment_coverage_mean": round(
                float(
                    group[
                        "digital_assessment_coverage"
                    ].mean()
                ),
                2,
            ),
            "digital_critical_event_count": int(
                (
                    alert_values
                    .astype(str)
                    .str.upper()
                    == "CRITICO"
                ).sum()
            ),
            "digital_high_event_count": int(
                (
                    alert_values
                    .astype(str)
                    .str.upper()
                    == "ALTO"
                ).sum()
            ),
            "digital_incomplete_event_count": int(
                (
                    alert_values
                    .astype(str)
                    .str.upper()
                    == "INCOMPLETO"
                ).sum()
            ),
            "digital_case_alert_level": (
                highest_risk_alert(
                    alert_values
                )
            ),
            "digital_latest_event_timestamp": (
                latest_timestamp.isoformat()
            ),
            "digital_recommended_actions": (
                _join_unique_values(
                    group[
                        "digital_recommended_action"
                    ]
                )
            ),
            "digital_triggered_signals": (
                _join_unique_values(
                    group[
                        "digital_triggered_signals"
                    ],
                    split_pipe=True,
                )
            ),
        }

        rows.append(row)

    return pd.DataFrame(rows)


def combine_case_alerts(
    general_alert: Any,
    digital_alert: Any,
) -> str:
    """
    Combine general and digital alerts without averaging scores.

    Returns:
        The highest observed categorical risk level across both engines.
    """

    return highest_risk_alert(
        [
            general_alert,
            digital_alert,
        ]
    )


def consolidated_action_for_alert(
    alert_level: str,
) -> str:
    """
    Map a consolidated alert level to an operational recommendation.

    The recommendation does not execute a block or determine that fraud
    occurred.

    Raises:
        ValueError: If the alert level has no configured action.
    """

    actions = {
        "CRITICO": "REVISION_PRIORITARIA",
        "ALTO": "REVISION_REFORZADA",
        "MEDIO": "REVISION_ESTANDAR",
        "BAJO": "REVISION_ESTANDAR",
        "INCOMPLETO": (
            "REQUIERE_MAS_ANTECEDENTES"
        ),
        "SIN_DATOS": (
            "SIN_ACCION_AUTOMATICA"
        ),
    }

    try:
        return actions[alert_level]
    except KeyError as exc:
        raise ValueError(
            "No existe una acción para la alerta: "
            f"{alert_level!r}"
        ) from exc


def consolidate_case_and_digital_data(
    general_df: pd.DataFrame,
    digital_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge general assessments with case-level digital summaries.

    All general cases are preserved through a left join. Cases without digital
    events receive explicit defaults instead of being removed.

    Args:
        general_df:
            One-row-per-case general fraud assessment table.
        digital_df:
            Event-level digital fraud assessment table.

    Returns:
        A consolidated one-row-per-case DataFrame sorted by consolidated alert
        severity and estimated loss.

    Raises:
        ValueError:
            If either source fails schema, identifier, timestamp, or numeric
            validation.
    """

    validate_general_cases(general_df)

    digital_summary = (
        aggregate_digital_events_by_case(
            digital_df
        )
    )

    result_df = general_df.copy()

    result_df["case_id"] = (
        result_df["case_id"]
        .astype(str)
    )

    digital_summary["case_id"] = (
        digital_summary["case_id"]
        .astype(str)
    )

    # A left join preserves every general case, including cases with no
    # observed digital events.
    result_df = result_df.merge(
        digital_summary,
        on="case_id",
        how="left",
        validate="one_to_one",
    )

    count_columns = [
        "digital_event_count",
        "digital_critical_event_count",
        "digital_high_event_count",
        "digital_incomplete_event_count",
    ]

    for column in count_columns:
        result_df[column] = (
            result_df[column]
            .fillna(0)
            .astype(int)
        )

    result_df["has_digital_events"] = (
        result_df["digital_event_count"] > 0
    )

    result_df[
        "digital_case_alert_level"
    ] = (
        result_df[
            "digital_case_alert_level"
        ]
        .fillna("SIN_EVENTOS")
    )

    text_columns_with_default = {
        "digital_latest_event_timestamp": "",
        "digital_recommended_actions": "",
        "digital_triggered_signals": "",
    }

    for column, default_value in (
        text_columns_with_default.items()
    ):
        result_df[column] = (
            result_df[column]
            .fillna(default_value)
        )

    # Consolidation selects the highest categorical risk signal; it does not
    # average scores produced by different engines.
    result_df[
        "consolidated_alert_level"
    ] = [
        combine_case_alerts(
            general_alert,
            digital_alert,
        )
        for general_alert, digital_alert
        in zip(
            result_df["alert_level"],
            result_df[
                "digital_case_alert_level"
            ],
        )
    ]

    # Data completeness is represented independently from risk severity.
    def determine_data_status(
        row: pd.Series,
    ) -> str:
        if not bool(
            row["has_digital_events"]
        ):
            return "SIN_EVENTOS_DIGITALES"

        general_incomplete = (
            str(
                row["alert_level"]
            ).upper()
            == "INCOMPLETO"
        )

        digital_incomplete = (
            int(
                row[
                    "digital_incomplete_event_count"
                ]
            )
            > 0
        )

        if (
            general_incomplete
            or digital_incomplete
        ):
            return (
                "REQUIERE_REVISION_DE_DATOS"
            )

        return "COMPLETO"

    result_df[
        "consolidated_data_status"
    ] = result_df.apply(
        determine_data_status,
        axis=1,
    )

    result_df[
        "consolidated_recommended_action"
    ] = [
        consolidated_action_for_alert(
            alert
        )
        for alert in result_df[
            "consolidated_alert_level"
        ]
    ]

    sort_rank = {
        "CRITICO": 4,
        "ALTO": 3,
        "MEDIO": 2,
        "BAJO": 1,
        "INCOMPLETO": 0,
        "SIN_DATOS": -1,
    }

    result_df[
        "_consolidated_priority_rank"
    ] = (
        result_df[
            "consolidated_alert_level"
        ]
        .map(sort_rank)
        .fillna(-1)
    )

    # Operational ordering surfaces higher consolidated risk and then larger
    # estimated financial exposure.
    result_df = result_df.sort_values(
        by=[
            "_consolidated_priority_rank",
            "estimated_loss_clp",
        ],
        ascending=[
            False,
            False,
        ],
    ).drop(
        columns=[
            "_consolidated_priority_rank"
        ]
    ).reset_index(
        drop=True
    )

    return result_df