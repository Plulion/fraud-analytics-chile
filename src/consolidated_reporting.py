"""
Consolidated fraud reporting and visualization.

This module transforms the one-row-per-case consolidated analytical view into
summary metrics and static charts suitable for operational review.

Business interpretation
-----------------------
The reports describe portfolio composition and analytical signals. They do not
confirm fraud, replace case investigation, or authorize operational actions.

Outputs
-------
- JSON-compatible summary metrics;
- case counts by consolidated alert level;
- comparison of general risk score and maximum observed digital risk score.

Governance principles
---------------------
- Required columns are validated before reporting.
- Empty consolidated views are rejected.
- Summary values are explicitly converted to standard Python types.
- Charts use deterministic category ordering.
- Missing digital risk is represented as zero only for visualization and may
  indicate that no digital events were available.
- Output directories are created without altering the source DataFrame.

Current limitations
-------------------
This educational reporting layer does not yet include interactive dashboards,
role-based access, report versioning, data masking, confidence intervals,
drill-down controls, or automated distribution.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_CONSOLIDATED_COLUMNS = {
    "case_id",
    "risk_score",
    "alert_level",
    "estimated_loss_clp",
    "digital_event_count",
    "digital_risk_score_max",
    "digital_case_alert_level",
    "has_digital_events",
    "consolidated_alert_level",
    "consolidated_data_status",
}


def validate_consolidated_data(
    df: pd.DataFrame,
) -> None:
    """
    Validate the consolidated case view required by reporting functions.

    Raises:
        ValueError:
            If required columns are missing or the DataFrame is empty.
    """
    missing_columns = (
        REQUIRED_CONSOLIDATED_COLUMNS
        .difference(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "No se puede generar el informe "
            "consolidado. Faltan las columnas: "
            f"{sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError(
            "La vista consolidada está vacía."
        )


def build_consolidated_summary(
    df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Build JSON-compatible metrics from the consolidated case view.

    Returns:
        A dictionary containing case counts, digital-event coverage, alert
        distribution, data-review volume, and estimated loss for cases with
        digital activity.
    """

    validate_consolidated_data(df)

    # Preserve the complete alert distribution for portfolio-level review.
    alert_counts = (
        df["consolidated_alert_level"]
        .value_counts()
        .to_dict()
    )

    # Estimated loss for digitally observed cases is separated from the
    # overall portfolio to avoid implying digital coverage where none exists.
    cases_with_events = df[
        df["has_digital_events"]
    ]

    summary: dict[str, Any] = {
        "total_cases": int(len(df)),
        "cases_with_digital_events": int(
            df["has_digital_events"].sum()
        ),
        "cases_without_digital_events": int(
            (
                ~df["has_digital_events"]
            ).sum()
        ),
        "total_digital_events": int(
            df["digital_event_count"].sum()
        ),
        "critical_consolidated_cases": int(
            (
                df[
                    "consolidated_alert_level"
                ]
                == "CRITICO"
            ).sum()
        ),
        "high_consolidated_cases": int(
            (
                df[
                    "consolidated_alert_level"
                ]
                == "ALTO"
            ).sum()
        ),
        "cases_requiring_data_review": int(
            (
                df[
                    "consolidated_data_status"
                ]
                == "REQUIERE_REVISION_DE_DATOS"
            ).sum()
        ),
        "estimated_loss_cases_with_digital_events_clp": int(
            round(
                cases_with_events[
                    "estimated_loss_clp"
                ].sum()
            )
        ),
        "cases_by_consolidated_alert": {
            str(alert): int(count)
            for alert, count
            in alert_counts.items()
        },
    }

    return summary


def save_consolidated_summary_json(
    summary: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Persist a consolidated summary as UTF-8 formatted JSON.

    Parent directories are created when necessary.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=4,
        )


def generate_consolidated_alert_chart(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Generate a bar chart of cases by consolidated alert level.

    Categories are reindexed to a fixed business order so absent alert levels
    remain visible with a zero count.
    """

    validate_consolidated_data(df)

    # Fixed ordering keeps charts comparable across reporting periods even
    # when one or more alert categories are absent.
    alert_order = [
        "CRITICO",
        "ALTO",
        "MEDIO",
        "BAJO",
        "INCOMPLETO",
        "SIN_DATOS",
    ]

    alert_counts = (
        df["consolidated_alert_level"]
        .value_counts()
        .reindex(
            alert_order,
            fill_value=0,
        )
    )

    figure, axis = plt.subplots(
        figsize=(9, 5),
    )

    alert_counts.plot(
        kind="bar",
        ax=axis,
    )

    axis.set_title(
        "Casos por nivel de alerta consolidada"
    )
    axis.set_xlabel(
        "Nivel de alerta"
    )
    axis.set_ylabel(
        "Cantidad de casos"
    )
    axis.tick_params(
        axis="x",
        rotation=0,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(figure)


def generate_risk_comparison_chart(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Compare general risk with maximum observed digital risk by case.

    A displayed digital value of zero may mean that no digital events were
    available. It must not automatically be interpreted as zero digital risk.
    """

    validate_consolidated_data(df)

    comparison = df[
        [
            "case_id",
            "risk_score",
            "digital_risk_score_max",
        ]
    ].copy()

    # Zero is a plotting default for missing digital scores. It does not prove
    # that a digital assessment found no risk.
    comparison[
        "digital_risk_score_max"
    ] = (
        comparison[
            "digital_risk_score_max"
        ]
        .fillna(0)
    )

    comparison = comparison.set_index(
        "case_id"
    )

    comparison = comparison.rename(
        columns={
            "risk_score": "Riesgo general",
            "digital_risk_score_max": (
                "Riesgo digital máximo"
            ),
        }
    )

    figure, axis = plt.subplots(
        figsize=(13, 6),
    )

    comparison.plot(
        kind="bar",
        ax=axis,
    )

    axis.set_title(
        "Riesgo general frente a riesgo digital"
    )
    axis.set_xlabel(
        "Caso"
    )
    axis.set_ylabel(
        "Puntaje"
    )
    axis.set_ylim(
        0,
        105,
    )
    axis.tick_params(
        axis="x",
        rotation=45,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(figure)


def generate_consolidated_reports(
    df: pd.DataFrame,
    output_directory: Path,
) -> list[Path]:
    """
    Generate all static charts for the consolidated reporting package.

    Returns:
        Paths of the generated chart files in deterministic order.
    """
    validate_consolidated_data(df)

    # Report generation writes only to the requested output directory and
    # never mutates the consolidated input DataFrame.
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    alert_chart_path = (
        output_directory
        / "casos_por_alerta_consolidada.png"
    )

    comparison_chart_path = (
        output_directory
        / "riesgo_general_vs_digital.png"
    )

    generate_consolidated_alert_chart(
        df,
        alert_chart_path,
    )

    generate_risk_comparison_chart(
        df,
        comparison_chart_path,
    )

    return [
        alert_chart_path,
        comparison_chart_path,
    ]