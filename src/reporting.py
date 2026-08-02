"""
General fraud reporting, summaries, and static visualizations.

This module transforms the validated case-level assessment dataset into
portfolio metrics, category summaries, CSV/JSON exports, and static charts.

Business interpretation
-----------------------
The reporting layer describes the case portfolio and its analytical signals.
It does not confirm fraud, authorize enforcement actions, or replace case-level
investigation.

Main outputs
------------
- overall portfolio metrics;
- metrics grouped by fraud category;
- JSON summary files;
- CSV category summaries;
- alert, sector-loss, detection-delay, category-count, and category-loss charts.

Governance principles
---------------------
- Required columns are validated before reporting.
- Empty input datasets are rejected.
- Numeric values are converted to standard Python types for JSON compatibility.
- Category and alert charts use stable ordering where business meaning requires
  it.
- Monetary values are presented in Chilean pesos and formatted in millions only
  for visualization.
- Report generation does not mutate the source DataFrame.

Current limitations
-------------------
This educational reporting layer does not yet provide interactive filtering,
role-based access, masking of sensitive attributes, report versioning,
confidence intervals, lineage metadata, automated distribution, or scheduled
refreshes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter


REQUIRED_REPORT_COLUMNS = {
    "case_id",
    "sector",
    "fraud_category",
    "fraud_type_name",
    "risk_score",
    "assessment_coverage",
    "alert_level",
    "estimated_loss_clp",
    "detection_delay_days",
    "detection_delay_level",
}


def validate_reporting_data(
    df: pd.DataFrame,
) -> None:
    """
    Validate the case-level dataset required by reporting functions.

    Raises:
        ValueError:
            If required columns are missing or the DataFrame is empty.
    """

    missing_columns = (
        REQUIRED_REPORT_COLUMNS.difference(
            df.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "No se pueden generar los informes. "
            "Faltan las columnas: "
            f"{sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError(
            "No se pueden generar informes "
            "con un DataFrame vacío."
        )


def build_summary_metrics(
    df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Build overall and grouped fraud portfolio metrics.

    The result is JSON-compatible and includes case volumes, risk counts,
    detection-delay counts, estimated losses, average analytical measures,
    category distributions, fraud-type distributions, and leading categories.

    Returns:
        A dictionary containing portfolio-level summary metrics.
    """

    validate_reporting_data(df)

    # Count categories independently from monetary exposure so volume and
    # financial impact can be interpreted as separate dimensions.
    category_counts = (
        df["fraud_category"]
        .value_counts()
        .sort_index()
    )

    loss_by_category = (
        df.groupby(
            "fraud_category"
        )["estimated_loss_clp"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    fraud_type_counts = (
        df["fraud_type_name"]
        .value_counts()
        .sort_index()
    )

    cases_by_category = {
        str(category): int(count)
        for category, count
        in category_counts.items()
    }

    losses_by_category = {
        str(category): int(
            round(loss)
        )
        for category, loss
        in loss_by_category.items()
    }

    cases_by_fraud_type = {
        str(fraud_type): int(count)
        for fraud_type, count
        in fraud_type_counts.items()
    }

    # Leading categories are descriptive portfolio indicators only; they do
    # not establish causal importance or confirmed fraud prevalence.
    top_category_by_cases = (
        str(category_counts.idxmax())
        if not category_counts.empty
        else None
    )

    top_category_by_loss = (
        str(loss_by_category.idxmax())
        if not loss_by_category.empty
        else None
    )

    metrics: dict[str, Any] = {
        "total_cases": int(len(df)),
        "critical_cases": int(
            (
                df["alert_level"]
                == "CRITICO"
            ).sum()
        ),
        "high_risk_cases": int(
            (
                df["alert_level"]
                == "ALTO"
            ).sum()
        ),
        "incomplete_cases": int(
            (
                df["alert_level"]
                == "INCOMPLETO"
            ).sum()
        ),
        "late_detection_cases": int(
            df["detection_delay_level"]
            .isin(
                [
                    "TARDIA",
                    "MUY_TARDIA",
                ]
            )
            .sum()
        ),
        "estimated_loss_total_clp": int(
            round(
                df[
                    "estimated_loss_clp"
                ].sum()
            )
        ),
        "average_risk_score": round(
            float(
                df["risk_score"].mean()
            ),
            2,
        ),
        "average_assessment_coverage": round(
            float(
                df[
                    "assessment_coverage"
                ].mean()
            ),
            2,
        ),
        "average_detection_delay_days": round(
            float(
                df[
                    "detection_delay_days"
                ].mean()
            ),
            2,
        ),
        "cases_by_fraud_category": (
            cases_by_category
        ),
        "estimated_loss_by_fraud_category_clp": (
            losses_by_category
        ),
        "cases_by_fraud_type": (
            cases_by_fraud_type
        ),
        "top_fraud_category_by_cases": (
            top_category_by_cases
        ),
        "top_fraud_category_by_loss": (
            top_category_by_loss
        ),
    }

    return metrics


def build_category_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a one-row-per-fraud-category summary table.

    The output includes case counts, alert counts, late-detection counts,
    estimated losses, and average risk, coverage, and detection delay.

    Returns:
        A DataFrame sorted by total estimated loss and then case volume.
    """

    validate_reporting_data(df)

    # Aggregate at fraud-category level while preserving separate counts,
    # exposure, risk, coverage, and detection-timing measures.
    summary = (
        df.groupby(
            "fraud_category",
            as_index=False,
        )
        .agg(
            total_cases=(
                "case_id",
                "size",
            ),
            critical_cases=(
                "alert_level",
                lambda values: int(
                    (
                        values
                        == "CRITICO"
                    ).sum()
                ),
            ),
            high_risk_cases=(
                "alert_level",
                lambda values: int(
                    (
                        values
                        == "ALTO"
                    ).sum()
                ),
            ),
            incomplete_cases=(
                "alert_level",
                lambda values: int(
                    (
                        values
                        == "INCOMPLETO"
                    ).sum()
                ),
            ),
            late_detection_cases=(
                "detection_delay_level",
                lambda values: int(
                    values.isin(
                        [
                            "TARDIA",
                            "MUY_TARDIA",
                        ]
                    ).sum()
                ),
            ),
            estimated_loss_total_clp=(
                "estimated_loss_clp",
                "sum",
            ),
            average_risk_score=(
                "risk_score",
                "mean",
            ),
            average_assessment_coverage=(
                "assessment_coverage",
                "mean",
            ),
            average_detection_delay_days=(
                "detection_delay_days",
                "mean",
            ),
        )
    )

    summary[
        "estimated_loss_total_clp"
    ] = (
        summary[
            "estimated_loss_total_clp"
        ]
        .round()
        .astype(int)
    )

    columns_to_round = [
        "average_risk_score",
        "average_assessment_coverage",
        "average_detection_delay_days",
    ]

    summary[
        columns_to_round
    ] = summary[
        columns_to_round
    ].round(2)

    # Sort by financial exposure first and case volume second to support
    # operational review without changing underlying risk classifications.
    summary = summary.sort_values(
        by=[
            "estimated_loss_total_clp",
            "total_cases",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )

    return summary


def save_summary_json(
    metrics: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Save portfolio summary metrics as formatted UTF-8 JSON.

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
            metrics,
            file,
            ensure_ascii=False,
            indent=4,
        )


def save_category_summary_csv(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Save the fraud-category summary as a UTF-8 CSV file.

    UTF-8 with BOM is used to improve compatibility with common spreadsheet
    applications.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )


def _format_clp_millions(
    value: float,
    position: int,
) -> str:
    """
    Format a numeric CLP axis value as rounded millions of pesos.

    ``position`` is required by Matplotlib's formatter protocol and is not used
    by the business calculation.
    """

    del position

    return (
        f"${value / 1_000_000:.0f}M"
    )


def generate_alert_level_chart(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Generate a bar chart of case counts by alert level.

    A fixed alert order keeps visual comparisons consistent across runs.
    """

    # Fixed business ordering keeps missing categories visible with zero
    # counts and preserves comparability across reporting periods.
    alert_order = [
        "CRITICO",
        "ALTO",
        "MEDIO",
        "BAJO",
        "INCOMPLETO",
    ]

    alert_counts = (
        df["alert_level"]
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
        "Casos por nivel de alerta"
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


def generate_sector_loss_chart(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Generate a horizontal bar chart of estimated loss by sector.

    Monetary values are aggregated in CLP and displayed in rounded millions.
    """

    # Sector totals are descriptive sums of estimated loss, not realized-loss
    # accounting values.
    loss_by_sector = (
        df.groupby(
            "sector"
        )["estimated_loss_clp"]
        .sum()
        .sort_values()
    )

    figure, axis = plt.subplots(
        figsize=(10, 7),
    )

    loss_by_sector.plot(
        kind="barh",
        ax=axis,
    )

    axis.set_title(
        "Pérdida estimada por sector"
    )
    axis.set_xlabel(
        "Pérdida estimada en CLP"
    )
    axis.set_ylabel(
        "Sector"
    )

    axis.xaxis.set_major_formatter(
        FuncFormatter(
            _format_clp_millions
        )
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(figure)


def generate_detection_delay_chart(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Generate a bar chart of cases by detection-delay classification.

    A fixed order preserves the intended operational progression from early to
    very late detection.
    """

    # Preserve the operational progression defined by case_metrics.py.
    delay_order = [
        "TEMPRANA",
        "OPORTUNA",
        "TARDIA",
        "MUY_TARDIA",
    ]

    delay_counts = (
        df[
            "detection_delay_level"
        ]
        .value_counts()
        .reindex(
            delay_order,
            fill_value=0,
        )
    )

    figure, axis = plt.subplots(
        figsize=(9, 5),
    )

    delay_counts.plot(
        kind="bar",
        ax=axis,
    )

    axis.set_title(
        "Casos por demora de detección"
    )
    axis.set_xlabel(
        "Clasificación de la demora"
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


def generate_fraud_category_cases_chart(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Generate a horizontal bar chart of case volume by fraud category.
    """

    cases_by_category = (
        df["fraud_category"]
        .value_counts()
        .sort_values()
    )

    figure, axis = plt.subplots(
        figsize=(10, 6),
    )

    cases_by_category.plot(
        kind="barh",
        ax=axis,
    )

    axis.set_title(
        "Casos por categoría de fraude"
    )
    axis.set_xlabel(
        "Cantidad de casos"
    )
    axis.set_ylabel(
        "Categoría de fraude"
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(figure)


def generate_fraud_category_loss_chart(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Generate a horizontal bar chart of estimated loss by fraud category.

    Monetary values are aggregated in CLP and displayed in rounded millions.
    """

    loss_by_category = (
        df.groupby(
            "fraud_category"
        )["estimated_loss_clp"]
        .sum()
        .sort_values()
    )

    figure, axis = plt.subplots(
        figsize=(10, 6),
    )

    loss_by_category.plot(
        kind="barh",
        ax=axis,
    )

    axis.set_title(
        "Pérdida estimada por categoría de fraude"
    )
    axis.set_xlabel(
        "Pérdida estimada en CLP"
    )
    axis.set_ylabel(
        "Categoría de fraude"
    )

    axis.xaxis.set_major_formatter(
        FuncFormatter(
            _format_clp_millions
        )
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(figure)


def generate_reports(
    df: pd.DataFrame,
    output_directory: Path,
) -> list[Path]:
    """
    Generate the complete static chart package.

    Args:
        df:
            Validated case-level fraud assessment dataset.
        output_directory:
            Directory where chart files will be created.

    Returns:
        Paths of all generated image files in deterministic order.
    """

    validate_reporting_data(df)

    # Create only the requested reporting directory. The source DataFrame is
    # never modified by report generation.
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated_files = [
        (
            output_directory
            / "alertas_por_nivel.png"
        ),
        (
            output_directory
            / "perdida_por_sector.png"
        ),
        (
            output_directory
            / "demora_deteccion.png"
        ),
        (
            output_directory
            / "casos_por_categoria_fraude.png"
        ),
        (
            output_directory
            / "perdida_por_categoria_fraude.png"
        ),
    ]

    generate_alert_level_chart(
        df,
        generated_files[0],
    )

    generate_sector_loss_chart(
        df,
        generated_files[1],
    )

    generate_detection_delay_chart(
        df,
        generated_files[2],
    )

    generate_fraud_category_cases_chart(
        df,
        generated_files[3],
    )

    generate_fraud_category_loss_chart(
        df,
        generated_files[4],
    )

    return generated_files