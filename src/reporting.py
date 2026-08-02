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
    Comprueba que estén disponibles las columnas
    necesarias para crear los informes.
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
    Construye las métricas generales y las métricas
    agrupadas por categoría de fraude.
    """

    validate_reporting_data(df)

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
    Construye una tabla con las principales métricas
    agrupadas por categoría de fraude.
    """

    validate_reporting_data(df)

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
    Guarda las métricas generales en JSON.
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
    Guarda las métricas agrupadas por categoría
    en un archivo CSV.
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
    Formatea montos como millones de pesos chilenos.
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
    Genera un gráfico de casos por nivel de alerta.
    """

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
    Genera un gráfico de pérdidas por sector.
    """

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
    Genera un gráfico con las categorías
    de demora de detección.
    """

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
    Genera un gráfico con la cantidad de casos
    agrupados por categoría de fraude.
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
    Genera un gráfico con la pérdida estimada
    agrupada por categoría de fraude.
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
    Genera todos los gráficos y devuelve
    las rutas de los archivos creados.
    """

    validate_reporting_data(df)

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