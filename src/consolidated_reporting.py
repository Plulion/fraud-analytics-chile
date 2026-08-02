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
    Construye las métricas de la vista consolidada.
    """

    validate_consolidated_data(df)

    alert_counts = (
        df["consolidated_alert_level"]
        .value_counts()
        .to_dict()
    )

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
    Genera un gráfico de casos según la alerta
    consolidada.
    """

    validate_consolidated_data(df)

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
    Compara el puntaje general con el máximo
    puntaje digital observado en cada caso.

    Un valor digital igual a cero en el gráfico puede
    significar que no había eventos digitales.
    """

    validate_consolidated_data(df)

    comparison = df[
        [
            "case_id",
            "risk_score",
            "digital_risk_score_max",
        ]
    ].copy()

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
    validate_consolidated_data(df)

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