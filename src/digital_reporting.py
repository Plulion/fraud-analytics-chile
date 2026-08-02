from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_DIGITAL_REPORT_COLUMNS = {
    "event_id",
    "customer_id",
    "digital_risk_score",
    "digital_assessment_coverage",
    "digital_alert_level",
    "digital_recommended_action",
    "digital_triggered_signals",
}


def validate_digital_report_data(
    df: pd.DataFrame,
) -> None:
    """
    Valida las columnas requeridas para generar
    métricas e informes digitales.
    """

    missing_columns = (
        REQUIRED_DIGITAL_REPORT_COLUMNS
        .difference(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "No se puede generar el informe "
            "digital. Faltan las columnas: "
            f"{sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError(
            "No se puede generar un informe "
            "digital con datos vacíos."
        )


def count_triggered_signals(
    df: pd.DataFrame,
) -> dict[str, int]:
    """
    Cuenta cuántas veces se activó cada señal.
    """

    validate_digital_report_data(df)

    signal_counter: Counter[str] = Counter()

    for value in df[
        "digital_triggered_signals"
    ].fillna(""):
        signals = [
            signal.strip()
            for signal in str(value).split("|")
            if signal.strip()
        ]

        signal_counter.update(signals)

    return dict(
        sorted(
            signal_counter.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )
    )


def build_digital_summary(
    df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Construye las métricas generales del motor
    de fraude digital.
    """

    validate_digital_report_data(df)

    alert_counts = (
        df["digital_alert_level"]
        .value_counts()
        .to_dict()
    )

    action_counts = (
        df[
            "digital_recommended_action"
        ]
        .value_counts()
        .to_dict()
    )

    unique_customers = int(
        df["customer_id"].nunique()
    )

    summary: dict[str, Any] = {
        "total_events": int(len(df)),
        "unique_customers": unique_customers,
        "critical_events": int(
            (
                df["digital_alert_level"]
                == "CRITICO"
            ).sum()
        ),
        "high_risk_events": int(
            (
                df["digital_alert_level"]
                == "ALTO"
            ).sum()
        ),
        "incomplete_events": int(
            (
                df["digital_alert_level"]
                == "INCOMPLETO"
            ).sum()
        ),
        "average_digital_risk_score": round(
            float(
                df[
                    "digital_risk_score"
                ].mean()
            ),
            2,
        ),
        "average_digital_coverage": round(
            float(
                df[
                    "digital_assessment_coverage"
                ].mean()
            ),
            2,
        ),
        "events_by_alert_level": {
            str(level): int(count)
            for level, count
            in alert_counts.items()
        },
        "events_by_recommended_action": {
            str(action): int(count)
            for action, count
            in action_counts.items()
        },
        "triggered_signals": (
            count_triggered_signals(df)
        ),
    }

    return summary


def save_digital_summary_json(
    summary: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Guarda el resumen digital en JSON.
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


def generate_digital_alert_chart(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Genera un gráfico de eventos por nivel
    de alerta digital.
    """

    validate_digital_report_data(df)

    alert_order = [
        "CRITICO",
        "ALTO",
        "MEDIO",
        "BAJO",
        "INCOMPLETO",
    ]

    alert_counts = (
        df["digital_alert_level"]
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
        "Eventos digitales por nivel de alerta"
    )
    axis.set_xlabel(
        "Nivel de alerta"
    )
    axis.set_ylabel(
        "Cantidad de eventos"
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


def generate_triggered_signals_chart(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Genera un gráfico con las señales técnicas
    activadas con mayor frecuencia.
    """

    signal_counts = count_triggered_signals(
        df
    )

    signal_series = pd.Series(
        signal_counts,
        dtype="int64",
    ).sort_values()

    figure, axis = plt.subplots(
        figsize=(11, 7),
    )

    if signal_series.empty:
        axis.text(
            0.5,
            0.5,
            "No se activaron señales",
            horizontalalignment="center",
            verticalalignment="center",
        )

        axis.set_axis_off()
    else:
        signal_series.plot(
            kind="barh",
            ax=axis,
        )

        axis.set_title(
            "Señales digitales activadas"
        )
        axis.set_xlabel(
            "Cantidad de activaciones"
        )
        axis.set_ylabel(
            "Señal"
        )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(figure)


def generate_digital_reports(
    df: pd.DataFrame,
    output_directory: Path,
) -> list[Path]:
    """
    Genera los gráficos especializados del
    motor de fraude digital.
    """

    validate_digital_report_data(df)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    alert_chart_path = (
        output_directory
        / "fraude_digital_por_nivel.png"
    )

    signals_chart_path = (
        output_directory
        / "senales_digitales_activadas.png"
    )

    generate_digital_alert_chart(
        df,
        alert_chart_path,
    )

    generate_triggered_signals_chart(
        df,
        signals_chart_path,
    )

    return [
        alert_chart_path,
        signals_chart_path,
    ]