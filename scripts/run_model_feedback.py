from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.model_feedback import (
    build_feedback_dataset,
    calculate_backtesting_metrics,
)


ASSESSMENTS_INPUT = Path(
    "outputs/evaluaciones_profesionales.csv"
)

ALERTS_INPUT = Path(
    "outputs/alertas_profesionales.csv"
)

OUTCOMES_INPUT = Path(
    "data/resultados_investigacion_demo.csv"
)

FEEDBACK_OUTPUT = Path(
    "outputs/retroalimentacion_modelo.csv"
)

METRICS_OUTPUT = Path(
    "outputs/backtesting_inicial.json"
)


def percentage_text(
    value: Any,
) -> str:
    """
    Convierte una proporción en porcentaje.
    """

    return (
        f"{float(value) * 100:.2f}%"
    )


def main() -> None:
    required_files = [
        ASSESSMENTS_INPUT,
        ALERTS_INPUT,
        OUTCOMES_INPUT,
    ]

    missing_files = [
        path
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        missing_text = ", ".join(
            str(path)
            for path in missing_files
        )

        raise FileNotFoundError(
            "Faltan archivos requeridos: "
            f"{missing_text}. "
            "Ejecute primero el pipeline "
            "profesional y cree el archivo "
            "de resultados de investigación."
        )

    assessments_df = pd.read_csv(
        ASSESSMENTS_INPUT
    )

    alerts_df = pd.read_csv(
        ALERTS_INPUT
    )

    outcomes_df = pd.read_csv(
        OUTCOMES_INPUT
    )

    feedback_df = build_feedback_dataset(
        assessments_df,
        alerts_df,
        outcomes_df,
    )

    metrics = (
        calculate_backtesting_metrics(
            feedback_df
        )
    )

    FEEDBACK_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    feedback_df.to_csv(
        FEEDBACK_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    with METRICS_OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            ensure_ascii=False,
            indent=4,
        )

    display_columns = [
        "case_id",
        "calculated_alert_level",
        "alert_created",
        "investigation_outcome",
        "prediction_result",
        "risk_score",
        "assessment_coverage",
    ]

    print(
        "\nRETROALIMENTACIÓN DEL MOTOR\n"
    )

    print(
        feedback_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\nMATRIZ DE CONFUSIÓN\n"
    )

    print(
        "Verdaderos positivos: "
        f"{metrics['true_positive']}"
    )

    print(
        "Falsos positivos: "
        f"{metrics['false_positive']}"
    )

    print(
        "Verdaderos negativos: "
        f"{metrics['true_negative']}"
    )

    print(
        "Falsos negativos: "
        f"{metrics['false_negative']}"
    )

    print(
        "\nMÉTRICAS INICIALES\n"
    )

    print(
        "Precisión: "
        f"{percentage_text(metrics['precision'])}"
    )

    print(
        "Recall o sensibilidad: "
        f"{percentage_text(metrics['recall'])}"
    )

    print(
        "Especificidad: "
        f"{percentage_text(metrics['specificity'])}"
    )

    print(
        "Exactitud: "
        f"{percentage_text(metrics['accuracy'])}"
    )

    print(
        "Tasa de falsos positivos: "
        f"{percentage_text(
            metrics['false_positive_rate']
        )}"
    )

    print(
        "Tasa de falsos negativos: "
        f"{percentage_text(
            metrics['false_negative_rate']
        )}"
    )

    print(
        "\nARCHIVOS GENERADOS\n"
    )

    print(
        f"- {FEEDBACK_OUTPUT.resolve()}"
    )

    print(
        f"- {METRICS_OUTPUT.resolve()}"
    )

    print(
        "\nIMPORTANTE\n"
    )

    print(
        "Estas métricas utilizan solamente "
        "datos sintéticos y no representan "
        "el desempeño de un sistema real."
    )


if __name__ == "__main__":
    main()