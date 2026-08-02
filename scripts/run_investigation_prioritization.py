from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.investigation_prioritization import (
    prioritize_investigations,
)


ALERTS_INPUT = Path(
    "outputs/alertas_profesionales.csv"
)

ASSESSMENTS_INPUT = Path(
    "outputs/evaluaciones_profesionales.csv"
)

CONSOLIDATED_INPUT = Path(
    "outputs/vista_consolidada_casos.csv"
)

QUEUE_OUTPUT = Path(
    "outputs/cola_priorizada_investigaciones.csv"
)

SELECTED_OUTPUT = Path(
    "outputs/"
    "alertas_seleccionadas_para_investigar.csv"
)

WAITLIST_OUTPUT = Path(
    "outputs/alertas_en_espera.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prioriza alertas profesionales "
            "según riesgo, impacto, actividad "
            "repetida, cobertura y capacidad."
        )
    )

    parser.add_argument(
        "--capacity",
        type=int,
        default=5,
        help=(
            "Cantidad máxima de alertas que "
            "pueden seleccionarse en esta ejecución."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    required_files = [
        ALERTS_INPUT,
        ASSESSMENTS_INPUT,
        CONSOLIDATED_INPUT,
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
            "Ejecute primero los pipelines "
            "consolidado y profesional."
        )

    alerts_df = pd.read_csv(
        ALERTS_INPUT
    )

    assessments_df = pd.read_csv(
        ASSESSMENTS_INPUT
    )

    consolidated_df = pd.read_csv(
        CONSOLIDATED_INPUT
    )

    queue_df = prioritize_investigations(
        alerts_df,
        assessments_df,
        consolidated_df,
        investigation_capacity=(
            args.capacity
        ),
    )

    selected_df = queue_df.loc[
        queue_df["queue_status"]
        == "SELECCIONADA"
    ].copy()

    waitlist_df = queue_df.loc[
        queue_df["queue_status"]
        == "EN_ESPERA"
    ].copy()

    QUEUE_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    queue_df.to_csv(
        QUEUE_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    selected_df.to_csv(
        SELECTED_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    waitlist_df.to_csv(
        WAITLIST_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    display_columns = [
        "queue_position",
        "alert_id",
        "case_id",
        "risk_level",
        "estimated_loss_clp",
        "digital_event_count",
        "assessment_coverage",
        "priority_score",
        "investigation_priority",
        "queue_status",
    ]

    print(
        "\nCOLA PRIORIZADA DE INVESTIGACIONES\n"
    )

    print(
        queue_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\nRESUMEN DE CAPACIDAD\n"
    )

    print(
        "Alertas disponibles: "
        f"{len(queue_df)}"
    )

    print(
        "Capacidad configurada: "
        f"{args.capacity}"
    )

    print(
        "Alertas seleccionadas: "
        f"{len(selected_df)}"
    )

    print(
        "Alertas en espera: "
        f"{len(waitlist_df)}"
    )

    print(
        "\nARCHIVOS GENERADOS\n"
    )

    print(
        f"- {QUEUE_OUTPUT.resolve()}"
    )

    print(
        f"- {SELECTED_OUTPUT.resolve()}"
    )

    print(
        f"- {WAITLIST_OUTPUT.resolve()}"
    )


if __name__ == "__main__":
    main()