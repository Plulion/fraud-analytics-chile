from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.digital_batch import (
    process_digital_events,
)
from src.digital_reporting import (
    build_digital_summary,
    generate_digital_reports,
    save_digital_summary_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Procesa eventos sintéticos mediante "
            "el motor de fraude digital."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/eventos_digitales_chile.csv"
        ),
        help="Archivo CSV de eventos digitales.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/evaluacion_fraude_digital.csv"
        ),
        help="Archivo CSV con las evaluaciones.",
    )

    return parser.parse_args()


def format_metric(
    value: Any,
) -> str:
    if value is None:
        return "SIN_DATOS"

    return str(value)


def print_digital_results(
    df: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    display_columns = [
        "event_id",
        "case_id",
        "customer_id",
        "channel",
        "region",
        "digital_risk_score",
        "digital_assessment_coverage",
        "digital_alert_level",
        "digital_recommended_action",
        "digital_triggered_signal_count",
    ]

    print(
        "\nEVENTOS DIGITALES PRIORIZADOS\n"
    )

    print(
        df[display_columns].to_string(
            index=False,
        )
    )

    print(
        "\nRESUMEN DE FRAUDE DIGITAL\n"
    )

    print(
        "Eventos procesados: "
        f"{summary['total_events']}"
    )

    print(
        "Clientes únicos: "
        f"{summary['unique_customers']}"
    )

    print(
        "Eventos críticos: "
        f"{summary['critical_events']}"
    )

    print(
        "Eventos de riesgo alto: "
        f"{summary['high_risk_events']}"
    )

    print(
        "Eventos incompletos: "
        f"{summary['incomplete_events']}"
    )

    print(
        "Puntaje digital promedio: "
        f"{summary[
            'average_digital_risk_score'
        ]}"
    )

    print(
        "Cobertura digital promedio: "
        f"{summary[
            'average_digital_coverage'
        ]}%"
    )

    triggered_signals = summary[
        "triggered_signals"
    ]

    print(
        "\nSEÑALES ACTIVADAS\n"
    )

    if triggered_signals:
        for signal, count in (
            triggered_signals.items()
        ):
            print(
                f"- {signal}: {count}"
            )
    else:
        print(
            "- No se activaron señales."
        )


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            "No existe el archivo de eventos: "
            f"{args.input}"
        )

    df = pd.read_csv(
        args.input
    )

    result_df = process_digital_events(
        df
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        args.output,
        index=False,
        encoding="utf-8-sig",
    )

    report_directory = (
        args.output.parent
        / "reportes_digitales"
    )

    summary = build_digital_summary(
        result_df
    )

    summary_path = (
        report_directory
        / "resumen_fraude_digital.json"
    )

    save_digital_summary_json(
        summary,
        summary_path,
    )

    generated_charts = (
        generate_digital_reports(
            result_df,
            report_directory,
        )
    )

    print_digital_results(
        result_df,
        summary,
    )

    print(
        "\nARCHIVOS GENERADOS\n"
    )

    print(
        "Evaluación digital: "
        f"{args.output.resolve()}"
    )

    print(
        "Resumen digital: "
        f"{summary_path.resolve()}"
    )

    print(
        "\nGRÁFICOS DIGITALES"
    )

    for chart_path in generated_charts:
        print(
            f"- {chart_path.resolve()}"
        )


if __name__ == "__main__":
    main()