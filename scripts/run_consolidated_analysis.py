from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.case_consolidation import (
    consolidate_case_and_digital_data,
)
from src.consolidated_reporting import (
    build_consolidated_summary,
    generate_consolidated_reports,
    save_consolidated_summary_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Relaciona casos generales con "
            "eventos digitales mediante case_id."
        )
    )

    parser.add_argument(
        "--general-input",
        type=Path,
        default=Path(
            "outputs/evaluacion_riesgo.csv"
        ),
        help=(
            "Archivo con la evaluación general "
            "de los casos."
        ),
    )

    parser.add_argument(
        "--digital-input",
        type=Path,
        default=Path(
            "outputs/evaluacion_fraude_digital.csv"
        ),
        help=(
            "Archivo con las evaluaciones "
            "de eventos digitales."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/vista_consolidada_casos.csv"
        ),
        help=(
            "Archivo de salida con la vista "
            "consolidada."
        ),
    )

    return parser.parse_args()


def format_metric(
    value: Any,
) -> str:
    if value is None:
        return "SIN_DATOS"

    return str(value)


def print_consolidated_results(
    df: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    display_columns = [
        "case_id",
        "sector",
        "fraud_type_name",
        "risk_score",
        "alert_level",
        "digital_event_count",
        "digital_risk_score_max",
        "digital_case_alert_level",
        "consolidated_alert_level",
        "consolidated_data_status",
        "consolidated_recommended_action",
    ]

    print(
        "\nVISTA CONSOLIDADA DE CASOS\n"
    )

    print(
        df[display_columns].to_string(
            index=False,
        )
    )

    print(
        "\nRESUMEN CONSOLIDADO\n"
    )

    print(
        "Casos totales: "
        f"{summary['total_cases']}"
    )

    print(
        "Casos con eventos digitales: "
        f"{summary[
            'cases_with_digital_events'
        ]}"
    )

    print(
        "Casos sin eventos digitales: "
        f"{summary[
            'cases_without_digital_events'
        ]}"
    )

    print(
        "Eventos digitales relacionados: "
        f"{summary[
            'total_digital_events'
        ]}"
    )

    print(
        "Casos críticos consolidados: "
        f"{summary[
            'critical_consolidated_cases'
        ]}"
    )

    print(
        "Casos altos consolidados: "
        f"{summary[
            'high_consolidated_cases'
        ]}"
    )

    print(
        "Casos que requieren revisar datos: "
        f"{summary[
            'cases_requiring_data_review'
        ]}"
    )


def main() -> None:
    args = parse_args()

    missing_files = [
        path
        for path in [
            args.general_input,
            args.digital_input,
        ]
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
            "Ejecute primero los análisis "
            "general y digital."
        )

    general_df = pd.read_csv(
        args.general_input
    )

    digital_df = pd.read_csv(
        args.digital_input
    )

    consolidated_df = (
        consolidate_case_and_digital_data(
            general_df,
            digital_df,
        )
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    consolidated_df.to_csv(
        args.output,
        index=False,
        encoding="utf-8-sig",
    )

    report_directory = (
        args.output.parent
        / "reportes_consolidados"
    )

    summary = build_consolidated_summary(
        consolidated_df
    )

    summary_path = (
        report_directory
        / "resumen_consolidado.json"
    )

    save_consolidated_summary_json(
        summary,
        summary_path,
    )

    generated_charts = (
        generate_consolidated_reports(
            consolidated_df,
            report_directory,
        )
    )

    print_consolidated_results(
        consolidated_df,
        summary,
    )

    print(
        "\nARCHIVOS GENERADOS\n"
    )

    print(
        "Vista consolidada: "
        f"{args.output.resolve()}"
    )

    print(
        "Resumen consolidado: "
        f"{summary_path.resolve()}"
    )

    print(
        "\nGRÁFICOS CONSOLIDADOS"
    )

    for chart_path in generated_charts:
        print(
            f"- {chart_path.resolve()}"
        )


if __name__ == "__main__":
    main()