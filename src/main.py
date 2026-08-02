from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.case_metrics import (
    calculate_average_loss_per_day,
    classify_detection_delay,
)
from src.reporting import (
    build_category_summary,
    build_summary_metrics,
    generate_reports,
    save_category_summary_csv,
    save_summary_json,
)
from src.risk_scoring import assess_case
from src.typology_enrichment import (
    enrich_cases_with_typology,
)


REQUIRED_INPUT_COLUMNS = {
    "case_id",
    "sector",
    "region",
    "fraud_type_code",
    "weak_controls",
    "privileged_access",
    "financial_pressure",
    "performance_pressure",
    "rationalization_signal",
    "collusion_signal",
    "high_competence",
    "estimated_loss_clp",
    "detection_delay_days",
}


SIGNAL_COLUMNS = [
    "weak_controls",
    "privileged_access",
    "financial_pressure",
    "performance_pressure",
    "rationalization_signal",
    "collusion_signal",
    "high_competence",
]


def parse_args() -> argparse.Namespace:
    """
    Lee los argumentos utilizados al ejecutar el programa.

    Ejecución normal:

        python -m src.main

    Ejecución con rutas personalizadas:

        python -m src.main \
            --input data/otro_archivo.csv \
            --output outputs/otro_resultado.csv
    """

    parser = argparse.ArgumentParser(
        description=(
            "Evalúa casos sintéticos chilenos mediante "
            "el Triángulo del Fraude, tipologías y "
            "métricas de detección."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/casos_sinteticos_chile.csv"
        ),
        help="Ruta del archivo CSV de entrada.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/evaluacion_riesgo.csv"
        ),
        help="Ruta del archivo CSV de salida.",
    )

    return parser.parse_args()


def validate_input_dataframe(
    df: pd.DataFrame,
) -> None:
    """
    Comprueba que el CSV contenga la estructura mínima
    necesaria para ejecutar el análisis.
    """

    missing_columns = (
        REQUIRED_INPUT_COLUMNS.difference(
            df.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "El archivo de entrada no contiene "
            "las columnas requeridas: "
            f"{sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError(
            "El archivo de entrada no contiene "
            "casos para evaluar."
        )

    if df["case_id"].isna().any():
        raise ValueError(
            "La columna 'case_id' contiene "
            "identificadores vacíos."
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
            "Existen identificadores de caso "
            "duplicados: "
            f"{sorted(duplicated_cases)}"
        )

    text_columns = [
        "case_id",
        "sector",
        "region",
        "fraud_type_code",
    ]

    for column in text_columns:
        empty_values = (
            df[column]
            .astype(str)
            .str.strip()
            .isin(["", "nan", "None"])
        )

        if empty_values.any():
            affected_cases = (
                df.loc[
                    empty_values,
                    "case_id",
                ]
                .astype(str)
                .tolist()
            )

            raise ValueError(
                f"La columna {column!r} contiene "
                "valores vacíos. "
                f"Casos afectados: {affected_cases}"
            )

    numeric_columns = [
        "estimated_loss_clp",
        "detection_delay_days",
    ]

    for column in numeric_columns:
        if df[column].isna().any():
            raise ValueError(
                f"La columna {column!r} "
                "contiene valores vacíos."
            )

        try:
            numeric_values = pd.to_numeric(
                df[column],
                errors="raise",
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"La columna {column!r} debe "
                "contener valores numéricos."
            ) from exc

        if (numeric_values < 0).any():
            affected_cases = (
                df.loc[
                    numeric_values < 0,
                    "case_id",
                ]
                .astype(str)
                .tolist()
            )

            raise ValueError(
                f"La columna {column!r} no puede "
                "contener valores negativos. "
                f"Casos afectados: {affected_cases}"
            )

    for column in SIGNAL_COLUMNS:
        invalid_values: list[str] = []

        for value in df[column]:
            if pd.isna(value):
                continue

            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                invalid_values.append(
                    str(value)
                )
                continue

            if numeric_value not in {
                0.0,
                1.0,
            }:
                invalid_values.append(
                    str(value)
                )

        if invalid_values:
            raise ValueError(
                f"La columna {column!r} solo "
                "puede contener 0, 1 o un "
                "valor vacío. "
                f"Valores inválidos: "
                f"{sorted(set(invalid_values))}"
            )


def add_risk_assessment_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ejecuta el motor del Triángulo del Fraude
    para cada caso.
    """

    result_df = df.copy()

    records = result_df.to_dict(
        orient="records"
    )

    assessments = [
        assess_case(record)
        for record in records
    ]

    result_df["opportunity_score"] = [
        item.opportunity
        for item in assessments
    ]

    result_df["pressure_score"] = [
        item.pressure
        for item in assessments
    ]

    result_df["rationalization_score"] = [
        item.rationalization
        for item in assessments
    ]

    result_df["aggravating_score"] = [
        item.aggravating_factors
        for item in assessments
    ]

    result_df["risk_score"] = [
        item.risk_score
        for item in assessments
    ]

    result_df["assessment_coverage"] = [
        item.assessment_coverage
        for item in assessments
    ]

    result_df["alert_level"] = [
        item.alert_level
        for item in assessments
    ]

    result_df["decision_status"] = [
        item.decision_status
        for item in assessments
    ]

    return result_df


def add_detection_metrics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcula la clasificación de demora y la
    pérdida diaria promedio descriptiva.
    """

    result_df = df.copy()

    result_df["detection_delay_days"] = (
        pd.to_numeric(
            result_df[
                "detection_delay_days"
            ],
            errors="raise",
        ).astype(int)
    )

    result_df["estimated_loss_clp"] = (
        pd.to_numeric(
            result_df[
                "estimated_loss_clp"
            ],
            errors="raise",
        )
    )

    result_df["detection_delay_level"] = [
        classify_detection_delay(
            int(days)
        )
        for days in result_df[
            "detection_delay_days"
        ]
    ]

    result_df[
        "average_loss_per_day_clp"
    ] = [
        calculate_average_loss_per_day(
            estimated_loss_clp=float(loss),
            detection_delay_days=int(days),
        )
        for loss, days in zip(
            result_df[
                "estimated_loss_clp"
            ],
            result_df[
                "detection_delay_days"
            ],
        )
    ]

    return result_df


def format_clp(
    value: int | float,
) -> str:
    """
    Formatea un número como pesos chilenos.
    """

    formatted = f"{value:,.0f}"

    formatted = formatted.replace(
        ",",
        ".",
    )

    return f"${formatted} CLP"


def format_metric_text(
    value: Any,
) -> str:
    """
    Evita mostrar la palabra None cuando una métrica
    todavía no tiene un valor disponible.
    """

    if value is None:
        return "SIN_DATOS"

    return str(value)


def print_category_summary(
    category_summary: pd.DataFrame,
) -> None:
    """
    Muestra en la terminal un resumen compacto
    por categoría de fraude.
    """

    category_display_columns = [
        "fraud_category",
        "total_cases",
        "critical_cases",
        "high_risk_cases",
        "late_detection_cases",
        "estimated_loss_total_clp",
        "average_risk_score",
        "average_detection_delay_days",
    ]

    print(
        "\nRESUMEN POR CATEGORÍA DE FRAUDE\n"
    )

    print(
        category_summary[
            category_display_columns
        ].to_string(
            index=False,
        )
    )


def print_results(
    df: pd.DataFrame,
    metrics: dict[str, Any],
    category_summary: pd.DataFrame,
) -> None:
    """
    Muestra los casos priorizados, las métricas
    generales y el resumen por categoría.
    """

    display_columns = [
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
        "average_loss_per_day_clp",
    ]

    print(
        "\nCASOS PRIORIZADOS PARA REVISIÓN\n"
    )

    print(
        df[display_columns].to_string(
            index=False,
        )
    )

    print(
        "\nRESUMEN GENERAL\n"
    )

    print(
        "Casos evaluados: "
        f"{metrics['total_cases']}"
    )

    print(
        "Casos críticos: "
        f"{metrics['critical_cases']}"
    )

    print(
        "Casos de riesgo alto: "
        f"{metrics['high_risk_cases']}"
    )

    print(
        "Casos incompletos: "
        f"{metrics['incomplete_cases']}"
    )

    print(
        "Casos con detección tardía: "
        f"{metrics['late_detection_cases']}"
    )

    print(
        "Pérdida total estimada: "
        + format_clp(
            metrics[
                "estimated_loss_total_clp"
            ]
        )
    )

    print(
        "Puntaje promedio de riesgo: "
        f"{metrics['average_risk_score']}"
    )

    print(
        "Cobertura promedio: "
        f"{metrics[
            'average_assessment_coverage'
        ]}%"
    )

    print(
        "Demora promedio de detección: "
        f"{metrics[
            'average_detection_delay_days'
        ]} días"
    )

    print(
        "Categoría con más casos: "
        + format_metric_text(
            metrics[
                "top_fraud_category_by_cases"
            ]
        )
    )

    print(
        "Categoría con mayor pérdida: "
        + format_metric_text(
            metrics[
                "top_fraud_category_by_loss"
            ]
        )
    )

    print_category_summary(
        category_summary
    )


def save_main_results(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Guarda el resultado completo del análisis.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            "No existe el archivo de entrada: "
            f"{args.input}"
        )

    if not args.input.is_file():
        raise ValueError(
            "La ruta de entrada no corresponde "
            "a un archivo: "
            f"{args.input}"
        )

    df = pd.read_csv(
        args.input
    )

    validate_input_dataframe(
        df
    )

    # 1. Agrega nombre, categoría, controles y datos
    # recomendados según la tipología.
    df = enrich_cases_with_typology(
        df
    )

    # 2. Calcula el puntaje basado en señales del
    # Triángulo del Fraude.
    df = add_risk_assessment_columns(
        df
    )

    # 3. Calcula la demora y la pérdida promedio
    # descriptiva por día.
    df = add_detection_metrics(
        df
    )

    # 4. Prioriza primero por puntaje y, en caso
    # de empate, por pérdida estimada.
    df = df.sort_values(
        by=[
            "risk_score",
            "estimated_loss_clp",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )

    save_main_results(
        df,
        args.output,
    )

    report_directory = (
        args.output.parent
        / "reportes"
    )

    metrics = build_summary_metrics(
        df
    )

    summary_path = (
        report_directory
        / "resumen_metricas.json"
    )

    save_summary_json(
        metrics,
        summary_path,
    )

    category_summary = build_category_summary(
        df
    )

    category_summary_path = (
        report_directory
        / "resumen_por_categoria.csv"
    )

    save_category_summary_csv(
        category_summary,
        category_summary_path,
    )

    generated_charts = generate_reports(
        df,
        report_directory,
    )

    print_results(
        df,
        metrics,
        category_summary,
    )

    print(
        "\nARCHIVOS GENERADOS\n"
    )

    print(
        "Resultado CSV: "
        f"{args.output.resolve()}"
    )

    print(
        "Resumen JSON: "
        f"{summary_path.resolve()}"
    )

    print(
        "Resumen por categoría: "
        f"{category_summary_path.resolve()}"
    )

    print(
        "\nGRÁFICOS GENERADOS"
    )

    for chart_path in generated_charts:
        print(
            f"- {chart_path.resolve()}"
        )


if __name__ == "__main__":
    main()