from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from src.operational_dashboard import (
    build_analyst_workload,
    build_operational_dashboard,
    calculate_operational_kpis,
)


CASES_INPUT = Path(
    "outputs/gestion_casos.csv"
)

AUDIT_INPUT = Path(
    "outputs/auditoria_casos.csv"
)

PRIORITIES_INPUT = Path(
    "outputs/cola_priorizada_investigaciones.csv"
)

DASHBOARD_OUTPUT = Path(
    "outputs/dashboard_operativo_casos.csv"
)

WORKLOAD_OUTPUT = Path(
    "outputs/dashboard_carga_analistas.csv"
)

KPI_OUTPUT = Path(
    "outputs/dashboard_kpis.json"
)


CHILE_TIME_ZONE = ZoneInfo(
    "America/Santiago"
)


REFERENCE_TIMESTAMP = datetime.now(
    CHILE_TIME_ZONE
)


def percentage_text(
    value: Any,
) -> str:
    return (
        f"{float(value) * 100:.2f}%"
    )


def main() -> None:
    required_files = [
        CASES_INPUT,
        AUDIT_INPUT,
        PRIORITIES_INPUT,
    ]

    missing_files = [
        path
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Faltan archivos requeridos: "
            + ", ".join(
                str(path)
                for path in missing_files
            )
            + ". Ejecute primero los procesos "
            "de workflow y priorización."
        )

    cases_df = pd.read_csv(
        CASES_INPUT
    )

    audit_df = pd.read_csv(
        AUDIT_INPUT
    )

    priorities_df = pd.read_csv(
        PRIORITIES_INPUT
    )

    dashboard_df = (
        build_operational_dashboard(
            cases_df,
            audit_df,
            priorities_df,
            reference_timestamp=(
                REFERENCE_TIMESTAMP
            ),
        )
    )

    workload_df = (
        build_analyst_workload(
            dashboard_df
        )
    )

    kpis = calculate_operational_kpis(
        dashboard_df
    )

    DASHBOARD_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dashboard_df.to_csv(
        DASHBOARD_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    workload_df.to_csv(
        WORKLOAD_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    with KPI_OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            kpis,
            file,
            ensure_ascii=False,
            indent=4,
        )

    dashboard_columns = [
        "case_id",
        "investigation_status",
        "assigned_analyst_id",
        "investigation_priority",
        "priority_score",
        "case_age_hours",
        "hours_to_assignment",
        "hours_to_start",
        "sla_target_hours",
        "sla_status",
    ]

    print(
        "\nPANEL OPERATIVO DE CASOS\n"
    )

    print(
        dashboard_df[
            dashboard_columns
        ].to_string(
            index=False
        )
    )

    workload_columns = [
        "assigned_analyst_id",
        "total_cases",
        "open_cases",
        "closed_cases",
        "in_investigation_cases",
        "requires_information_cases",
        "escalated_cases",
        "overdue_cases",
        "average_open_age_hours",
    ]

    print(
        "\nCARGA POR ANALISTA\n"
    )

    print(
        workload_df[
            workload_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\nKPI OPERATIVOS\n"
    )

    print(
        "Casos totales: "
        f"{kpis['total_cases']}"
    )

    print(
        "Casos abiertos: "
        f"{kpis['open_cases']}"
    )

    print(
        "Casos cerrados: "
        f"{kpis['closed_cases']}"
    )

    print(
        "Casos sin asignar: "
        f"{kpis['unassigned_cases']}"
    )

    print(
        "Casos vencidos o incumplidos: "
        f"{kpis[
            'overdue_or_breached_cases'
        ]}"
    )

    print(
        "Tasa de cierre: "
        f"{percentage_text(
            kpis['closure_rate']
        )}"
    )

    print(
        "Cumplimiento SLA medible: "
        f"{percentage_text(
            kpis['sla_compliance_rate']
        )}"
    )

    print(
        "Horas promedio hasta asignación: "
        f"{kpis[
            'average_hours_to_assignment'
        ]:.2f}"
    )

    print(
        "Horas promedio hasta inicio: "
        f"{kpis[
            'average_hours_to_start'
        ]:.2f}"
    )

    print(
        "Edad promedio de casos abiertos: "
        f"{kpis[
            'average_open_case_age_hours'
        ]:.2f}"
    )

    print(
        "\nARCHIVOS GENERADOS\n"
    )

    print(
        f"- {DASHBOARD_OUTPUT.resolve()}"
    )

    print(
        f"- {WORKLOAD_OUTPUT.resolve()}"
    )

    print(
        f"- {KPI_OUTPUT.resolve()}"
    )


if __name__ == "__main__":
    main()