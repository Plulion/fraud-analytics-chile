from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.investigation_workflow import (
    assign_case,
    change_investigation_status,
    create_empty_audit_log,
    initialize_case_management,
)


CONSOLIDATED_INPUT = Path(
    "outputs/vista_consolidada_casos.csv"
)

CASES_OUTPUT = Path(
    "outputs/gestion_casos.csv"
)

AUDIT_OUTPUT = Path(
    "outputs/auditoria_casos.csv"
)


def main() -> None:
    if not CONSOLIDATED_INPUT.exists():
        raise FileNotFoundError(
            "No existe la vista consolidada: "
            f"{CONSOLIDATED_INPUT}. "
            "Ejecute primero "
            "'python -m scripts.run_consolidated_analysis'."
        )

    consolidated_df = pd.read_csv(
        CONSOLIDATED_INPUT
    )

    cases_df = initialize_case_management(
        consolidated_df
    )

    audit_log = create_empty_audit_log()

    first_case_id = str(
        cases_df.iloc[0]["case_id"]
    )

    assignment = assign_case(
        cases_df,
        audit_log,
        case_id=first_case_id,
        analyst_id="ANA-001",
        supervisor_id="SUP-001",
        actor_id="SUP-001",
        comment=(
            "Asignación inicial según "
            "prioridad consolidada."
        ),
    )

    cases_df = assignment.cases
    audit_log = assignment.audit_log

    investigation_start = (
        change_investigation_status(
            cases_df,
            audit_log,
            case_id=first_case_id,
            new_status="EN_INVESTIGACION",
            actor_id="ANA-001",
            comment=(
                "Se inicia revisión de "
                "eventos, documentos y señales."
            ),
        )
    )

    cases_df = investigation_start.cases
    audit_log = investigation_start.audit_log

    CASES_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cases_df.to_csv(
        CASES_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    audit_log.to_csv(
        AUDIT_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    display_columns = [
        "case_id",
        "consolidated_alert_level",
        "investigation_status",
        "assigned_analyst_id",
        "supervisor_id",
        "investigation_updated_at",
    ]

    print(
        "\nGESTIÓN DE CASOS\n"
    )

    print(
        cases_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\nBITÁCORA DEL CASO DEMOSTRATIVO\n"
    )

    case_audit = audit_log.loc[
        audit_log["case_id"]
        == first_case_id
    ]

    print(
        case_audit.to_string(
            index=False
        )
    )

    print(
        "\nARCHIVOS GENERADOS\n"
    )

    print(
        f"- {CASES_OUTPUT.resolve()}"
    )

    print(
        f"- {AUDIT_OUTPUT.resolve()}"
    )


if __name__ == "__main__":
    main()