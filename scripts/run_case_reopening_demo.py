from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.case_reopening import (
    reopen_closed_case,
)


CASES_INPUT = Path(
    "outputs/gestion_casos_cierre_demo.csv"
)

RESOLUTIONS_INPUT = Path(
    "outputs/resoluciones_casos.csv"
)

AUDIT_INPUT = Path(
    "outputs/auditoria_casos_cierre_demo.csv"
)

FEEDBACK_INPUT = Path(
    "outputs/feedback_casos_cerrados.csv"
)


CASES_OUTPUT = Path(
    "outputs/gestion_casos_reapertura_demo.csv"
)

RESOLUTIONS_OUTPUT = Path(
    "outputs/resoluciones_casos_reapertura_demo.csv"
)

AUDIT_OUTPUT = Path(
    "outputs/auditoria_casos_reapertura_demo.csv"
)

FEEDBACK_OUTPUT = Path(
    "outputs/feedback_casos_reapertura_demo.csv"
)


def main() -> None:
    required_files = [
        CASES_INPUT,
        RESOLUTIONS_INPUT,
        AUDIT_INPUT,
        FEEDBACK_INPUT,
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
        )

    cases_df = pd.read_csv(
        CASES_INPUT
    )

    resolutions_df = pd.read_csv(
        RESOLUTIONS_INPUT
    )

    audit_df = pd.read_csv(
        AUDIT_INPUT
    )

    feedback_df = pd.read_csv(
        FEEDBACK_INPUT
    )

    result = reopen_closed_case(
        cases_df,
        resolutions_df,
        audit_df,
        feedback_df,
        case_id="CL-002",
        supervisor_id="SUP-001",
        reopening_reason=(
            "Se recibió nueva evidencia documental "
            "que podría modificar la conclusión "
            "aprobada previamente."
        ),
        target_status="EN_INVESTIGACION",
        reopened_at=(
            "2026-08-02T09:00:00-04:00"
        ),
    )

    CASES_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.cases.to_csv(
        CASES_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    result.resolutions.to_csv(
        RESOLUTIONS_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    result.audit.to_csv(
        AUDIT_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    result.feedback.to_csv(
        FEEDBACK_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    reopened_case = result.cases.loc[
        result.cases["case_id"]
        .astype(str)
        .str.strip()
        .eq("CL-002")
    ]

    print(
        "\nCASO REABIERTO\n"
    )

    print(
        reopened_case[
            [
                "case_id",
                "investigation_status",
                "assigned_analyst_id",
                "supervisor_id",
                "investigation_updated_at",
                "investigation_closed_at",
                "resolution",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nRESOLUCIÓN ANTERIOR\n"
    )

    print(
        result.resolutions[
            [
                "resolution_id",
                "case_id",
                "proposed_outcome",
                "approval_status",
                "resolution_lifecycle_status",
                "superseded_at",
                "superseded_by",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nFEEDBACK INVALIDADO\n"
    )

    print(
        result.feedback[
            [
                "case_id",
                "resolution_id",
                "investigation_outcome",
                "outcome_label",
                "feedback_status",
                "invalidated_at",
                "invalidated_by",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nNUEVOS EVENTOS DE AUDITORÍA\n"
    )

    print(
        result.audit.tail(
            3
        )[
            [
                "audit_id",
                "case_id",
                "actor_id",
                "action_type",
                "field_name",
                "previous_value",
                "new_value",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nARCHIVOS GENERADOS\n"
    )

    for path in [
        CASES_OUTPUT,
        RESOLUTIONS_OUTPUT,
        AUDIT_OUTPUT,
        FEEDBACK_OUTPUT,
    ]:
        print(
            f"- {path.resolve()}"
        )


if __name__ == "__main__":
    main()