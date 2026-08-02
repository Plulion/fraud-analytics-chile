from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.case_closure import (
    close_case_with_supervisor_approval,
    create_empty_feedback_registry,
    create_empty_resolution_registry,
)


CASES_INPUT = Path(
    "outputs/gestion_casos.csv"
)

AUDIT_INPUT = Path(
    "outputs/auditoria_casos.csv"
)

CASES_OUTPUT = Path(
    "outputs/gestion_casos_cierre_demo.csv"
)

RESOLUTIONS_OUTPUT = Path(
    "outputs/resoluciones_casos.csv"
)

AUDIT_OUTPUT = Path(
    "outputs/auditoria_casos_cierre_demo.csv"
)

FEEDBACK_OUTPUT = Path(
    "outputs/feedback_casos_cerrados.csv"
)


def main() -> None:
    required_files = [
        CASES_INPUT,
        AUDIT_INPUT,
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

    audit_df = pd.read_csv(
        AUDIT_INPUT
    )

    resolutions_df = (
        create_empty_resolution_registry()
    )

    feedback_df = (
        create_empty_feedback_registry()
    )

    result = (
        close_case_with_supervisor_approval(
            cases_df,
            resolutions_df,
            audit_df,
            feedback_df,
            case_id="CL-002",
            proposed_outcome="CONFIRMADO",
            resolution_summary=(
                "Se confirma un conflicto de interés "
                "y una concentración incompatible "
                "de funciones en el proceso revisado."
            ),
            resolution_rationale=(
                "La revisión identificó que una misma "
                "persona intervino en solicitud, "
                "aprobación y recepción, junto con "
                "antecedentes de relación con el "
                "proveedor. El resultado corresponde "
                "a un caso sintético de entrenamiento."
            ),
            proposed_by="ANA-001",
            supervisor_id="SUP-001",
            confirmed_loss_clp=42_000_000,
            recovered_amount_clp=5_000_000,
            proposed_at=(
                "2026-08-01T23:20:00-04:00"
            ),
            approved_at=(
                "2026-08-01T23:30:00-04:00"
            ),
        )
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

    closed_case = result.cases.loc[
        result.cases["case_id"]
        .astype(str)
        .str.strip()
        == "CL-002"
    ]

    print(
        "\nCASO CERRADO\n"
    )

    print(
        closed_case[
            [
                "case_id",
                "investigation_status",
                "assigned_analyst_id",
                "supervisor_id",
                "investigation_closed_at",
                "resolution",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nRESOLUCIÓN APROBADA\n"
    )

    print(
        result.resolutions[
            [
                "resolution_id",
                "case_id",
                "proposed_outcome",
                "proposed_by",
                "supervisor_id",
                "approval_status",
                "confirmed_loss_clp",
                "recovered_amount_clp",
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
            4
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
        "\nFEEDBACK PARA EL MODELO\n"
    )

    print(
        result.feedback.to_string(
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
        f"- {RESOLUTIONS_OUTPUT.resolve()}"
    )

    print(
        f"- {AUDIT_OUTPUT.resolve()}"
    )

    print(
        f"- {FEEDBACK_OUTPUT.resolve()}"
    )


if __name__ == "__main__":
    main()