from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.professional_pipeline import (
    process_professional_pipeline,
)


INPUT_PATH = Path(
    "outputs/vista_consolidada_casos.csv"
)

ASSESSMENTS_OUTPUT = Path(
    "outputs/evaluaciones_profesionales.csv"
)

ALERTS_OUTPUT = Path(
    "outputs/alertas_profesionales.csv"
)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "No existe la vista consolidada: "
            f"{INPUT_PATH}. "
            "Ejecute primero "
            "'python -m scripts."
            "run_consolidated_analysis'."
        )

    consolidated_df = pd.read_csv(
        INPUT_PATH
    )

    result = (
        process_professional_pipeline(
            consolidated_df
        )
    )

    ASSESSMENTS_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.assessments.to_csv(
        ASSESSMENTS_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    result.alerts.to_csv(
        ALERTS_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    assessment_columns = [
        "assessment_id",
        "case_id",
        "engine_type",
        "engine_version",
        "policy_version",
        "risk_score",
        "assessment_coverage",
        "calculated_alert_level",
        "data_quality_status",
    ]

    print(
        "\nEVALUACIONES PROFESIONALES\n"
    )

    print(
        result.assessments[
            assessment_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\nALERTAS SELECCIONADAS\n"
    )

    if result.alerts.empty:
        print(
            "La política no generó alertas."
        )
    else:
        alert_columns = [
            "alert_id",
            "assessment_id",
            "case_id",
            "alert_status",
            "alert_priority",
            "risk_level",
            "estimated_loss_clp",
            "alert_reason",
        ]

        print(
            result.alerts[
                alert_columns
            ].to_string(
                index=False
            )
        )

    print(
        "\nRESUMEN\n"
    )

    print(
        "Evaluaciones creadas: "
        f"{len(result.assessments)}"
    )

    print(
        "Alertas creadas: "
        f"{len(result.alerts)}"
    )

    print(
        "Evaluaciones sin alerta: "
        f"{len(result.assessments) - len(result.alerts)}"
    )

    print(
        "\nARCHIVOS GENERADOS\n"
    )

    print(
        f"- {ASSESSMENTS_OUTPUT.resolve()}"
    )

    print(
        f"- {ALERTS_OUTPUT.resolve()}"
    )


if __name__ == "__main__":
    main()