from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.training_dataset import (
    build_training_dataset,
)


ACTIVE_FEEDBACK_INPUT = Path(
    "outputs/feedback_casos_cerrados.csv"
)

INVALIDATED_FEEDBACK_INPUT = Path(
    "outputs/feedback_casos_reapertura_demo.csv"
)


DATASET_OUTPUT = Path(
    "outputs/dataset_entrenamiento_limpio.csv"
)

EXCLUSIONS_OUTPUT = Path(
    "outputs/registros_excluidos_entrenamiento.csv"
)

REPORT_OUTPUT = Path(
    "outputs/reporte_dataset_entrenamiento.json"
)


FEATURE_COLUMNS = [
    "amount_clp",
    "transaction_count_24h",
    "new_device",
    "geolocation_mismatch",
    "amount_spike_ratio",
]


def build_demo_features() -> pd.DataFrame:
    """
    Snapshot educativo de variables disponibles antes
    de conocer el resultado final de la investigación.
    """

    return pd.DataFrame(
        {
            "case_id": [
                "CL-001",
                "CL-002",
                "CL-003",
            ],
            "feature_snapshot_at": [
                "2026-08-01T21:00:00-04:00",
                "2026-08-01T22:35:00-04:00",
                "2026-08-01T21:15:00-04:00",
            ],
            "amount_clp": [
                8_500_000,
                42_000_000,
                1_200_000,
            ],
            "transaction_count_24h": [
                3,
                18,
                2,
            ],
            "new_device": [
                0,
                1,
                0,
            ],
            "geolocation_mismatch": [
                0,
                1,
                0,
            ],
            "amount_spike_ratio": [
                1.2,
                7.8,
                1.1,
            ],
        }
    )


def build_demo_feedback() -> pd.DataFrame:
    if not ACTIVE_FEEDBACK_INPUT.exists():
        raise FileNotFoundError(
            "Falta el archivo de feedback de cierre: "
            f"{ACTIVE_FEEDBACK_INPUT}"
        )

    active_feedback = pd.read_csv(
        ACTIVE_FEEDBACK_INPUT
    )

    frames = [
        active_feedback
    ]

    if INVALIDATED_FEEDBACK_INPUT.exists():
        invalidated_feedback = pd.read_csv(
            INVALIDATED_FEEDBACK_INPUT
        )

        frames.append(
            invalidated_feedback
        )

    synthetic_active_feedback = pd.DataFrame(
        {
            "case_id": [
                "CL-001",
                "CL-003",
            ],
            "resolution_id": [
                "RES-DEMO-001",
                "RES-DEMO-003",
            ],
            "investigation_outcome": [
                "DESCARTADO",
                "CONFIRMADO",
            ],
            "outcome_label": [
                0,
                1,
            ],
            "closed_at": [
                "2026-08-01T22:00:00-04:00",
                "2026-08-01T22:45:00-04:00",
            ],
            "supervisor_id": [
                "SUP-DEMO",
                "SUP-DEMO",
            ],
            "confirmed_loss_clp": [
                0.0,
                1_200_000.0,
            ],
            "recovered_amount_clp": [
                0.0,
                0.0,
            ],
            "feedback_source": [
                "SYNTHETIC_EDUCATIONAL_FEEDBACK",
                "SYNTHETIC_EDUCATIONAL_FEEDBACK",
            ],
            "feedback_status": [
                "ACTIVO",
                "ACTIVO",
            ],
        }
    )

    frames.append(
        synthetic_active_feedback
    )

    return pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )


def main() -> None:
    features_df = build_demo_features()
    feedback_df = build_demo_feedback()

    result = build_training_dataset(
        features_df,
        feedback_df,
        feature_columns=FEATURE_COLUMNS,
    )

    DATASET_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.dataset.to_csv(
        DATASET_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    result.exclusions.to_csv(
        EXCLUSIONS_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    REPORT_OUTPUT.write_text(
        json.dumps(
            result.report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nDATASET DE ENTRENAMIENTO LIMPIO\n"
    )

    if result.dataset.empty:
        print(
            "No existen registros elegibles."
        )
    else:
        print(
            result.dataset.to_string(
                index=False
            )
        )

    print(
        "\nREGISTROS EXCLUIDOS\n"
    )

    if result.exclusions.empty:
        print(
            "No existen registros excluidos."
        )
    else:
        print(
            result.exclusions.to_string(
                index=False
            )
        )

    print(
        "\nREPORTE DEL DATASET\n"
    )

    print(
        json.dumps(
            result.report,
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        "\nARCHIVOS GENERADOS\n"
    )

    for path in [
        DATASET_OUTPUT,
        EXCLUSIONS_OUTPUT,
        REPORT_OUTPUT,
    ]:
        print(
            f"- {path.resolve()}"
        )


if __name__ == "__main__":
    main()