from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_governance import (
    build_data_quality_report,
    build_lineage_records,
    save_json,
    validate_raci_matrix,
)


CONSOLIDATED_INPUT = Path(
    "outputs/vista_consolidada_casos.csv"
)

SOURCE_REGISTRY_INPUT = Path(
    "data/source_system_registry.csv"
)

RACI_INPUT = Path(
    "data/raci_matrix.csv"
)

LINEAGE_OUTPUT = Path(
    "outputs/data_lineage_records.csv"
)

QUALITY_OUTPUT = Path(
    "outputs/data_quality_report.json"
)

RACI_OUTPUT = Path(
    "outputs/raci_matrix_validated.csv"
)


def main() -> None:
    required_files = [
        CONSOLIDATED_INPUT,
        SOURCE_REGISTRY_INPUT,
        RACI_INPUT,
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

    consolidated_df = pd.read_csv(
        CONSOLIDATED_INPUT
    )

    registry_df = pd.read_csv(
        SOURCE_REGISTRY_INPUT
    )

    raci_df = pd.read_csv(
        RACI_INPUT,
        keep_default_na=False,
    )

    lineage_df = build_lineage_records(
        consolidated_df,
        registry_df,
        source_system_id="SRC-001",
        source_file_name=(
            CONSOLIDATED_INPUT.name
        ),
        received_at=(
            "2026-08-01T21:00:00-04:00"
        ),
        pipeline_id=(
            "CONSOLIDATED_CASE_PIPELINE"
        ),
        pipeline_version="1.0.0",
    )

    quality_report = (
        build_data_quality_report(
            consolidated_df,
            lineage_df,
        )
    )

    validated_raci_df = (
        validate_raci_matrix(
            raci_df
        )
    )

    LINEAGE_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lineage_df.to_csv(
        LINEAGE_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    save_json(
        quality_report,
        QUALITY_OUTPUT,
    )

    validated_raci_df.to_csv(
        RACI_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\nLINAJE DE DATOS\n"
    )

    lineage_columns = [
        "case_id",
        "source_system_id",
        "schema_version",
        "pipeline_version",
        "data_owner_id",
        "data_steward_id",
        "classification",
        "record_hash",
    ]

    display_lineage = (
        lineage_df[
            lineage_columns
        ].copy()
    )

    display_lineage[
        "record_hash"
    ] = (
        display_lineage[
            "record_hash"
        ]
        .str.slice(
            0,
            16,
        )
        + "..."
    )

    print(
        display_lineage.to_string(
            index=False
        )
    )

    print(
        "\nCALIDAD DE DATOS\n"
    )

    print(
        "Estado general: "
        f"{quality_report['overall_status']}"
    )

    print(
        "Completitud: "
        f"{quality_report[
            'completeness'
        ]['completeness_rate'] * 100:.2f}%"
    )

    print(
        "Validez: "
        f"{quality_report[
            'validity'
        ]['validity_rate'] * 100:.2f}%"
    )

    print(
        "Unicidad: "
        f"{quality_report[
            'uniqueness'
        ]['uniqueness_rate'] * 100:.2f}%"
    )

    print(
        "Trazabilidad: "
        f"{quality_report[
            'traceability'
        ]['traceability_rate'] * 100:.2f}%"
    )

    print(
        "\nMATRIZ RACI\n"
    )

    print(
        validated_raci_df[
            [
                "activity_id",
                "activity_name",
                "accountable_count",
                "responsible_count",
                "raci_validation_status",
            ]
        ].to_string(
            index=False
        )
    )

    invalid_raci_count = int(
        (
            validated_raci_df[
                "raci_validation_status"
            ]
            != "VALIDA"
        ).sum()
    )

    print(
        "\nActividades RACI que requieren "
        f"revisión: {invalid_raci_count}"
    )

    print(
        "\nARCHIVOS GENERADOS\n"
    )

    print(
        f"- {LINEAGE_OUTPUT.resolve()}"
    )

    print(
        f"- {QUALITY_OUTPUT.resolve()}"
    )

    print(
        f"- {RACI_OUTPUT.resolve()}"
    )


if __name__ == "__main__":
    main()