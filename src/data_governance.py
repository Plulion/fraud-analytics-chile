from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


REQUIRED_SOURCE_REGISTRY_COLUMNS = {
    "source_system_id",
    "source_system_name",
    "source_type",
    "data_owner_id",
    "data_steward_id",
    "schema_id",
    "schema_version",
    "classification",
    "active",
}


REQUIRED_RACI_COLUMNS = {
    "activity_id",
    "activity_name",
}


VALID_RACI_VALUES = {
    "R",
    "A",
    "C",
    "I",
    "",
}


VALID_CLASSIFICATIONS = {
    "INTERNAL",
    "CONFIDENTIAL",
    "RESTRICTED",
}


LINEAGE_REQUIRED_COLUMNS = {
    "case_id",
    "risk_score",
    "assessment_coverage",
    "consolidated_alert_level",
    "consolidated_data_status",
    "estimated_loss_clp",
}


DEFAULT_HASH_COLUMNS = [
    "case_id",
    "risk_score",
    "assessment_coverage",
    "consolidated_alert_level",
    "consolidated_data_status",
    "estimated_loss_clp",
    "digital_event_count",
    "digital_risk_score_max",
]


def normalize_text(
    value: Any,
) -> str:
    """
    Convierte un valor a texto sin espacios
    laterales.
    """

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def normalize_upper_text(
    value: Any,
) -> str:
    return normalize_text(
        value
    ).upper()


def canonicalize_value(
    value: Any,
) -> Any:
    """
    Convierte valores en una representación estable
    para calcular un hash reproducible.
    """

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(
        value,
        bool,
    ):
        return value

    if isinstance(
        value,
        int,
    ):
        return value

    if isinstance(
        value,
        float,
    ):
        if value.is_integer():
            return int(value)

        return round(
            value,
            10,
        )

    if isinstance(
        value,
        pd.Timestamp,
    ):
        return value.isoformat()

    return str(value).strip()


def calculate_record_hash(
    record: dict[str, Any],
    *,
    columns: Iterable[str],
) -> str:
    """
    Calcula SHA-256 usando una representación JSON
    estable y ordenada.
    """

    canonical_record = {
        column: canonicalize_value(
            record.get(column)
        )
        for column in columns
    }

    serialized = json.dumps(
        canonical_record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    )

    return hashlib.sha256(
        serialized.encode(
            "utf-8"
        )
    ).hexdigest()


def validate_source_registry(
    registry_df: pd.DataFrame,
) -> None:
    missing_columns = (
        REQUIRED_SOURCE_REGISTRY_COLUMNS
        .difference(registry_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "El registro de sistemas no contiene "
            "las columnas requeridas: "
            f"{sorted(missing_columns)}"
        )

    if registry_df.empty:
        raise ValueError(
            "El registro de sistemas está vacío."
        )

    identifier_columns = [
        "source_system_id",
        "source_system_name",
    ]

    for column in identifier_columns:
        empty_values = (
            registry_df[column]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
        )

        if empty_values.any():
            raise ValueError(
                f"La columna {column!r} contiene "
                "valores vacíos."
            )

        if registry_df[
            column
        ].duplicated().any():
            raise ValueError(
                f"La columna {column!r} contiene "
                "valores duplicados."
            )

    for classification in registry_df[
        "classification"
    ]:
        normalized = normalize_upper_text(
            classification
        )

        if normalized not in (
            VALID_CLASSIFICATIONS
        ):
            raise ValueError(
                "Clasificación de datos desconocida: "
                f"{classification!r}"
            )

    active_values = pd.to_numeric(
        registry_df["active"],
        errors="coerce",
    )

    if active_values.isna().any():
        raise ValueError(
            "La columna 'active' debe contener "
            "solamente 0 o 1."
        )

    if not active_values.isin(
        [
            0,
            1,
        ]
    ).all():
        raise ValueError(
            "La columna 'active' debe contener "
            "solamente 0 o 1."
        )


def get_active_source_system(
    registry_df: pd.DataFrame,
    source_system_id: str,
) -> pd.Series:
    validate_source_registry(
        registry_df
    )

    normalized_id = normalize_text(
        source_system_id
    )

    matches = registry_df.loc[
        registry_df[
            "source_system_id"
        ].astype(str)
        == normalized_id
    ]

    if matches.empty:
        raise ValueError(
            "No existe el sistema de origen: "
            f"{normalized_id}"
        )

    row = matches.iloc[0]

    if int(
        row["active"]
    ) != 1:
        raise ValueError(
            "El sistema de origen está inactivo: "
            f"{normalized_id}"
        )

    return row


def validate_lineage_input(
    data_df: pd.DataFrame,
) -> None:
    missing_columns = (
        LINEAGE_REQUIRED_COLUMNS
        .difference(data_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Los datos no contienen las columnas "
            "requeridas para linaje: "
            f"{sorted(missing_columns)}"
        )

    if data_df.empty:
        raise ValueError(
            "No se puede generar linaje desde "
            "un DataFrame vacío."
        )

    if data_df[
        "case_id"
    ].isna().any():
        raise ValueError(
            "Existen registros sin case_id."
        )

    if data_df[
        "case_id"
    ].duplicated().any():
        duplicated = (
            data_df.loc[
                data_df[
                    "case_id"
                ].duplicated(
                    keep=False
                ),
                "case_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            "Existen case_id duplicados: "
            f"{sorted(duplicated)}"
        )


def build_lineage_records(
    data_df: pd.DataFrame,
    registry_df: pd.DataFrame,
    *,
    source_system_id: str,
    source_file_name: str,
    received_at: str,
    pipeline_id: str,
    pipeline_version: str,
    hash_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Genera un registro de linaje por caso.
    """

    validate_lineage_input(
        data_df
    )

    source = get_active_source_system(
        registry_df,
        source_system_id,
    )

    parsed_received_at = pd.to_datetime(
        received_at,
        errors="coerce",
    )

    if pd.isna(
        parsed_received_at
    ):
        raise ValueError(
            "received_at contiene una fecha "
            "inválida."
        )

    selected_hash_columns = (
        hash_columns
        if hash_columns is not None
        else DEFAULT_HASH_COLUMNS
    )

    missing_hash_columns = [
        column
        for column in selected_hash_columns
        if column not in data_df.columns
    ]

    if missing_hash_columns:
        raise ValueError(
            "No se puede calcular el hash. "
            "Faltan columnas: "
            f"{missing_hash_columns}"
        )

    rows: list[
        dict[str, Any]
    ] = []

    for record in data_df.to_dict(
        orient="records"
    ):
        record_hash = (
            calculate_record_hash(
                record,
                columns=(
                    selected_hash_columns
                ),
            )
        )

        rows.append(
            {
                "case_id": normalize_text(
                    record["case_id"]
                ),
                "source_system_id": normalize_text(
                    source[
                        "source_system_id"
                    ]
                ),
                "source_system_name": normalize_text(
                    source[
                        "source_system_name"
                    ]
                ),
                "source_type": normalize_upper_text(
                    source[
                        "source_type"
                    ]
                ),
                "source_file_name": (
                    normalize_text(
                        source_file_name
                    )
                ),
                "schema_id": normalize_text(
                    source[
                        "schema_id"
                    ]
                ),
                "schema_version": normalize_text(
                    source[
                        "schema_version"
                    ]
                ),
                "pipeline_id": normalize_text(
                    pipeline_id
                ),
                "pipeline_version": normalize_text(
                    pipeline_version
                ),
                "received_at": (
                    parsed_received_at.isoformat()
                ),
                "data_owner_id": normalize_text(
                    source[
                        "data_owner_id"
                    ]
                ),
                "data_steward_id": normalize_text(
                    source[
                        "data_steward_id"
                    ]
                ),
                "classification": (
                    normalize_upper_text(
                        source[
                            "classification"
                        ]
                    )
                ),
                "hash_algorithm": "SHA-256",
                "hash_columns": " | ".join(
                    selected_hash_columns
                ),
                "record_hash": record_hash,
            }
        )

    return pd.DataFrame(
        rows
    )


def calculate_completeness(
    data_df: pd.DataFrame,
    required_columns: list[str],
) -> dict[str, Any]:
    """
    Calcula completitud por celda requerida.
    """

    missing_columns = [
        column
        for column in required_columns
        if column not in data_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "No se puede medir completitud. "
            "Faltan columnas: "
            f"{missing_columns}"
        )

    total_required_cells = (
        len(data_df)
        * len(required_columns)
    )

    missing_cells = 0

    missing_by_column: dict[
        str,
        int
    ] = {}

    for column in required_columns:
        missing_mask = (
            data_df[column]
            .isna()
            | data_df[column]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
        )

        column_missing = int(
            missing_mask.sum()
        )

        missing_by_column[
            column
        ] = column_missing

        missing_cells += column_missing

    complete_cells = (
        total_required_cells
        - missing_cells
    )

    completeness_rate = (
        complete_cells
        / total_required_cells
        if total_required_cells
        else 0.0
    )

    return {
        "total_required_cells": (
            total_required_cells
        ),
        "complete_cells": complete_cells,
        "missing_cells": missing_cells,
        "completeness_rate": round(
            completeness_rate,
            4,
        ),
        "missing_by_column": (
            missing_by_column
        ),
    }


def calculate_validity(
    data_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Evalúa reglas técnicas iniciales.
    """

    validity_errors: list[
        dict[str, str]
    ] = []

    numeric_ranges = {
        "risk_score": (
            0,
            100,
        ),
        "assessment_coverage": (
            0,
            100,
        ),
        "estimated_loss_clp": (
            0,
            None,
        ),
        "digital_event_count": (
            0,
            None,
        ),
    }

    checked_values = 0

    for column, (
        minimum,
        maximum,
    ) in numeric_ranges.items():
        if column not in data_df.columns:
            continue

        numeric_values = pd.to_numeric(
            data_df[column],
            errors="coerce",
        )

        for index, value in (
            numeric_values.items()
        ):
            checked_values += 1

            case_id = normalize_text(
                data_df.loc[
                    index,
                    "case_id",
                ]
            )

            if pd.isna(value):
                validity_errors.append(
                    {
                        "case_id": case_id,
                        "column": column,
                        "reason": (
                            "NON_NUMERIC_VALUE"
                        ),
                    }
                )

                continue

            if value < minimum:
                validity_errors.append(
                    {
                        "case_id": case_id,
                        "column": column,
                        "reason": (
                            "VALUE_BELOW_MINIMUM"
                        ),
                    }
                )

            if (
                maximum is not None
                and value > maximum
            ):
                validity_errors.append(
                    {
                        "case_id": case_id,
                        "column": column,
                        "reason": (
                            "VALUE_ABOVE_MAXIMUM"
                        ),
                    }
                )

    valid_values = (
        checked_values
        - len(validity_errors)
    )

    validity_rate = (
        valid_values / checked_values
        if checked_values
        else 0.0
    )

    return {
        "checked_values": checked_values,
        "valid_values": valid_values,
        "invalid_values": len(
            validity_errors
        ),
        "validity_rate": round(
            validity_rate,
            4,
        ),
        "errors": validity_errors,
    }


def calculate_uniqueness(
    data_df: pd.DataFrame,
    identifier_column: str,
) -> dict[str, Any]:
    if identifier_column not in (
        data_df.columns
    ):
        raise ValueError(
            "No existe la columna identificadora: "
            f"{identifier_column}"
        )

    duplicated_mask = (
        data_df[
            identifier_column
        ].duplicated(
            keep=False
        )
    )

    duplicated_values = sorted(
        data_df.loc[
            duplicated_mask,
            identifier_column,
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    unique_rate = (
        data_df[
            identifier_column
        ].nunique(
            dropna=True
        )
        / len(data_df)
        if len(data_df)
        else 0.0
    )

    return {
        "identifier_column": (
            identifier_column
        ),
        "total_records": int(
            len(data_df)
        ),
        "duplicated_record_count": int(
            duplicated_mask.sum()
        ),
        "duplicated_values": (
            duplicated_values
        ),
        "uniqueness_rate": round(
            unique_rate,
            4,
        ),
    }


def build_data_quality_report(
    data_df: pd.DataFrame,
    lineage_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Genera un informe inicial de calidad.
    """

    validate_lineage_input(
        data_df
    )

    completeness_columns = [
        "case_id",
        "risk_score",
        "assessment_coverage",
        "consolidated_alert_level",
        "consolidated_data_status",
        "estimated_loss_clp",
    ]

    completeness = (
        calculate_completeness(
            data_df,
            completeness_columns,
        )
    )

    validity = calculate_validity(
        data_df
    )

    uniqueness = calculate_uniqueness(
        data_df,
        "case_id",
    )

    expected_lineage_case_ids = set(
        data_df[
            "case_id"
        ].astype(str)
    )

    actual_lineage_case_ids = set(
        lineage_df[
            "case_id"
        ].astype(str)
    )

    missing_lineage = sorted(
        expected_lineage_case_ids
        - actual_lineage_case_ids
    )

    traceability_rate = (
        len(
            actual_lineage_case_ids
            & expected_lineage_case_ids
        )
        / len(
            expected_lineage_case_ids
        )
        if expected_lineage_case_ids
        else 0.0
    )

    overall_status = "APROBADO"

    if (
        completeness[
            "completeness_rate"
        ] < 1.0
        or validity[
            "validity_rate"
        ] < 1.0
        or uniqueness[
            "uniqueness_rate"
        ] < 1.0
        or traceability_rate < 1.0
    ):
        overall_status = (
            "REQUIERE_REVISION"
        )

    return {
        "report_generated_at": (
            datetime.now()
            .astimezone()
            .isoformat(
                timespec="seconds"
            )
        ),
        "total_records": int(
            len(data_df)
        ),
        "overall_status": (
            overall_status
        ),
        "completeness": completeness,
        "validity": validity,
        "uniqueness": uniqueness,
        "traceability": {
            "records_with_lineage": int(
                len(
                    actual_lineage_case_ids
                    & expected_lineage_case_ids
                )
            ),
            "missing_lineage_case_ids": (
                missing_lineage
            ),
            "traceability_rate": round(
                traceability_rate,
                4,
            ),
        },
    }


def validate_raci_matrix(
    raci_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Valida que cada actividad tenga exactamente
    un Accountable y al menos un Responsible.
    """

    missing_columns = (
        REQUIRED_RACI_COLUMNS
        .difference(raci_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "La matriz RACI no contiene "
            "las columnas requeridas: "
            f"{sorted(missing_columns)}"
        )

    if raci_df.empty:
        raise ValueError(
            "La matriz RACI está vacía."
        )

    role_columns = [
        column
        for column in raci_df.columns
        if column not in {
            "activity_id",
            "activity_name",
        }
    ]

    if not role_columns:
        raise ValueError(
            "La matriz RACI no contiene roles."
        )

    result_df = raci_df.copy()

    validation_statuses: list[
        str
    ] = []

    accountable_counts: list[
        int
    ] = []

    responsible_counts: list[
        int
    ] = []

    for _, row in result_df.iterrows():
        role_values = [
            normalize_upper_text(
                row[column]
            )
            for column in role_columns
        ]

        invalid_values = [
            value
            for value in role_values
            if value not in (
                VALID_RACI_VALUES
            )
        ]

        if invalid_values:
            raise ValueError(
                "La matriz RACI contiene códigos "
                "inválidos: "
                f"{sorted(set(invalid_values))}"
            )

        accountable_count = (
            role_values.count("A")
        )

        responsible_count = (
            role_values.count("R")
        )

        accountable_counts.append(
            accountable_count
        )

        responsible_counts.append(
            responsible_count
        )

        if (
            accountable_count == 1
            and responsible_count >= 1
        ):
            validation_statuses.append(
                "VALIDA"
            )
        else:
            validation_statuses.append(
                "REQUIERE_REVISION"
            )

    result_df[
        "accountable_count"
    ] = accountable_counts

    result_df[
        "responsible_count"
    ] = responsible_counts

    result_df[
        "raci_validation_status"
    ] = validation_statuses

    return result_df


def save_json(
    data: dict[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=4,
        )