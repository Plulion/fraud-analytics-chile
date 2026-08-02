from __future__ import annotations

import pandas as pd

from src.fraud_taxonomy import (
    FraudTypology,
    get_fraud_typology,
)


def validate_fraud_type_column(
    df: pd.DataFrame,
) -> None:
    """
    Comprueba que todos los casos tengan un código
    de tipología utilizable.
    """

    if "fraud_type_code" not in df.columns:
        raise ValueError(
            "Falta la columna 'fraud_type_code'."
        )

    if df["fraud_type_code"].isna().any():
        cases_without_typology = (
            df.loc[
                df["fraud_type_code"].isna(),
                "case_id",
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "Existen casos sin tipología: "
            f"{cases_without_typology}"
        )


def load_typologies(
    df: pd.DataFrame,
) -> list[FraudTypology]:
    """
    Recupera desde el catálogo la ficha
    correspondiente a cada fila.
    """

    validate_fraud_type_column(df)

    return [
        get_fraud_typology(
            str(fraud_type_code)
        )
        for fraud_type_code
        in df["fraud_type_code"]
    ]


def enrich_cases_with_typology(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Agrega al DataFrame la información descriptiva
    y operacional de cada tipología.
    """

    result_df = df.copy()

    typologies = load_typologies(
        result_df
    )

    result_df["fraud_type_code"] = [
        typology.code
        for typology in typologies
    ]

    result_df["fraud_category"] = [
        typology.category
        for typology in typologies
    ]

    result_df["fraud_type_name"] = [
        typology.display_name
        for typology in typologies
    ]

    result_df["fraud_type_description"] = [
        typology.description
        for typology in typologies
    ]

    result_df["common_tactics"] = [
        " | ".join(
            typology.common_tactics
        )
        for typology in typologies
    ]

    result_df["recommended_data"] = [
        " | ".join(
            typology.recommended_data
        )
        for typology in typologies
    ]

    result_df["recommended_controls"] = [
        " | ".join(
            typology.recommended_controls
        )
        for typology in typologies
    ]

    return result_df