from __future__ import annotations

from pathlib import Path

import pandas as pd


DATA_PATH = Path(
    "data/casos_sinteticos_chile.csv"
)


CASE_FRAUD_TYPES = {
    "CL-001": "ACCOUNT_TAKEOVER",
    "CL-002": "CORRUPTION",
    "CL-003": "PAYMENT_CARD_FRAUD",
    "CL-004": "INSURANCE_SOFT_FRAUD",
    "CL-005": "FINANCIAL_STATEMENT_FRAUD",
    "CL-006": "EMBEZZLEMENT",
    "CL-007": "IDENTITY_THEFT",
    "CL-008": "PAYMENT_CARD_FRAUD",
    "CL-009": "EMBEZZLEMENT",
    "CL-010": "SYNTHETIC_IDENTITY",
    "CL-011": "IDENTITY_THEFT",
    "CL-012": "PHISHING",
}


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "No se encontró el archivo: "
            f"{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if "case_id" not in df.columns:
        raise ValueError(
            "El CSV no contiene la columna 'case_id'."
        )

    if df["case_id"].duplicated().any():
        duplicated_cases = (
            df.loc[
                df["case_id"].duplicated(),
                "case_id",
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "Existen identificadores duplicados: "
            f"{duplicated_cases}"
        )

    unknown_cases = sorted(
        set(df["case_id"])
        - set(CASE_FRAUD_TYPES)
    )

    if unknown_cases:
        raise ValueError(
            "No existe una tipología asignada "
            "para estos casos: "
            f"{unknown_cases}"
        )

    df["fraud_type_code"] = (
        df["case_id"]
        .map(CASE_FRAUD_TYPES)
    )

    if df["fraud_type_code"].isna().any():
        raise ValueError(
            "Al menos un caso quedó sin tipología."
        )

    df.to_csv(
        DATA_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "Tipologías agregadas correctamente."
    )

    print(
        df[
            [
                "case_id",
                "sector",
                "fraud_type_code",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nArchivo actualizado en: "
        f"{DATA_PATH.resolve()}"
    )


if __name__ == "__main__":
    main()