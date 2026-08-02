import pandas as pd
import pytest

from src.typology_enrichment import (
    enrich_cases_with_typology,
)


def test_enrich_account_takeover_case() -> None:
    df = pd.DataFrame(
        {
            "case_id": ["CL-001"],
            "fraud_type_code": [
                "ACCOUNT_TAKEOVER"
            ],
        }
    )

    result = enrich_cases_with_typology(
        df
    )

    assert (
        result.loc[0, "fraud_category"]
        == "FRAUDE_DIGITAL"
    )

    assert (
        result.loc[0, "fraud_type_name"]
        == "Toma de cuenta"
    )

    assert (
        "mfa"
        in result.loc[
            0,
            "recommended_controls",
        ]
    )


def test_missing_fraud_type_is_rejected() -> None:
    df = pd.DataFrame(
        {
            "case_id": ["CL-999"],
            "fraud_type_code": [None],
        }
    )

    with pytest.raises(
        ValueError,
        match="casos sin tipología",
    ):
        enrich_cases_with_typology(
            df
        )