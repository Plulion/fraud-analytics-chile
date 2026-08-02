import pytest

from src.fraud_taxonomy import (
    get_fraud_typology,
    list_typologies_by_category,
    normalize_fraud_code,
)


def test_normalize_fraud_code() -> None:
    result = normalize_fraud_code(
        "account takeover"
    )

    assert result == "ACCOUNT_TAKEOVER"


def test_get_account_takeover() -> None:
    typology = get_fraud_typology(
        "ACCOUNT_TAKEOVER"
    )

    assert typology.category == "FRAUDE_DIGITAL"
    assert typology.display_name == "Toma de cuenta"
    assert "mfa" in typology.recommended_controls
    assert (
        "analisis_de_comportamiento"
        in typology.recommended_controls
    )


def test_insurance_types_are_different() -> None:
    hard_fraud = get_fraud_typology(
        "INSURANCE_HARD_FRAUD"
    )

    soft_fraud = get_fraud_typology(
        "INSURANCE_SOFT_FRAUD"
    )

    assert hard_fraud.code != soft_fraud.code

    assert (
        "siniestro_inventado"
        in hard_fraud.common_tactics
    )

    assert (
        "inflacion_del_monto"
        in soft_fraud.common_tactics
    )


def test_list_digital_typologies() -> None:
    typologies = list_typologies_by_category(
        "FRAUDE_DIGITAL"
    )

    codes = {
        typology.code
        for typology in typologies
    }

    assert codes == {
        "ACCOUNT_TAKEOVER",
        "PHISHING",
    }


def test_unknown_typology_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="Tipología de fraude desconocida",
    ):
        get_fraud_typology(
            "TIPO_INVENTADO"
        )