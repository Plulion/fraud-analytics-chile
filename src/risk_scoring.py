from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from pandas import isna


REQUIRED_COLUMNS = {
    "weak_controls",
    "privileged_access",
    "financial_pressure",
    "performance_pressure",
    "rationalization_signal",
    "collusion_signal",
    "high_competence",
}


SIGNAL_WEIGHTS = {
    "weak_controls": 25.0,
    "privileged_access": 25.0,
    "financial_pressure": 15.0,
    "performance_pressure": 15.0,
    "rationalization_signal": 15.0,
    "collusion_signal": 2.5,
    "high_competence": 2.5,
}


@dataclass(frozen=True)
class FraudTriangleAssessment:
    opportunity: float
    pressure: float
    rationalization: float
    aggravating_factors: float
    risk_score: float
    assessment_coverage: float
    alert_level: str
    decision_status: str


def _optional_binary(
    value: object,
    field_name: str,
) -> int | None:
    """
    Convierte un valor en 0, 1 o None.

    0    = factor evaluado y no observado.
    1    = factor evaluado y observado.
    None = factor todavía desconocido.
    """

    if value is None or isna(value):
        return None

    if isinstance(value, str) and not value.strip():
        return None

    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field_name} debe ser 0, 1 o vacío; "
            f"recibido: {value!r}"
        ) from exc

    if parsed not in (0, 1):
        raise ValueError(
            f"{field_name} debe ser 0, 1 o vacío; "
            f"recibido: {parsed}"
        )

    return parsed


def _calculate_component(
    signals: list[tuple[int | None, float]],
) -> float:
    """
    Suma los puntos de las señales observadas.

    Las señales desconocidas no suman puntos,
    pero tampoco se consideran evaluadas.
    """

    return sum(
        value * weight
        for value, weight in signals
        if value is not None
    )


def assess_case(
    row: Mapping[str, object],
) -> FraudTriangleAssessment:
    """
    Evalúa un caso mediante señales del Triángulo del Fraude.

    El resultado es un puntaje de priorización, no una
    probabilidad de fraude ni una declaración de culpabilidad.
    """

    missing = REQUIRED_COLUMNS.difference(row.keys())

    if missing:
        raise KeyError(
            f"Faltan columnas requeridas: {sorted(missing)}"
        )

    values = {
        field_name: _optional_binary(
            row[field_name],
            field_name,
        )
        for field_name in REQUIRED_COLUMNS
    }

    opportunity = _calculate_component(
        [
            (
                values["weak_controls"],
                SIGNAL_WEIGHTS["weak_controls"],
            ),
            (
                values["privileged_access"],
                SIGNAL_WEIGHTS["privileged_access"],
            ),
        ]
    )

    pressure = _calculate_component(
        [
            (
                values["financial_pressure"],
                SIGNAL_WEIGHTS["financial_pressure"],
            ),
            (
                values["performance_pressure"],
                SIGNAL_WEIGHTS["performance_pressure"],
            ),
        ]
    )

    rationalization = _calculate_component(
        [
            (
                values["rationalization_signal"],
                SIGNAL_WEIGHTS["rationalization_signal"],
            )
        ]
    )

    aggravating_factors = _calculate_component(
        [
            (
                values["collusion_signal"],
                SIGNAL_WEIGHTS["collusion_signal"],
            ),
            (
                values["high_competence"],
                SIGNAL_WEIGHTS["high_competence"],
            ),
        ]
    )

    risk_score = round(
        opportunity
        + pressure
        + rationalization
        + aggravating_factors,
        2,
    )

    total_weight = sum(SIGNAL_WEIGHTS.values())

    assessed_weight = sum(
        SIGNAL_WEIGHTS[field_name]
        for field_name, value in values.items()
        if value is not None
    )

    assessment_coverage = round(
        100.0 * assessed_weight / total_weight,
        2,
    )

    if assessment_coverage < 70.0:
        alert_level = "INCOMPLETO"
        decision_status = "REQUIERE_MAS_ANTECEDENTES"

    elif risk_score >= 70.0:
        alert_level = "CRITICO"
        decision_status = "REVISION_PRIORITARIA"

    elif risk_score >= 50.0:
        alert_level = "ALTO"
        decision_status = "REVISION_REFORZADA"

    elif risk_score >= 30.0:
        alert_level = "MEDIO"
        decision_status = "REVISION_ESTANDAR"

    else:
        alert_level = "BAJO"
        decision_status = "REVISION_ESTANDAR"

    return FraudTriangleAssessment(
        opportunity=round(opportunity, 2),
        pressure=round(pressure, 2),
        rationalization=round(rationalization, 2),
        aggravating_factors=round(
            aggravating_factors,
            2,
        ),
        risk_score=risk_score,
        assessment_coverage=assessment_coverage,
        alert_level=alert_level,
        decision_status=decision_status,
    )