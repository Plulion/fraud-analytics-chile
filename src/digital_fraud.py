from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from pandas import isna


DIGITAL_SIGNAL_WEIGHTS = {
    "new_device": 15.0,
    "unusual_ip": 15.0,
    "geolocation_mismatch": 10.0,
    "failed_mfa_attempts": 15.0,
    "recent_password_reset": 10.0,
    "new_beneficiary": 10.0,
    "amount_spike_ratio": 15.0,
    "rapid_transaction_count_10m": 10.0,
}


@dataclass(frozen=True)
class DigitalFraudAssessment:
    """
    Resultado de la evaluación de un evento digital.

    El puntaje no confirma fraude. Se utiliza para
    priorizar eventos que requieren revisión.
    """

    device_score: float
    network_score: float
    authentication_score: float
    transaction_score: float
    digital_risk_score: float
    assessment_coverage: float
    alert_level: str
    recommended_action: str
    triggered_signals: tuple[str, ...]


def _is_unknown(value: Any) -> bool:
    """
    Determina si un valor no fue informado.

    Reconoce:
    - None
    - NaN de Pandas
    - cadenas vacías
    """

    if value is None:
        return True

    if isinstance(value, str):
        return value.strip() == ""

    try:
        return bool(isna(value))
    except (TypeError, ValueError):
        return False


def _optional_binary(
    value: Any,
    field_name: str,
) -> int | None:
    """
    Convierte una señal binaria a 0, 1 o None.

    0:
        La señal fue evaluada y está ausente.

    1:
        La señal fue evaluada y está presente.

    None:
        La señal todavía no fue evaluada.
    """

    if _is_unknown(value):
        return None

    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"El campo {field_name!r} debe contener "
            "0, 1 o un valor vacío."
        ) from exc

    if numeric_value not in {
        0.0,
        1.0,
    }:
        raise ValueError(
            f"El campo {field_name!r} debe contener "
            "solamente 0, 1 o un valor vacío."
        )

    return int(numeric_value)


def _optional_non_negative_number(
    value: Any,
    field_name: str,
) -> float | None:
    """
    Convierte un valor numérico a float.

    Acepta None cuando el antecedente todavía
    no fue recopilado.
    """

    if _is_unknown(value):
        return None

    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"El campo {field_name!r} debe ser "
            "numérico o estar vacío."
        ) from exc

    if numeric_value < 0:
        raise ValueError(
            f"El campo {field_name!r} no puede "
            "ser negativo."
        )

    return numeric_value


def _score_failed_mfa_attempts(
    attempts: float | None,
) -> tuple[float, str | None]:
    """
    Asigna puntaje según la cantidad de intentos
    fallidos de autenticación multifactor.
    """

    if attempts is None:
        return 0.0, None

    if attempts >= 5:
        return (
            15.0,
            "MFA_FAILURES_HIGH",
        )

    if attempts >= 2:
        return (
            7.5,
            "MFA_FAILURES_MEDIUM",
        )

    return 0.0, None


def _score_amount_spike(
    ratio: float | None,
) -> tuple[float, str | None]:
    """
    Evalúa cuántas veces el monto supera el
    comportamiento normal del cliente.

    Ejemplo:
        ratio = 5.0

    significa que el monto es cinco veces superior
    al valor normal definido para ese cliente.
    """

    if ratio is None:
        return 0.0, None

    if ratio >= 5:
        return (
            15.0,
            "AMOUNT_SPIKE_HIGH",
        )

    if ratio >= 2:
        return (
            7.5,
            "AMOUNT_SPIKE_MEDIUM",
        )

    return 0.0, None


def _score_transaction_velocity(
    transaction_count: float | None,
) -> tuple[float, str | None]:
    """
    Evalúa cuántas transacciones fueron ejecutadas
    en una ventana de diez minutos.
    """

    if transaction_count is None:
        return 0.0, None

    if transaction_count >= 5:
        return (
            10.0,
            "RAPID_TRANSACTIONS_HIGH",
        )

    if transaction_count >= 3:
        return (
            5.0,
            "RAPID_TRANSACTIONS_MEDIUM",
        )

    return 0.0, None


def _classify_digital_risk(
    digital_risk_score: float,
    assessment_coverage: float,
) -> tuple[str, str]:
    """
    Determina el nivel de alerta y la acción sugerida.

    La acción es una recomendación para el proceso
    de revisión. Esta función no bloquea cuentas.
    """

    if assessment_coverage < 70:
        return (
            "INCOMPLETO",
            "REQUIERE_MAS_ANTECEDENTES",
        )

    if digital_risk_score >= 70:
        return (
            "CRITICO",
            "BLOQUEO_TEMPORAL_Y_REVISION",
        )

    if digital_risk_score >= 50:
        return (
            "ALTO",
            "DESAFIO_MFA_Y_REVISION",
        )

    if digital_risk_score >= 30:
        return (
            "MEDIO",
            "MONITOREO_REFORZADO",
        )

    return (
        "BAJO",
        "MONITOREO_NORMAL",
    )


def assess_digital_event(
    event: Mapping[str, Any],
) -> DigitalFraudAssessment:
    """
    Evalúa un inicio de sesión o una transacción
    utilizando señales técnicas de fraude digital.
    """

    new_device = _optional_binary(
        event.get("new_device"),
        "new_device",
    )

    unusual_ip = _optional_binary(
        event.get("unusual_ip"),
        "unusual_ip",
    )

    geolocation_mismatch = _optional_binary(
        event.get("geolocation_mismatch"),
        "geolocation_mismatch",
    )

    recent_password_reset = _optional_binary(
        event.get("recent_password_reset"),
        "recent_password_reset",
    )

    new_beneficiary = _optional_binary(
        event.get("new_beneficiary"),
        "new_beneficiary",
    )

    failed_mfa_attempts = (
        _optional_non_negative_number(
            event.get("failed_mfa_attempts"),
            "failed_mfa_attempts",
        )
    )

    amount_spike_ratio = (
        _optional_non_negative_number(
            event.get("amount_spike_ratio"),
            "amount_spike_ratio",
        )
    )

    rapid_transaction_count = (
        _optional_non_negative_number(
            event.get(
                "rapid_transaction_count_10m"
            ),
            "rapid_transaction_count_10m",
        )
    )

    triggered_signals: list[str] = []

    device_score = 0.0

    if new_device == 1:
        device_score = (
            DIGITAL_SIGNAL_WEIGHTS[
                "new_device"
            ]
        )

        triggered_signals.append(
            "NEW_DEVICE"
        )

    network_score = 0.0

    if unusual_ip == 1:
        network_score += (
            DIGITAL_SIGNAL_WEIGHTS[
                "unusual_ip"
            ]
        )

        triggered_signals.append(
            "UNUSUAL_IP"
        )

    if geolocation_mismatch == 1:
        network_score += (
            DIGITAL_SIGNAL_WEIGHTS[
                "geolocation_mismatch"
            ]
        )

        triggered_signals.append(
            "GEOLOCATION_MISMATCH"
        )

    authentication_score = 0.0

    mfa_score, mfa_signal = (
        _score_failed_mfa_attempts(
            failed_mfa_attempts
        )
    )

    authentication_score += mfa_score

    if mfa_signal is not None:
        triggered_signals.append(
            mfa_signal
        )

    if recent_password_reset == 1:
        authentication_score += (
            DIGITAL_SIGNAL_WEIGHTS[
                "recent_password_reset"
            ]
        )

        triggered_signals.append(
            "RECENT_PASSWORD_RESET"
        )

    transaction_score = 0.0

    if new_beneficiary == 1:
        transaction_score += (
            DIGITAL_SIGNAL_WEIGHTS[
                "new_beneficiary"
            ]
        )

        triggered_signals.append(
            "NEW_BENEFICIARY"
        )

    amount_score, amount_signal = (
        _score_amount_spike(
            amount_spike_ratio
        )
    )

    transaction_score += amount_score

    if amount_signal is not None:
        triggered_signals.append(
            amount_signal
        )

    velocity_score, velocity_signal = (
        _score_transaction_velocity(
            rapid_transaction_count
        )
    )

    transaction_score += velocity_score

    if velocity_signal is not None:
        triggered_signals.append(
            velocity_signal
        )

    digital_risk_score = round(
        device_score
        + network_score
        + authentication_score
        + transaction_score,
        2,
    )

    evaluated_values = {
        "new_device": new_device,
        "unusual_ip": unusual_ip,
        "geolocation_mismatch": (
            geolocation_mismatch
        ),
        "failed_mfa_attempts": (
            failed_mfa_attempts
        ),
        "recent_password_reset": (
            recent_password_reset
        ),
        "new_beneficiary": new_beneficiary,
        "amount_spike_ratio": (
            amount_spike_ratio
        ),
        "rapid_transaction_count_10m": (
            rapid_transaction_count
        ),
    }

    assessed_weight = sum(
        DIGITAL_SIGNAL_WEIGHTS[field_name]
        for field_name, value
        in evaluated_values.items()
        if value is not None
    )

    total_weight = sum(
        DIGITAL_SIGNAL_WEIGHTS.values()
    )

    assessment_coverage = round(
        (
            assessed_weight
            / total_weight
        )
        * 100,
        2,
    )

    alert_level, recommended_action = (
        _classify_digital_risk(
            digital_risk_score,
            assessment_coverage,
        )
    )

    return DigitalFraudAssessment(
        device_score=round(
            device_score,
            2,
        ),
        network_score=round(
            network_score,
            2,
        ),
        authentication_score=round(
            authentication_score,
            2,
        ),
        transaction_score=round(
            transaction_score,
            2,
        ),
        digital_risk_score=(
            digital_risk_score
        ),
        assessment_coverage=(
            assessment_coverage
        ),
        alert_level=alert_level,
        recommended_action=(
            recommended_action
        ),
        triggered_signals=tuple(
            triggered_signals
        ),
    )