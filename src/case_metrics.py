from __future__ import annotations


def classify_detection_delay(
    detection_delay_days: int,
) -> str:
    """
    Clasifica cuánto tardó la organización en detectar el caso.

    Los límites utilizados pertenecen a este ejercicio educativo;
    no representan una norma oficial.
    """

    if detection_delay_days < 0:
        raise ValueError(
            "La demora de detección no puede ser negativa."
        )

    if detection_delay_days <= 7:
        return "TEMPRANA"

    if detection_delay_days <= 30:
        return "OPORTUNA"

    if detection_delay_days <= 90:
        return "TARDIA"

    return "MUY_TARDIA"


def calculate_average_loss_per_day(
    estimated_loss_clp: float,
    detection_delay_days: int,
) -> float:
    """
    Calcula una pérdida diaria promedio descriptiva.

    No supone que la pérdida haya ocurrido uniformemente
    durante todos los días.
    """

    if estimated_loss_clp < 0:
        raise ValueError(
            "La pérdida estimada no puede ser negativa."
        )

    if detection_delay_days < 0:
        raise ValueError(
            "La demora de detección no puede ser negativa."
        )

    exposure_days = max(detection_delay_days, 1)

    return round(
        estimated_loss_clp / exposure_days,
        2,
    )