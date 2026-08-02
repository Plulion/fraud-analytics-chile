"""
Case-level descriptive fraud metrics.

This module provides small, explainable metrics used by the project to
describe detection timing and estimated financial exposure.

Business interpretation
-----------------------
- Detection delay is grouped into educational categories.
- The thresholds are project-specific and do not represent an official rule,
  regulation, or service-level agreement.
- Average loss per day is a descriptive ratio only.
- The ratio must not be interpreted as proof that the loss accumulated evenly
  over time.

Validation principles
---------------------
- Negative detection delays are rejected.
- Negative estimated losses are rejected.
- A zero-day delay uses one exposure day to avoid division by zero while
  preserving a meaningful descriptive output.
"""

from __future__ import annotations


def classify_detection_delay(
    detection_delay_days: int,
) -> str:
    """
    Classify how long the organization took to detect a case.

    The thresholds are educational values defined for this project and do not
    represent an official standard.

    Args:
        detection_delay_days:
            Number of calendar days between case occurrence and detection.

    Returns:
        ``TEMPRANA``, ``OPORTUNA``, ``TARDIA``, or ``MUY_TARDIA``.

    Raises:
        ValueError: If the detection delay is negative.
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
    Calculate a descriptive average estimated loss per exposure day.

    This ratio does not assume that losses occurred uniformly throughout the
    detection period.

    Args:
        estimated_loss_clp:
            Estimated financial loss in Chilean pesos.
        detection_delay_days:
            Number of days between occurrence and detection.

    Returns:
        Estimated loss divided by exposure days, rounded to two decimals.

    Raises:
        ValueError:
            If estimated loss or detection delay is negative.
    """

    if estimated_loss_clp < 0:
        raise ValueError(
            "La pérdida estimada no puede ser negativa."
        )

    if detection_delay_days < 0:
        raise ValueError(
            "La demora de detección no puede ser negativa."
        )

    # Use one day for same-day detection to avoid division by zero without
    # changing the reported estimated loss.
    exposure_days = max(detection_delay_days, 1)

    return round(
        estimated_loss_clp / exposure_days,
        2,
    )