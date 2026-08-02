import pytest

from src.case_metrics import (
    calculate_average_loss_per_day,
    classify_detection_delay,
)


def test_early_detection() -> None:
    result = classify_detection_delay(3)

    assert result == "TEMPRANA"


def test_very_late_detection() -> None:
    result = classify_detection_delay(210)

    assert result == "MUY_TARDIA"


def test_municipality_average_loss() -> None:
    result = calculate_average_loss_per_day(
        estimated_loss_clp=42_000_000,
        detection_delay_days=210,
    )

    assert result == 200_000.0


def test_negative_delay_is_invalid() -> None:
    with pytest.raises(ValueError):
        classify_detection_delay(-1)