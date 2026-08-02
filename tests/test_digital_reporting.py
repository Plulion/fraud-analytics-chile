from pathlib import Path

import pandas as pd

from src.digital_reporting import (
    build_digital_summary,
    count_triggered_signals,
    generate_digital_reports,
)


def build_reporting_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": [
                "EV-001",
                "EV-002",
            ],
            "customer_id": [
                "CUS-001",
                "CUS-002",
            ],
            "digital_risk_score": [
                100.0,
                0.0,
            ],
            "digital_assessment_coverage": [
                100.0,
                100.0,
            ],
            "digital_alert_level": [
                "CRITICO",
                "BAJO",
            ],
            "digital_recommended_action": [
                "BLOQUEO_TEMPORAL_Y_REVISION",
                "MONITOREO_NORMAL",
            ],
            "digital_triggered_signals": [
                (
                    "NEW_DEVICE | "
                    "UNUSUAL_IP | "
                    "NEW_BENEFICIARY"
                ),
                "",
            ],
        }
    )


def test_count_triggered_signals() -> None:
    df = build_reporting_dataframe()

    counts = count_triggered_signals(
        df
    )

    assert counts == {
        "NEW_BENEFICIARY": 1,
        "NEW_DEVICE": 1,
        "UNUSUAL_IP": 1,
    }


def test_build_digital_summary() -> None:
    df = build_reporting_dataframe()

    summary = build_digital_summary(
        df
    )

    assert summary["total_events"] == 2
    assert summary["unique_customers"] == 2
    assert summary["critical_events"] == 1

    assert (
        summary[
            "average_digital_risk_score"
        ]
        == 50.0
    )


def test_generate_digital_reports(
    tmp_path: Path,
) -> None:
    df = build_reporting_dataframe()

    files = generate_digital_reports(
        df,
        tmp_path,
    )

    assert len(files) == 2

    for file_path in files:
        assert file_path.exists()
        assert file_path.stat().st_size > 0