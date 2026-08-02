from pathlib import Path

import pandas as pd
import pytest

from src.reporting import (
    build_category_summary,
    build_summary_metrics,
    generate_reports,
)


def build_test_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": [
                "CL-001",
                "CL-002",
                "CL-003",
            ],
            "sector": [
                "Banco",
                "Retail",
                "Fintech",
            ],
            "fraud_category": [
                "FRAUDE_DIGITAL",
                "FRAUDE_FINANCIERO",
                "FRAUDE_DIGITAL",
            ],
            "fraud_type_name": [
                "Toma de cuenta",
                "Apropiación indebida",
                "Phishing",
            ],
            "risk_score": [
                80.0,
                60.0,
                10.0,
            ],
            "assessment_coverage": [
                100.0,
                100.0,
                50.0,
            ],
            "alert_level": [
                "CRITICO",
                "ALTO",
                "BAJO",
            ],
            "estimated_loss_clp": [
                10_000_000,
                5_000_000,
                0,
            ],
            "detection_delay_days": [
                120,
                20,
                5,
            ],
            "detection_delay_level": [
                "MUY_TARDIA",
                "OPORTUNA",
                "TEMPRANA",
            ],
        }
    )


def test_build_summary_metrics() -> None:
    df = build_test_dataframe()

    metrics = build_summary_metrics(df)

    assert metrics["total_cases"] == 3
    assert metrics["critical_cases"] == 1
    assert metrics["high_risk_cases"] == 1
    assert metrics["incomplete_cases"] == 0
    assert metrics["late_detection_cases"] == 1

    assert (
        metrics[
            "estimated_loss_total_clp"
        ]
        == 15_000_000
    )

    assert (
        metrics["average_risk_score"]
        == pytest.approx(50.0)
    )

    assert (
        metrics[
            "average_assessment_coverage"
        ]
        == pytest.approx(
            83.33,
            abs=0.01,
        )
    )

    assert (
        metrics[
            "average_detection_delay_days"
        ]
        == pytest.approx(
            48.33,
            abs=0.01,
        )
    )


def test_metrics_include_fraud_categories() -> None:
    df = build_test_dataframe()

    metrics = build_summary_metrics(df)

    assert metrics[
        "cases_by_fraud_category"
    ] == {
        "FRAUDE_DIGITAL": 2,
        "FRAUDE_FINANCIERO": 1,
    }

    assert metrics[
        "estimated_loss_by_fraud_category_clp"
    ] == {
        "FRAUDE_DIGITAL": 10_000_000,
        "FRAUDE_FINANCIERO": 5_000_000,
    }

    assert (
        metrics[
            "top_fraud_category_by_cases"
        ]
        == "FRAUDE_DIGITAL"
    )

    assert (
        metrics[
            "top_fraud_category_by_loss"
        ]
        == "FRAUDE_DIGITAL"
    )


def test_build_category_summary() -> None:
    df = build_test_dataframe()

    summary = build_category_summary(df)

    digital_row = summary.loc[
        summary["fraud_category"]
        == "FRAUDE_DIGITAL"
    ].iloc[0]

    assert digital_row["total_cases"] == 2
    assert digital_row["critical_cases"] == 1

    assert (
        digital_row[
            "estimated_loss_total_clp"
        ]
        == 10_000_000
    )

    assert (
        digital_row[
            "average_risk_score"
        ]
        == pytest.approx(45.0)
    )


def test_generate_reports(
    tmp_path: Path,
) -> None:
    df = build_test_dataframe()

    generated_files = generate_reports(
        df,
        tmp_path,
    )

    assert len(generated_files) == 5

    for generated_file in generated_files:
        assert generated_file.exists()

        assert (
            generated_file.stat().st_size
            > 0
        )