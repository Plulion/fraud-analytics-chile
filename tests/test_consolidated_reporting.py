from pathlib import Path

from src.case_consolidation import (
    consolidate_case_and_digital_data,
)
from src.consolidated_reporting import (
    build_consolidated_summary,
    generate_consolidated_reports,
)
from tests.test_case_consolidation import (
    build_digital_dataframe,
    build_general_dataframe,
)


def build_consolidated_dataframe():
    return consolidate_case_and_digital_data(
        build_general_dataframe(),
        build_digital_dataframe(),
    )


def test_build_consolidated_summary() -> None:
    df = build_consolidated_dataframe()

    summary = build_consolidated_summary(
        df
    )

    assert summary["total_cases"] == 2

    assert (
        summary[
            "cases_with_digital_events"
        ]
        == 1
    )

    assert (
        summary[
            "cases_without_digital_events"
        ]
        == 1
    )

    assert (
        summary[
            "total_digital_events"
        ]
        == 2
    )

    assert (
        summary[
            "critical_consolidated_cases"
        ]
        == 1
    )


def test_generate_consolidated_reports(
    tmp_path: Path,
) -> None:
    df = build_consolidated_dataframe()

    generated_files = (
        generate_consolidated_reports(
            df,
            tmp_path,
        )
    )

    assert len(generated_files) == 2

    for file_path in generated_files:
        assert file_path.exists()
        assert file_path.stat().st_size > 0