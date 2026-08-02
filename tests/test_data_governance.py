import pandas as pd
import pytest

from src.data_governance import (
    build_data_quality_report,
    build_lineage_records,
    calculate_record_hash,
    validate_raci_matrix,
)


def build_registry_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_system_id": [
                "SRC-001",
            ],
            "source_system_name": [
                "CASE_SYSTEM",
            ],
            "source_type": [
                "INTERNAL_APPLICATION",
            ],
            "data_owner_id": [
                "OWNER-001",
            ],
            "data_steward_id": [
                "STEWARD-001",
            ],
            "schema_id": [
                "CASE_SCHEMA",
            ],
            "schema_version": [
                "1.0.0",
            ],
            "classification": [
                "CONFIDENTIAL",
            ],
            "active": [
                1,
            ],
        }
    )


def build_case_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": [
                "CL-001",
                "CL-002",
            ],
            "risk_score": [
                80.0,
                40.0,
            ],
            "assessment_coverage": [
                100.0,
                90.0,
            ],
            "consolidated_alert_level": [
                "CRITICO",
                "MEDIO",
            ],
            "consolidated_data_status": [
                "COMPLETO",
                "COMPLETO",
            ],
            "estimated_loss_clp": [
                10_000_000,
                2_000_000,
            ],
            "digital_event_count": [
                3,
                1,
            ],
            "digital_risk_score_max": [
                100.0,
                40.0,
            ],
        }
    )


def test_record_hash_is_reproducible() -> None:
    record = {
        "case_id": "CL-001",
        "risk_score": 80.0,
    }

    first_hash = calculate_record_hash(
        record,
        columns=[
            "case_id",
            "risk_score",
        ],
    )

    second_hash = calculate_record_hash(
        record,
        columns=[
            "case_id",
            "risk_score",
        ],
    )

    assert first_hash == second_hash
    assert len(first_hash) == 64


def test_record_hash_changes_with_data() -> None:
    first_record = {
        "case_id": "CL-001",
        "risk_score": 80.0,
    }

    second_record = {
        "case_id": "CL-001",
        "risk_score": 81.0,
    }

    first_hash = calculate_record_hash(
        first_record,
        columns=[
            "case_id",
            "risk_score",
        ],
    )

    second_hash = calculate_record_hash(
        second_record,
        columns=[
            "case_id",
            "risk_score",
        ],
    )

    assert first_hash != second_hash


def test_build_lineage_records() -> None:
    result = build_lineage_records(
        build_case_dataframe(),
        build_registry_dataframe(),
        source_system_id="SRC-001",
        source_file_name="cases.csv",
        received_at=(
            "2026-08-01T20:00:00-04:00"
        ),
        pipeline_id="PIPELINE-001",
        pipeline_version="1.0.0",
    )

    assert len(result) == 2

    assert (
        result["source_system_id"]
        .eq("SRC-001")
        .all()
    )

    assert (
        result["data_owner_id"]
        .eq("OWNER-001")
        .all()
    )

    assert (
        result["record_hash"]
        .str.len()
        .eq(64)
        .all()
    )


def test_quality_report_is_approved() -> None:
    cases_df = build_case_dataframe()

    lineage_df = build_lineage_records(
        cases_df,
        build_registry_dataframe(),
        source_system_id="SRC-001",
        source_file_name="cases.csv",
        received_at=(
            "2026-08-01T20:00:00-04:00"
        ),
        pipeline_id="PIPELINE-001",
        pipeline_version="1.0.0",
    )

    report = build_data_quality_report(
        cases_df,
        lineage_df,
    )

    assert (
        report["overall_status"]
        == "APROBADO"
    )

    assert (
        report["completeness"][
            "completeness_rate"
        ]
        == 1.0
    )

    assert (
        report["traceability"][
            "traceability_rate"
        ]
        == 1.0
    )


def test_quality_report_detects_missing_value() -> None:
    cases_df = build_case_dataframe()

    cases_df.loc[
        0,
        "consolidated_data_status",
    ] = ""

    lineage_df = build_lineage_records(
        cases_df,
        build_registry_dataframe(),
        source_system_id="SRC-001",
        source_file_name="cases.csv",
        received_at=(
            "2026-08-01T20:00:00-04:00"
        ),
        pipeline_id="PIPELINE-001",
        pipeline_version="1.0.0",
    )

    report = build_data_quality_report(
        cases_df,
        lineage_df,
    )

    assert (
        report["overall_status"]
        == "REQUIERE_REVISION"
    )

    assert (
        report["completeness"][
            "missing_cells"
        ]
        == 1
    )


def test_valid_raci_matrix() -> None:
    raci_df = pd.DataFrame(
        {
            "activity_id": [
                "ACT-001",
            ],
            "activity_name": [
                "INVESTIGATE_ALERT",
            ],
            "ANALYST": [
                "R",
            ],
            "SUPERVISOR": [
                "A",
            ],
            "LEGAL": [
                "C",
            ],
        }
    )

    result = validate_raci_matrix(
        raci_df
    )

    assert (
        result.iloc[0][
            "raci_validation_status"
        ]
        == "VALIDA"
    )


def test_raci_with_two_accountables_requires_review() -> None:
    raci_df = pd.DataFrame(
        {
            "activity_id": [
                "ACT-001",
            ],
            "activity_name": [
                "INVESTIGATE_ALERT",
            ],
            "ANALYST": [
                "R",
            ],
            "SUPERVISOR": [
                "A",
            ],
            "DATA_OWNER": [
                "A",
            ],
        }
    )

    result = validate_raci_matrix(
        raci_df
    )

    assert (
        result.iloc[0][
            "raci_validation_status"
        ]
        == "REQUIERE_REVISION"
    )

    assert (
        result.iloc[0][
            "accountable_count"
        ]
        == 2
    )


def test_inactive_source_is_rejected() -> None:
    registry_df = (
        build_registry_dataframe()
    )

    registry_df.loc[
        0,
        "active",
    ] = 0

    with pytest.raises(
        ValueError,
        match="sistema de origen está inactivo",
    ):
        build_lineage_records(
            build_case_dataframe(),
            registry_df,
            source_system_id="SRC-001",
            source_file_name="cases.csv",
            received_at=(
                "2026-08-01T20:00:00-04:00"
            ),
            pipeline_id="PIPELINE-001",
            pipeline_version="1.0.0",
        )