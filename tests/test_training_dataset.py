import pandas as pd
import pytest

from src.training_dataset import (
    build_training_dataset,
)


FEATURE_COLUMNS = [
    "amount_clp",
    "new_device",
]


def build_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": [
                "CL-001",
                "CL-002",
                "CL-003",
            ],
            "feature_snapshot_at": [
                "2026-08-01T20:00:00-04:00",
                "2026-08-01T20:00:00-04:00",
                "2026-08-02T10:00:00-04:00",
            ],
            "amount_clp": [
                1000,
                2000,
                3000,
            ],
            "new_device": [
                0,
                1,
                1,
            ],
        }
    )


def build_feedback() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": [
                "CL-001",
                "CL-002",
                "CL-003",
            ],
            "resolution_id": [
                "RES-001",
                "RES-002",
                "RES-003",
            ],
            "investigation_outcome": [
                "CONFIRMADO",
                "DESCARTADO",
                "CONFIRMADO",
            ],
            "outcome_label": [
                1,
                0,
                1,
            ],
            "closed_at": [
                "2026-08-01T22:00:00-04:00",
                "2026-08-01T22:00:00-04:00",
                "2026-08-02T09:00:00-04:00",
            ],
            "feedback_source": [
                "TEST",
                "TEST",
                "TEST",
            ],
            "feedback_status": [
                "ACTIVO",
                "INVALIDADO",
                "ACTIVO",
            ],
        }
    )


def test_only_active_non_leaking_records_are_included() -> None:
    result = build_training_dataset(
        build_features(),
        build_feedback(),
        feature_columns=FEATURE_COLUMNS,
    )

    assert (
        result.dataset[
            "case_id"
        ].tolist()
        == ["CL-001"]
    )

    assert (
        result.dataset.iloc[0][
            "outcome_label"
        ]
        == 1
    )


def test_invalidated_feedback_is_excluded() -> None:
    result = build_training_dataset(
        build_features(),
        build_feedback(),
        feature_columns=FEATURE_COLUMNS,
    )

    exclusion = result.exclusions.loc[
        result.exclusions[
            "case_id"
        ].eq("CL-002")
    ].iloc[0]

    assert (
        exclusion[
            "exclusion_reason_code"
        ]
        == "FEEDBACK_NO_ACTIVO"
    )


def test_temporal_leakage_is_excluded() -> None:
    result = build_training_dataset(
        build_features(),
        build_feedback(),
        feature_columns=FEATURE_COLUMNS,
    )

    exclusion = result.exclusions.loc[
        result.exclusions[
            "case_id"
        ].eq("CL-003")
    ].iloc[0]

    assert (
        exclusion[
            "exclusion_reason_code"
        ]
        == "FUGA_TEMPORAL"
    )


def test_missing_feedback_status_defaults_to_active() -> None:
    feedback = build_feedback().drop(
        columns=[
            "feedback_status"
        ]
    )

    result = build_training_dataset(
        build_features().iloc[
            [0]
        ].copy(),
        feedback.iloc[
            [0]
        ].copy(),
        feature_columns=FEATURE_COLUMNS,
    )

    assert len(
        result.dataset
    ) == 1


def test_inconsistent_label_is_rejected() -> None:
    feedback = build_feedback()
    feedback.loc[
        0,
        "outcome_label",
    ] = 0

    with pytest.raises(
        ValueError,
        match="no coincide",
    ):
        build_training_dataset(
            build_features(),
            feedback,
            feature_columns=FEATURE_COLUMNS,
        )


def test_non_binary_label_is_rejected() -> None:
    feedback = build_feedback()
    feedback.loc[
        0,
        "outcome_label",
    ] = 2

    with pytest.raises(
        ValueError,
        match="0 o 1",
    ):
        build_training_dataset(
            build_features(),
            feedback,
            feature_columns=FEATURE_COLUMNS,
        )


def test_forbidden_feature_is_rejected() -> None:
    features = build_features()
    features[
        "confirmed_loss_clp"
    ] = [
        100,
        200,
        300,
    ]

    with pytest.raises(
        ValueError,
        match="fuga de información",
    ):
        build_training_dataset(
            features,
            build_feedback(),
            feature_columns=[
                "amount_clp",
                "confirmed_loss_clp",
            ],
        )


def test_duplicate_feature_case_is_rejected() -> None:
    features = pd.concat(
        [
            build_features(),
            build_features().iloc[
                [0]
            ],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="case_id duplicados",
    ):
        build_training_dataset(
            features,
            build_feedback(),
            feature_columns=FEATURE_COLUMNS,
        )


def test_invalid_feature_timestamp_is_rejected() -> None:
    features = build_features()
    features.loc[
        0,
        "feature_snapshot_at",
    ] = "fecha-invalida"

    with pytest.raises(
        ValueError,
        match="feature_snapshot_at",
    ):
        build_training_dataset(
            features,
            build_feedback(),
            feature_columns=FEATURE_COLUMNS,
        )


def test_invalidated_version_overrides_active_version() -> None:
    feedback = pd.concat(
        [
            build_feedback().iloc[
                [0]
            ].assign(
                feedback_status="ACTIVO"
            ),
            build_feedback().iloc[
                [0]
            ].assign(
                feedback_status="INVALIDADO",
                invalidated_at=(
                    "2026-08-02T09:00:00-04:00"
                ),
            ),
        ],
        ignore_index=True,
    )

    result = build_training_dataset(
        build_features().iloc[
            [0]
        ].copy(),
        feedback,
        feature_columns=FEATURE_COLUMNS,
    )

    assert result.dataset.empty

    assert (
        result.exclusions.iloc[0][
            "exclusion_reason_code"
        ]
        == "FEEDBACK_NO_ACTIVO"
    )


def test_multiple_active_labels_for_same_case_are_excluded() -> None:
    feedback = pd.DataFrame(
        {
            "case_id": [
                "CL-001",
                "CL-001",
            ],
            "resolution_id": [
                "RES-001",
                "RES-002",
            ],
            "investigation_outcome": [
                "CONFIRMADO",
                "DESCARTADO",
            ],
            "outcome_label": [
                1,
                0,
            ],
            "closed_at": [
                "2026-08-01T22:00:00-04:00",
                "2026-08-02T22:00:00-04:00",
            ],
            "feedback_source": [
                "TEST",
                "TEST",
            ],
            "feedback_status": [
                "ACTIVO",
                "ACTIVO",
            ],
        }
    )

    result = build_training_dataset(
        build_features().iloc[
            [0]
        ].copy(),
        feedback,
        feature_columns=FEATURE_COLUMNS,
    )

    assert result.dataset.empty

    assert (
        result.exclusions[
            "exclusion_reason_code"
        ].eq(
            "MULTIPLES_ETIQUETAS_ACTIVAS"
        ).sum()
        == 2
    )


def test_report_counts_are_correct() -> None:
    result = build_training_dataset(
        build_features(),
        build_feedback(),
        feature_columns=FEATURE_COLUMNS,
    )

    assert (
        result.report[
            "included_training_records"
        ]
        == 1
    )

    assert (
        result.report[
            "excluded_records"
        ]
        == 2
    )

    assert (
        result.report[
            "positive_labels"
        ]
        == 1
    )