from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from services.wqi_predictor import config
from services.wqi_predictor.data.drift import _psi_numeric
from services.wqi_predictor.data.ingestion import (
    IngestedHemisphere,
    build_feature_frame,
    drop_forbidden_columns,
)
from services.wqi_predictor.data.splitter import split_chronologically
from services.wqi_predictor.data.validator import (
    DataValidationError,
    validate_schema,
    validate_temporal_index,
)
from services.wqi_predictor.seeding import seed_everything


def test_seed_everything_makes_numpy_deterministic() -> None:
    seed_everything(42)
    first = np.random.rand(5)

    seed_everything(42)
    second = np.random.rand(5)

    assert np.array_equal(first, second)


def test_validate_schema_accepts_expected_schema(synthetic_frame: pd.DataFrame) -> None:
    validate_schema(synthetic_frame, "north")


def test_validate_schema_rejects_missing_column(synthetic_frame: pd.DataFrame) -> None:
    frame = synthetic_frame.drop(columns=["daily_rainfall_mm"])

    with pytest.raises(DataValidationError):
        validate_schema(frame, "north")


def test_future_leakage_column_is_rejected(synthetic_frame: pd.DataFrame) -> None:
    frame = synthetic_frame.copy()
    frame["antecedent_moisture_condition_lead5"] = 0.0

    with pytest.raises(DataValidationError):
        validate_schema(frame, "north")


def test_anomaly_score_is_dropped_not_raised(synthetic_frame: pd.DataFrame) -> None:
    frame = synthetic_frame.copy()
    frame["anomaly_score"] = 0.1

    validate_schema(frame, "north")

    cleaned, dropped = drop_forbidden_columns(frame)

    assert "anomaly_score" not in cleaned.columns
    assert dropped == ["anomaly_score"]


def test_build_feature_frame_removes_blacklisted_and_excluded_columns(
    synthetic_frame: pd.DataFrame,
) -> None:
    feature_frame = build_feature_frame(synthetic_frame)

    forbidden = {
        "hemisphere",
        "day_index",
        "year_index",
        "water_quality_index",
        "is_weekend",
        "holiday_weekend_flag",
    }

    assert forbidden.isdisjoint(feature_frame.columns)


def _make_ingested(frame: pd.DataFrame) -> IngestedHemisphere:
    feature_frame = build_feature_frame(frame)
    target = frame[config.TARGET_COLUMN].copy().reset_index(drop=True)

    return IngestedHemisphere(
        hemisphere="north",
        raw_frame=frame,
        feature_frame=feature_frame,
        target=target,
        data_hash="test-hash",
        metadata={},
    )


def test_chronological_split_sizes_and_order(synthetic_frame: pd.DataFrame) -> None:
    validate_temporal_index(synthetic_frame)

    ingested = _make_ingested(synthetic_frame)
    split = split_chronologically(ingested)

    assert len(split.x_train) == 1_095
    assert len(split.x_validation) == 365
    assert len(split.x_test) == 365

    assert len(split.y_train) == 1_095
    assert len(split.y_validation) == 365
    assert len(split.y_test) == 365

    train_max = split.metadata["day_index"]["train"]["max"]
    validation_min = split.metadata["day_index"]["validation"]["min"]
    validation_max = split.metadata["day_index"]["validation"]["max"]
    test_min = split.metadata["day_index"]["test"]["min"]

    assert train_max < validation_min
    assert validation_max < test_min


def test_feature_matrix_does_not_contain_target_or_identifiers(
    synthetic_frame: pd.DataFrame,
) -> None:
    validate_temporal_index(synthetic_frame)

    ingested = _make_ingested(synthetic_frame)
    split = split_chronologically(ingested)

    for frame in (split.x_train, split.x_validation, split.x_test):
        assert config.TARGET_COLUMN not in frame.columns
        assert config.DAY_INDEX_COLUMN not in frame.columns
        assert config.YEAR_INDEX_COLUMN not in frame.columns
        assert config.HEMISPHERE_COLUMN not in frame.columns


def test_psi_numeric_is_zero_for_identical_distributions() -> None:
    rng = np.random.default_rng(42)
    series = pd.Series(rng.normal(size=500))

    psi = _psi_numeric(series, series, bins=10)

    assert psi == pytest.approx(0.0, abs=1e-6)


def test_psi_numeric_is_positive_for_shifted_distribution() -> None:
    rng = np.random.default_rng(42)

    base = pd.Series(rng.normal(loc=0.0, scale=1.0, size=500))
    compare = pd.Series(rng.normal(loc=2.0, scale=1.0, size=500))

    psi = _psi_numeric(base, compare, bins=10)

    assert psi > 0.1
    