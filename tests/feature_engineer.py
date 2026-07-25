from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from typing import Literal

from services.wqi_predictor.config import TARGET_COLUMN, YEAR_INDEX_COLUMN
from services.wqi_predictor.data.ingestion import (
    IngestedHemisphere,
    build_feature_frame,
)
from services.wqi_predictor.features.engineer import build_engineered_dataset
from services.wqi_predictor.features.feature_registry import CLUSTER_COLUMNS


@pytest.fixture
def feature_engineer_frame(synthetic_frame: pl.DataFrame) -> pl.DataFrame:
    rng = np.random.default_rng(42)

    return synthetic_frame.with_columns(
        [
            pl.Series(
                name=column,
                values=rng.uniform(100.0, 1000.0, size=len(synthetic_frame)).astype("float32"),
            )
            for column in CLUSTER_COLUMNS
        ]
    )


def _make_ingested(
    frame: pl.DataFrame,
    hemisphere: Literal["north", "south"],
) -> IngestedHemisphere:
    governed_frame = frame
    feature_frame = build_feature_frame(governed_frame)
    target = governed_frame[TARGET_COLUMN]

    return IngestedHemisphere(
        hemisphere=hemisphere,
        raw_frame=governed_frame,
        feature_frame=feature_frame,
        target=target,
        data_hash="feature_engineer-test-hash",
        metadata={},
    )


def test_engineered_dataset_preserves_row_count_and_has_no_nulls(
    feature_engineer_frame: pl.DataFrame,
) -> None:
    ingested = _make_ingested(feature_engineer_frame, "north")
    engineered = build_engineered_dataset(ingested)

    assert engineered.feature_frame.shape[0] == 1825
    assert engineered.feature_frame.null_count().to_numpy().sum() == 0


def test_target_lag_features_use_only_past_target_values(
    feature_engineer_frame: pl.DataFrame,
) -> None:
    ingested = _make_ingested(feature_engineer_frame, "north")
    engineered = build_engineered_dataset(ingested)

    features = engineered.feature_frame
    target = engineered.target

    assert features["wqi_lag1"][0] == pytest.approx(50.0)

    assert np.allclose(
        features["wqi_lag1"][1:].to_numpy(),
        target[:-1].to_numpy(),
    )


def test_wqi_roll_mean_7_uses_only_past_window(
    feature_engineer_frame: pl.DataFrame,
) -> None:
    ingested = _make_ingested(feature_engineer_frame, "north")
    engineered = build_engineered_dataset(ingested)

    features = engineered.feature_frame
    target = engineered.target

    row = 10
    expected = target[3:10].mean()

    assert features["wqi_roll_mean_7"][row] == pytest.approx(expected)


def test_seasonal_target_encoding_year_zero_uses_cold_start(
    feature_engineer_frame: pl.DataFrame,
) -> None:
    ingested = _make_ingested(feature_engineer_frame, "north")
    engineered = build_engineered_dataset(ingested)

    year_zero_mask = feature_engineer_frame[YEAR_INDEX_COLUMN] == 0

    year_zero_mean = engineered.feature_frame.filter(year_zero_mask)["seasonal_wqi_mean_doy"]

    assert np.allclose(year_zero_mean.to_numpy(), 50.0)


def test_seasonal_target_encoding_year_one_is_not_cold_start(
    feature_engineer_frame: pl.DataFrame,
) -> None:
    ingested = _make_ingested(feature_engineer_frame, "north")
    engineered = build_engineered_dataset(ingested)

    year_one_mask = feature_engineer_frame[YEAR_INDEX_COLUMN] == 1

    year_one_mean = engineered.feature_frame.filter(year_one_mask)["seasonal_wqi_mean_doy"]

    assert not np.allclose(year_one_mean.to_numpy(), 50.0)


def test_exogenous_lag1_uses_past_values(
    feature_engineer_frame: pl.DataFrame,
) -> None:
    ingested = _make_ingested(feature_engineer_frame, "north")
    engineered = build_engineered_dataset(ingested)

    features = engineered.feature_frame
    source = feature_engineer_frame

    assert np.allclose(
        features["daily_rainfall_mm_lag1"][1:].to_numpy(),
        source["daily_rainfall_mm"][:-1].to_numpy(),
    )


def test_north_includes_policy_features(
    feature_engineer_frame: pl.DataFrame,
) -> None:
    ingested = _make_ingested(feature_engineer_frame, "north")
    engineered = build_engineered_dataset(ingested)

    columns = set(engineered.feature_frame.columns)

    assert "season_label" in columns
    assert "tiered_pricing_regime" in columns
    assert "watering_ban_active" in columns
    assert "watering_ban_x_outdoor_landscape_cluster" in columns
    assert "tiered_pricing_x_standard_consumer_cluster" in columns


def test_south_excludes_policy_and_north_only_features(
    feature_engineer_frame: pl.DataFrame,
) -> None:
    south_frame = feature_engineer_frame.with_columns(pl.lit("south").alias("hemisphere"))

    ingested = _make_ingested(south_frame, "south")
    engineered = build_engineered_dataset(ingested)

    columns = set(engineered.feature_frame.columns)

    assert "season_label" not in columns
    assert "tiered_pricing_regime" not in columns
    assert "watering_ban_active" not in columns
    assert "demand_x_runoff_pressure" not in columns
    assert "drought_x_heat_stress" not in columns
    assert "watering_ban_x_outdoor_landscape_cluster" not in columns
    assert "tiered_pricing_x_standard_consumer_cluster" not in columns


def test_cluster_aggregates_are_present_and_finite(
    feature_engineer_frame: pl.DataFrame,
) -> None:
    ingested = _make_ingested(feature_engineer_frame, "north")
    engineered = build_engineered_dataset(ingested)

    features = engineered.feature_frame

    required_cluster_features = {
        "total_cluster_demand",
        "heavy_share",
        "conservationist_share",
        "outdoor_share",
        "standard_share",
        "cluster_demand_std",
        "heavy_to_conservation_ratio",
        "cluster_demand_pc1",
    }

    assert required_cluster_features.issubset(features.columns)

    assert np.isfinite(features["cluster_demand_pc1"].to_numpy()).all()


def test_calendar_features_are_bounded(
    feature_engineer_frame: pl.DataFrame,
) -> None:
    ingested = _make_ingested(feature_engineer_frame, "north")
    engineered = build_engineered_dataset(ingested)

    features = engineered.feature_frame

    calendar_columns = [
        "day_of_year_sin",
        "day_of_year_cos",
        "month_sin",
        "month_cos",
    ]

    for column in calendar_columns:
        values = features[column].to_numpy()
        assert values.min() >= -1.0
        assert values.max() <= 1.0
