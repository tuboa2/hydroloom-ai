from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
import polars.selectors as cs

from ..config import (
    DAY_INDEX_COLUMN,
    HEMISPHERE_COLUMN,
    TARGET_COLUMN,
    TEST_YEAR,
    TRAIN_YEARS,
    VALIDATION_YEAR,
    YEAR_INDEX_COLUMN,
)
from ..data.ingestion import IngestedHemisphere
from ..features.registry import (
    COMMON_EXCLUDED_COLUMNS,
    POLICY_COLUMNS,
    HemisphereFeatureConfig,
    get_feature_config,
)
from ..features.temporal import (
    add_calendar_features,
    add_cluster_aggregates,
    add_cluster_lag_features,
    add_domain_interactions,
    add_exogenous_features,
    add_seasonal_target_encoding,
    add_target_features,
)


@dataclass(frozen=True)
class EngineeredDataset:
    hemisphere: str
    feature_frame: pl.DataFrame
    target: pl.Series
    split_masks: dict[str, np.ndarray]
    metadata: dict[str, Any]


def _select_raw_feature_columns(
    dataframe: pl.DataFrame,
    config: HemisphereFeatureConfig,
) -> list[str]:
    excluded = {HEMISPHERE_COLUMN, DAY_INDEX_COLUMN, YEAR_INDEX_COLUMN, TARGET_COLUMN}

    excluded.update(COMMON_EXCLUDED_COLUMNS)

    if not config.include_policy:
        excluded.update(POLICY_COLUMNS)
    if not config.include_demand_runoff_pressure:
        excluded.add("demand_x_runoff_pressure")
    if not config.include_drought_heat_stress:
        excluded.add("drought_x_heat_stress")

    return [column for column in dataframe.columns if column not in excluded]


def build_engineered_dataset(
    ingested: IngestedHemisphere, cold_start_target: float = 50.0
) -> EngineeredDataset:
    source = ingested.raw_frame.clone()
    config = get_feature_config(ingested.hemisphere)

    target = source[TARGET_COLUMN].cast(pl.Float64)
    year_index = source[YEAR_INDEX_COLUMN]
    day_index = source[DAY_INDEX_COLUMN]
    train_mask = year_index.is_in(TRAIN_YEARS).to_numpy()
    raw_feature_columns = _select_raw_feature_columns(source, config)

    pieces: list[pl.DataFrame] = [
        source[raw_feature_columns],
        add_calendar_features(day_index),
        add_target_features(target, cold_start_target),
        add_seasonal_target_encoding(
            year_index=year_index,
            day_index=day_index,
            target=target,
            smoothing=10.0,
            cold_start_target=cold_start_target,
        ),
        add_exogenous_features(source, config),
        add_cluster_lag_features(source, config),
        add_domain_interactions(source, config),
        add_cluster_aggregates(source, train_mask),
    ]

    seen_cols = set()
    cleaned_pieces = []
    
    for piece in pieces:
        new_cols = [col for col in piece.columns if col not in seen_cols]
        if new_cols:
            cleaned_pieces.append(piece.select(new_cols))
            seen_cols.update(new_cols)
    
    engineered = pl.concat(cleaned_pieces, how="horizontal")
    engineered = engineered.select(list(dict.fromkeys(engineered.columns)))
    engineered = engineered.drop(
        [
            HEMISPHERE_COLUMN,
            DAY_INDEX_COLUMN,
            YEAR_INDEX_COLUMN,
            TARGET_COLUMN,
        ],
        strict=False,
    )

    numeric_columns = engineered.select(cs.numeric()).columns
    categorical_columns = engineered.select(cs.categorical()).columns

    engineered[numeric_columns] = engineered[numeric_columns].fill_null(0.0)
    if categorical_columns:
        engineered = engineered.with_columns(
            pl.col(categorical_columns).fill_null("missing")
        )

    null_columns = [
        col
        for col in engineered.columns
        if engineered[col].has_nulls()
        or (engineered[col].dtype.is_float() and engineered[col].is_nan().any())
    ]

    if null_columns:
        raise ValueError(
            f"{ingested.hemisphere}: engineered feature frame contains nulls or NaNs "
            f"in columns: {null_columns}"
        )

    split_masks = {
        "train": year_index.is_in(TRAIN_YEARS).to_numpy(),
        "val": (year_index == VALIDATION_YEAR).to_numpy(),
        "test": (year_index == TEST_YEAR).to_numpy(),
    }

    metadata: dict[str, Any] = {
        "hemisphere": ingested.hemisphere,
        "cold_start_target": cold_start_target,
        "feature_count": int(engineered.shape[1]),
        "numeric_feature_count": len(numeric_columns),
        "categorical_feature_count": len(categorical_columns),
        "feature_columns": list(engineered.columns),
        "categorical_feature_columns": list(categorical_columns),
        "numeric_feature_columns": list(numeric_columns),
        "split_sizes": {
            "train": int(split_masks["train"].sum()),
            "val": int(split_masks["val"].sum()),
            "test": int(split_masks["test"].sum()),
        },
        "feature_config": {
            "include_policy": config.include_policy,
            "include_demand_runoff_pressure": config.include_demand_runoff_pressure,
            "include_drought_heat_stress": config.include_drought_heat_stress,
            "heat_index_optimal_lag": config.heat_index_optimal_lag,
            "cluster_optimal_lag": config.cluster_optimal_lag,
            "include_policy_interactions": config.include_policy_interactions,
        },
    }

    return EngineeredDataset(
        hemisphere=ingested.hemisphere,
        feature_frame=engineered,
        target=target,
        split_masks=split_masks,
        metadata=metadata,
    )
