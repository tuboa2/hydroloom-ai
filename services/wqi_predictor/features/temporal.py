from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..config import RANDOM_STATE
from .registry import (
    CLUSTER_COLUMNS,
    EXOGENOUS_DRIVER_COLUMNS,
    HemisphereFeatureConfig,
)

EPSILON = 1e-6
DAYS_PER_YEAR = 365
MONTH_END_DOY = np.array([31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365], dtype=int)


def _past_expanding_mean(series: pl.Series, fill_value: float) -> pl.Series:
    past = series.shift(1)
    return (past.cum_sum() / past.cum_count()).fill_null(fill_value)


def add_calendar_features(day_index: pl.Series) -> pl.DataFrame:
    col_name = day_index.name or "day_index"
    day_of_year = pl.col(col_name) % DAYS_PER_YEAR
    month = (
        day_of_year.cut(breaks=MONTH_END_DOY[:-1], labels=[str(i) for i in range(1, 13)])
        .cast(pl.String)
        .cast(pl.Float64)
    )

    return day_index.to_frame(col_name).with_columns(
        day_of_year_sin=(2.0 * np.pi * day_of_year / DAYS_PER_YEAR).sin(),
        day_of_year_cos=(2.0 * np.pi * day_of_year / DAYS_PER_YEAR).cos(),
        month_sin=(2.0 * np.pi * month / 12.0).sin(),
        month_cos=(2.0 * np.pi * month / 12.0).cos(),
    )


def add_target_features(target: pl.Series, cold_start_target: float = 50.0) -> pl.DataFrame:
    target = target.cast(pl.Float64)
    out = target.to_frame()
    past = target.shift(1)
    past_expanding_mean = _past_expanding_mean(target, cold_start_target)

    features = []

    for lag in (1, 2, 3, 7, 14):
        lagged = (
            target.shift(lag)
            .fill_null(past_expanding_mean)
            .fill_null(cold_start_target)
            .alias(f"wqi_lag{lag}")
        )
        features.append(lagged)

    for window in (7, 14, 28):
        features.extend(
            [
                past.rolling_mean(window_size=window, min_samples=1)
                .fill_null(cold_start_target)
                .alias(f"wqi_roll_mean_{window}"),
                past.rolling_min(window_size=window, min_samples=1)
                .fill_null(cold_start_target)
                .alias(f"wqi_roll_min_{window}"),
                past.rolling_max(window_size=window, min_samples=1)
                .fill_null(cold_start_target)
                .alias(f"wqi_roll_max_{window}"),
                past.rolling_std(window_size=window, min_samples=2)
                .fill_null(0.0)
                .alias(f"wqi_roll_std_{window}"),
            ]
        )

    features.extend(
        [
            past.ewm_mean(alpha=0.3, adjust=False, min_samples=1)
            .fill_null(cold_start_target)
            .alias("wqi_ewm_mean_0.3"),
            past.ewm_mean(alpha=0.1, adjust=False, min_samples=1)
            .fill_null(cold_start_target)
            .alias("wqi_ewm_mean_0.1"),
        ]
    )

    out = out.with_columns(features)

    lag1 = out["wqi_lag1"]
    lag2 = out["wqi_lag2"]

    lag8 = target.shift(8).fill_null(past_expanding_mean).fill_null(cold_start_target)

    out = out.with_columns(
        (lag1 - lag2).fill_null(0.0).alias("wqi_diff1"),
        (lag1 - lag8).fill_null(0.0).alias("wqi_diff7"),
    )

    return out


def add_seasonal_target_encoding(
    year_index: pl.Series,
    day_index: pl.Series,
    target: pl.Series,
    smoothing: float = 10.0,
    cold_start_target: float = 50.0,
) -> pl.DataFrame:
    out = target.to_frame()
    years = year_index.unique().sort().to_list()
    day_of_year = (day_index.to_numpy() % DAYS_PER_YEAR).astype(int)

    mean_values = np.empty(len(target), dtype=float)
    median_values = np.empty(len(target), dtype=float)
    std_values = np.empty(len(target), dtype=float)

    prior_doy: list[int] = []
    prior_target: list[float] = []

    for year in years:
        year_mask = year_index.to_numpy() == year
        current_doy = day_of_year[year_mask]

        if not prior_target:
            mean_values[year_mask] = cold_start_target
            median_values[year_mask] = cold_start_target
            std_values[year_mask] = 0.0
        else:
            prior = pl.DataFrame(
                {
                    "doy": prior_doy,
                    "target": prior_target,
                }
            )

            global_mean = prior["target"].mean() or 0.0
            global_median = prior["target"].median() or 0.0
            global_std = (prior["target"].std(ddof=0) if len(prior) > 1 else 0.0) or 0.0

            stats = prior.group_by("doy").agg(
                pl.col("target").count().alias("count"),
                pl.col("target").mean().alias("mean"),
                pl.col("target").median().alias("median"),
                pl.col("target").std().alias("std"),
            )

            stats = pl.DataFrame({"doy": current_doy}).join(stats, on="doy", how="left")

            count = stats["count"].fill_null(0.0).to_numpy()
            mean = stats["mean"].fill_null(global_mean).to_numpy()
            median = stats["median"].fill_null(global_median).to_numpy()
            std = stats["std"].fill_null(0.0).to_numpy()

            mean_values[year_mask] = (count * mean + smoothing * global_mean) / (count + smoothing)

            median_values[year_mask] = (count * median + smoothing * global_median) / (
                count + smoothing
            )

            std_values[year_mask] = (count * std + smoothing * global_std) / (count + smoothing)

        current_target = target.to_numpy()[year_mask]
        prior_doy.extend(current_doy.tolist())
        prior_target.extend(current_target.tolist())

    out = out.with_columns(
        pl.Series("seasonal_wqi_mean_doy", mean_values),
        pl.Series("seasonal_wqi_median_doy", median_values),
        pl.Series("seasonal_wqi_std_doy", std_values),
    )

    return out


def add_exogenous_features(source: pl.DataFrame, config: HemisphereFeatureConfig) -> pl.DataFrame:
    out = source.select()
    cols = []

    for column in EXOGENOUS_DRIVER_COLUMNS:
        if column not in source.columns:
            continue

        series = source[column].cast(pl.Float64)
        past = series.shift(1)
        past_expanding_mean = _past_expanding_mean(series, 0.0)

        lags = {1, 7}
        if column == "cumulative_heat_index":
            lags.add(config.heat_index_optimal_lag)

        for lag in sorted(lags):
            lagged = (
                series.shift(lag)
                .fill_null(past_expanding_mean)
                .fill_null(0.0)
                .alias(f"{column}_lag{lag}")
            )
            cols.append(lagged)

        cols.append(
            past.rolling_mean(window_size=7, min_samples=1)
            .fill_null(0.0)
            .alias(f"{column}_roll_mean_7")
        )
        cols.append(
            past.rolling_std(window_size=7, min_samples=1)
            .fill_nan(0.0)
            .fill_null(0.0)
            .alias(f"{column}_roll_std_7")
        )

        cols.append((series - series.shift(1)).fill_null(0.0).alias(f"{column}_diff1"))
        cols.append((series - series.shift(7)).fill_null(0.0).alias(f"{column}_diff7"))

        expanding_mean = past.cum_sum() / past.cum_count()
        expanding_std = past.cumulative_eval(pl.element().std(), min_samples=2).fill_null(1.0)

        zscore = (series - expanding_mean) / (expanding_std + EPSILON)
        zscore = (
            zscore.fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0).alias(f"{column}_zscore_expanding")
        )

        cols.append(zscore)

    return out.with_columns(cols)


def add_cluster_lag_features(
    source: pl.DataFrame,
    config: HemisphereFeatureConfig,
) -> pl.DataFrame:
    out = source.clone()
    lag = config.cluster_optimal_lag

    if lag <= 0:
        return out

    cluster_lag_features = []

    for column in CLUSTER_COLUMNS:
        if column not in out.columns:
            continue

        series = out[column].cast(pl.Float64)
        past_expanding_mean = _past_expanding_mean(series, 0.0)

        lagged = (
            series.shift(lag)
            .fill_null(past_expanding_mean)
            .fill_null(0.0)
            .alias(f"{column}_lag{lag}")
        )

        cluster_lag_features.append(lagged)

    if not cluster_lag_features:
        return out

    return out.with_columns(cluster_lag_features)


def add_domain_interactions(
    source: pl.DataFrame,
    config: HemisphereFeatureConfig,
) -> pl.DataFrame:
    out = source.clone()

    def add_product(left: str, right: str, name: str):
        nonlocal out
        out = out.with_columns(
            (pl.col(left).cast(pl.Float64) * pl.col(right).cast(pl.Float64)).alias(name)
        )

    add_product(
        "daily_rainfall_mm",
        "antecedent_moisture_condition",
        "rainfall_x_antecedent_moisture",
    )

    add_product(
        "cumulative_storm_rainfall_mm",
        "total_suspended_solids_mg_L",
        "storm_rainfall_x_tss",
    )

    add_product(
        "daily_runoff_volume_m3",
        "total_suspended_solids_mg_L",
        "runoff_x_tss",
    )

    add_product(
        "temp_anomaly_celsius",
        "nutrient_load_index",
        "temp_anomaly_x_nutrient_load",
    )

    add_product(
        "cumulative_heat_index",
        "consecutive_dry_days",
        "heat_x_dry_days",
    )

    add_product(
        "rolling_7d_rainfall_mm",
        "nutrient_load_index",
        "rolling_rain_x_nutrient_load",
    )

    add_product(
        "daily_runoff_volume_m3",
        "nutrient_load_index",
        "runoff_x_nutrient_load",
    )

    add_product(
        "cumulative_heat_index",
        "daily_runoff_volume_m3",
        "heat_x_runoff",
    )

    if config.include_policy_interactions:
        add_product(
            "watering_ban_active",
            "cluster_outdoor_landscape_daily_mean_liters",
            "watering_ban_x_outdoor_landscape_cluster",
        )

        add_product(
            "tiered_pricing_regime",
            "cluster_standard_consumers_daily_mean_liters",
            "tiered_pricing_x_standard_consumer_cluster",
        )

    return out.fill_null(0.0)


def add_cluster_aggregates(
    source: pl.DataFrame,
    train_mask: np.ndarray,
) -> pl.DataFrame:
    cluster_columns = [column for column in CLUSTER_COLUMNS if column in source.columns]

    if not cluster_columns:
        return source

    cluster_frame = source.select(pl.col(cluster_columns).cast(pl.Float64))

    share_names = {
        "cluster_heavy_users_daily_mean_liters": "heavy_share",
        "cluster_conservationists_daily_mean_liters": "conservationist_share",
        "cluster_outdoor_landscape_daily_mean_liters": "outdoor_share",
        "cluster_standard_consumers_daily_mean_liters": "standard_share",
    }

    exprs = [
        pl.sum_horizontal(cluster_columns).alias("total_cluster_demand"),
        pl.concat_list(cluster_columns).list.std(ddof=0).alias("cluster_demand_std"),
    ]

    total_demand_expr = pl.sum_horizontal(cluster_columns)
    for col in cluster_columns:
        share_name = share_names.get(col, f"{col}_share")
        exprs.append((pl.col(col) / (total_demand_expr + EPSILON)).fill_null(0.0).alias(share_name))

    if (
        "cluster_heavy_users_daily_mean_liters" in cluster_columns
        and "cluster_conservationists_daily_mean_liters" in cluster_columns
    ):
        exprs.append(
            (
                pl.col("cluster_heavy_users_daily_mean_liters")
                / (pl.col("cluster_conservationists_daily_mean_liters") + EPSILON)
            )
            .fill_null(0.0)
            .alias("heavy_to_conservation_ratio")
        )

    out = source.with_columns(exprs)

    scaler = StandardScaler()
    train_cluster_values = cluster_frame.filter(train_mask)

    scaler.fit(train_cluster_values)
    scaled_all = scaler.transform(cluster_frame)

    pca = PCA(n_components=1, random_state=RANDOM_STATE)
    pca.fit(scaled_all[train_mask])

    pc1_all = pca.transform(scaled_all)[:, 0]

    return out.with_columns(pl.Series("cluster_demand_pc1", pc1_all))
