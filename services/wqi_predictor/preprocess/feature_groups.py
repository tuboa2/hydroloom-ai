from __future__ import annotations

from typing import Final

CATEGORICAL_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "season_label",
        "tiered_pricing_regime",
        "watering_ban_active",
    }
)

POLICY_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "season_label",
        "tiered_pricing_regime",
        "watering_ban_active",
    }
)

POLICY_INTERACTION_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "watering_ban_x_outdoor_landscape_cluster",
        "tiered_pricing_x_standard_consumer_cluster",
    }
)

RAW_INTERACTION_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "heat_x_nutrient_synergy",
        "demand_x_runoff_pressure",
        "drought_x_heat_stress",
    }
)

NEAR_NORMAL_BASES: Final[frozenset[str]] = frozenset(
    {
        "daily_max_temp_celsius",
        "temp_anomaly_celsius",
        "cumulative_heat_index",
    }
)

EXTREME_SKEW_BASES: Final[frozenset[str]] = frozenset(
    {
        "daily_rainfall_mm",
        "rolling_7d_rainfall_mm",
        "cumulative_storm_rainfall_mm",
        "daily_runoff_volume_m3",
        "total_suspended_solids_mg_L",
        "nutrient_load_index",
        "heat_x_nutrient_synergy",
        "drought_x_heat_stress",
        "demand_x_runoff_pressure",
    }
)

EXTREME_SKEW_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "rain",
        "runoff",
        "solids",
        "tss",
        "nutrient",
        "demand_x_runoff",
        "drought_x_heat",
    }
)

MODERATE_BASES: Final[frozenset[str]] = frozenset(
    {
        "antecedent_moisture_condition",
        "consecutive_dry_days",
    }
)

RAW_CLUSTER_FEATURES: Final[frozenset[str]] = frozenset(
    {
        "cluster_heavy_users_daily_mean_liters",
        "cluster_conservationists_daily_mean_liters",
        "cluster_outdoor_landscape_daily_mean_liters",
        "cluster_standard_consumers_daily_mean_liters",
    }
)

CLUSTER_AGGREGATE_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "total_cluster_demand",
        "heavy_share",
        "conservationist_share",
        "outdoor_share",
        "standard_share",
        "cluster_demand_std",
        "heavy_to_conservation_ratio",
        "cluster_demand_pc1",
    }
)

CALENDAR_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "day_of_year_sin",
        "day_of_year_cos",
        "month_sin",
        "month_cos",
    }
)

DERIVED_SUFFIXES: Final[tuple[str, ...]] = (
    "_lag0",
    "_lag1",
    "_lag2",
    "_lag3",
    "_lag7",
    "_lag14",
    "_roll_mean_7",
    "_roll_std_7",
    "_roll_min_7",
    "_roll_max_7",
    "_roll_mean_14",
    "_roll_std_14",
    "_roll_mean_28",
    "_roll_std_28",
    "_diff1",
    "_diff7",
    "_zscore_expanding",
)

REQUIRED_FEATURES: Final[tuple[str, ...]] = (
    "wqi_lag1",
    "wqi_lag7",
    "seasonal_wqi_mean_doy",
)

FEATURE_CAPS: Final[dict[str, int]] = {
    "north": 35,
    "south": 28,
}

FAMILY_PRIORITY: Final[dict[str, int]] = {
    "target_lags": 10,
    "seasonal_target": 15,
    "raw_environment": 20,
    "cluster_raw": 30,
    "cluster_aggregates": 35,
    "exogenous_derived": 40,
    "interactions": 50,
    "policy": 60,
    "calendar": 70,
}

def infer_feature_family(column: str) -> str:
    if column in POLICY_COLUMNS or column in POLICY_INTERACTION_COLUMNS:
        return "policy"
    if column.startswith("seasonal_wqi_"):
        return "seasonal_target"
    if column.startswith("wqi_"):
        return "target_lags"
    if column in CALENDAR_COLUMNS or column.startswith(("day_of_year_", "month_")):
        return "calendar"
    if column in CLUSTER_AGGREGATE_COLUMNS:
        return "cluster_aggregates"
    if column in RAW_CLUSTER_FEATURES or any(
        column.startswith(prefix) for prefix in RAW_CLUSTER_FEATURES
    ):
        return "cluster_raw"
    if column in RAW_INTERACTION_COLUMNS:
        return "raw_environment"
    if "_x_" in column:
        return "interactions"
    if any(column.endswith(suffix) for suffix in DERIVED_SUFFIXES):
        return "exogenous_derived"

    return "raw_environment"

def assign_numeric_transformer(column: str) -> str:
    if column in CALENDAR_COLUMNS or column.startswith(("day_of_year_", "month_")):
        return "standard"
    if column in NEAR_NORMAL_BASES or any(
        column.startswith(base) for base in NEAR_NORMAL_BASES
    ):
        return "standard"
    if column in RAW_CLUSTER_FEATURES or any(
        column.startswith(base) for base in RAW_CLUSTER_FEATURES
    ):
        return "standard"
    if column in EXTREME_SKEW_BASES:
        return "power"

    lowered = column.lower()

    if any(token in lowered for token in EXTREME_SKEW_TOKENS):
        return "power"
    if column in CLUSTER_AGGREGATE_COLUMNS:
        return "robust"
    if column.startswith(("wqi_", "seasonal_wqi_")):
        return "robust"
    if column in MODERATE_BASES or any(
        column.startswith(base) for base in MODERATE_BASES
    ):
        return "robust"
    if any(column.endswith(suffix) for suffix in DERIVED_SUFFIXES):
        return "robust"

    return "robust"
