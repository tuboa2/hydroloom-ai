from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Final

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR: Final[Path] = PROJECT_ROOT / "data" / "processed"
ARTIFACT_DIR: Final[Path] = PROJECT_ROOT / "artifacts"
MLFLOW_DIR: Final[Path] = PROJECT_ROOT / "mlruns"
LOG_DIR: Final[Path] = PROJECT_ROOT / "logs"

RANDOM_STATE: Final[int] = 42

TARGET_COLUMN: Final[str] = "water_quality_index"
DAY_INDEX_COLUMN: Final[str] = "day_index"
YEAR_INDEX_COLUMN: Final[str] = "year_index"
HEMISPHERE_COLUMN: Final[str] = "hemisphere"

EXPECTED_ROW_COUNT: Final[int] = 1_825
EXPECTED_YEAR_COUNT: Final[int] = 5
DAYS_PER_YEAR: Final[int] = 365

HEMISPHERES: Final[tuple[str, ...]] = ("north", "south")

EXPECTED_SCHEMA: Final[tuple[str, ...]] = (
    "hemisphere",
    "day_index",
    "year_index",
    "is_weekend",
    "season_label",
    "daily_max_temp_celsius",
    "temp_anomaly_celsius",
    "cumulative_heat_index",
    "daily_rainfall_mm",
    "consecutive_dry_days",
    "rolling_7d_rainfall_mm",
    "cumulative_storm_rainfall_mm",
    "antecedent_moisture_condition",
    "daily_runoff_volume_m3",
    "total_suspended_solids_mg_L",
    "nutrient_load_index",
    "heat_x_nutrient_synergy",
    "cluster_heavy_users_daily_mean_liters",
    "cluster_conservationists_daily_mean_liters",
    "cluster_outdoor_landscape_daily_mean_liters",
    "cluster_standard_consumers_daily_mean_liters",
    "watering_ban_active",
    "holiday_weekend_flag",
    "tiered_pricing_regime",
    "drought_x_heat_stress",
    "demand_x_runoff_pressure",
    "water_quality_index",
)

REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(EXPECTED_SCHEMA)

LEAKAGE_BLACKLIST: Final[frozenset[str]] = frozenset(
    {HEMISPHERE_COLUMN, DAY_INDEX_COLUMN, YEAR_INDEX_COLUMN}
)

EVIDENCE_EXCLUSIONS: Final[frozenset[str]] = frozenset({"is_weekend", "holiday_weekend_flag"})

FEATURE_EXCLUSIONS: Final[frozenset[str]] = frozenset(
    LEAKAGE_BLACKLIST | EVIDENCE_EXCLUSIONS | {TARGET_COLUMN}
)

FORBIDDEN_DROP_PATTERNS: Final[tuple[str, ...]] = (
    r"^anomaly_score",
    r"^var_anomaly",
    r"^.*_anomaly_score$",
    r"^.*_var_residual$",
)

FORBIDDEN_RAISE_PATTERNS: Final[tuple[str, ...]] = (
    r"^antecedent_moisture_condition_lead",
    r"^antecedent_moisture_condition_plus",
    r"^antecedent_moisture_condition_future",
    r"^antecedent_moisture_condition_lag_minus",
    r"^.*_future$",
    r"^.*_lead\d+$",
)

TRAIN_YEARS: Final[frozenset[int]] = frozenset({0, 1, 2})
VALIDATION_YEAR: Final[int] = 3
TEST_YEAR: Final[int] = 4

PSI_BINS: Final[int] = 10

CLUSTER_COLUMNS: Final[tuple[str, ...]] = (
    "cluster_heavy_users_daily_mean_liters",
    "cluster_conservationists_daily_mean_liters",
    "cluster_outdoor_landscape_daily_mean_liters",
    "cluster_standard_consumers_daily_mean_liters",
)

POLICY_COLUMNS: Final[tuple[str, ...]] = (
    "season_label",
    "tiered_pricing_regime",
    "watering_ban_active",
)

COMMON_EXCLUDED_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "is_weekend",
        "holiday_weekend_flag",
    }
)

EXOGENOUS_DRIVER_COLUMNS: Final[tuple[str, ...]] = (
    "daily_rainfall_mm",
    "rolling_7d_rainfall_mm",
    "cumulative_storm_rainfall_mm",
    "daily_runoff_volume_m3",
    "total_suspended_solids_mg_L",
    "nutrient_load_index",
    "cumulative_heat_index",
    "temp_anomaly_celsius",
    "consecutive_dry_days",
)

INTERACTION_SOURCE_COLUMNS: Final[tuple[str, ...]] = (
    "daily_rainfall_mm",
    "antecedent_moisture_condition",
    "cumulative_storm_rainfall_mm",
    "total_suspended_solids_mg_L",
    "daily_runoff_volume_m3",
    "temp_anomaly_celsius",
    "nutrient_load_index",
    "cumulative_heat_index",
    "consecutive_dry_days",
    "rolling_7d_rainfall_mm",
)

SOUTH_DEFAULT_EXCLUSIONS: frozenset[str] = frozenset(
    {
        "season_label",
        "tiered_pricing_regime",
        "watering_ban_active",
        "demand_x_runoff_pressure",
        "drought_x_heat_stress",
    }
)

MODEL_CONFIG = {
    "random_state": RANDOM_STATE,
    "target_column": TARGET_COLUMN,
    "prediction_clip_min": 0.0,
    "prediction_clip_max": 100.0,
    "primary_metric": "rmse",
    "scientific_metric": "nse",
    "test_set_policy": "untouched",
}

CV_CONFIG = {
    "n_splits": 5,
    "gap_primary": 0,
    "gap_robustness": 7,
    "shuffle": False,
}

OPTUNA_CONFIG = {
    "sampler": "TPE",
    "sampler_seed": 42,
    "pruner": "MedianPruner",
    "n_startup_trials": 15,
    "n_warmup_steps": 3,
    "direction": "minimize",
    "storage": "sqlite:///artifacts/model/optuna/hydromind_model.db",
}

EARLY_STOPPING_ROUNDS: int = 100

FORBIDDEN_FEATURES = frozenset(
    {
        HEMISPHERE_COLUMN,
        DAY_INDEX_COLUMN,
        YEAR_INDEX_COLUMN,
        TARGET_COLUMN,
    }
)

CATEGORICAL_FEATURES = frozenset(
    {
        "season_label",
        "tiered_pricing_regime",
    }
)

PASSTHROUGH_FEATURES = frozenset(
    {
        "watering_ban_active",
        "is_weekend",
        "holiday_weekend_flag",
    }
)

STANDARD_FEATURES = frozenset(
    {
        "daily_max_temp_celsius",
        "temp_anomaly_celsius",
        "cumulative_heat_index",
        "seasonal_wqi_mean_doy",
        "seasonal_wqi_median_doy",
        "seasonal_wqi_std_doy",
    }
)

POWER_FEATURES = frozenset(
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
        "rainfall_x_antecedent_moisture",
        "storm_rainfall_x_tss",
        "runoff_x_tss",
        "temp_anomaly_x_nutrient_load",
        "heat_x_dry_days",
        "rolling_rain_x_nutrient_load",
        "runoff_x_nutrient_load",
        "heat_x_runoff",
    }
)

_FAMILY_ABBREVIATIONS: Mapping[str, str] = {
    "xgboost": "xgb",
    "lightgbm": "lgbm",
    "catboost": "catboost",
    "linear": "linear",
}
