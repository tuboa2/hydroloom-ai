from __future__ import annotations
import os
from pathlib import Path
from typing import Final

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "hydromind" / "data" / "processed"
ARTIFACT_DIR: Final[Path] = PROJECT_ROOT / "hydromind" / "artifacts" / "setup"
MLFLOW_DIR: Final[Path] = PROJECT_ROOT / "hydromind" / "mlruns"

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
    {
        HEMISPHERE_COLUMN,
        DAY_INDEX_COLUMN,
        YEAR_INDEX_COLUMN
    }
)

EVIDENCE_EXCLUSIONS: Final[frozenset[str]] = frozenset(
    {
        "is_weekend",
        "holiday_weekend_flag"
    }
)

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
