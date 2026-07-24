from __future__ import annotations
from dataclasses import dataclass
from typing import Final
from ..config import (
    DAY_INDEX_COLUMN,
    HEMISPHERE_COLUMN,
    TARGET_COLUMN,
    YEAR_INDEX_COLUMN,
    CLUSTER_COLUMNS,
    POLICY_COLUMNS,
    COMMON_EXCLUDED_COLUMNS,
    EXOGENOUS_DRIVER_COLUMNS,
    INTERACTION_SOURCE_COLUMNS,
)

@dataclass(frozen=True)
class HemisphereFeatureConfig:
    name: str
    include_policy: bool
    include_demand_runoff_pressure: bool
    include_drought_heat_stress: bool
    heat_index_optimal_lag: int
    cluster_optimal_lag: int
    include_policy_interactions: bool

NORTH_FEATURE_CONFIG: Final[HemisphereFeatureConfig] = HemisphereFeatureConfig(
    name="north",
    include_policy=True,
    include_demand_runoff_pressure=True,
    include_drought_heat_stress=True,
    heat_index_optimal_lag=7,
    cluster_optimal_lag=1,
    include_policy_interactions=True,
)

SOUTH_FEATURE_CONFIG: Final[HemisphereFeatureConfig] = HemisphereFeatureConfig(
    name="south",
    include_policy=False,
    include_demand_runoff_pressure=False,
    include_drought_heat_stress=False,
    heat_index_optimal_lag=3,
    cluster_optimal_lag=0,
    include_policy_interactions=False,
)

def get_feature_config(hemisphere: str) -> HemisphereFeatureConfig:
    normalized = hemisphere.strip().lower()
    if normalized == "north":
        return NORTH_FEATURE_CONFIG
    if normalized == "south":
        return SOUTH_FEATURE_CONFIG
    raise ValueError(f"Unsupported hemisphere: {hemisphere}.")

__all__ = [
    "CLUSTER_COLUMNS",
    "POLICY_COLUMNS",
    "COMMON_EXCLUDED_COLUMNS",
    "EXOGENOUS_DRIVER_COLUMNS",
    "INTERACTION_SOURCE_COLUMNS",
    "HemisphereFeatureConfig",
    "NORTH_FEATURE_CONFIG",
    "SOUTH_FEATURE_CONFIG",
    "get_feature_config",
    "DAY_INDEX_COLUMN",
    "YEAR_INDEX_COLUMN",
    "HEMISPHERE_COLUMN",
    "TARGET_COLUMN",
]
