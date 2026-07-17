from dataclasses import dataclass, field

@dataclass(frozen=True)
class HemisphereFeatures:
    # explicitly lists the features and their scaling algorithms for a hemisphere
    normal: list[str] = field(default_factory=list)
    robust: list[str] = field(default_factory=list)
    power_transform: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    cluster: list[str] = field(default_factory=list)

# north hemisphere feature configurations
NORTH_REGISTRY = HemisphereFeatures(
    normal=[
        "daily_max_temp_celsius",
        "temp_anomaly_celsius",
        "cumulative_heat_index_lag7",
    ],
    robust=[
        "antecedent_moisture_condition",
    ],
    power_transform=[
        "daily_rainfall_mm",
        "rolling_7d_rainfall_mm",
        "cumulative_storm_rainfall_mm",
        "daily_runoff_volume_m3",
        "total_suspended_solids_mg_L",
        "nutrient_load_index",
        "heat_x_nutrient_synergy",
        "demand_x_runoff_pressure",
        "drought_x_heat_stress",
    ],
    categorical=[
        "season_label",
        "tiered_pricing_regime",
        "watering_ban_active"
    ],
    cluster=[
        "cluster_heavy_users_daily_mean_liters_lag1",
        "cluster_conservationists_daily_mean_liters_lag1",
        "cluster_standard_consumers_daily_mean_liters_lag1"
    ]
)

SOUTH_REGISTRY = HemisphereFeatures(
    normal=[
        "daily_max_temp_celsius",
        "temp_anomaly_celsius",
        "cumulative_heat_index_lag3",
    ],
    robust=[
        "antecedent_moisture_condition",
    ],
    power_transform=[
        "daily_rainfall_mm",
        "rolling_7d_rainfall_mm",
        "cumulative_storm_rainfall_mm",
        "daily_runoff_volume_m3",
        "total_suspended_solids_mg_L",
        "nutrient_load_index",
        "heat_x_nutrient_synergy",
    ],
    categorical = [
        "cluster_heavy_users_daily_mean_liters",
        "cluster_conservationists_daily_mean_liters",
        "cluster_standard_consumers_daily_mean_liters",
        "cluster_outdoor_landscape_daily_mean_liters"
    ]
)

BLACKLIST = [
    "hemisphere",
    "day_index",
    "year_index",
    "is_weekend",
    "holiday_weekend_flag"
]
