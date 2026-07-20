from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, OrdinalEncoder
from category_encoders import LeaveOneOutEncoder
from typing import Literal

def build_preprocessor(hemisphere: Literal["north", "south"], use_loo_te: bool = True) -> ColumnTransformer:
    # constructs the column transformer pipeline dynamically
    transformers = []

    # near-normal features
    normal_features = ["daily_max_temp_celsius", "temp_anomaly_celsius"]
    cluster_features = [
        "cluster_heavy_users_daily_mean_liters",
        "cluster_conservationists_daily_mean_liters",
        "cluster_standard_consumers_daily_mean_liters",
        "cluster_outdoor_landscape_daily_mean_liters"
    ]

    if hemisphere == "north":
        normal_features.append("cumulative_heat_index_lag7")
        cluster_features = [f + "_lag1" for f in cluster_features[:3]]
    else:
        normal_features.append("cumulative_heat_index_lag3")

    # standard scaler for normal and cluster features
    transformers.append(("standard", StandardScaler(), normal_features + cluster_features))

    # robust scaler for moderate skew features bounded to [0, 1]
    transformers.append(("robust", RobustScaler(), ["antecedent_moisture_condition"]))

    # power transformer through yeo johnson for extreme skewness
    power_features = [
        "daily_rainfall_mm",
        "rolling_7d_rainfall_mm",
        "cumulative_storm_rainfall_mm",
        "daily_runoff_volume_m3",
        "total_suspended_solids_mg_L",
        "nutrient_load_index",
        "heat_x_nutrient_synergy"
    ]

    if hemisphere == "north":
        power_features.extend(["demand_x_runoff_pressure", "drought_x_heat_stress"])
    transformers.append(("power", PowerTransformer(method="yeo-johnson"), power_features))

    # categorical encodings
    if hemisphere == "north":
        if use_loo_te:
            transformers.append((
                "loo_te",
                LeaveOneOutEncoder(sigma=0.05, random_state=42),
                ["season_label"]
            ))
        else:
            transformers.append((
                "ordinal_season",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                ["season_label"]
            ))
        transformers.append((
            "ordinal_polcy",
            OrdinalEncoder(handle_unknown="use_encoded_value",  unknown_value=-1),
            ["tiered_pricing_regime", "watering_ban_active"]
        ))

    return ColumnTransformer(transformers=transformers, remainder="drop")

def get_pipeline(hemisphere: Literal["north", "south"], model=None) -> Pipeline:
    preprocessor = build_preprocessor(hemisphere)
    steps = [("preprocessor", preprocessor)]
    if model is not None:
        steps.append(("model", model))
    return Pipeline(steps=steps)
    