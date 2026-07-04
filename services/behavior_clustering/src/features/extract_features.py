import numpy as np
import pandas as pd
from typing import Literal
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    RobustScaler,
    PowerTransformer,
)
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import clone

class BehavioralFeatureExtractor(BaseEstimator, TransformerMixin):
    # extracts 10 behavioral features from the raw data
    def __init__(self, hemisphere: Literal["north", "south"]) -> None:
        self.hemisphere: str = hemisphere
        self.feature_names_: list[str] = []
        self._dry_day_mask: np.ndarray | None = None
        self._wet_day_mask: np.ndarray | None = None
        self._weekend_mask: np.ndarray | None = None
        self._weekday_mask: np.ndarray | None = None
        self._summer_mask: np.ndarray | None = None
        self._winter_mask: np.ndarray | None = None
        self._temp_array: np.ndarray | None = None

    def fit(self, X: dict, y=None):
        # learn temporal masks from environment data
        env = X["environment"]
        rainfall = env["daily_rainfall_mm"].to_numpy()
        self._dry_day_mask = (rainfall < 0.1)
        self._wet_day_mask = ~self._dry_day_mask

        day_indices = np.arange(365)
        self._weekend_mask = (day_indices % 7) >= 5
        self._weekday_mask = ~self._weekend_mask

        north_summer_index = np.arange(91, 274)
        north_winter_index = np.concatenate([np.arange(0, 91), np.arange(274, 365)])

        if self.hemisphere == "north":
            summer_index = north_summer_index
            winter_index = north_winter_index
        else:
            summer_index = north_winter_index
            winter_index = north_summer_index

        self._summer_mask = np.isin(day_indices, summer_index)
        self._winter_mask = np.isin(day_indices, winter_index)

        self._temp_array = env["daily_max_temp_celsius"].to_numpy()

        self.feature_names_ = [
            "log_per_capita_usage",
            "efficiency_penalty_ratio",
            "water_usage_cv",
            "dry_day_spike_factor",
            "temp_sensitivity_corr",
            "weekend_weekday_ratio",
            "landscape_demand_index",
            "seasonal_amplitude_ratio",
            "drought_responsiveness_index",
            "baseline_peak_ratio"
        ]

        return self

    def transform(self, X: dict) -> pd.DataFrame:
        household = X["household"]
        water_usage = X["water_usage"]
        occupancy = household["occupancy_count"].to_numpy().astype(np.float32)
        efficiency = household["appliance_efficiency_score"].to_numpy().astype(np.float32)
        landscape = household["landscape_type"].to_list()

        n = water_usage.shape[0]
        features = np.empty((n, 10), dtype=np.float32)

        mean_water_usage = water_usage.mean(axis=1)
        std_water_usage = water_usage.std(axis=1)

        # feature 1: log per capita usage
        per_capita_water_usage = mean_water_usage / np.maximum(occupancy, 1.0)
        features[:, 0] = np.log1p(per_capita_water_usage)

        # feature 2: efficiency penalty ratio
        features[:, 1] = 1.0 + (1.0 - efficiency)

        # feature 3: water usage coeff of variation
        features[:, 2] = np.where(
            mean_water_usage > 0,
            std_water_usage / mean_water_usage,
            0.0
        )

        # feature 4: dry-day spike factor
        dry_mean = water_usage[:, self._dry_day_mask].mean(axis=1)
        wet_mean = water_usage[:, self._wet_day_mask].mean(axis=1)
        features[:, 3] = np.where(
            wet_mean > 0,
            dry_mean / wet_mean,
            1.0
        )

        # feature 5: temperature sensitivity correlation
        # fixed: vectorized correlation
        x_mean = water_usage.mean(axis=1, keepdims=True)
        y_mean = self._temp_array.mean()
        x_std = water_usage.std(axis=1)
        y_std = self._temp_array.std()

        with np.errstate(divide='ignore', invalid='ignore'):
            covariance = ((water_usage - x_mean) * (self._temp_array - y_mean)).mean(axis=1)
            corr = covariance / (x_std * y_std)
            # fixed: clip raw correlations upto -0.5 only
            features[:, 4] = np.clip(np.where(np.isnan(corr), 0.0, corr), -0.5, None)

        # feature 6: weekend/weekday ratio
        weekend_mean = water_usage[:, self._weekend_mask].mean(axis=1)
        weekday_mean = water_usage[:, self._weekday_mask].mean(axis=1)
        # fixed: clipped ratio upto 0.75 minimum
        features[:, 5] = np.clip(np.where(
            weekday_mean > 0,
            weekend_mean / weekday_mean,
            1.0
        ), 0.75, None)

        # feature 7: landscape demand index
        landscape_map = {
            "turfgrass_dominant": 1.0,
            "hardscape_dominant": 0.8,
            "container_balcony": 0.6,
            "xeriscape_native": 0.3,
            "food_homegarden": 0.2,
        }
        features[:, 6] = np.array(
            [landscape_map.get(lt, 0.5) for lt in landscape],
            dtype=np.float32
        )

        # feature 8: seasonal amplitude ratio
        summer_mean = water_usage[:, self._summer_mask].mean(axis=1)
        winter_mean = water_usage[:, self._winter_mask].mean(axis=1)
        features[:, 7] = np.where(
            winter_mean > 0,
            summer_mean / winter_mean,
            1.0
        )

        # feature 9: drought responsiveness index
        rolling_rain = np.convolve(
            self._dry_day_mask.astype(np.float32),
            np.ones(7),
            mode='same'
        )
        drought_days = rolling_rain >= 6.0
        if drought_days.sum() > 0:
            drought_usage = water_usage[:, drought_days].mean(axis=1)
            features[:, 8] = np.where(
                mean_water_usage > 0,
                drought_usage / mean_water_usage,
                1.0
            )
        else:
            features[:, 8] = 1.0

        # feature 10: baseline vs peak ratio
        p25 = np.percentile(water_usage, 25, axis=1)
        p95 = np.percentile(water_usage, 95, axis=1)
        features[:, 9] = np.where(
            p95 > 0,
            p25 / p95,
            1.0
        )

        # fixed: return a polars dataframe
        return pd.DataFrame(features, columns=self.feature_names_)
    
    # new: method to get feature names out
    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)

# scaling pipeline
# fixed: self documenting column names in string
scaling_pipeline = ColumnTransformer(
    transformers=[
        ("robust", RobustScaler(quantile_range=(5.0, 95.0)), [
            "log_per_capita_usage",
            "dry_day_spike_factor",
            "drought_responsiveness_index",
            # fixed: used robust scaler for the efficiency score and weekend weekday ratio to scale extreme outliers
            "efficiency_penalty_ratio",
            "weekend_weekday_ratio",
            # fixed: landscape demand index, seasonal amplitude ratio and baseline peak ratio from standard to robust scaler
            "landscape_demand_index",
            "seasonal_amplitude_ratio",
            "baseline_peak_ratio"
        ]),
        ("power", PowerTransformer(method='yeo-johnson', standardize=True), [
            "water_usage_cv",
            # fixed: used power transform for temp sensitivity corr to shift the skewness from negative
            "temp_sensitivity_corr",
        ]),
    ],
    remainder="drop"
)

# north hemisphere pipeline
north_pipeline = Pipeline([
    ("extract", BehavioralFeatureExtractor(hemisphere="north")),
    ("scale", clone(scaling_pipeline)) # fixed: clone scaling pipeline to isolate weights for each hemisphere
])

# south hemisphere data pipeline
south_pipeline = Pipeline([
    ("extract", BehavioralFeatureExtractor(hemisphere="south")),
    ("scale", clone(scaling_pipeline)) # fixed: clone scaling pipeline to isolate weights for each hemisphere
])
