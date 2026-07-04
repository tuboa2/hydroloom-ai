# ul core library
from __future__ import annotations
from typing import Literal
import logging
import uuid_utils as uuid
import numpy as np
import polars as pl
from numpy.random import Generator
from config import (
    GlobalInitializer,
    SimulationConfig
)
from params import (
    OCCUPANCY_PARAMS,
    APPLIANCE_EFFICIENCY_PARAMS,
    LANDSCAPE_TYPE_PARAMS,
    HEMISPHERE_TEMPERATURE_PARAMS,
    SOUTH_PHASE_SHIFT,
    SEASON_BOUNDARIES_NORTH
)

__all__ = [
    "EnvironmentalSimulator",
    "HouseholdDemographicSimulator",
]

# environmental time series simulator (temperature)
class EnvironmentalSimulator:
    # generates time series max daily temperature
    def __init__(self, temporal_index: np.ndarray, rng: Generator) -> None:
        self._temporal_index = temporal_index
        self._simulation_days: int = len(temporal_index)
        self._rng = rng
        self._logger = logging.getLogger(__name__)
        self._logger.info(
            "Environmental Simulator Initialized. Days: %d",
            self._simulation_days,
        )

    # private helper methods
    def _generate_ou_noise(
        self,
        theta: float = 0.25,
        sigma: float = 1.2,
        noise_bounds: tuple[float, float] = (5.0, 10.0),
    ) -> np.ndarray:
        # Ornstein-Uhlenbeck noise
        mid_point = (noise_bounds[0] + noise_bounds[1]) / 2.0
        noise = np.full(self._simulation_days, mid_point)

        for t in range(1, self._simulation_days):
            drift = -theta * (noise[t - 1] - mid_point)
            shock = sigma * self._rng.standard_normal()
            noise[t] = noise[t - 1] + drift + shock

        return np.clip(noise, a_min=noise_bounds[0], a_max=noise_bounds[1])

    # public methods
    def generate_daily_max_temp(
        self, 
        hemisphere: Literal["north", "south"]
    ) -> pl.DataFrame:
        temp_params = HEMISPHERE_TEMPERATURE_PARAMS[hemisphere]

        # base seasonal curve
        angular_curve = (2 * np.pi / 365) * (
            self._temporal_index - temp_params.phase_shift
        )

        if hemisphere == "north":
            # fix: amplitude creates a summer peak in the middle of the year
            base_curve = temp_params.annual_mean + temp_params.amplitude * np.sin(angular_curve) 
        else:
            # fix: amplitude creates a winter trough in the middle of the year
            base_curve = temp_params.annual_mean - temp_params.amplitude * np.sin(angular_curve)
        
        weather_noise = self._generate_ou_noise(theta=temp_params.ou_theta, sigma=temp_params.ou_sigma, noise_bounds=temp_params.anomaly_bounds)
        
        # 5. Package into a structured DataFrame
        return pl.DataFrame(
            {
                "day_index": self._temporal_index,
                "base_seasonal_temp_celsius": np.round(base_curve, 2),
                "ou_volatility_noise": np.round(weather_noise, 2),
                "daily_max_temp_celsius": np.round(base_curve + weather_noise, 2).clip(temp_params.physical_bounds[0], temp_params.physical_bounds[1]),
            },
            strict=False,
        ).select(
            pl.col("day_index").cast(pl.Int32),
            pl.col("base_seasonal_temp_celsius").cast(pl.Float32),
            pl.col("ou_volatility_noise").cast(pl.Float32),
            pl.col("daily_max_temp_celsius").cast(pl.Float32),
        )
    
    # generate daily rainfall
    def generate_daily_rainfall_mm(
        self,
        daily_temp: np.ndarray,
        *,
        hemisphere: str,
        gamma_shape: float = 0.2984420954,
        gamma_scale: float = 28.4812368282,
        wet_floor: float = 0.1,
        cap: float = 75.0
    ) -> pl.DataFrame:
        # cast temp input to float32
        temp_f32 = daily_temp.astype(np.float32)

        k: np.float32
        t0: np.float32

        # 1. sigmoid probability
        if (hemisphere == 'north'):
            k = np.float32(0.20)
            t0 = np.float32(11.35)
        else:
            k = np.float32(0.39)
            t0 = np.float32(11.33)
        
        exponent = k * (temp_f32 - t0)
        prob_rain = np.float32(1.0) / (np.float32(1.0) + np.exp(exponent))

        # 2. bernoulli wet/dry mask
        uniform = self._rng.uniform(0.0, 1.0, size=self._simulation_days).astype(np.float32)
        wet_mask = (uniform < prob_rain).astype(np.uint8)

        # 3. gamma volume
        raw_volume = self._rng.gamma(
            shape=gamma_shape,
            scale=gamma_scale,
            size=self._simulation_days
        ).astype(np.float32)

        # 4. floor + cap
        np.maximum(raw_volume, np.float32(wet_floor), out=raw_volume)
        np.minimum(raw_volume, np.float32(cap), out=raw_volume)

        # 5. mask * volume
        daily_rainfall_mm = (wet_mask.astype(np.float32) * raw_volume)

        # package
        return pl.DataFrame(
            {
                "day_index": self._temporal_index,
                "rain_probability": np.round(prob_rain, 4),
                "wet_dry_state": wet_mask,
                "daily_rainfall_mm": np.round(daily_rainfall_mm, 2)
            },
            strict=False,
        ).select(
            pl.col("day_index").cast(pl.Int32),
            pl.col("rain_probability").cast(pl.Float32),
            pl.col("wet_dry_state").cast(pl.Int32),
            pl.col("daily_rainfall_mm").cast(pl.Float32),
        )

    # new: generate seasonal labels
    def generate_season_labels(
        self,
        day_index: np.ndarray,
        *,
        hemisphere: Literal["north", "south"]
    ) -> np.ndarray:
        # vectorized season assignment O(n)
        assert day_index.dtype == np.int32, (
            f"day_index must be int32, got {day_index.dtype}"
        )
        assert day_index.shape == (1825,), (
            f"Expected shape (1825,), got {day_index.shape}"
        )

        day_of_year: np.ndarray = day_index % 365
        
        if hemisphere == "south":
            day_of_year = (day_of_year + SOUTH_PHASE_SHIFT) % 365

        # 365 day master lookup array
        labels = np.empty(365, dtype="U6")

        for season, ranges in SEASON_BOUNDARIES_NORTH.items():
            for low, high in ranges:
               # direct slice assignment
               labels[low : high + 1] = season

        assert np.all(labels != ""), (
            "Unlabeled day detected - Season boundary gaps"
        )
        
        return labels[day_of_year]

    # new: generate temporal framework
    def generate_temporal_framework(
        self,
        config: SimulationConfig,
        *,
        hemisphere: Literal["north", "south"]
    ) -> dict[str, np.ndarray]:
        # generate all 4 temporal features as aligned numpy arrays
        sim_days: int = config.simulation_days
        days_per_year: int = config.days_per_year
        
        # feature 2.1: day_index
        day_index: np.ndarray = np.arange(sim_days, dtype=np.int32)
    
        # feature 2.2: year_index
        year_index: np.ndarray = (
            day_index // days_per_year
        ).astype(np.int8)
        
        # feature 2.3: is_weekend
        is_weekend: np.ndarray = (
            (day_index % 7) >= 5
        ).astype(np.int8)
    
        # feature 2.4: season_label
        season_label: np.ndarray = self.generate_season_labels(
            day_index, hemisphere=hemisphere
        )

        # shape invariants
        assert day_index.shape == (sim_days,)
        assert year_index.shape == (sim_days,)
        assert is_weekend.shape == (sim_days,)
        assert season_label.shape == (sim_days,)

        # value invariants
        assert day_index[0] == 0 and day_index[-1] == sim_days - 1
        assert year_index[0] == 0 and year_index[-1] == 4
        assert is_weekend.min() == 0 and is_weekend.max() == 1
        assert (season_label != "").all(), "Unlabeled day detected - Season boundary gaps"
        
        return {
            "day_index": day_index,
            "year_index": year_index,
            "is_weekend": is_weekend,
            "season_label": season_label
        }
        
class HouseholdDemographicSimulator:
    # generates occupancy_count arrays via Negative Binomial
    def __init__(self, global_config: GlobalInitializer) -> None:
        self._rng = global_config.rng
        self._population_size = global_config.population_size
        self._simulation_days = global_config.simulation_days
        self._logger = logging.getLogger(__name__)

    def generate_household_ids(self, population_size: int) -> np.ndarray:
        # generates uuid v7
        self._logger.info(
            "Generating %d household UUIDs (v7)....",
            population_size
        )
        # 1. generate uuids via iteration
        id_set: set[str] = set()
        id_list: list[str] = []

        for _ in range(population_size):
            new_id = str(uuid.uuid7())
            id_set.add(new_id)
            id_list.append(new_id)

        # 2. hash-set uniqueness assertion
        assert len(id_set) == population_size, (
            f"UUID Collision Detected: Generated {len(id_set)} unique IDs from {population_size} attempts."
        )

        result = np.array(id_list, dtype=object)

        self._logger.info(
            "Household ID Stats | unique=%d | sample='%s'", len(id_set), result[0]
        )

        return result

    def generate_occupancy_count(
        self,
        population_size: int,
        hemisphere: Literal["north", "south"],    
    ) -> np.ndarray:
        params = OCCUPANCY_PARAMS[hemisphere]
        self._logger.info(
            "Generating Occupancy Count for %sern Hemisphere.\nParams: n=%.2f | r=%.2f | | μ'=%.2f | p=%.5f",
            hemisphere, population_size, params.r, params.mu, params.p
        )

        # 1. vectorized negative binomial draw (failures before r successes)
        rate = self._rng.gamma(
            shape=params.r,
            scale=(1.0 - params.p) / params.p,
            size=population_size
        )
        raw_draws = self._rng.poisson(lam=rate)

        # 2. Shift +1
        occupancy = raw_draws + 1

        # 3. cap at structural limit
        np.minimum(occupancy, params.cap, out=occupancy)

        # 4. downcast to int8
        result = occupancy.astype(np.int8)

        self._logger.info(
            "Occupancy Count Stats | mean=%.3f | std=%.3f | min=%d | max=%d",
            result.mean(), result.std(), result.min(), result.max(),
        )

        return result

    # generate hemispheric appliance efficiency score
    def generate_appliance_efficiency_score(
        self,
        population_size: int,
        hemisphere: Literal["north", "south"]
    ) -> np.ndarray:
        params = APPLIANCE_EFFICIENCY_PARAMS[hemisphere]
        self._logger.info(
            "Generating Appliance Efficiency Score for %s.\n"
            "Params: α=%.2f | β=%.2f | bounds=[%.2f, %.2f] | E[X]=%.4f | Mode=%.4f",
            params.label, params.alpha, params.beta, params.lower_bound, params.upper_bound, params.theoretical_mean, params.theoretical_mode
        )

        # 1. vectorized beta draw
        raw_scores = self._rng.beta(
            a=params.alpha,
            b=params.beta,
            size=population_size
        )

        # 2. domain clamp
        np.clip(raw_scores, params.lower_bound, params.upper_bound, out=raw_scores)

        # 3. downcast float64 -> float32
        result = raw_scores.astype(np.float32)

        self._logger.info(
            "Appliance Efficiency Stats | mean=%.4f | std=%.4f | min=%.4f | max=%.4f",
            result.mean(), result.std(), result.min(), result.max()
        )

        return result
    
    # generate hemispheric landscape type
    def generate_landscape_type(
        self,
        population_size: int,
        hemisphere: Literal["north", "south"]
    ) -> np.ndarray:
        # weighted categorical draw for landscape_type
        params = LANDSCAPE_TYPE_PARAMS[hemisphere]
        self._logger.info(
            "Generating Landscape Type for %s.\n"
            "Categories: %s\nWeights: %s\nFallback: '%s'",
            params.label, params.categories, params.weights, params.fallback_category 
        )

        # vectorized weighted random selection
        try:
            result = self._rng.choice(
                a=np.array(params.categories, dtype=object),
                size=population_size,
                replace=True,
                p=params.weight_array
            )
        except ValueError as e:
            # FP normalization fallback
            self._logger.warning(
                "RNG weight normalization failed (%s)."
                "Applying fallback: all -> '%s'.",
                e, params.fallback_category
            )
            result = np.full(
                population_size,
                params.fallback_category,
                dtype=object
            )

        self._logger.info(
            "Landscape Type Stats | unique=%d | mode='%s' | n=%d",
            len(np.unique(result)), pl.Series(result).mode().item(0), population_size,
        )

        return result

    def generate_daily_water_usage_liters(
        self,
        occupancy_count: np.ndarray,
        appliance_efficiency_score: np.ndarray,
        landscape_type: np.ndarray,
        daily_max_temp_celsius: np.ndarray,
        daily_rainfall_mm: np.ndarray,
        hemisphere: Literal["north", "south"],
    ) -> np.ndarray:
        # computes daily_water_usage_liters through simd vectorization
        total_usage_liters = np.empty((self._population_size, self._simulation_days), dtype=np.float32)

        # hemispheric parameter initialization
        if (hemisphere == "north"):
            weekend_multiplier = np.float32(1.45)
            physiological_intake = np.float32(2.725)
            per_capita_baseline = np.float32(200)
            baseline_temp = np.float32(15.2)
            # fix: add hemispheric yard area medians (sqm)
            yard_area_sqm = np.float32(115.0)
        elif (hemisphere == "south"):
            weekend_multiplier = np.float32(1.3)
            physiological_intake = np.float32(2.35)
            per_capita_baseline = np.float32(150)
            baseline_temp = np.float32(13.3)
            # fix: add hemispheric yard area medians (sqm)
            yard_area_sqm = np.float32(85.0)
        
        # broadcasting dimenstionality alignment
        occupancy_2d = occupancy_count.astype(np.float32)[:, np.newaxis]
        appliance_efficiency_2d = appliance_efficiency_score.astype(np.float32)[:, np.newaxis]

        # indoor baseline vector operations
        day_indices = np.arange(self._simulation_days, dtype=np.int32)
        is_weekend = ((day_indices % 7) >= 5).astype(np.float32)
        temporal_multiplier = np.where(is_weekend == 1, weekend_multiplier, np.float32(1.0))[np.newaxis, :]
        
        efficiency_penalty = 1.0 + (1.0 - appliance_efficiency_2d) 
        indoor_baseline = (physiological_intake + (occupancy_2d * temporal_multiplier * per_capita_baseline)) * efficiency_penalty
        
        # outdoor demand vector operations
        temperature_2d = daily_max_temp_celsius.astype(np.float32)[np.newaxis, :]
        rainfall_2d = daily_rainfall_mm.astype(np.float32)[np.newaxis, :]

        # simplified hargreaves-samani evapotranspiration heuristic
        et_rate = np.maximum(temperature_2d - baseline_temp, 0.0)
        env_deficit = np.maximum(et_rate - rainfall_2d, 0.5)
        
        # landscape type string mapping to float coefs
        landscape_map = {
            "turfgrass_dominant": 1.0,
            "hardscape_dominant": 0.8,
            "container_balcony": 0.6,
            "xeriscape_native": 0.3,
            "food_homegarden": 0.2
        }
        landscape_coeffs = np.array(
            [landscape_map.get(lt, 0.5) for lt in landscape_type],
            dtype=np.float32
        )[:, np.newaxis]

        outdoor_demand = env_deficit * yard_area_sqm *landscape_coeffs

        # axiomatic noise injection
        human_noise = self._rng.normal(loc=0.0, scale=4.5, size=(self._population_size, self._simulation_days)).astype(np.float32)

        # matrix assembly
        total_usage_liters = indoor_baseline + outdoor_demand + human_noise

        # enforce physiological baseline to block mathematically impossible values
        abs_floor = occupancy_2d * physiological_intake
        np.maximum(total_usage_liters, abs_floor, out=total_usage_liters)
        
        # fix: add macro grid noise
        macro_grid_noise = self._rng.normal(loc=0.0, scale=15.0, size=self._simulation_days).astype(np.float32)
        total_usage_liters += macro_grid_noise[np.newaxis, :]

        return total_usage_liters
