import logging
import numpy as np
import polars as pl
import uuid_utils as uuid
from typing import Literal
from config import GlobalInitializer
from params import (
    OCCUPANCY_PARAMS,
    APPLIANCE_EFFICIENCY_PARAMS,
    LANDSCAPE_TYPE_PARAMS
)

class HouseholdSimulator:
    # generates occupancy_count arrays via Negative Binomial
    def __init__(self, global_config: GlobalInitializer) -> None:
        self._rng = global_config.rng
        self._population_size = global_config.population_size
        self._simulation_days = global_config.simulation_days
        self._logger = logging.getLogger(__name__)

    def generate_household_ids(self) -> np.ndarray:
        # generates uuid v7
        self._logger.info(
            "Generating %d household UUIDs (v7)....",
            self._population_size
        )
        # 1. generate uuids via iteration
        id_set: set[str] = set()
        id_list: list[str] = []

        for _ in range(self._population_size):
            new_id = str(uuid.uuid7())
            id_set.add(new_id)
            id_list.append(new_id)

        # 2. hash-set uniqueness assertion
        assert len(id_set) == self._population_size, (
            f"UUID Collision Detected: Generated {len(id_set)} unique IDs from {self._population_size} attempts."
        )

        result = np.array(id_list, dtype=object)

        self._logger.info(
            "Household ID Stats | unique=%d | sample='%s'", len(id_set), result[0]
        )

        return result

    def generate_occupancy_count(
        self,
        hemisphere: Literal["north", "south"],    
    ) -> np.ndarray:
        params = OCCUPANCY_PARAMS[hemisphere]
        self._logger.info(
            "Generating Occupancy Count for %sern Hemisphere.\nParams: n=%.2f | r=%.2f | | μ'=%.2f | p=%.5f",
            hemisphere, self._population_size, params.r, params.mu, params.p
        )

        # 1. vectorized negative binomial draw (failures before r successes)
        rate = self._rng.gamma(
            shape=params.r,
            scale=(1.0 - params.p) / params.p,
            size=self._population_size
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
            size=self._population_size
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
                size=self._population_size,
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
                self._population_size,
                params.fallback_category,
                dtype=object
            )

        self._logger.info(
            "Landscape Type Stats | unique=%d | mode='%s' | n=%d",
            len(np.unique(result)), pl.Series(result).mode().item(0), self._population_size,
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
        