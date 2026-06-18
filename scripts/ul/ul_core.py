# ul core library
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import logging
import numpy as np
import pandas as pd
from numpy.random import Generator

__all__ = [
    "SimulationConfig",
    "GlobalInitializer",
    "EnvironmentalSimulator",
    "HouseholdDemographicSimulator",
    "OCCUPANCY_PARAMS",
    "APPLIANCE_EFFICIENCY_PARAMS",
    "LANDSCAPE_TYPE_PARAMS",
]

# 1. global initialization
@dataclass(frozen=True)
class SimulationConfig:
    # Immutable configuration parameters for the entire simulation.
    simulation_days: int = 365
    population_size: int = 1000
    random_seed: int = 2026

    def __post_init__(self) -> None:
        if self.simulation_days <= 0:
            raise ValueError(
                f"simulation_days must be > 0. Received: {self.simulation_days}"
            )
        if self.population_size <= 0:
            raise ValueError(
                f"population_size must be > 0. Received: {self.population_size}"
            )
        if self.random_seed < 0:
            raise ValueError(
                f"random_seed must be a positive number. Received: {self.random_seed}"
            )

class GlobalInitializer:
    def __init__(self, config: SimulationConfig) -> None:
        self._config = config
        self._logger = logging.getLogger(__name__)
        self._rng: Generator = self._initialize_rng()
        self._temporal_index: np.ndarray = self._initialize_temporal_horizon()

        self._logger.info("Global Environment Initialization Complete.")
        self._logger.info(
            "Parameters -> Days: %d | Population: %d | Seed: %d",
            self._config.simulation_days,
            self._config.population_size,
            self._config.random_seed,
        )

    # private helper methods
    def _initialize_rng(self) -> Generator:
        self._logger.debug(
            "Initializing NumPy Generator with seed: %d",
            self._config.random_seed,
        )
        return np.random.default_rng(seed=self._config.random_seed)

    def _initialize_temporal_horizon(self) -> np.ndarray:
        self._logger.debug(
            "Generating Temporal Index Array of Length %d",
            self._config.simulation_days,
        )
        return np.arange(self._config.simulation_days, dtype=np.int32)

    # public attributes/properties
    @property
    def config(self) -> SimulationConfig:
        return self._config

    @property
    def rng(self) -> Generator:
        return self._rng

    @property
    def temporal_index(self) -> np.ndarray:
        return self._temporal_index

    @property
    def population_size(self) -> int:
        return self._config.population_size

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
        self, hemisphere: Literal["north", "south"]
    ) -> pd.DataFrame:
        # 1. Hemispheric parameters
        if hemisphere == "north":
            annual_mean = 15.2
            amplitude = 7.15
            phase_shift = 30
            bounds = (7.0, 35)
            noise_bounds = (-5.0, 5.0)
            self._logger.info("Generating Northern Hemisphere Temperature Data...")
        elif hemisphere == "south":
            annual_mean = 13.3
            amplitude = 3.65
            phase_shift = 40
            bounds = (9.0, 25)
            noise_bounds = (-2.0, 2.0)
            self._logger.info("Generating Southern Hemisphere Temperature Data...")
        else:
            raise ValueError("Hemisphere must be strictly 'north' or 'south'.")

        # base seasonal curve
        angular_curve = (2 * np.pi / 365) * (
            self._temporal_index - phase_shift
        )

        base_curve = annual_mean - amplitude * np.sin(angular_curve)
        
        # OU Distribution for Noise
        if hemisphere == "north":
            weather_noise = self._generate_ou_noise(theta=0.25, sigma=0.88, noise_bounds=noise_bounds)
        elif hemisphere == "south":
            weather_noise = self._generate_ou_noise(theta=0.25, sigma=0.71, noise_bounds=noise_bounds)

        # 5. Package into a structured DataFrame
        return pd.DataFrame(
            {
                "day_index": self._temporal_index,
                "base_seasonal_temp_celsius": np.round(base_curve, 2),
                "ou_volatility_noise": np.round(weather_noise, 2),
                "daily_max_temp_celsius": np.round(base_curve + weather_noise, 2).clip(bounds[0], bounds[1]),
            }
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
    ) -> pd.DataFrame:
        # cast temp input to float32
        temp_f32 = daily_temp.astype(np.float32)

        # 1. sigmoid probability
        if (hemisphere == 'north'):
            k: np.float32 = 0.20
            t0: np.float32 = 11.35
        else:
            k = 0.39
            t0: np.float32 = 11.33
        
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
        return pd.DataFrame(
            {
                "day_index": self._temporal_index,
                "rain_probability": np.round(prob_rain, 4),
                "wet_dry_state": wet_mask,
                "daily_rainfall_mm": np.round(daily_rainfall_mm, 2)
            }
        )

@dataclass(frozen=True)
class OccupancyParams:
    # hemispheric parameters for occupancy_count generation
    r: float
    mu: float
    label: str
    cap: int = 8

    @property
    def p(self) -> np.float64:
        return self.r / (self.r + self.mu)

@dataclass(frozen=True)
class ApplianceEfficiencyParams:
    # immutable beta distribution parameters for appliance_efficiency_score
    alpha: float
    beta: float
    label: str
    lower_bound: float = 0.15
    upper_bound: float = 0.95

    @property
    def theoretical_mean(self) -> np.float64:
        return self.alpha / (self.alpha + self.beta)

    @property
    def theoretical_mode(self) -> np.float64:
        # only valid when alpha > 1 and beta > 1
        return (self.alpha - 1.0) / (self.alpha + self.beta - 2.0)

@dataclass(frozen=True)
class LandscapeTypeParams:
    # immutable parameters for landscape type categorical sampling
    categories: tuple[str, ...]
    weights: tuple[str, ...]
    label: str

    def __post_init__(self) -> None:
        if len(self.categories) != len(self.weights):
            raise ValueError(
                f"Categories ({len(self.categories)}) and weights "
                f"({len(self.weights)}) must have an equal length."
            )
        w_sum = sum(self.weights)
        if abs(w_sum - 1.0) > 1e-9:
            raise ValueError(
                f"Weights must sum to 1.0, got {w_sum:.12f}"
            )
        if any(w < 0.0 for w in self.weights):
            raise ValueError("All weights must be non-negative.")
        
    @property
    def weight_array(self) -> np.ndarray:
        # returns weights as float64 ndarray
        return np.array(self.weights, dtype=np.float64)
    
    @property
    def fallback_category(self) -> str:
        # modal category used as FP fallback
        return self.categories[self.weights.index(max(self.weights))]

OCCUPANCY_PARAMS: dict[str, OccupancyParams] = {
    "north": OccupancyParams(r=3, mu=2.44, cap=8, label="North Hemisphere"),
    "south": OccupancyParams(r=5.72, mu=2.85, cap=8, label="South Hemisphere")
}

APPLIANCE_EFFICIENCY_PARAMS: dict[str, ApplianceEfficiencyParams] = {
    "north": ApplianceEfficiencyParams(
        alpha=9.36, beta=3.64, label="North Hemisphere"
    ),
    "south": ApplianceEfficiencyParams(
        alpha=4.2, beta=4.55, label="South Hemisphere"
    )
}

# hemisphereic constants
LANDSCAPE_TYPE_PARAMS: dict[str, LandscapeTypeParams] = {
    "north": LandscapeTypeParams(
        categories=(
            "turfgrass_dominant",
            "hardscape_dominant",
            "container_balcony",
            "xeriscape_native",
            "food_homegarden",
        ),
        weights=(0.40, 0.25, 0.15, 0.12, 0.08),
        label="North Hemisphere"
    ),
    "south": LandscapeTypeParams(
        categories=(
            "turfgrass_dominant",
            "hardscape_dominant",
            "container_balcony",
            "xeriscape_native",
            "food_homegarden",
        ),
        weights=(0.25, 0.30, 0.05, 0.20, 0.20),
        label="South Hemisphere",
    ),
}

class HouseholdDemographicSimulator:
    # generates occupancy_count arrays via Negative Binomial
    def __init__(self, rng: Generator) -> None:
        self._rng = rng
        self._logger = logging.getLogger(__name__)

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
            len(np.unique(result)), pd.Series(result).mode().iloc[0], population_size,
        )

        return result
