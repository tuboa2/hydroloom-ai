import logging
import numpy as np
import polars as pl
from typing import Literal
from utils import ou_loop
from config import GlobalInitializer
from params import (
    HEMISPHERE_TEMPERATURE_PARAMS,
    SOUTH_PHASE_SHIFT,
    SEASON_BOUNDARIES_NORTH
)

class EnvironmentalSimulator:
    # generates time series max daily temperature
    def __init__(self, temporal_index: np.ndarray, rng: np.random.Generator) -> None:
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
        mu: float = 0.0,
        sigma: float = 1.2,
    ) -> np.ndarray:
        # Ornstein-Uhlenbeck noise
        noise = np.full(self._simulation_days, mu, dtype=np.float64)
        
        if self._simulation_days > 1:
            shocks = sigma * self._rng.standard_normal(self._simulation_days - 1)
            ou_loop(theta, mu, shocks, noise)

        return noise

    # new: enhanced base seasonal curve with red noise using AR(1) algorithm
    def _generate_base_seasonal_curve(
        self,
        hemisphere: Literal["north", "south"],
        day_index: np.ndarray,
        year_index: np.ndarray,
    ) -> np.ndarray:
        # vectorized seasonal baseline with per-year climate shifts
        assert day_index.shape == year_index.shape == (1825,)
        # get hemisphere parameters
        params = HEMISPHERE_TEMPERATURE_PARAMS[hemisphere]
        # what day is it
        day_of_year: np.ndarray = (day_index % 365).astype(np.float32)
        # Generate AR(1) noise at a MONTHLY resolution to allow seasonal variability within a year
        total_months = (day_index.size // 30) + 12
        shocks = self._rng.normal(0.0, params.climate_sigma, size=total_months).astype(np.float32)
        climate_shifts_monthly = np.zeros(total_months, dtype=np.float32)
        climate_shifts_monthly[0] = shocks[0]
        for t in range(1, total_months):
            climate_shifts_monthly[t] = (params.climate_rho * climate_shifts_monthly[t-1]) + shocks[t]
        
        # Interpolate monthly shifts to daily resolution
        month_indices = np.arange(day_index.size, dtype=np.float32) / 30.416
        delta_climate: np.ndarray = np.interp(month_indices, np.arange(total_months), climate_shifts_monthly).astype(np.float32)
        # angular frequency
        angular_freq: np.float32 = np.float32(2.0 * (np.pi / 365.0))
        angular_curve: np.ndarray = angular_freq * (day_of_year - params.phase_shift)
        # hemispheric sinusoidal curve
        if hemisphere == "north":
            base_curve = (
                (params.annual_mean + delta_climate)
                - params.amplitude * np.cos(angular_curve)
            )
        else:
            base_curve = (
                (params.annual_mean + delta_climate)
                + params.amplitude * np.cos(angular_curve)
            )

        return base_curve.astype(np.float32)

    # enhanced: add red noise + anomaly + cumulative heat index
    def generate_daily_max_temp(
        self,
        day_index: np.ndarray,
        year_index: np.ndarray,
        *,
        hemisphere: Literal["north", "south"]
    ) -> dict[str, np.ndarray]:  
        params = HEMISPHERE_TEMPERATURE_PARAMS[hemisphere]
        n = day_index.size
        
        # 1. Base seasonal curve & 2. OU stochastic noise
        base_curve = self._generate_base_seasonal_curve(
            hemisphere=hemisphere, day_index=day_index, year_index=year_index
        )
        ou_noise = self._generate_ou_noise(
            theta=params.ou_theta, mu=0.0, sigma=params.ou_sigma
        )
        # Lowered probability from 0.05 to 0.005 for realistic extreme events
        extremes = self._rng.gumbel(loc=0.0, scale=2.0, size=n).astype(np.float32)
        extreme_mask = self._rng.binomial(1, 0.005, size=n).astype(np.float32)
        extreme_events = (extremes * extreme_mask).astype(np.float32)
    
        # PRE-ALLOCATION
        temp_anomaly = np.empty(n, dtype=np.float32)
        daily_max_temp = np.empty(n, dtype=np.float32)
        excess = np.empty(n, dtype=np.float32)
        
        # Removed hard clip on anomaly to allow natural Gaussian tails
        np.copyto(temp_anomaly, ou_noise)
        np.add(base_curve, temp_anomaly, out=daily_max_temp)
        
        # Add extreme events AFTER base calculation so they break through normal bounds
        np.add(daily_max_temp, extreme_events, out=daily_max_temp)
        
        np.clip(daily_max_temp, params.physical_bounds[0], params.physical_bounds[1], out=daily_max_temp)
        
        np.subtract(daily_max_temp, params.baseline_temp, out=excess)
        np.maximum(excess, 0.0, out=excess)
        
        # VECTORIZED GROUPED CUMSUM: Replaces the slow Python for-loop with pure C-level math
        cum_excess = np.cumsum(excess)
        
        # Shift the year grouping for the South so it resets in winter, not mid-summer
        effective_year = year_index
        if hemisphere == "south":
            effective_year = ((day_index + SOUTH_PHASE_SHIFT) // 365).astype(np.int8)

        year_boundaries = np.flatnonzero(np.diff(effective_year, prepend=-1))
        
        end_of_year_vals = np.zeros(year_boundaries.size, dtype=np.float32)
        
        if year_boundaries.size > 1:
            # Subtract the cumulative sum at the exact end of the previous effective year
            end_of_year_vals[1:] = cum_excess[year_boundaries[1:] - 1]
            
        # Normalize index to 0 to safely map to end_of_year_vals array
        normalized_year_idx = effective_year - effective_year[0]
        cumulative_heat_index = cum_excess - end_of_year_vals[normalized_year_idx]
        
        # ASSERTIONS: Wrapped in __debug__ so they are entirely stripped out in production (-O flag).
        # Also fixed the syntax error from your original code: `arr.shape(n,)` -> `arr.shape == (n,)`
        if __debug__:
            for name, arr in [
                ("base_curve", base_curve),
                ("daily_max_temp", daily_max_temp),
                ("temp_anomaly", temp_anomaly),
                ("cumulative_heat_index", cumulative_heat_index)
            ]:
                assert arr.shape == (n,), f"{name}: expected ({n},), got {arr.shape}"
                assert arr.dtype == np.float32, f"{name}: expected float32, got {arr.dtype}"
        
        # IN-PLACE ROUNDING: Avoids allocating 4 brand new arrays for the final dictionary return
        np.round(base_curve, 2, out=base_curve)
        np.round(daily_max_temp, 1, out=daily_max_temp)
        np.round(temp_anomaly, 2, out=temp_anomaly)
        np.round(cumulative_heat_index, 2, out=cumulative_heat_index)
        
        return {
            "base_seasonal_curve": base_curve,
            "daily_max_temp_celsius": daily_max_temp,
            "temp_anomaly_celsius": temp_anomaly,
            "cumulative_heat_index": cumulative_heat_index,
        }
        
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
        config: GlobalInitializer,
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

        # new: add sin and cos day
        day_of_year = (day_index % 365).astype(np.float32)
        angular_freq = np.float32(2.0 * np.pi / 365.0)
        
        sin_day = np.sin(day_of_year * angular_freq).astype(np.float32)
        cos_day = np.cos(day_of_year * angular_freq).astype(np.float32)
        
        return {
            "day_index": day_index,
            "year_index": year_index,
            "is_weekend": is_weekend,
            "season_label": season_label,
            "sin_day_of_year": sin_day,
            "cos_day_of_year": cos_day
        }