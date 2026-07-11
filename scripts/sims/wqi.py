import numpy as np
from typing import Dict
from numpy.random import Generator
from params import WQIParams

class WQISimulator: 
    def __init__(self, rng: Generator):
        self.rng = rng

    def generate_target(
        self,
        daily_max_temp_celsius: np.ndarray,
        daily_rainfall_mm: np.ndarray,
        total_suspended_solids_mg_L: np.ndarray,
        daily_runoff_volume_m3: np.ndarray,
        nutrient_load_index: np.ndarray,
        heat_x_nutrient_synergy: np.ndarray,
        consecutive_dry_days: np.ndarray,
        cluster_heavy_users_mean: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        n_days = len(daily_max_temp_celsius)
        
        # Base config
        wqi_base = WQIParams.wqi_base
        demand_threshold = WQIParams.demand_threshold
        recovery_divisor = WQIParams.recovery_volume_divisor
        
        # 1. Temperature penalty
        delta_temp = -0.035 * np.maximum(daily_max_temp_celsius - 18.0, 0.0)**2
        
        # 2. Runoff penalty
        runoff_load = total_suspended_solids_mg_L * daily_runoff_volume_m3 / 100.0
        delta_runoff = -8.0 * np.log10(1.0 + runoff_load)
        
        # 3. Nutrient penalty
        delta_nutrient = -2.5 * nutrient_load_index
        
        # 4. Synergy penalty
        delta_synergy = -0.8 * heat_x_nutrient_synergy
        
        # 5. Drought penalty
        delta_drought = -1.5 * np.maximum(consecutive_dry_days - 7.0, 0.0) / 10.0
        
        # 6. Demand penalty
        delta_demand = -3.0 * np.maximum(cluster_heavy_users_mean - demand_threshold, 0.0) / 1000.0
        
        # 7. Lagged lake recovery (Cumulative Volume Fix)
        # 3 to 9 day lag cumulative rainfall
        delta_recovery = np.zeros(n_days, dtype=np.float32)
        for t in range(n_days):
            start_idx = max(0, t - 9)
            end_idx = max(0, t - 2) # t-3 inclusive, so t-2 exclusive
            if start_idx < end_idx:
                rain_sum = np.sum(daily_rainfall_mm[start_idx:end_idx])
            else:
                rain_sum = 0.0
            delta_recovery[t] = 1.0 * (rain_sum / recovery_divisor)
            
        # 8. Latent Unobserved Penalties
        # These will be dropped later to prevent leakage (INV-027)
        latent_groundwater = self.rng.gamma(shape=2.0, scale=1.5, size=n_days)
        latent_industrial = self.rng.lognormal(mean=0.5, sigma=0.5, size=n_days)
        delta_latent = -2.0 * latent_groundwater - 1.5 * latent_industrial
        
        # 9. Environmental Noise
        epsilon_noise = self.rng.normal(loc=0.0, scale=np.sqrt(2.25), size=n_days)
        
        # Assembly
        wqi_raw = (
            wqi_base + delta_temp + delta_runoff + delta_nutrient +
            delta_synergy + delta_drought + delta_demand +
            delta_recovery + delta_latent + epsilon_noise
        )
        
        # Hard clipping
        wqi_final = np.clip(wqi_raw, 0.0, 100.0)
        
        return {
            "water_quality_index": wqi_final.astype(np.float32),
            "latent_groundwater": latent_groundwater.astype(np.float32),
            "latent_industrial": latent_industrial.astype(np.float32)
        }
        