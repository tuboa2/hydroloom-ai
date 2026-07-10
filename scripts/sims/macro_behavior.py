from __future__ import annotations

import logging
import numpy as np
from typing import Dict
from params import MacroBehaviorParams

logger = logging.getLogger(__name__)

class MacroBehavioralSimulator:
    def __init__(
        self, 
        rng: np.random.Generator, 
        n_days: int = 1825,
        params: MacroBehaviorParams | None = None
    ):
        self.rng = rng
        self.n_days = n_days
        # Seamlessly inject params without requiring gen_data.py to pass it
        self.params = params or MacroBehaviorParams()
        self._logger = logging.getLogger(__name__)
        self._logger.info(
            "Macro-Behavioral Simulator Initialized. Days: %d", self.n_days
        )

    def _generate_watering_ban(
        self, 
        cdd: np.ndarray, 
        season_label: np.ndarray
    ) -> np.ndarray:
        """
        Implements §7.1: watering_ban_active with an independent state tracker.
        """
        watering_ban_active = np.zeros(self.n_days, dtype=np.int8)
        ban_expiry_day = -1
        
        for t in range(self.n_days):
            # Check trigger condition
            if (cdd[t] >= self.params.ban_cdd_trigger and 
                season_label[t] in ("spring", "summer") and 
                t > ban_expiry_day):
                # Initiate ban for 7 to 21 days
                ban_duration = self.rng.integers(
                    self.params.ban_duration_min, 
                    self.params.ban_duration_max + 1
                )
                ban_expiry_day = t + ban_duration
                
            # Enforce state hysteresis
            if t <= ban_expiry_day:
                watering_ban_active[t] = 1
                
        return watering_ban_active

    def _generate_holidays(self, day_index: np.ndarray) -> np.ndarray:
        """
        Implements §7.2: holiday_weekend_flag.
        """
        holiday_flag = np.zeros(self.n_days, dtype=np.int8)
        
        for t in range(self.n_days):
            day_of_year = day_index[t] % 365
            if day_of_year in self.params.annual_holidays:
                holiday_flag[t] = 1
                # Bridge day (day after holiday) with 50% probability
                if t + 1 < self.n_days and self.rng.random() > 0.5:
                    holiday_flag[t + 1] = 1
                    
        return holiday_flag

    def _generate_tiered_pricing(self, city_daily_mean: np.ndarray) -> np.ndarray:
        """
        Implements §7.3: tiered_pricing_regime with expanding window and lag.
        """
        rolling_30d = np.zeros(self.n_days, dtype=np.float32)
        for t in range(self.n_days):
            start_idx = max(0, t - 30 + 1)
            rolling_30d[t] = np.mean(city_daily_mean[start_idx:t+1])
            
        raw_pricing_tier = np.zeros(self.n_days, dtype=np.int8)
        for t in range(self.n_days):
            # Strict expanding window for thresholds (INV-025)
            hist_end = max(1, t - 30)
            historical_data = city_daily_mean[:hist_end]
            thresh_75 = np.percentile(historical_data, 75)
            thresh_90 = np.percentile(historical_data, 90)
            
            if rolling_30d[t] > thresh_90:
                raw_pricing_tier[t] = 2
            elif rolling_30d[t] > thresh_75:
                raw_pricing_tier[t] = 1
            else:
                raw_pricing_tier[t] = 0
                
        # Enforce 3-day lag for municipal policy execution
        tiered_pricing_regime = np.zeros(self.n_days, dtype=np.int8)
        lag_days = self.params.pricing_lag_days
        if self.n_days > lag_days:
            tiered_pricing_regime[lag_days:] = raw_pricing_tier[:-lag_days]
            
        return tiered_pricing_regime

    def generate_features(
        self,
        day_index: np.ndarray,
        season_label: np.ndarray,
        consecutive_dry_days: np.ndarray,
        cluster_means_dict: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Orchestrates the generation of all Macro-Behavioral Stochastic Events.
        """
        self._logger.info("=== Domain 6: Macro-Behavioral Stochastic Events ===")
        
        # 1. Generate core macro-events
        ban_active = self._generate_watering_ban(
            cdd=consecutive_dry_days, 
            season_label=season_label
        )
        holidays = self._generate_holidays(day_index=day_index)
        
        # Calculate unadjusted city mean for pricing thresholds
        stacked_means = np.vstack(list(cluster_means_dict.values()))
        city_daily_mean = np.mean(stacked_means, axis=0).astype(np.float32)
        pricing_regime = self._generate_tiered_pricing(city_daily_mean=city_daily_mean)
        
        # 2. Apply Compliance effects & Multipliers to Cluster Means
        adjusted_clusters = {}
        for cluster_name, usage_arr in cluster_means_dict.items():
            base_usage = usage_arr.copy()
            
            # Holiday multiplier
            base_usage = np.where(
                holidays == 1, 
                base_usage * self.params.holiday_multiplier, 
                base_usage
            )
            
            # Pricing penalty
            base_usage = np.where(
                pricing_regime == 1, 
                base_usage * self.params.pricing_tier_1_multiplier, 
                base_usage
            )
            base_usage = np.where(
                pricing_regime == 2, 
                base_usage * self.params.pricing_tier_2_multiplier, 
                base_usage
            )
            
            # Watering ban compliance
            if "outdoor" in cluster_name.lower() or "landscape" in cluster_name.lower():
                base_usage = np.where(
                    ban_active == 1, 
                    base_usage * self.params.ban_compliance_outdoor, 
                    base_usage
                )
            else:
                base_usage = np.where(
                    ban_active == 1, 
                    base_usage * self.params.ban_compliance_base, 
                    base_usage
                )
                
            adjusted_clusters[f"{cluster_name}_adj"] = base_usage.astype(np.float32)
            
        # Post-conditions
        assert ban_active.shape == (self.n_days,)
        assert holidays.shape == (self.n_days,)
        assert pricing_regime.shape == (self.n_days,)
        for name, arr in adjusted_clusters.items():
            assert arr.shape == (self.n_days,), f"{name}: shape mismatch"
            assert arr.dtype == np.float32, f"{name}: dtype mismatch"
            
        self._logger.info("Macro-Behavioral features generated successfully.")
        
        return {
            "watering_ban_active": ban_active,
            "holiday_weekend_flag": holidays,
            "tiered_pricing_regime": pricing_regime,
            **adjusted_clusters
        }
        