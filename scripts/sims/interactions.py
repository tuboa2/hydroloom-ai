import numpy as np
from typing import Literal

class InteractionSimulator:
    def __init__(self) -> None:
        pass

    def generate_features(
        self,
        consecutive_dry_days: np.ndarray,
        daily_max_temp_celsius: np.ndarray,
        cluster_standard_consumers_daily_mean: np.ndarray,
        daily_runoff_volume_m3: np.ndarray,
        hemisphere: Literal["north", "south"],
        baseline_temp: float
    ) -> dict[str, np.ndarray]:
        # Validate inputs
        n_days = len(consecutive_dry_days)
        if not (
            n_days == len(daily_max_temp_celsius) 
            == len(cluster_standard_consumers_daily_mean) 
            == len(daily_runoff_volume_m3)
        ):
            raise ValueError("All input arrays must have the identical length (n_days).")
            
        # Feature 8.1: drought_x_heat_stress
        # Logic: max(CDD(t) - 7, 0) * max(T(t) - T_baseline, 0)
        cdd_excess = np.maximum(consecutive_dry_days - 7.0, 0.0)
        temp_excess = np.maximum(daily_max_temp_celsius - baseline_temp, 0.0)
        drought_x_heat_stress = cdd_excess * temp_excess
        
        # Feature 8.2: demand_x_runoff_pressure
        # Logic: cluster_standard_consumers_daily_mean(t) * daily_runoff_volume_m3(t) / 10000
        demand_x_runoff_pressure = (
            cluster_standard_consumers_daily_mean * daily_runoff_volume_m3 / 10000.0
        )
        
        return {
            "drought_x_heat_stress": drought_x_heat_stress.astype(np.float32),
            "demand_x_runoff_pressure": demand_x_runoff_pressure.astype(np.float32)
        }
        