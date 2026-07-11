import pytest
import numpy as np
from scripts.sims.wqi import WQISimulator

@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)

@pytest.fixture
def wqi_config():
    return {
        "wqi_base": 85.0,
        "demand_threshold": 1000.0,
        "recovery_volume_divisor": 50.0
    }

def test_wqi_bounds(rng):
    sim = WQISimulator(rng)
    n = 100
    
    # Create extreme inputs to push bounds
    res = sim.generate_target(
        daily_max_temp_celsius=np.full(n, 45.0), # Extreme heat
        daily_rainfall_mm=np.zeros(n),
        total_suspended_solids_mg_L=np.full(n, 1000.0),
        daily_runoff_volume_m3=np.full(n, 5000.0),
        nutrient_load_index=np.full(n, 20.0),
        heat_x_nutrient_synergy=np.full(n, 60.0),
        consecutive_dry_days=np.full(n, 50.0),
        cluster_heavy_users_mean=np.full(n, 2500.0)
    )
    
    wqi = res["water_quality_index"]
    assert np.all(wqi >= 0.0), "FATAL: WQI dropped below 0"
    assert np.all(wqi <= 100.0), "FATAL: WQI exceeded 100"

def test_latent_variables_generated(rng):
    sim = WQISimulator(rng)
    res = sim.generate_target(
        daily_max_temp_celsius=np.zeros(10),
        daily_rainfall_mm=np.zeros(10),
        total_suspended_solids_mg_L=np.zeros(10),
        daily_runoff_volume_m3=np.zeros(10),
        nutrient_load_index=np.zeros(10),
        heat_x_nutrient_synergy=np.zeros(10),
        consecutive_dry_days=np.zeros(10),
        cluster_heavy_users_mean=np.zeros(10)
    )
    
    assert "latent_groundwater" in res
    assert "latent_industrial" in res
