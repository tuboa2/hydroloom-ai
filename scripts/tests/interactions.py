import pytest
import numpy as np
from scripts.sims.interactions import InteractionSimulator

def test_drought_x_heat_stress_computation():
    """Validates the mathematical formula: max(CDD - 7, 0) * max(T - T_baseline, 0)"""
    sim = InteractionSimulator()
    cdd = np.array([5.0, 7.0, 10.0]) # Excess CDD: 0, 0, 3
    temp = np.array([15.0, 20.0, 25.0]) # Excess Temp (Baseline 18): 0, 2, 7
    demand = np.array([100.0, 200.0, 300.0])
    runoff = np.array([0.0, 5000.0, 10000.0])
    baseline_temp = 18.0
    
    features = sim.generate_features(
        consecutive_dry_days=cdd,
        daily_max_temp_celsius=temp,
        cluster_standard_consumers_daily_mean=demand,
        daily_runoff_volume_m3=runoff,
        hemisphere="north",
        baseline_temp=baseline_temp
    )
    
    expected_drought_x_heat = np.array([0.0, 0.0, 21.0], dtype=np.float32)
    np.testing.assert_allclose(
        features["drought_x_heat_stress"], 
        expected_drought_x_heat, 
        err_msg="drought_x_heat_stress logic failed."
    )

def test_demand_x_runoff_pressure_computation():
    """Validates the mathematical formula: demand * runoff / 10000"""
    sim = InteractionSimulator()
    cdd = np.array([5.0, 7.0, 10.0])
    temp = np.array([15.0, 20.0, 25.0])
    demand = np.array([100.0, 200.0, 300.0])
    runoff = np.array([0.0, 5000.0, 10000.0])
    baseline_temp = 18.0
    
    features = sim.generate_features(
        consecutive_dry_days=cdd,
        daily_max_temp_celsius=temp,
        cluster_standard_consumers_daily_mean=demand,
        daily_runoff_volume_m3=runoff,
        hemisphere="north",
        baseline_temp=baseline_temp
    )
    
    expected_demand_x_runoff = np.array([0.0, 100.0, 300.0], dtype=np.float32)
    np.testing.assert_allclose(
        features["demand_x_runoff_pressure"], 
        expected_demand_x_runoff, 
        err_msg="demand_x_runoff_pressure logic failed."
    )

def test_mismatched_vector_lengths():
    """Validates defense against mismatched input lengths."""
    sim = InteractionSimulator()
    with pytest.raises(ValueError, match="identical length"):
        sim.generate_features(
            consecutive_dry_days=np.array([1, 2]),
            daily_max_temp_celsius=np.array([1]), # Intentionally wrong shape
            cluster_standard_consumers_daily_mean=np.array([1, 2]),
            daily_runoff_volume_m3=np.array([1, 2]),
            hemisphere="north",
            baseline_temp=18.0
        )
