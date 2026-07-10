import numpy as np
import pytest
from sims.macro_behavior import MacroBehavioralSimulator

@pytest.fixture
def macro_sim():
    rng = np.random.default_rng(42)
    return MacroBehavioralSimulator(rng=rng, n_days=365)

def test_watering_ban_hysteresis_invariant(macro_sim):
    """
    Validates INV-Hysteresis: Ban persists beyond rain events.
    """
    cdd = np.zeros(365, dtype=np.int32)
    season = np.array(["summer"] * 365)
    
    # Spike CDD at day 10, then drop to 0
    cdd[10] = 12
    cdd[11] = 0
    
    ban_flags = macro_sim._generate_watering_ban(cdd, season)
    
    # Assert ban starts on day 10
    assert ban_flags[10] == 1
    # Assert ban persists on day 11 despite CDD dropping to 0
    assert ban_flags[11] == 1
    
    # Assert ban duration is strictly between 7 and 21 days
    total_ban_days = np.sum(ban_flags)
    assert 7 <= total_ban_days <= 21

def test_tiered_pricing_leakage_invariant(macro_sim):
    """
    Validates INV-025: Expanding window preventing future leakage.
    """
    # Create sudden spike in usage at day 100
    city_usage = np.ones(365) * 100
    city_usage[100:150] = 500  
    
    pricing = macro_sim._generate_tiered_pricing(city_usage)
    
    # Assert the pricing lag (INV-Phase-Lag)
    # The rolling 30-day average hits the threshold around day 100+.
    # Due to expanding window up to t-30, the threshold is computed on early data (~100).
    # Since lag is 3 days, pricing[0:3] must be absolute 0.
    assert np.all(pricing[0:3] == 0)

def test_cluster_multiplier_application(macro_sim):
    """
    Validates INV-Sparsity: Multipliers are applied correctly.
    """
    day_idx = np.arange(365)
    season = np.array(["summer"] * 365)
    cdd = np.zeros(365)
    cdd[50] = 15 # Trigger ban at day 50
    
    clusters = {
        "cluster_outdoor_landscape_daily_mean_liters": np.ones(365) * 1000,
        "cluster_standard_consumers_daily_mean_liters": np.ones(365) * 500
    }
    
    features = macro_sim.generate_features(day_idx, season, cdd, clusters)
    
    # Find day where ban is active but no holiday
    ban_day = 51
    assert features["watering_ban_active"][ban_day] == 1
    
    # Check compliance multipliers (assuming no pricing tier hit for simplicity)
    outdoor_adj = features["cluster_outdoor_landscape_daily_mean_liters_adj"][ban_day]
    standard_adj = features["cluster_standard_consumers_daily_mean_liters_adj"][ban_day]
    
    assert np.isclose(outdoor_adj, 1000 * 0.60)
    assert np.isclose(standard_adj, 500 * 0.90)
