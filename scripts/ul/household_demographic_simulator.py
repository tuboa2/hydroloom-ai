import numpy as np
import pandas as pd
import logging
from ul_core import (
    HouseholdDemographicSimulator,
    OCCUPANCY_PARAMS,
    APPLIANCE_EFFICIENCY_PARAMS,
    LANDSCAPE_TYPE_PARAMS
)
from typing import Literal
from config import GlobalInitializer

logger = logging.getLogger(__name__)

def run(
    global_config: GlobalInitializer,
    population_size: int,
    hemisphere: Literal["north", "south"],
    daily_max_temp_celsius: np.ndarray,
    daily_rainfall_mm: np.ndarray
) -> pd.DataFrame:
    # get the household demographic simulator instance
    household_demo_sim = HouseholdDemographicSimulator(global_config)

    # generate household ids
    household_ids = household_demo_sim.generate_household_ids(
        population_size=population_size
    )
    _validate_household_ids(
        household_ids,
        population_size=population_size,
        label=f"{hemisphere.capitalize()} Hemisphere",
    )

    # get the params
    occupancy_params = OCCUPANCY_PARAMS[hemisphere]

    # generate occupancy counts
    occupancy_count = household_demo_sim.generate_occupancy_count(population_size=population_size, hemisphere=hemisphere)

    # validate
    _validate_occupancy(
        occupancy_count,
        population_size=2500,
        label=occupancy_params.label,
        expected_mean=occupancy_params.mu + 1
    )

    logger.info("Occupancy Count Generation Complete for %s.\n", occupancy_params.label)

    # generate appliance efficiency scores
    appliance_params = APPLIANCE_EFFICIENCY_PARAMS[hemisphere]

    appliance_scores = household_demo_sim.generate_appliance_efficiency_score(
        population_size=population_size,
        hemisphere=hemisphere
    )

    _validate_efficiency(
        appliance_scores,
        population_size=population_size,
        label=appliance_params.label,
        expected_mean=float(appliance_params.theoretical_mean),
        lower_bound=appliance_params.lower_bound,
        upper_bound=appliance_params.upper_bound,
    )

    # generate landscape type
    landscape_params = LANDSCAPE_TYPE_PARAMS[hemisphere]
    landscape_type = household_demo_sim.generate_landscape_type(
        population_size=population_size,
        hemisphere=hemisphere,
    )

    _validate_landscape_type(
        landscape_type,
        population_size=population_size,
        label=landscape_params.label,
        expected_categories=landscape_params.categories,
        expected_weights=landscape_params.weights,
    )

    # generate daily water usage in liters
    daily_water_usage_liters = household_demo_sim.generate_daily_water_usage_liters(
        occupancy_count=occupancy_count,
        appliance_efficiency_score=appliance_scores,
        landscape_type=landscape_type,
        daily_max_temp_celsius=daily_max_temp_celsius,
        daily_rainfall_mm=daily_rainfall_mm,
        hemisphere=hemisphere
    )

    _validate_water_usage(
        daily_water_usage_liters,
        population_size=population_size,
        simulation_days=global_config.simulation_days,
        occupancy_count=occupancy_count,
        hemisphere=hemisphere,
        label = landscape_params.label
    )

    logger.info(
        "Landscape Type Generation Complete for %s.\n",
        landscape_params.label,
    )

    df_traits = pd.DataFrame({
        "household_id": household_ids,
        "occupancy_count": occupancy_count,
        "appliance_efficiency_score": appliance_scores,
        "landscape_type": landscape_type,
    })

    return df_traits, pd.DataFrame(daily_water_usage_liters)

# validate household ids
def _validate_household_ids(
    arr: np.ndarray,
    *,
    population_size: int,
    label: str,
) -> None:
    """Post-generation invariant checks for household_id."""
    # Shape check
    assert arr.shape == (population_size,), (
        f"{label}: expected shape ({population_size},), got {arr.shape}"
    )

    # Uniqueness check (redundant with generation, but validates post-hoc)
    unique_count = len(np.unique(arr))
    assert unique_count == population_size, (
        f"{label}: expected {population_size} unique IDs, got {unique_count}"
    )

    # Format check — all entries must be non-empty strings
    assert all(isinstance(x, str) and len(x) > 0 for x in arr), (
        f"{label}: invalid ID format detected"
    )

    logger.info(
        "%s: %d unique household IDs verified.",
        label, unique_count,
    )

# validate occupancy count
def _validate_occupancy(
    arr: np.ndarray,
    *,
    population_size: int,
    label: str,
    expected_mean: float,
    tolerance: float = 0.15,
    cap: int = 8,
) -> None:
    """Post-generation invariant checks for occupancy_count."""
    assert arr.shape == (population_size,), (
        f"{label}: expected shape ({population_size},), got {arr.shape}"
    )
    assert arr.dtype == np.int8, (
        f"{label}: expected int8, got {arr.dtype}"
    )
    assert arr.min() >= 1, (
        f"{label}: floor violation — min {arr.min()} < 1"
    )
    assert arr.max() <= cap, (
        f"{label}: cap violation — max {arr.max()} > {cap}"
    )
    assert not np.isnan(arr.astype(np.float32)).any(), (
        f"{label}: NaN detected"
    )
    
    # Statistical sanity (soft check — logged, not asserted)
    observed_mean = arr.mean()
    deviation = abs(observed_mean - expected_mean) / expected_mean
    if deviation > tolerance:
        logger.warning(
            "%s: observed mean %.3f deviates %.1f%% from expected %.3f",
            label, observed_mean, deviation * 100, expected_mean,
        )
    else:
        logger.info(
            "%s: mean=%.3f (expected ≈%.2f, deviation=%.2f%%)",
            label, observed_mean, expected_mean, deviation * 100,
        )

# validate appliance efficiency scores
def _validate_efficiency(
    arr: np.ndarray,
    *,
    population_size: int,
    label: str,
    expected_mean: float,
    lower_bound: float,
    upper_bound: float,
    tolerance: float = 0.15,
) -> None:
    """Post-generation invariant checks for appliance_efficiency_score."""
    assert arr.shape == (population_size,), (
        f"{label}: expected shape ({population_size},), got {arr.shape}"
    )
    assert arr.dtype == np.float32, (
        f"{label}: expected float32, got {arr.dtype}"
    )
    assert arr.min() >= lower_bound, (
        f"{label}: lower bound violation — min {arr.min():.4f} < {lower_bound}"
    )
    assert arr.max() <= upper_bound, (
        f"{label}: upper bound violation — max {arr.max():.4f} > {upper_bound}"
    )
    assert not np.isnan(arr).any(), (
        f"{label}: NaN detected"
    )

    # Statistical sanity (soft check — logged, not asserted)
    observed_mean = arr.mean()
    deviation = abs(observed_mean - expected_mean) / expected_mean
    if deviation > tolerance:
        logger.warning(
            "%s: observed mean %.4f deviates %.1f%% from expected %.4f",
            label, observed_mean, deviation * 100, expected_mean,
        )
    else:
        logger.info(
            "%s: mean=%.4f (expected ≈%.4f, deviation=%.2f%%)",
            label, observed_mean, expected_mean, deviation * 100,
        )

def _validate_landscape_type(
    arr: np.ndarray,
    *,
    population_size: int,
    label: str,
    expected_categories: tuple[str, ...],
    expected_weights: tuple[float, ...],
    tolerance: float = 0.15,
) -> None:
    """Post-generation invariant checks for landscape_type."""
    # Shape check
    assert arr.shape == (population_size,), (
        f"{label}: expected shape ({population_size},), got {arr.shape}"
    )

    # Category integrity — no invalid labels
    unique_observed = set(arr)
    valid_set = set(expected_categories)
    invalid = unique_observed - valid_set
    assert not invalid, (
        f"{label}: invalid categories detected: {invalid}"
    )

    # Statistical sanity (soft check)
    for cat, w in zip(expected_categories, expected_weights):
        observed_frac = np.sum(arr == cat) / population_size
        deviation = abs(observed_frac - w) / w if w > 0 else 0.0
        if deviation > tolerance:
            logger.warning(
                "%s: category '%s' observed %.3f, expected %.3f "
                "(deviation %.1f%%)",
                label, cat, observed_frac, w, deviation * 100,
            )
        else:
            logger.info(
                "%s: category '%s' frac=%.3f (expected ≈%.2f, "
                "deviation=%.2f%%)",
                label, cat, observed_frac, w, deviation * 100,
            )

def _validate_water_usage(
    arr: np.ndarray,
    *,
    population_size: int,
    simulation_days: int,
    occupancy_count: np.ndarray,
    hemisphere: Literal["north", "south"],
    label: str,
) -> None:
    """Post-generation invariant checks for daily_water_usage_liters."""
    # 1. Shape check (Must be a 2D matrix: Households x Days)
    assert arr.shape == (population_size, simulation_days), (
        f"{label}: expected shape ({population_size}, {simulation_days}), got {arr.shape}"
    )
    
    # 2. Dtype check
    assert arr.dtype == np.float32, (
        f"{label}: expected float32, got {arr.dtype}"
    )
    
    # 3. NaN / Inf integrity check
    assert not np.isnan(arr).any(), f"{label}: NaN detected in water usage matrix"
    assert not np.isinf(arr).any(), f"{label}: Inf detected in water usage matrix"
    
    # 4. Physiological floor check
    # Re-establish the exact hemispheric constants used in ul_core.py
    physiological_intake = np.float32(3.0) if hemisphere == "north" else np.float32(3.2)
    
    # Broadcast 1D occupancy to 2D floor matrix (Population, 1)
    abs_floor = (occupancy_count.astype(np.float32) * physiological_intake)[:, np.newaxis]
    
    # Assert that no value falls below the absolute physiological minimum
    # (Using a microscopic epsilon to guard against floating-point comparison artifacts)
    assert (arr >= (abs_floor - 1e-5)).all(), (
        f"{label}: values detected below the physiological intake floor"
    )
    
    # 5. Statistical sanity (soft check — logged, not asserted)
    logger.info(
        "%s Water Usage Matrix Stats | global_mean=%.2f L/day | std=%.2f | min=%.2f | max=%.2f",
        label, arr.mean(), arr.std(), arr.min(), arr.max()
    )
