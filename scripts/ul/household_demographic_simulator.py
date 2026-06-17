import numpy as np
import pandas as pd
import logging
from ul_core import (
    HouseholdDemographicSimulator,
    GlobalInitializer,
    OCCUPANCY_PARAMS,
    APPLIANCE_EFFICIENCY_PARAMS
)
from typing import Literal

logger = logging.getLogger(__name__)

def run(
    global_config: GlobalInitializer,
    population_size: int,
    hemisphere: Literal["north", "south"]
) -> pd.DataFrame:
    # get the household demographic simulator instance
    household_demo_sim = HouseholdDemographicSimulator(global_config.rng)

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

    return pd.DataFrame(occupancy_count), pd.DataFrame(appliance_scores)

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

