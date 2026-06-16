import numpy as np
import pandas as pd
import logging
from ul_core import (
    HouseholdDemographicSimulator,
    GlobalInitializer,
    OCCUPANCY_PARAMS
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
    params = OCCUPANCY_PARAMS[hemisphere]

    # generate occupancy counts
    occupancy_count = household_demo_sim.generate_occupancy_count(population_size=population_size, hemisphere=hemisphere)

    # validate
    _validate_occupancy(
        occupancy_count,
        population_size=2500,
        label=params.label,
        expected_mean=params.mu + 1
    )

    logger.info("Occupancy Count Generation Complete for %s.\n", params.label)

    return pd.DataFrame(occupancy_count)

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
