# run environmental simulator
from __future__ import annotations
import logging
import pandas as pd
from ul_core import (
    EnvironmentalSimulator,
    GlobalInitializer,
    SimulationConfig,
)

logger = logging.getLogger(__name__)

_HEMISPHERE_BOUNDS: dict[str, dict] = {
    "north": {
        "lower": 7.0,
        "upper": 26.0,
        "label": "Northern Hemisphere",
    },
    "south": {
        "lower": 9.0,
        "upper": 20.0,
        "label": "Southern Hemisphere",
    },
}

def run(
    global_config: GlobalInitializer,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sim = EnvironmentalSimulator(
        temporal_index=global_config.temporal_index,
        rng=global_config.rng,
    )

    # generate data
    df_north = sim.generate_daily_max_temp(hemisphere="north")
    df_south = sim.generate_daily_max_temp(hemisphere="south")

    # validate hemisphere
    for hemisphere, df in [("north", df_north), ("south", df_south)]:
        meta = _HEMISPHERE_BOUNDS[hemisphere]
        _validate(df, **meta)

    # log summary
    logger.info(
        "North Temp Range: %.2f°C – %.2f°C",
        df_north["daily_max_temp_celsius"].min(),
        df_north["daily_max_temp_celsius"].max(),
    )
    logger.info(
        "South Temp Range: %.2f°C – %.2f°C",
        df_south["daily_max_temp_celsius"].min(),
        df_south["daily_max_temp_celsius"].max(),
    )
    logger.info("Phase 2 validation passed. Ready for the next step.")

    return df_north, df_south

# helper functions
def _validate(
    df: pd.DataFrame,
    *,
    lower: float,
    upper: float,
    label: str,
) -> None:
    expected_cols = 4
    assert df.shape == (365, expected_cols), (
        f"{label}: expected shape (365, {expected_cols}), got {df.shape}"
    )
    assert df["daily_max_temp_celsius"].min() >= lower, (
        f"{label}: lower bound violation — "
        f"min {df['daily_max_temp_celsius'].min():.2f} < {lower}"
    )
    assert df["daily_max_temp_celsius"].max() <= upper, (
        f"{label}: upper bound violation — "
        f"max {df['daily_max_temp_celsius'].max():.2f} > {upper}"
    )
    assert not df.isnull().any().any(), (
        f"{label}: NaN values detected in output DataFrame"
    )
    assert df["day_index"].iloc[0] == 0, (
        f"{label}: day_index must start at 0"
    )
    assert df["day_index"].iloc[-1] == 364, (
        f"{label}: day_index must end at 364"
    )

# isolation test
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%m-%d-%Y %H:%M:%S",
    )

    # Mock Phase 1 outputs for isolated testing
    mock_config = SimulationConfig(
        simulation_days=365,
        population_size=1000,
        random_seed=2026,
    )
    mock_global = GlobalInitializer(config=mock_config)

    df_n, df_s = run(mock_global)

    print("Phase 2 standalone verification passed.")
    print(f"  North: {df_n.shape[0]} rows, "
          f"range {df_n['daily_max_temp_celsius'].min():.2f}°C "
          f"– {df_n['daily_max_temp_celsius'].max():.2f}°C")
    print(f"  South: {df_s.shape[0]} rows, "
          f"range {df_s['daily_max_temp_celsius'].min():.2f}°C "
          f"– {df_s['daily_max_temp_celsius'].max():.2f}°C")