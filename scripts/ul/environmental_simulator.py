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
        "upper": 35,
        "label": "Northern Hemisphere",
    },
    "south": {
        "lower": 9.0,
        "upper": 25,
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

    # generate temp
    df_north_temp = sim.generate_daily_max_temp(hemisphere="north")
    df_south_temp = sim.generate_daily_max_temp(hemisphere="south")

    # generate rainfall
    df_north_rainfall = sim.generate_daily_rainfall_mm(daily_temp=df_north_temp["daily_max_temp_celsius"], hemisphere="north")
    df_south_rainfall = sim.generate_daily_rainfall_mm(daily_temp=df_south_temp["daily_max_temp_celsius"], hemisphere="south")

    # validate temp
    for hemisphere, df in [("north", df_north_temp), ("south", df_south_temp)]:
        meta = _HEMISPHERE_BOUNDS[hemisphere]
        _validate_temp(df, **meta)

    # validate rainfall
    _validate_rainfall(df=df_north_rainfall, label="North Hemisphere")
    _validate_rainfall(df=df_south_rainfall, label="South Hemisphere")

    # log summary
    logger.info(
        "North Temp Range: %.2f°C – %.2f°C",
        df_north_temp["daily_max_temp_celsius"].min(),
        df_north_temp["daily_max_temp_celsius"].max(),
    )
    logger.info(
        "South Temp Range: %.2f°C – %.2f°C",
        df_south_temp["daily_max_temp_celsius"].min(),
        df_south_temp["daily_max_temp_celsius"].max(),
    )
    logger.info(
        "North Rainfall Range: %.2fmm - %.2fmm",
        df_north_rainfall["daily_rainfall_mm"].min(),
        df_north_rainfall["daily_rainfall_mm"].max(),  
    )
    logger.info(
        "South Rainfall Range: %.2fmm - %.2fmm",
        df_south_rainfall["daily_rainfall_mm"].min(),
        df_south_rainfall["daily_rainfall_mm"].max(),  
    )
    logger.info("Phase 2 validation passed. Ready for the next step.")

    return df_north_temp, df_south_temp, df_north_rainfall, df_south_rainfall

# helper functions
def _validate_temp(
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

def _validate_rainfall(df: pd.DataFrame, *, label: str) -> None:
    """Post-generation invariant checks."""
    col = "daily_rainfall_mm"
    assert df.shape[0] == 365, f"{label}: expected 365 rows, got {df.shape[0]}"
    assert not df.isnull().any().any(), f"{label}: NaN values detected"
    assert (df[col] >= 0.0).all(), f"{label}: negative rainfall detected"
    assert (df[col] <= 75.0).all(), f"{label}: rainfall exceeds 75mm cap"

    # Wet-day floor check: all non-zero values must be >= 0.1
    wet_days = df[df["wet_dry_state"] == 1]
    if len(wet_days) > 0:
        assert (wet_days[col] >= 0.1).all(), (
            f"{label}: wet-day rainfall below 0.1mm floor"
        )

    # Dry-day exactness: all dry days must be exactly 0.0
    dry_days = df[df["wet_dry_state"] == 0]
    if len(dry_days) > 0:
        assert (dry_days[col] == 0.0).all(), (
            f"{label}: dry-day rainfall is not exactly 0.0"
        )

    # Statistical sanity (soft, logged not asserted)
    wet_frac = wet_days.shape[0] / 365
    wet_mean = wet_days[col].mean() if len(wet_days) > 0 else 0.0
    logger.info(
        "%s Rainfall Stats: wet_frac=%.2f, wet_mean=%.2f mm, annual_avg=%.2f mm",
        label, wet_frac, wet_mean, df[col].mean(),
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
        random_seed=None,
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