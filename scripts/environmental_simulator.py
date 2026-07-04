# run environmental simulator
from __future__ import annotations
import logging
import polars as pl
from params import HEMISPHERE_TEMPERATURE_PARAMS
from core import EnvironmentalSimulator
from config import (
    GlobalInitializer,
    SimulationConfig,
)

logger = logging.getLogger(__name__)

def run(
    global_config: GlobalInitializer,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    sim = EnvironmentalSimulator(
        temporal_index=global_config.temporal_index,
        rng=global_config.rng,
    )
    
    # generate temp
    df_north_temp = sim.generate_daily_max_temp(hemisphere="north")
    df_south_temp = sim.generate_daily_max_temp(hemisphere="south")

    # generate rainfall
    df_north_rainfall = sim.generate_daily_rainfall_mm(daily_temp=df_north_temp["daily_max_temp_celsius"].to_numpy(), hemisphere="north")
    df_south_rainfall = sim.generate_daily_rainfall_mm(daily_temp=df_south_temp["daily_max_temp_celsius"].to_numpy(), hemisphere="south")

    # validate temp
    for hemisphere, df in [("north", df_north_temp), ("south", df_south_temp)]:
        meta = HEMISPHERE_TEMPERATURE_PARAMS[hemisphere]
        _validate_temp(df, physical_bounds=meta.physical_bounds, label=meta.label)

    # validate rainfall
    _validate_rainfall(df=df_north_rainfall, label="North Hemisphere")
    _validate_rainfall(df=df_south_rainfall, label="South Hemisphere")

    # log summary
    logger.info(
        "North Temp Range: %.2f°C – %.2f°C",
        df_north_temp.select(pl.col("daily_max_temp_celsius").min()).item(),
        df_north_temp.select(pl.col("daily_max_temp_celsius").max()).item(),
    )
    logger.info(
        "South Temp Range: %.2f°C – %.2f°C",
        df_south_temp.select(pl.col("daily_max_temp_celsius").min()).item(),
        df_south_temp.select(pl.col("daily_max_temp_celsius").max()).item(),
    )
    logger.info(
        "North Rainfall Range: %.2fmm - %.2fmm",
        df_north_rainfall.select(pl.col("daily_rainfall_mm").min()).item(),
        df_north_rainfall.select(pl.col("daily_rainfall_mm").max()).item(),  
    )
    logger.info(
        "South Rainfall Range: %.2fmm - %.2fmm",
        df_south_rainfall.select(pl.col("daily_rainfall_mm").min()).item(),
        df_south_rainfall.select(pl.col("daily_rainfall_mm").max()).item(),  
    )
    logger.info("Temperature and rainfall validation passed. Ready for the next step.")

    return df_north_temp, df_south_temp, df_north_rainfall, df_south_rainfall

# helper functions
def _validate_temp(
    df: pl.DataFrame,
    *,
    physical_bounds: tuple[float, float],
    label: str,
) -> None:
    expected_cols = 4
    assert df.shape == (365, expected_cols), (
        f"{label}: expected shape (365, {expected_cols}), got {df.shape}"
    )
    _min_temp = df.select(pl.col("daily_max_temp_celsius").min()).item()
    _max_temp = df.select(pl.col("daily_max_temp_celsius").max()).item()
    assert _min_temp >= physical_bounds[0], (
        f"{label}: lower bound violation — "
        f"min {_min_temp:.2f} < {physical_bounds[0]}"
    )
    assert _max_temp <= physical_bounds[1], (
        f"{label}: upper bound violation — "
        f"max {_max_temp:.2f} > {physical_bounds[1]}"
    )
    assert df.null_count().sum_horizontal().item() == 0, (
        f"{label}: NaN values detected in output DataFrame"
    )
    assert df.select(pl.col("day_index").first()).item() == 0, (
        f"{label}: day_index must start at 0"
    )
    assert df.select(pl.col("day_index").last()).item() == 364, (
        f"{label}: day_index must end at 364"
    )

def _validate_rainfall(df: pl.DataFrame, *, label: str) -> None:
    """Post-generation invariant checks."""
    col = "daily_rainfall_mm"
    assert df.shape[0] == 365, f"{label}: expected 365 rows, got {df.shape[0]}"
    assert df.null_count().sum_horizontal().item() == 0, f"{label}: NaN values detected"
    assert df.select((pl.col(col) >= 0.0).all()).item(), f"{label}: negative rainfall detected"
    assert df.select((pl.col(col) <= 75.0).all()).item(), f"{label}: rainfall exceeds 75mm cap"

    # Wet-day floor check: all non-zero values must be >= 0.1
    wet_days = df.filter(pl.col("wet_dry_state") == 1)
    if len(wet_days) > 0:
        assert wet_days.select((pl.col(col) >= 0.1).all()).item(), (
            f"{label}: wet-day rainfall below 0.1mm floor"
        )

    # Dry-day exactness: all dry days must be exactly 0.0
    dry_days = df.filter(pl.col("wet_dry_state") == 0)
    if len(dry_days) > 0:
        assert dry_days.select((pl.col(col) == 0.0).all()).item(), (
            f"{label}: dry-day rainfall is not exactly 0.0"
        )

    # Statistical sanity (soft, logged not asserted)
    wet_frac = wet_days.shape[0] / 365
    wet_mean = wet_days.select(pl.col(col).mean()).item() if len(wet_days) > 0 else 0.0
    logger.info(
        "%s Rainfall Stats: wet_frac=%.2f, wet_mean=%.2f mm, annual_avg=%.2f mm",
        label, wet_frac, wet_mean, df.select(pl.col(col).mean()).item(),
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
        random_seed=42,
    )
    mock_global = GlobalInitializer(config=mock_config)

    df_n, df_s, _, _ = run(mock_global)

    print("Phase 2 standalone verification passed.")
    print(f"  North: {df_n.shape[0]} rows, "
          f"range {df_n.select(pl.col('daily_max_temp_celsius').min()).item():.2f}°C "
          f"– {df_n.select(pl.col('daily_max_temp_celsius').max()).item():.2f}°C")
    print(f"  South: {df_s.shape[0]} rows, "
          f"range {df_s.select(pl.col('daily_max_temp_celsius').min()).item():.2f}°C "
          f"– {df_s.select(pl.col('daily_max_temp_celsius').max()).item():.2f}°C")
    