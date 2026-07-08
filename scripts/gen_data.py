# full data generation pipeline for the ul dataset
from __future__ import annotations
from pathlib import Path
import logging
import polars as pl
import global_init  # global initialization
from sims.environment import EnvironmentalSimulator
from sims.precipitation import PrecipitationSimulator
from sims.runoff import RunoffSimulator

# logging config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%m-%d-%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)

# get data directory
parent_dir = Path(__file__).resolve().parent.parent
data_dir = parent_dir / "data/raw"
data_dir.mkdir(parents=True, exist_ok=True)

def run() -> None:
    # run the data gen pipeline
    logger.info("Data Generation Pipeline Execution Started")

    # 1. global initialization
    logger.info("Creating Global Initialization...")
    global_config = global_init.run(
        simulation_days=1825,
        days_per_year=365,
        num_years=5,
        population_size=100_000,
        subsample=5_000,
        random_seed=2032,
    )
    logger.info("Global Initialization Complete.\n")

    # 2. simulator initialization
    env_sim = EnvironmentalSimulator(
        temporal_index=global_config.temporal_index,
        rng=global_config.rng
    )
    precipitation_sim = PrecipitationSimulator(
        temporal_index=global_config.temporal_index,
        rng=global_config.rng
    )
    runoff_sim = RunoffSimulator(
        temporal_index=global_config.temporal_index,
        rng=global_config.rng
    )

    # 3. generate temporal framework
    north_temporal_framework = env_sim.generate_temporal_framework(
        config=global_config,
        hemisphere="north"
    )
    south_temporal_framework = env_sim.generate_temporal_framework(
        config=global_config,
        hemisphere="south"
    )

    # 4. generate daily max temp
    north_daily_max_temp = env_sim.generate_daily_max_temp(
        day_index=north_temporal_framework["day_index"],
        year_index=north_temporal_framework["year_index"],
        hemisphere="north"
    )
    south_daily_max_temp = env_sim.generate_daily_max_temp(
        day_index=south_temporal_framework["day_index"],
        year_index=south_temporal_framework["year_index"],
        hemisphere="south"
    )

    north_temp_df = pl.DataFrame(north_daily_max_temp)
    south_temp_df = pl.DataFrame(south_daily_max_temp)

    north_temp_df.write_csv(data_dir / "north_temp.csv")
    south_temp_df.write_csv(data_dir / "south_temp.csv")

    # 5. generate precipitation features
    north_precipitation_features = precipitation_sim.generate_features(
        daily_temp=north_daily_max_temp["daily_max_temp_celsius"],
        season_label=north_temporal_framework["season_label"],
        hemisphere="north"
    )
    south_precipitation_features = precipitation_sim.generate_features(
        daily_temp=south_daily_max_temp["daily_max_temp_celsius"],
        season_label=south_temporal_framework["season_label"],
        hemisphere="south"
    )

    north_precipitation_df = pl.DataFrame(north_precipitation_features)
    south_precipitation_df = pl.DataFrame(south_precipitation_features)

    north_precipitation_df.write_csv(data_dir / "north_precipitation.csv")
    south_precipitation_df.write_csv(data_dir / "south_precipitation.csv")

    # 6. generate runoff pollutant features
    north_runoff_features = runoff_sim.generate_features(
        daily_rainfall_mm=north_precipitation_features["daily_rainfall_mm"],
        antecedent_moisture_condition=north_precipitation_features["antecedent_moisture_condition"],
        consecutive_dry_days=north_precipitation_features["consecutive_dry_days"],
        cumulative_storm_rainfall_mm=north_precipitation_features["cumulative_storm_rainfall_mm"],
        daily_max_temp_celsius=north_daily_max_temp["daily_max_temp_celsius"],
        temp_anomaly_celsius=north_daily_max_temp["temp_anomaly_celsius"],
        hemisphere="north"
    )
    south_runoff_features = runoff_sim.generate_features(
        daily_rainfall_mm=south_precipitation_features["daily_rainfall_mm"],
        antecedent_moisture_condition=south_precipitation_features["antecedent_moisture_condition"],
        consecutive_dry_days=south_precipitation_features["consecutive_dry_days"],
        cumulative_storm_rainfall_mm=south_precipitation_features["cumulative_storm_rainfall_mm"],
        daily_max_temp_celsius=south_daily_max_temp["daily_max_temp_celsius"],
        temp_anomaly_celsius=south_daily_max_temp["temp_anomaly_celsius"],
        hemisphere="south"
    )

    north_runoff_df = pl.DataFrame(north_runoff_features)
    south_runoff_df = pl.DataFrame(south_runoff_features)

    north_runoff_df.write_csv(data_dir / "north_runoff.csv")
    south_runoff_df.write_csv(data_dir / "south_runoff.csv")

if __name__ == "__main__":
    run()
