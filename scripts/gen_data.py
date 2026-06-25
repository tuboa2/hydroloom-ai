# full data generation pipeline for the ul dataset
from __future__ import annotations
from pathlib import Path
import logging
import polars as pl
import global_init as global_init  # global initialization
import environmental_simulator as environmental_simulator  # temp generator
import household_demographic_simulator as household_demographic_simulator

# logging config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%m-%d-%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)

# get data directory
parent_dir = Path(__file__).resolve().parent.parent
print(parent_dir)
data_dir = parent_dir / "data"
data_dir.mkdir(parents=True, exist_ok=True)

def run() -> None:
    # run the data gen pipeline
    logger.info("UL Data Gen Pipeline Execution Started")

    # 1. global initialization
    logger.info("Creating Global Initialization...")
    global_config = global_init.run(
        simulation_days=365,
        population_size=100000,
        random_seed=2030,
    )
    logger.info("Global Initialization Complete.\n")

    # 2. generate temperature data through environment simulation
    logger.info("Running Environmental Simulation...")
    df_north_temp, df_south_temp, df_north_rainfall, df_south_rainfall = (
        environmental_simulator.run(global_config)
    )
    logger.info(
        "Generated %d rows for North Temp, %d rows for South Temp",
        len(df_north_temp),
        len(df_south_temp),
    )
    logger.info(
        "Generated %d rows for North Rainfall, %d rows for South Rainfall",
        len(df_north_rainfall),
        len(df_south_rainfall),
    )
    df_north_env = pl.concat(
        [
            df_north_temp.select("daily_max_temp_celsius"),
            df_north_rainfall.select("daily_rainfall_mm"),
        ],
        how="horizontal",
    )
    df_south_env = pl.concat(
        [
            df_south_temp.select("daily_max_temp_celsius"),
            df_south_rainfall.select("daily_rainfall_mm"),
        ],
        how="horizontal",
    )
    logger.info("Environmental Simulation Complete.\n")

    # 3. generate occupancy count and appliance efficiency score
    logger.info("Running Household Demographic Simulation...")
    df_north_household, df_north_water_usage = household_demographic_simulator.run(
        global_config=global_config,
        population_size=global_config.population_size,
        hemisphere="north",
        daily_max_temp_celsius=df_north_env["daily_max_temp_celsius"].to_numpy(),
        daily_rainfall_mm=df_north_env["daily_rainfall_mm"].to_numpy(),
    )
    df_south_household, df_south_water_usage = household_demographic_simulator.run(
        global_config=global_config,
        population_size=global_config.population_size,
        hemisphere="south",
        daily_max_temp_celsius=df_south_env["daily_max_temp_celsius"].to_numpy(),
        daily_rainfall_mm=df_south_env["daily_rainfall_mm"].to_numpy(),
    )
    logger.info("House Demographic Simulation Complete.\n")

    # export dataframe
    df_north_env.write_csv(data_dir / "north_environment.csv")
    df_south_env.write_csv(data_dir / "south_environment.csv")
    df_north_household.write_csv(data_dir / "north_household.csv")
    df_south_household.write_csv(data_dir / "south_household.csv")
    df_north_water_usage.write_parquet(
        data_dir / "north_water_usage.parquet"
    )
    df_south_water_usage.write_parquet(
        data_dir / "south_water_usage.parquet"
    )

    # done
    logger.info("UL Data Gen Execution Successfully Finished")

if __name__ == "__main__":
    run()
