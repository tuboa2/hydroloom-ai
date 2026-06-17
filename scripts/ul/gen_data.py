# full data generation pipeline for the ul dataset
from __future__ import annotations
from pathlib import Path
import logging
import global_init as global_init # global initialization 
import environmental_simulator as environmental_simulator # temp generator
import household_demographic_simulator as household_demographic_simulator

# logging config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%m-%d-%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)

# get data directory
parent_dir = Path(__file__).resolve().parent.parent.parent
print(parent_dir)
data_dir = parent_dir / "data"
data_dir.mkdir(parents=True,exist_ok=True)

def main() -> None:
    # run the data gen pipeline
    logger.info("UL Data Gen Pipeline Execution Started")

    # 1. global initialization
    logger.info("Creating Global Initialization...")
    global_config = global_init.run(
        simulation_days=365,
        population_size=2500,
        random_seed=2026,
    )
    logger.info("Global Initialization Complete.\n")

    # 2. generate temperature data through environment simulation
    logger.info("Running Environmental Simulation...")
    df_north_temp, df_south_temp, df_north_rainfall, df_south_rainfall = environmental_simulator.run(global_config)
    logger.info(
        "Generated %d rows for North Temp, %d rows for South Temp",
        len(df_north_temp),
        len(df_south_temp),
    )
    logger.info(
        "Generated %d rows for North Rainfall, %d rows for South Rainfall",
        len(df_north_rainfall),
        len(df_south_rainfall)
    )
    logger.info("Environmental Simulation Complete.\n")

    # 3. generate occupancy count and appliance efficiency score
    logger.info("Running Household Demographic Simulation...")
    df_occupancy_count_north, df_appliance_efficiency_north = household_demographic_simulator.run(global_config=global_config, population_size=global_config.population_size, hemisphere="north")
    df_occupancy_count_south, df_appliance_efficiency_south = household_demographic_simulator.run(global_config=global_config, population_size=global_config.population_size, hemisphere="south")
    logger.info(
        "Generated %d rows for North Occupancy Counts, %d rows for South Occupancy Counts\n",
        len(df_occupancy_count_north),
        len(df_occupancy_count_south),
    )
    logger.info(
        "Generated %d rows for North Appliance Efficiency Score, %d rows for South Appliance Efficiency Score",
        len(df_appliance_efficiency_north),
        len(df_appliance_efficiency_south),
    )
    logger.info("House Demographic Simulation Complete.\n")

    # export dataframe
    df_north_temp.to_csv(data_dir / "north_temp.csv", index=False)
    df_south_temp.to_csv(data_dir / "south_temp.csv", index=False)
    df_north_rainfall.to_csv(data_dir / "north_rainfall.csv", index=False)
    df_south_rainfall.to_csv(data_dir / "south_rainfall.csv", index=False)
    df_occupancy_count_north.to_csv(data_dir / "north_occupancy.csv", index=False)
    df_occupancy_count_south.to_csv(data_dir / "south_occupancy.csv", index=False)
    df_appliance_efficiency_north.to_csv(data_dir / "north_appliance_efficiency.csv", index=False)
    df_appliance_efficiency_south.to_csv(data_dir / "south_appliance_efficiency.csv", index=False)

    # done
    logger.info("UL Data Gen Execution Successfully Finished")

if __name__ == "__main__":
    main()
