# full data generation pipeline for the ul dataset
from __future__ import annotations
from pathlib import Path
import logging
import global_init as global_init # global initialization 
import environmental_simulator as environmental_simulator # temp generator

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

    # export dataframe
    df_north_temp.to_csv(data_dir / "north_temp.csv", index=False)
    df_south_temp.to_csv(data_dir / "south_temp.csv", index=False)
    df_north_rainfall.to_csv(data_dir / "north_rainfall.csv", index=False)
    df_south_rainfall.to_csv(data_dir / "south_rainfall.csv", index=False)

    # done
    logger.info("UL Data Gen Execution Successfully Finished")

if __name__ == "__main__":
    main()
