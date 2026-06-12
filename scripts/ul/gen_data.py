# full data generation pipeline for the ul dataset
from __future__ import annotations
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

def main() -> None:
    # run the data gen pipeline
    logger.info("UL Data Gen Pipeline Execution Started")

    # 1. global initialization
    logger.info("Creating Global Initialization...")
    global_config = global_init.run(
        simulation_days=365,
        population_size=1000,
        random_seed=2026,
    )
    logger.info("Global Initialization Complete.\n")

    # 2. generate temperature data through environment simulation
    logger.info("Running Environmental Simulation...")
    df_north, df_south = environmental_simulator.run(global_config)
    logger.info(
        "Generated %d North rows, %d South rows.",
        len(df_north),
        len(df_south),
    )
    logger.info("Environmental Simulation Complete.\n")

    # done
    logger.info("UL Data Gen Execution Successfully Finished")

if __name__ == "__main__":
    main()
