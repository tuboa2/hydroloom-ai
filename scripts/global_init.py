# global initialization
from __future__ import annotations
import logging
from config import GlobalInitializer, SimulationConfig

# get the logger
logger = logging.getLogger(__name__)

def run(
    simulation_days: int = 1825,
    days_per_year: int = 365,
    num_years: int = 5,
    population_size: int = 1000,
    subsample: int = 5_000,
    random_seed: int = 2032,
) -> GlobalInitializer:
    # create global initialization
    config = SimulationConfig(
        simulation_days=simulation_days,
        days_per_year=days_per_year,
        num_years=num_years,
        population_size=population_size,
        subsample=subsample,
        random_seed=random_seed,
    )

    global_init = GlobalInitializer(config=config)

    # validate outputs
    assert global_init.temporal_index.shape == (simulation_days,), (
        "Temporal index shape mismatch."
    )
    assert (
        global_init.temporal_index[0] == 0
        and global_init.temporal_index[-1] == simulation_days - 1
    ), "Incorrect temporal bounds."
    assert global_init.population_size == population_size, (
        "Population size mismatch."
    )

    # verify reproducibility
    test_draw_1 = global_init.rng.uniform(0, 1)
    global_init_2 = GlobalInitializer(config=config)
    test_draw_2 = global_init_2.rng.uniform(0, 1)
    assert test_draw_1 == test_draw_2, (
        "RNG failed to reproduce identical sequences from the same seed."
    )

    logger.info("Phase 1 validation passed. Ready for the next step.")
    return global_init

# isolation test
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%m-%d-%Y %H:%M:%S",
    )

    result = run()
    print("Phase 1 standalone verification passed.")
    print(f"  Temporal Index: {result.temporal_index.shape[0]} days")
    print(f"  Population:     {result.population_size} households")
