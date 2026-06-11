from __future__ import annotations
from dataclasses import dataclass
from numpy.random import Generator

import logging
import numpy as np

# config structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%m-%d-%Y %H:%M:%S"
)
logger = logging.getLogger("Gen Data (Unsupervised)")

@dataclass(frozen=True)
class SimulationConfig:
    # immutable config params for the simulation
    simulation_days: int = 365
    population_size: int = 1000
    random_seed: int = 42

    def __post_init__(self) -> None:
        # evaluates config params upon instantiation
        if self.simulation_days <= 0:
            raise ValueError(f"simulation_days must be > 0. Received: {self.simulation_days}")
        if self.population_size <= 0:
            raise ValueError(f"population_size must be > 0. Received: {self.population_size}")
        if self.random_seed < 0:
            raise ValueError(f"random_seed must be a positive number. Received: {self.random_seed}")       
        
class GlobalInitializer:
    # global initialization and config
    def __init__(self, config: SimulationConfig) -> None:
        self._config = config
        self._rng: Generator = self._initialize_rng()
        self._temporal_index: np.ndarray = self._initialize_temporal_horizon()

        logger.info("Global Environment Initialization Complete.")
        logger.info(
            "Parameters -> Days: %d | Population: %d | Seed: %d",
            self._config.simulation_days,
            self._config.population_size,
            self._config.random_seed
        )

    def _initialize_rng(self) -> Generator:
        # instantiates modern numpy pseudo-random number generator
        logger.debug(
            "Initializing NumPy Generator with seed: %d", 
            self._config.random_seed
        )
        return np.random.default_rng(seed=self._config.random_seed)
    
    def _initialize_temporal_horizon(self) -> np.ndarray:
        # creates the sequential index array that represents the total simulation days
        logger.debug(
            "Generating Temporal Index Array of Length %d",
            self._config.simulation_days
        )
        return np.arange(self._config.simulation_days, dtype=np.int32)
    
    @property
    def config(self) -> SimulationConfig:
        # exposes the immutable simulation config
        return self._config
    
    @property
    def rng(self) -> Generator:
        # exposes the initialized random number generator
        return self._rng
    
    @property
    def temporal_index(self) -> np.ndarray:
        # exposes the sequential index array
        return self._temporal_index
    
    @property
    def population_size(self) -> int:
        # exposes the simulation config total population size
        return self._config.population_size
    

if __name__ == "__main__":
    # initialize variables
    days = 365
    size = 1000

    # define the configuration
    hydromind_config = SimulationConfig(
        simulation_days=days,
        population_size=size,
        random_seed=2026
    )

    # initialize the environment
    global_init = GlobalInitializer(config=hydromind_config)

    # validate the outputs
    assert global_init.temporal_index.shape == (days, ), "Temporal index shape mismatch."
    assert global_init.temporal_index[0] == 0 and global_init.temporal_index[-1] == days - 1, "Incorrect temporal bounds."
    assert global_init.population_size == size, "Population size mismatch."

    # verify random number generator reproducibility
    test_draw_1 = global_init.rng.uniform(0, 1)
    global_init_2 = GlobalInitializer(config=hydromind_config)
    test_draw_2 = global_init_2.rng.uniform(0,1)
    assert test_draw_1 == test_draw_2, "Random Number Generator failed to reproduce indentical sequences from the same seed."

    logger.info("Standalone verification passed. Ready for the next step.")
