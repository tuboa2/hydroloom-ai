from dataclasses import dataclass
from numpy.random import Generator
import logging
import numpy as np

__all__ = [
    "SimulationConfig",
    "GlobalInitializer",
]

# 1. global initialization
@dataclass(frozen=True)
class SimulationConfig:
    # Immutable configuration parameters for the entire simulation.
    simulation_days: int = 1825
    days_per_year: int = 365
    num_years: int = 5
    population_size: int = 100_000
    subsample: int = 5_000
    random_seed: int = 2032

    def __post_init__(self) -> None:
        if self.simulation_days <= 0:
            raise ValueError(
                f"simulation_days must be > 0. Received: {self.simulation_days}"
            )
        if self.population_size <= 0:
            raise ValueError(
                f"population_size must be > 0. Received: {self.population_size}"
            )
        if self.random_seed < 0:
            raise ValueError(
                f"random_seed must be a positive number. Received: {self.random_seed}"
            )
        if self.simulation_days != self.days_per_year * self.num_years:
            raise ValueError(
                f"simulation_days ({self.simulation_days}) must equal "
                f"days_per_year × num_years "
                f"({self.days_per_year} × {self.num_years})"
            )
        if self.subsample > self.population_size:
            raise ValueError(
                f"catchment_subsample ({self.subsample}) "
                f"cannot exceed population_size ({self.population_size})"
            )

class GlobalInitializer:
    def __init__(self, config: SimulationConfig) -> None:
        self._config = config
        self._logger = logging.getLogger(__name__)
        self._rng: Generator = self._initialize_rng()
        self._temporal_index: np.ndarray = self._initialize_temporal_horizon()

        self._logger.info("Global Environment Initialization Complete.")
        self._logger.info(
            "Parameters -> Days: %d | Population: %d | Seed: %d",
            self._config.simulation_days,
            self._config.population_size,
            self._config.random_seed,
        )

    # private helper methods
    def _initialize_rng(self) -> Generator:
        self._logger.debug(
            "Initializing NumPy Generator with seed: %d",
            self._config.random_seed,
        )
        return np.random.default_rng(seed=self._config.random_seed)

    def _initialize_temporal_horizon(self) -> np.ndarray:
        self._logger.debug(
            "Generating Temporal Index Array of Length %d",
            self._config.simulation_days,
        )
        return np.arange(self._config.simulation_days, dtype=np.int32)

    # public attributes/properties
    @property
    def sim_config(self) -> SimulationConfig:
        return self._config

    @property
    def rng(self) -> Generator:
        return self._rng

    @property
    def temporal_index(self) -> np.ndarray:
        return self._temporal_index

    @property
    def simulation_days(self) -> int:
        return len(self._temporal_index)

    @property
    def days_per_year(self) -> int:
        return self._config.days_per_year

    @property
    def population_size(self) -> int:
        return self._config.population_size
