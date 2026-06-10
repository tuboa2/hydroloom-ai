from __future__ import annotations
from dataclasses import dataclass
from numpy.random import Generator

import logging
import numpy

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
        
# TODO: GlobalInitializer Class
