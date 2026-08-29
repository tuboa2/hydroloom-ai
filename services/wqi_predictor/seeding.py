from __future__ import annotations

import os
import random

import numpy as np

from .config import RANDOM_STATE
from .utils.logging_config import get_logger

logger = get_logger(__name__)


def env_seed(seed: int = RANDOM_STATE) -> None:
    """Lock global randomness for reproducible execution."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    logger.info("Global random seed set to %d.", seed)


seed_everything = env_seed

__all__ = ["RANDOM_STATE", "env_seed", "seed_everything"]
