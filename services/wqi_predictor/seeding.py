from __future__ import annotations
import os
import random
import numpy as np
from config import RANDOM_STATE

def env_seed(seed: int = RANDOM_STATE) -> None:
    # lock global randomness for reproducible execution
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
