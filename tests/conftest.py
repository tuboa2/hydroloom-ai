from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.wqi_predictor import config


def _synthetic_frame(hemisphere: str = "north") -> pd.DataFrame:
    rng = np.random.default_rng(config.RANDOM_STATE)

    n = config.EXPECTED_ROW_COUNT
    day_index = np.arange(n)
    year_index = day_index // config.DAYS_PER_YEAR

    data: dict[str, object] = {
        "hemisphere": hemisphere,
        "day_index": day_index.astype("int64"),
        "year_index": year_index.astype("int64"),
        "season_label": rng.choice(
            ["Winter", "Spring", "Summer", "Autumn"],
            size=n,
        ),
    }

    binary_columns = {
        "is_weekend",
        "watering_ban_active",
        "holiday_weekend_flag",
    }

    ordinal_columns = {
        "tiered_pricing_regime",
    }

    already_created = set(data.keys())

    for column in config.EXPECTED_SCHEMA:
        if column in already_created:
            continue

        if column in binary_columns:
            data[column] = rng.integers(0, 2, size=n).astype("int8")
        elif column in ordinal_columns:
            data[column] = rng.integers(0, 3, size=n).astype("int8")
        elif column == config.TARGET_COLUMN:
            data[column] = rng.uniform(0.0, 85.0, size=n).astype("float32")
        else:
            data[column] = rng.normal(loc=10.0, scale=2.0, size=n).astype("float32")

    return pd.DataFrame(data)


@pytest.fixture
def synthetic_frame() -> pd.DataFrame:
    return _synthetic_frame("north")