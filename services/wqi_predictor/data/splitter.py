from __future__ import annotations
import polars as pl
from dataclasses import dataclass
from typing import Any, Literal
from .. import config
from .ingestion import IngestedHemisphere
from .validator import validate_split_integrity

@dataclass(frozen=True)
class ChronologicalSplit:
    hemisphere: Literal["north", "south"]
    x_train: pl.DataFrame
    y_train: pl.Series
    x_val: pl.DataFrame
    y_val: pl.Series
    x_test: pl.DataFrame
    y_test: pl.Series
    metadata: dict[str, Any]

def to_float(val):
    return float(val) if val is not None else None

def _series_summary(series: pl.Series) -> dict[str, Any]:
    return {
        "mean": to_float(series.mean()),
        "std": to_float(series.std()),
        "min": to_float(series.min()),
        "max": to_float(series.max()),
        "zero_count": int((series == 0).sum())
    }

def split_chronologically(ingested: IngestedHemisphere) -> ChronologicalSplit:
    raw = ingested.raw_frame
    features = ingested.feature_frame
    target = ingested.target

    train_mask = raw[config.YEAR_INDEX_COLUMN].is_in(config.TRAIN_YEARS).to_numpy()
    val_mask = (raw[config.YEAR_INDEX_COLUMN] == config.VALIDATION_YEAR).to_numpy()
    test_mask = (raw[config.YEAR_INDEX_COLUMN] == config.TEST_YEAR).to_numpy()

    if not (train_mask | val_mask | test_mask).all():
        raise ValueError(
            f"{ingested.hemisphere}: one or more rows are outside on what is defined in "
            "train/validation/test year assignments."
        )

    train_days = raw.filter(train_mask).get_column(config.DAY_INDEX_COLUMN)
    val_days = raw.filter(val_mask).get_column(config.DAY_INDEX_COLUMN)
    test_days = raw.filter(test_mask).get_column(config.DAY_INDEX_COLUMN)

    validate_split_integrity(train_days, val_days, test_days)

    x_train = features.filter(train_mask)
    x_val = features.filter(val_mask)
    x_test = features.filter(test_mask)

    y_train = target.filter(train_mask)
    y_val = target.filter(val_mask)
    y_test = target.filter(test_mask)

    metadata: dict[str, Any] = {
        "sizes": {
            "train": int(len(x_train)),
            "validation": int(len(x_val)),
            "test": int(len(x_test)),
        },
        "day_index": {
            "train": {
                "min": int(train_days.min()),
                "max": int(train_days.max()),
            },
            "validation": {
                "min": int(val_days.min()),
                "max": int(val_days.max()),
            },
            "test": {
                "min": int(test_days.min()),
                "max": int(test_days.max()),
            },
        },
        "target": {
            "train": _series_summary(y_train),
            "validation": _series_summary(y_val),
            "test": _series_summary(y_test),
        },
        "feature_columns": list(features.columns)
    }

    return ChronologicalSplit(
        hemisphere=ingested.hemisphere,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        metadata=metadata,
    )
