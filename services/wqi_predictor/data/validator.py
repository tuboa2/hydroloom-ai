from __future__ import annotations
import re
import numpy as np
import polars as pl
from .. import config

class DataValidationError(ValueError):
    # raised when a data governance, schema, or temporal integrity fails
    pass

def matches_any_pattern(column: str, patterns: tuple[str, ...]) -> bool:
    return any(re.match(pattern, column) for pattern in patterns)

def validate_schema(dataframe: pl.DataFrame, hemisphere: str) -> None:
    missing = config.REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing:
        raise DataValidationError(
            f"{hemisphere}: missing required columns: {sorted(missing)}"
        )
        
    unexpected = set(dataframe.columns) - set(config.REQUIRED_COLUMNS)
    if not unexpected:
        return
        
    future_leakage = {
        column
        for column in unexpected
        if matches_any_pattern(column, config.FORBIDDEN_RAISE_PATTERNS)
    }
    if future_leakage:
        raise DataValidationError(
            f"{hemisphere}: forbidden future-leakage columns detected:"
            f"{sorted(future_leakage)}"
        )

    droppable = {
        column
        for column in unexpected
        if matches_any_pattern(column, config.FORBIDDEN_DROP_PATTERNS)
    }

    unknown = unexpected - future_leakage - droppable

    if unknown:
        raise DataValidationError(
            f"{hemisphere}: unexpected unknown columns detected: {sorted(unknown)}"
        )

def validate_no_nulls(dataframe: pl.DataFrame) -> None:
    null_counts = dataframe.null_count().to_dicts()[0]
    columns_with_nulls = {
        col: count for col, count in null_counts.items() if count > 0
    }
    if columns_with_nulls:
        raise DataValidationError(
            f"Null values detected: {columns_with_nulls}"
        )

def validate_hemisphere_constant(dataframe: pl.DataFrame, hemisphere: str) -> None:
    values = (
        dataframe[config.HEMISPHERE_COLUMN]
        .cast(pl.String)
        .str.to_lowercase()
        .unique()
    )
    if len(values) != 1 or values[0] != hemisphere.lower():
        raise DataValidationError(
            f"Hemisphere column must contain only '{hemisphere}'. Found: {values}"
        )

def validate_temporal_index(dataframe: pl.DataFrame) -> None:
    day_index = dataframe[config.DAY_INDEX_COLUMN]
    if not day_index.dtype.is_integer():
        raise DataValidationError("day_index must be an integer dtype.")
    if not day_index.is_unique:
        raise DataValidationError("day_index contains duplicate values.")
    if not day_index.is_sorted():
        raise DataValidationError("day_index is not monotonically increasing")

    expected_days = np.arange(config.EXPECTED_ROW_COUNT, dtype=day_index.dtype.to_python())
    if not np.array_equal(day_index.to_numpy(), expected_days):
        raise DataValidationError(
            "day_index must be exactly 0 through 1824 in ascending order."
        )

    year_index = dataframe[config.YEAR_INDEX_COLUMN]
    if not year_index.dtype.is_integer():
        raise DataValidationError("year_index must be an integer dtype.")

    unique_years = set(year_index.unique().cast(int))
    expected_years = set(range(config.EXPECTED_YEAR_COUNT))
    if unique_years != expected_years:
        raise DataValidationError(
            f"year_index must contain exactly {sorted(expected_years)}."
            f"Found: {sorted(unique_years)}"
        )

    year_counts = year_index.value_counts().sort(config.YEAR_INDEX_COLUMN)
    if not (year_counts["count"] == config.DAYS_PER_YEAR).all():
        raise DataValidationError(
            "Each year_index must contain exactly 365 daily observations."
            f"Found counts: {year_counts.to_dict()}"
        )

def validate_target(dataframe: pl.DataFrame) -> None:
    target = dataframe[config.TARGET_COLUMN]
    if not target.dtype.is_numeric:
        raise DataValidationError("water_quality_index must be numeric.")
    if not np.isfinite(target).all():
        raise DataValidationError("water_quality_index contains non-finite values.")
    if (target < 0).any() or (target > 100).any():
        raise DataValidationError(
            "water_quality_index must be within the valid range of [0, 100]."
        )

def validate_split_integrity(
    train_days: pl.Series,
    validation_days: pl.Series,
    test_days: pl.Series
) -> None:
    expected_train_rows = config.DAYS_PER_YEAR * len(config.TRAIN_YEARS)
    expected_validation_rows = config.DAYS_PER_YEAR
    expected_test_rows = config.DAYS_PER_YEAR

    if len(train_days) != expected_train_rows:
        raise DataValidationError(
            f"Train split must contain {expected_train_rows} rows."
            f"Found {len(train_days)}."
        )

    if len(validation_days) != expected_validation_rows:
        raise DataValidationError(
            f"Validation split must contain {expected_validation_rows} rows."
            f"Found {len(validation_days)}."
        )

    if len(test_days) != expected_test_rows:
        raise DataValidationError(
            f"Test split must contain {expected_test_rows} rows."
            f"Found {len(test_days)}."
        )

    train_min_expected = 0
    train_max_expected = expected_train_rows - 1

    validation_min_expected = expected_train_rows
    validation_max_expected = validation_min_expected + expected_validation_rows - 1

    test_min_expected = validation_max_expected + 1
    test_max_expected = test_min_expected + expected_test_rows - 1

    if train_days.min() != train_min_expected or train_days.max() != train_max_expected:
            raise DataValidationError(
                f"Train day_index must span {train_min_expected} to {train_max_expected}. "
                f"Found {train_days.min()} to {train_days.max()}."
            )
    
    if (
        validation_days.min() != validation_min_expected
        or validation_days.max() != validation_max_expected
    ):
        raise DataValidationError(
            f"Validation day_index must span {validation_min_expected} to "
            f"{validation_max_expected}. "
            f"Found {validation_days.min()} to {validation_days.max()}."
        )

    if test_days.min() != test_min_expected or test_days.max() != test_max_expected:
        raise DataValidationError(
            f"Test day_index must span {test_min_expected} to {test_max_expected}. "
            f"Found {test_days.min()} to {test_days.max()}."
        )

    if train_days.max() >= validation_days.min():
        raise DataValidationError("Train split overlaps validation split.")

    if validation_days.max() >= test_days.min():
        raise DataValidationError("Validation split overlaps test split.")

    if not train_days.is_sorted():
        raise DataValidationError("Train day_index is not monotonically increasing.")

    if not validation_days.is_sorted():
        raise DataValidationError(
            "Validation day_index is not monotonically increasing."
        )

    if not test_days.is_sorted():
        raise DataValidationError("Test day_index is not monotonically increasing.")
