from __future__ import annotations

import re

import numpy as np
import polars as pl

from .. import config
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class DataValidationError(ValueError):
    # raised when a data governance, schema, or temporal integrity fails
    pass


def matches_any_pattern(column: str, patterns: tuple[str, ...]) -> bool:
    return any(re.match(pattern, column) for pattern in patterns)


def validate_schema(dataframe: pl.DataFrame, hemisphere: str) -> None:
    missing = config.REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing:
        logger.error("%s: missing required columns: %s", hemisphere, sorted(missing))
        raise DataValidationError(f"{hemisphere}: missing required columns: {sorted(missing)}")

    unexpected = set(dataframe.columns) - set(config.REQUIRED_COLUMNS)
    if not unexpected:
        logger.debug("%s: schema validation passed.", hemisphere)
        return

    future_leakage = {
        column
        for column in unexpected
        if matches_any_pattern(column, config.FORBIDDEN_RAISE_PATTERNS)
    }
    if future_leakage:
        logger.error(
            "%s: forbidden future-leakage columns: %s",
            hemisphere,
            sorted(future_leakage),
        )
        raise DataValidationError(
            f"{hemisphere}: forbidden future-leakage columns detected:{sorted(future_leakage)}"
        )

    droppable = {
        column
        for column in unexpected
        if matches_any_pattern(column, config.FORBIDDEN_DROP_PATTERNS)
    }

    unknown = unexpected - future_leakage - droppable

    if unknown:
        logger.error("%s: unexpected unknown columns: %s", hemisphere, sorted(unknown))
        raise DataValidationError(
            f"{hemisphere}: unexpected unknown columns detected: {sorted(unknown)}"
        )

    logger.debug(
        "%s: schema validation passed with %d droppable extra columns.", hemisphere, len(droppable)
    )


def validate_no_nulls(dataframe: pl.DataFrame) -> None:
    null_counts = dataframe.null_count().to_dicts()[0]
    columns_with_nulls = {col: count for col, count in null_counts.items() if count > 0}
    if columns_with_nulls:
        logger.error(
            "Null values detected in %d columns: %s", len(columns_with_nulls), columns_with_nulls
        )
        raise DataValidationError(f"Null values detected: {columns_with_nulls}")


def validate_hemisphere_constant(dataframe: pl.DataFrame, hemisphere: str) -> None:
    values = dataframe[config.HEMISPHERE_COLUMN].cast(pl.String).str.to_lowercase().unique()
    if len(values) != 1 or values[0] != hemisphere.lower():
        logger.error("Hemisphere column expected '%s', found: %s", hemisphere, values)
        raise DataValidationError(
            f"Hemisphere column must contain only '{hemisphere}'. Found: {values}"
        )


def validate_temporal_index(dataframe: pl.DataFrame) -> None:
    day_index = dataframe[config.DAY_INDEX_COLUMN]
    if not day_index.dtype.is_integer():
        logger.error("day_index has non-integer dtype: %s", day_index.dtype)
        raise DataValidationError("day_index must be an integer dtype.")
    if not day_index.is_unique:
        logger.error("day_index contains duplicate values.")
        raise DataValidationError("day_index contains duplicate values.")
    if not day_index.is_sorted():
        logger.error("day_index is not monotonically increasing.")
        raise DataValidationError("day_index is not monotonically increasing")

    expected_days = np.arange(config.EXPECTED_ROW_COUNT, dtype=day_index.dtype.to_python())
    if not np.array_equal(day_index.to_numpy(), expected_days):
        logger.error("day_index does not span 0–1824 in ascending order.")
        raise DataValidationError("day_index must be exactly 0 through 1824 in ascending order.")

    year_index = dataframe[config.YEAR_INDEX_COLUMN]
    if not year_index.dtype.is_integer():
        logger.error("year_index has non-integer dtype: %s", year_index.dtype)
        raise DataValidationError("year_index must be an integer dtype.")

    unique_years = set(year_index.unique().cast(int))
    expected_years = set(range(config.EXPECTED_YEAR_COUNT))
    if unique_years != expected_years:
        logger.error(
            "year_index expected %s, found %s",
            sorted(expected_years),
            sorted(unique_years),
        )
        raise DataValidationError(
            f"year_index must contain exactly {sorted(expected_years)}."
            f"Found: {sorted(unique_years)}"
        )

    year_counts = year_index.value_counts().sort(config.YEAR_INDEX_COLUMN)
    if not (year_counts["count"] == config.DAYS_PER_YEAR).all():
        logger.error(
            "year_index year counts are not all %d: %s",
            config.DAYS_PER_YEAR,
            year_counts.to_dict(),
        )
        raise DataValidationError(
            "Each year_index must contain exactly 365 daily observations."
            f"Found counts: {year_counts.to_dict()}"
        )


def validate_target(dataframe: pl.DataFrame) -> None:
    target = dataframe[config.TARGET_COLUMN]
    if not target.dtype.is_numeric:
        logger.error("water_quality_index is not numeric: %s", target.dtype)
        raise DataValidationError("water_quality_index must be numeric.")
    if not target.is_finite().all():
        logger.error("water_quality_index contains non-finite values.")
        raise DataValidationError("water_quality_index contains non-finite values.")
    if (target < 0).any() or (target > 100).any():
        logger.error(
            "water_quality_index out of [0, 100] range: min=%.4f, max=%.4f",
            float(target.min()),
            float(target.max()),
        )
        raise DataValidationError("water_quality_index must be within the valid range of [0, 100].")


def validate_split_integrity(
    train_days: pl.Series, validation_days: pl.Series, test_days: pl.Series
) -> None:
    expected_train_rows = config.DAYS_PER_YEAR * len(config.TRAIN_YEARS)
    expected_validation_rows = config.DAYS_PER_YEAR
    expected_test_rows = config.DAYS_PER_YEAR

    if len(train_days) != expected_train_rows:
        logger.error(
            "Train split row count: expected %d, got %d.",
            expected_train_rows,
            len(train_days),
        )
        raise DataValidationError(
            f"Train split must contain {expected_train_rows} rows.Found {len(train_days)}."
        )

    if len(validation_days) != expected_validation_rows:
        logger.error(
            "Validation split row count: expected %d, got %d.",
            expected_validation_rows,
            len(validation_days),
        )
        raise DataValidationError(
            f"Validation split must contain {expected_validation_rows} rows."
            f"Found {len(validation_days)}."
        )

    if len(test_days) != expected_test_rows:
        logger.error(
            "Test split row count: expected %d, got %d.",
            expected_test_rows,
            len(test_days),
        )
        raise DataValidationError(
            f"Test split must contain {expected_test_rows} rows.Found {len(test_days)}."
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
        raise DataValidationError("Validation day_index is not monotonically increasing.")

    if not test_days.is_sorted():
        raise DataValidationError("Test day_index is not monotonically increasing.")

    logger.debug(
        "Split integrity validated: train=%d, val=%d, test=%d rows.",
        len(train_days),
        len(validation_days),
        len(test_days),
    )
