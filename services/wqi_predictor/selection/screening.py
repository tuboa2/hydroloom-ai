from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import polars as pl

import wandb

from ..config import (
    DAY_INDEX_COLUMN,
    FORBIDDEN_DROP_PATTERNS,
    FORBIDDEN_RAISE_PATTERNS,
    HEMISPHERE_COLUMN,
    TARGET_COLUMN,
    YEAR_INDEX_COLUMN,
)
from ..data.validator import matches_any_pattern
from ..preprocess.feature_groups import (
    FAMILY_PRIORITY,
    REQUIRED_FEATURES,
    infer_feature_family,
)
from ..utils.logging_config import get_logger

logger = get_logger(__name__)

LEAKAGE_IDENTIFIER_COLUMNS = frozenset(
    {
        HEMISPHERE_COLUMN,
        DAY_INDEX_COLUMN,
        YEAR_INDEX_COLUMN,
        TARGET_COLUMN,
        "target",
    }
)


def _feature_priority(column: str) -> int:
    if column in REQUIRED_FEATURES:
        return 0

    family = infer_feature_family(column)
    return FAMILY_PRIORITY.get(family, 100)


def _choose_collinear_drop(left: str, right: str) -> str:
    left_priority = _feature_priority(left)
    right_priority = _feature_priority(right)

    if left_priority > right_priority:
        return left

    if right_priority > left_priority:
        return right

    return max(left, right)


def run_screening(
    x_train: pl.DataFrame,
    feature_columns: Sequence[str] | None = None,
    missingness_threshold: float = 0.999,
    collinearity_threshold: float = 0.999999,
) -> tuple[list[str], dict[str, Any]]:
    if feature_columns is None:
        feature_columns = list(x_train.columns)
    else:
        feature_columns = list(feature_columns)

    logger.info("Running feature screening on %d candidates.", len(feature_columns))

    if wandb.run is not None:
        wandb.log({"screening/input_count": len(feature_columns)})

    dropped_leakage: list[str] = []
    dropped_missingness: list[str] = []
    dropped_zero_variance: list[str] = []
    dropped_collinearity: list[str] = []
    collinear_pairs: list[dict[str, object]] = []

    retained_after_leakage: list[str] = []

    for column in feature_columns:
        if column in LEAKAGE_IDENTIFIER_COLUMNS:
            dropped_leakage.append(column)
            continue

        if matches_any_pattern(column, FORBIDDEN_RAISE_PATTERNS):
            dropped_leakage.append(column)
            continue

        if matches_any_pattern(column, FORBIDDEN_DROP_PATTERNS):
            dropped_leakage.append(column)
            continue

        retained_after_leakage.append(column)

    if wandb.run is not None:
        wandb.log(
            {
                "screening/retained_count": len(retained_after_leakage),
                "screening/dropped_leakage_count": len(dropped_leakage),
            }
        )

    retained_after_missingness: list[str] = []

    for column in retained_after_leakage:
        missingness = float(x_train[column].is_null().mean())

        if missingness > missingness_threshold:
            dropped_missingness.append(column)
            continue

        retained_after_missingness.append(column)

    if wandb.run is not None:
        wandb.log(
            {
                "screening/retained_count": len(retained_after_missingness),
                "screening/dropped_missingness_count": len(dropped_missingness),
            }
        )

    retained_after_zero_variance: list[str] = []

    for column in retained_after_missingness:
        series = x_train[column]

        if series.dtype.is_numeric():
            filled = series.fill_null(series.median())
            unique_count = int(filled.drop_nulls().n_unique())
            std_value = float(filled.std())

            if unique_count <= 1 or std_value == 0.0:
                dropped_zero_variance.append(column)
                continue

        else:
            mode = series.drop_nulls().mode()

            if mode.is_empty():
                fill_value = "missing"
            else:
                fill_value = mode[0]

            filled = series.fill_null(fill_value)
            unique_count = int(filled.drop_nulls().n_unique())

            if unique_count <= 1:
                dropped_zero_variance.append(column)
                continue

        retained_after_zero_variance.append(column)

    if wandb.run is not None:
        wandb.log(
            {
                "screening/retained_count": len(retained_after_zero_variance),
                "screening/dropped_zero_variance_count": len(dropped_zero_variance),
            }
        )

    numeric_columns = [
        column for column in retained_after_zero_variance if x_train[column].dtype.is_numeric()
    ]

    collinear_drop_set: set[str] = set()

    if len(numeric_columns) > 1:
        numeric_frame = x_train[numeric_columns].clone()
        medians = numeric_frame.median()
        numeric_frame = numeric_frame.with_columns(pl.all().fill_null(pl.all().median()))

        correlation = numeric_frame.corr().select(pl.all().abs())

        for i in range(len(numeric_columns)):
            for j in range(i + 1, len(numeric_columns)):
                left = numeric_columns[i]
                right = numeric_columns[j]

                if left in collinear_drop_set or right in collinear_drop_set:
                    continue

                correlation_value = float(correlation[i, right])

                if correlation_value >= collinearity_threshold:
                    drop_column = _choose_collinear_drop(left, right)
                    collinear_drop_set.add(drop_column)

                    collinear_pairs.append(
                        {
                            "feature_a": left,
                            "feature_b": right,
                            "abs_correlation": correlation_value,
                            "dropped": drop_column,
                        }
                    )

    dropped_collinearity = sorted(collinear_drop_set)

    retained_final = [
        column for column in retained_after_zero_variance if column not in collinear_drop_set
    ]

    if wandb.run is not None:
        wandb.log(
            {
                "screening/retained_count": len(retained_final),
                "screening/dropped_collinearity_count": len(dropped_collinearity),
            }
        )

    report: dict[str, Any] = {
        "input_features": feature_columns,
        "retained_features": retained_final,
        "dropped": {
            "leakage": sorted(dropped_leakage),
            "missingness": sorted(dropped_missingness),
            "zero_variance": sorted(dropped_zero_variance),
            "collinearity": dropped_collinearity,
        },
        "collinear_pairs": collinear_pairs,
        "counts": {
            "input": len(feature_columns),
            "retained": len(retained_final),
            "dropped_leakage": len(dropped_leakage),
            "dropped_missingness": len(dropped_missingness),
            "dropped_zero_variance": len(dropped_zero_variance),
            "dropped_collinearity": len(dropped_collinearity),
        },
    }

    logger.info(
        "Screening complete: %d → %d retained "
        "(leakage=%d, missingness=%d, zero_var=%d, collinear=%d).",
        len(feature_columns),
        len(retained_final),
        len(dropped_leakage),
        len(dropped_missingness),
        len(dropped_zero_variance),
        len(dropped_collinearity),
    )

    if wandb.run is not None:
        dropped_records = []
        for stage, features in report["dropped"].items():
            for feature in features:
                dropped_records.append([stage, feature])

        dropped_table = wandb.Table(columns=["Stage", "Feature"], data=dropped_records)
        wandb.log({"screening/dropped_summary_table": dropped_table})

        collinear_records = [
            [p["feature_a"], p["feature_b"], p["abs_correlation"], p["dropped"]]
            for p in collinear_pairs
        ]
        collinear_table = wandb.Table(
            columns=["feature_a", "feature_b", "abs_correlation", "dropped"], data=collinear_records
        )
        wandb.log({"screening/collinear_pairs_table": collinear_table})

    return retained_final, report
