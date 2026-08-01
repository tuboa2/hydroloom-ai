from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from ..preprocess.pipeline_factory import build_preprocessor
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


def make_time_series_cv(n_splits: int = 5, gap: int = 0) -> TimeSeriesSplit:
    return TimeSeriesSplit(n_splits=n_splits, gap=gap)


def cv_fold_boundaries(
    x: pl.DataFrame,
    n_splits: int = 5,
    gap: int = 0,
) -> list[dict[str, int]]:
    cv = make_time_series_cv(n_splits, gap=gap)

    boundaries: list[dict[str, int]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(x)):
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue

        boundaries.append(
            {
                "fold": fold_idx,
                "train_start": int(train_idx[0]),
                "train_end": int(train_idx[-1]),
                "val_start": int(val_idx[0]),
                "val_end": int(val_idx[-1]),
                "train_size": int(len(train_idx)),
                "val_size": int(len(val_idx)),
            }
        )

    return boundaries


def evaluate_features_cv(
    x_train: pl.DataFrame,
    y_train: pl.Series,
    feature_columns: Sequence[str],
    n_splits: int = 5,
    gap: int = 0,
    random_state: int = 42,
    n_jobs: int | None = -1,
) -> dict[str, float]:
    feature_columns = list(feature_columns)

    if not feature_columns:
        return {
            "mean_rmse": float("nan"),
            "std_rmse": float("nan"),
            "n_folds": 0,
        }

    logger.debug(
        "Evaluating %d features with %d-fold TSCV (gap=%d).",
        len(feature_columns),
        n_splits,
        gap,
    )

    cv = make_time_series_cv(n_splits=n_splits, gap=gap)

    fold_scores: list[float] = []

    for train_idx, val_idx in cv.split(x_train):
        x_fold_train = x_train[train_idx][feature_columns]
        x_fold_val = x_train[val_idx][feature_columns]

        y_fold_train = y_train[train_idx]
        y_fold_val = y_train[val_idx]

        if len(x_fold_train) < 2 or len(x_fold_val) < 2:
            continue

        preprocessor = build_preprocessor(feature_columns)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        random_state=random_state,
                        n_jobs=n_jobs,
                    ),
                ),
            ]
        )

        pipeline.fit(x_fold_train, y_fold_train)

        predictions = pipeline.predict(x_fold_val)

        rmse = float(np.sqrt(mean_squared_error(y_fold_val, predictions)))
        fold_scores.append(rmse)

    if not fold_scores:
        return {
            "mean_rmse": float("nan"),
            "std_rmse": float("nan"),
            "n_folds": 0,
        }

    return {
        "mean_rmse": float(np.mean(fold_scores)),
        "std_rmse": float(np.std(fold_scores)),
        "n_folds": int(len(fold_scores)),
    }


def gap_robustness_check(
    x_train: pl.DataFrame,
    y_train: pl.Series,
    feature_columns: Sequence[str],
    n_splits: int = 5,
    random_state: int = 42,
    n_jobs: int | None = -1,
) -> dict[str, dict[str, float]]:
    gap_zero = evaluate_features_cv(
        x_train=x_train,
        y_train=y_train,
        feature_columns=feature_columns,
        n_splits=n_splits,
        gap=0,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    gap_seven = evaluate_features_cv(
        x_train=x_train,
        y_train=y_train,
        feature_columns=feature_columns,
        n_splits=n_splits,
        gap=7,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    logger.info(
        "Gap robustness check: gap_0_rmse=%.4f, gap_7_rmse=%.4f.",
        gap_zero.get("mean_rmse", float("nan")),
        gap_seven.get("mean_rmse", float("nan")),
    )

    return {
        "gap_0": gap_zero,
        "gap_7": gap_seven,
    }
