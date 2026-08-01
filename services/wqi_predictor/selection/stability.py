from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from ..preprocess.pipeline_factory import build_preprocessor

STABILITY_REPORT_COLUMNS = (
    "feature",
    "stability",
    "mean_score",
    "mean_perm_importance",
    "mean_split_importance",
    "ranking_score",
)


def _normalize_non_negative(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), 0.0, None)

    if clipped.size == 0:
        return clipped

    max_value = float(np.max(clipped))

    if max_value <= 0.0:
        return np.zeros_like(clipped)

    return clipped / max_value


def _transformed_feature_names(
    pipeline: Pipeline,
    fallback_features: Sequence[str],
) -> list[str]:
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        names = preprocessor.get_feature_names_out()
        return [str(name) for name in names]
    except Exception:
        return list(fallback_features)


def _align_importances(
    transformed_names: Sequence[str],
    importances: np.ndarray,
    feature_columns: Sequence[str],
) -> np.ndarray:
    importances = np.asarray(importances, dtype=float)

    if importances.size == len(feature_columns) and list(transformed_names) == list(
        feature_columns
    ):
        return importances

    aligned = np.zeros(len(feature_columns), dtype=float)

    if importances.size != len(transformed_names):
        return aligned

    name_to_importance = dict(zip(transformed_names, importances))

    for i, column in enumerate(feature_columns):
        if column in name_to_importance:
            aligned[i] = float(name_to_importance[column])
            continue

        matching_values = [
            float(value)
            for name, value in name_to_importance.items()
            if str(name).split("__")[-1] == column
        ]

        if matching_values:
            aligned[i] = float(np.mean(matching_values))

    return aligned


def run_stability_selection(
    x_train: pl.DataFrame,
    y_train: pl.Series,
    feature_columns: Sequence[str],
    n_splits: int = 5,
    gap: int = 0,
    seeds: Sequence[int] = (42, 1337, 2024, 2025, 2026),
    min_stability: float = 0.70,
    run_score_threshold: float = 0.02,
    candidate_cap: int | None = None,
    min_fallback_features: int = 10,
    n_repeats: int = 5,
    n_jobs: int = -1,
    permutation_weight: float = 0.5,
    split_weight: float = 0.5,
) -> tuple[list[str], pl.DataFrame, dict[str, float]]:
    feature_columns = list(feature_columns)

    if not feature_columns:
        empty_report = pl.DataFrame(schema=list(STABILITY_REPORT_COLUMNS))
        return [], empty_report, {}

    weight_sum = float(permutation_weight) + float(split_weight)

    if weight_sum <= 0.0:
        permutation_weight = 0.5
        split_weight = 0.5
    else:
        permutation_weight = float(permutation_weight) / weight_sum
        split_weight = float(split_weight) / weight_sum

    x = x_train[feature_columns].clone()
    y = y_train.clone()

    cv = TimeSeriesSplit(n_splits=n_splits, gap=gap)

    selected_by_run: dict[str, list[bool]] = {column: [] for column in feature_columns}

    score_by_run: dict[str, list[float]] = {column: [] for column in feature_columns}

    perm_importance_by_run: dict[str, list[float]] = {column: [] for column in feature_columns}

    split_importance_by_run: dict[str, list[float]] = {column: [] for column in feature_columns}

    for seed in seeds:
        estimators = [
            (
                "random_forest",
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=seed,
                    n_jobs=n_jobs,
                ),
            ),
            (
                "extra_trees",
                ExtraTreesRegressor(
                    n_estimators=300,
                    random_state=seed,
                    n_jobs=n_jobs,
                ),
            ),
        ]

        for _, estimator in estimators:
            fold_perm_importances: list[np.ndarray] = []
            fold_split_importances: list[np.ndarray] = []

            for train_idx, val_idx in cv.split(x):
                x_fold_train = x[train_idx]
                x_fold_val = x[val_idx]

                y_fold_train = y[train_idx]
                y_fold_val = y[val_idx]

                if len(x_fold_train) < 2 or len(x_fold_val) < 2:
                    continue

                preprocessor = build_preprocessor(feature_columns)

                pipeline = Pipeline(
                    steps=[
                        ("preprocessor", preprocessor),
                        ("model", clone(estimator)),
                    ]
                )

                pipeline.fit(x_fold_train, y_fold_train)

                model = pipeline.named_steps["model"]
                split_importances = getattr(model, "feature_importances_", None)

                if split_importances is None:
                    split_aligned = np.zeros(len(feature_columns), dtype=float)
                else:
                    transformed_names = _transformed_feature_names(
                        pipeline=pipeline, fallback_features=feature_columns
                    )

                    split_aligned = _align_importances(
                        transformed_names=transformed_names,
                        importances=np.asarray(split_importances),
                        feature_columns=feature_columns,
                    )

                fold_split_importances.append(split_aligned)

                X_eval = x_fold_val.to_pandas()
                y_eval = y_fold_val.to_pandas()
                
                permutation_result = permutation_importance(
                    estimator=pipeline,
                    X=X_eval,
                    y=y_eval,
                    n_repeats=n_repeats,
                    random_state=seed,
                    scoring="neg_root_mean_squared_error",
                    n_jobs=n_jobs,
                )

                fold_perm_importances.append(
                    np.asarray(permutation_result.importances_mean, dtype=float)
                )

            if fold_perm_importances:
                mean_perm_importance = np.mean(fold_perm_importances, axis=0)
            else:
                mean_perm_importance = np.zeros(len(feature_columns), dtype=float)

            if fold_split_importances:
                mean_split_importance = np.mean(fold_split_importances, axis=0)
            else:
                mean_split_importance = np.zeros(len(feature_columns), dtype=float)

            normalized_perm_importance = _normalize_non_negative(mean_perm_importance)
            normalized_split_importance = _normalize_non_negative(mean_split_importance)

            combined_score = (
                permutation_weight * normalized_perm_importance
                + split_weight * normalized_split_importance
            )

            selected = (combined_score >= run_score_threshold) & (
                (mean_perm_importance > 0.0) | (mean_split_importance > 0.0)
            )

            for i, column in enumerate(feature_columns):
                selected_by_run[column].append(bool(selected[i]))
                score_by_run[column].append(float(combined_score[i]))
                perm_importance_by_run[column].append(float(mean_perm_importance[i]))
                split_importance_by_run[column].append(float(mean_split_importance[i]))

    records: list[dict[str, Any]] = []

    for column in feature_columns:
        selected_runs = selected_by_run[column]
        scores = score_by_run[column]
        perm_values = perm_importance_by_run[column]
        split_values = split_importance_by_run[column]

        stability = float(np.mean(selected_runs)) if selected_runs else 0.0
        mean_score = float(np.mean(scores)) if scores else 0.0
        mean_perm_importance = float(np.mean(perm_values)) if perm_values else 0.0
        mean_split_importance = float(np.mean(split_values)) if split_values else 0.0

        ranking_score = (0.6 * stability) + (0.4 * mean_score)

        records.append(
            {
                "feature": column,
                "stability": stability,
                "mean_score": mean_score,
                "mean_perm_importance": mean_perm_importance,
                "mean_split_importance": mean_split_importance,
                "ranking_score": ranking_score,
            }
        )

    report = pl.DataFrame(records)

    if report.is_empty():
        report = pl.DataFrame(schema=list(STABILITY_REPORT_COLUMNS))
        return [], report, {}

    report = report.sort("ranking_score", descending=True)

    selected_features = (
        report.filter(pl.col("stability") >= min_stability).get_column("feature").to_list()
    )

    if len(selected_features) < min_fallback_features:
        fallback_count = max(min_fallback_features, len(selected_features))
        fallback_features = report.head(fallback_count)["feature"].to_list()
        selected_features = list(dict.fromkeys(selected_features + fallback_features))

    if candidate_cap is not None:
        selected_features = selected_features[:candidate_cap]

    scores = dict(zip(report["feature"], report["ranking_score"]))

    return selected_features, report, scores
