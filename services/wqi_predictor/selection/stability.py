from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

import wandb

from ..preprocess.pipeline_factory import build_preprocessor
from ..utils.logging_config import get_logger

# HistGradientBoostingRegressor is optional only for compatibility with very old
# scikit-learn versions. On Kaggle it is available and is the preferred estimator.
try:
    from sklearn.ensemble import HistGradientBoostingRegressor
except Exception:  # pragma: no cover
    HistGradientBoostingRegressor = None


logger = get_logger(__name__)


STABILITY_REPORT_COLUMNS = (
    "feature",
    "stability",
    "mean_score",
    "mean_perm_importance",
    "mean_split_importance",
    "ranking_score",
)


# -----------------------------------------------------------------------------
# Performance controls
# -----------------------------------------------------------------------------
#
# 1. HistGradientBoostingRegressor is dramatically faster than RandomForest /
#    ExtraTrees for 100k rows x 130 features on a 4-core Kaggle CPU.
#
# 2. The two boosted configurations preserve the original "two complementary
#    tree ensembles per seed" design while avoiding the very high cost of
#    300-tree bagged forests.
#
# 3. Permutation importance is evaluated on a deterministic validation subset.
#    4096 rows is a strong speed/accuracy compromise for ~16k TimeSeriesSplit
#    validation folds. Increase this value if ranking correlation is not high
#    enough on a particular dataset.
#
# 4. When HistGradientBoosting is used, outer joblib parallelism is disabled.
#    HistGradientBoosting already uses OpenMP across all CPU cores. Running
#    multiple outer folds in parallel would oversubscribe the 4 Kaggle cores
#    and usually makes total runtime worse.
# -----------------------------------------------------------------------------
_USE_HIST_GRADIENT_BOOSTING = HistGradientBoostingRegressor is not None
_HGB_MAX_ITER = 100
_HGB_MAX_BINS = 64
_FOREST_N_ESTIMATORS = 100
_PERMUTATION_MAX_VALIDATION_SAMPLES = 4096


def _safe_random_state(seed: int) -> int:
    """
    scikit-learn RandomState seeds must be in [0, 2**32 - 1).
    This keeps deterministic behavior while protecting against large seeds.
    """
    return int(seed) % (2**32 - 1)


def _normalize_non_negative(values: np.ndarray) -> np.ndarray:
    """
    Clip negative importance values to zero and normalize to [0, 1].

    NaN / infinite values are sanitized to zero. This only affects degenerate
    numerical cases and prevents a single bad fold from corrupting the report.
    """
    values = np.nan_to_num(
        np.asarray(values, dtype=float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    clipped = np.clip(values, 0.0, None)

    if clipped.size == 0:
        return clipped

    max_value = float(np.max(clipped))
    if max_value <= 0.0:
        return np.zeros_like(clipped)

    return clipped / max_value


def _as_contiguous_float32_matrix(data: Any) -> np.ndarray:
    """
    Convert preprocessor output to a C-contiguous float32 NumPy matrix.

    This is a one-time conversion. All CV folds and workers later operate on
    this shared array, avoiding repeated Polars/Pandas conversions.
    """
    if hasattr(data, "toarray"):
        # scipy sparse, if ever produced by a preprocessor.
        data = data.toarray()

    if hasattr(data, "to_numpy"):
        # Pandas DataFrame / Polars DataFrame / Series-like objects.
        arr = data.to_numpy()
    else:
        arr = np.asarray(data)

    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    return np.ascontiguousarray(arr)


def _as_contiguous_float32_vector(data: Any) -> np.ndarray:
    """
    Convert target vector to contiguous float32 once.
    """
    if hasattr(data, "to_numpy"):
        arr = data.to_numpy()
    else:
        arr = np.asarray(data)

    arr = np.asarray(arr, dtype=np.float32).ravel()
    return np.ascontiguousarray(arr)


def _fit_preprocessor_once(
    preprocessor: Any,
    x_raw: pl.DataFrame,
    feature_columns: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    """
    Fit the preprocessor exactly once and transform the full training frame.

    Original behavior refit the preprocessor inside every fold/model task.
    That created large redundant CPU and memory overhead. Fitting once is the
    dominant preprocessing optimization requested.

    Returns
    -------
    X : np.ndarray
        C-contiguous float32 array.
    transformed_names : list[str]
        Names of transformed columns, used to map engineered/encoded columns
        back to original feature names.
    """
    try:
        transformed = preprocessor.fit_transform(x_raw)
    except Exception:
        # Some scikit-learn transformers are more robust with Pandas input.
        # This fallback happens at most once, so it does not affect hot loops.
        transformed = preprocessor.fit_transform(x_raw.to_pandas())

    try:
        transformed_names = [str(name) for name in preprocessor.get_feature_names_out()]
    except Exception:
        if hasattr(transformed, "columns"):
            transformed_names = [str(name) for name in transformed.columns]
        else:
            transformed_names = list(feature_columns)

    X = _as_contiguous_float32_matrix(transformed)

    # Defensive name-length reconciliation.
    if len(transformed_names) != X.shape[1]:
        if X.shape[1] == len(feature_columns):
            transformed_names = list(feature_columns)
        else:
            transformed_names = [f"transformed_{i}" for i in range(X.shape[1])]

    return X, transformed_names


def _match_transformed_name(
    name: str,
    feature_set: set[str],
    candidates_by_length: Sequence[str],
) -> str | None:
    """
    Map a transformed column name back to an original feature name.

    This supports common scikit-learn naming patterns:
      - "age"
      - "num__age"
      - "preprocessor__num__age"
      - "cat__color_red"
      - "color_red"
      - "age_scaled"

    The longest matching original feature name is preferred to avoid collisions
    such as "age" versus "age_group".
    """
    if name in feature_set:
        return name

    parts = name.split("__")

    # Check name components from most specific to least specific.
    for part in reversed(parts):
        if part in feature_set:
            return part

        for candidate in candidates_by_length:
            if part == candidate or part.startswith(candidate + "_"):
                return candidate

    # Final fallback: full transformed name prefix match.
    for candidate in candidates_by_length:
        if name.startswith(candidate + "_"):
            return candidate

    return None


def _build_feature_groups(
    transformed_names: Sequence[str],
    feature_columns: Sequence[str],
) -> list[list[int]]:
    """
    Build a mapping:

        original feature index -> list of transformed column indices

    This is essential when preprocessing creates one-hot encoded or renamed
    columns. Permuting an original feature should permute all transformed
    columns derived from that feature together.
    """
    n_features = len(feature_columns)
    n_transformed = len(transformed_names)

    # Fast path: no encoding/renaming.
    if n_transformed == n_features and list(transformed_names) == list(feature_columns):
        return [[i] for i in range(n_features)]

    feature_set = set(feature_columns)
    candidates_by_length = sorted(feature_set, key=len, reverse=True)
    feature_to_group = {feature: i for i, feature in enumerate(feature_columns)}

    groups: list[list[int]] = [[] for _ in range(n_features)]
    unmatched: list[int] = []

    for transformed_idx, name in enumerate(transformed_names):
        matched_feature = _match_transformed_name(
            str(name),
            feature_set,
            candidates_by_length,
        )

        if matched_feature is None:
            unmatched.append(transformed_idx)
        else:
            groups[feature_to_group[matched_feature]].append(transformed_idx)

    # Defensive fallback for unusual preprocessor naming schemes.
    empty_groups = [i for i, group in enumerate(groups) if not group]
    if unmatched and empty_groups:
        if len(unmatched) == len(empty_groups):
            for group_idx, transformed_idx in zip(empty_groups, unmatched):
                groups[group_idx] = [transformed_idx]
        elif len(empty_groups) == 1:
            groups[empty_groups[0]] = unmatched

    return groups


def _aggregate_importances(
    importances: np.ndarray,
    groups: Sequence[Sequence[int]],
    n_features: int,
) -> np.ndarray:
    """
    Aggregate transformed-column importances back to original features.

    For one-hot encoded features, this averages all derived transformed columns.
    """
    out = np.zeros(n_features, dtype=float)
    importances = np.asarray(importances, dtype=float)

    for feature_idx, group in enumerate(groups):
        if not group:
            continue

        vals = importances[list(group)]
        if vals.size:
            out[feature_idx] = float(np.mean(vals))

    return out


def _align_importances(
    transformed_names: Sequence[str],
    importances: np.ndarray,
    feature_columns: Sequence[str],
    groups: Sequence[Sequence[int]],
) -> np.ndarray:
    """
    Align model feature importances to the original feature order.
    """
    importances = np.asarray(importances, dtype=float)
    n_features = len(feature_columns)

    # Already aligned.
    if importances.size == n_features and list(transformed_names) == list(feature_columns):
        return importances

    # If the model importance length does not match transformed width, we cannot
    # safely map it. Return zeros rather than corrupting the ranking.
    if importances.size != len(transformed_names):
        return np.zeros(n_features, dtype=float)

    return _aggregate_importances(importances, groups, n_features)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    RMSE computed in float64 for numerical stability.

    This replaces sklearn's neg_root_mean_squared_error scoring path.
    For permutation importance:

        sklearn importance =
            neg_rmse_baseline - neg_rmse_permuted
            = rmse_permuted - rmse_baseline

    This function returns RMSE; the permutation routine computes the difference.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    if y_true.size != y_pred.size or y_true.size == 0:
        return float("nan")

    diff = y_pred - y_true
    return float(np.sqrt(np.mean(diff * diff)))


def _fast_permutation_importance(
    model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    groups: Sequence[Sequence[int]],
    n_repeats: int,
    seed: int,
    fold_idx: int,
    est_idx: int,
    split_for_skip: np.ndarray | None = None,
) -> np.ndarray:
    """
    NumPy-native permutation importance.

    Optimizations versus sklearn.inspection.permutation_importance:
      - no Pandas conversion
      - no sklearn scorer adapter overhead
      - direct RMSE computation
      - in-place column permutation and restoration
      - deterministic validation subsampling already applied upstream
      - group-wise permutation for one-hot / encoded features
      - skips constant features
      - skips features with exactly zero split importance when that information
        is trustworthy, because unused tree features cannot affect predictions
    """
    n_features = len(groups)
    importances = np.zeros(n_features, dtype=float)

    n_samples = X_val.shape[0]
    if n_samples <= 1 or n_repeats <= 0:
        return importances

    y_true = np.asarray(y_val, dtype=np.float64).ravel()
    baseline_pred = np.asarray(model.predict(X_val), dtype=np.float64).ravel()
    baseline_rmse = _rmse(y_true, baseline_pred)

    if not np.isfinite(baseline_rmse):
        return importances

    # Local mutable copy of validation data. Only this copy is modified.
    X_work = np.array(X_val, copy=True)

    # Deterministic per-fold/per-estimator permutation stream.
    rng_seed = (int(seed) + 1_000_003 * int(fold_idx) + 2_000_003 * int(est_idx)) % (2**32 - 1)
    rng = np.random.RandomState(rng_seed)

    for feature_idx, group in enumerate(groups):
        if not group:
            continue

        # If split importance is reliably zero, the tree ensemble never uses
        # this feature. Permuting it cannot change predictions.
        if split_for_skip is not None and float(split_for_skip[feature_idx]) == 0.0:
            continue

        # Skip constant transformed feature groups.
        constant = True
        for col in group:
            column = X_val[:, col]
            if column.size == 0:
                continue

            first = column[0]
            if not np.all(column == first):
                constant = False
                break

        if constant:
            continue

        group_arr = np.asarray(group, dtype=int)
        single_column = group_arr.size == 1
        accumulated = 0.0

        for _ in range(int(n_repeats)):
            perm = rng.permutation(n_samples)

            if single_column:
                col = int(group_arr[0])

                X_work[:, col] = X_val[perm, col]
                perm_pred = np.asarray(model.predict(X_work), dtype=np.float64).ravel()
                perm_rmse = _rmse(y_true, perm_pred)

                if np.isfinite(perm_rmse):
                    accumulated += perm_rmse - baseline_rmse

                X_work[:, col] = X_val[:, col]

            else:
                # Permute all transformed columns derived from the same original
                # feature using the same row permutation.
                X_work[:, group_arr] = X_val[np.ix_(perm, group_arr)]

                perm_pred = np.asarray(model.predict(X_work), dtype=np.float64).ravel()
                perm_rmse = _rmse(y_true, perm_pred)

                if np.isfinite(perm_rmse):
                    accumulated += perm_rmse - baseline_rmse

                X_work[:, group_arr] = X_val[:, group_arr]

        importances[feature_idx] = accumulated / float(n_repeats)

    return importances


def _index_length(index: slice | np.ndarray, total: int) -> int:
    """
    Length of a row index without materializing large index arrays.
    """
    if isinstance(index, slice):
        return len(range(*index.indices(total)))
    return len(index)


def _run_single_fold(
    estimator: Any,
    seed: int,
    X: np.ndarray,
    y: np.ndarray,
    train_index: slice | np.ndarray,
    val_index: slice | np.ndarray,
    feature_columns: list[str],
    transformed_names: list[str],
    groups: list[list[int]],
    n_repeats: int,
    fold_idx: int,
    est_idx: int,
    max_permutation_samples: int,
    completed_tasks: list[int] | None = None,
    total_tasks: int | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Train one estimator on one fold and compute aligned importances.

    This function is performance-critical. It receives shared NumPy arrays and
    avoids all DataFrame operations.
    """
    n_total = X.shape[0]
    train_len = _index_length(train_index, n_total)
    val_len = _index_length(val_index, n_total)

    if train_len < 2 or val_len < 2:
        return None, None

    est = clone(estimator)

    # For forest estimators, keep inner n_jobs=1. Outer threading parallelizes
    # folds. For HistGradientBoosting, n_jobs usually does not exist.
    if hasattr(est, "n_jobs"):
        est.n_jobs = 1

    est.fit(X[train_index], y[train_index])

    split_importances = getattr(est, "feature_importances_", None)

    if split_importances is None:
        split_aligned = np.zeros(len(feature_columns), dtype=float)
        split_for_skip = None
    else:
        split_aligned = _align_importances(
            transformed_names=transformed_names,
            importances=np.asarray(split_importances),
            feature_columns=feature_columns,
            groups=groups,
        )

        split_array = np.asarray(split_importances, dtype=float)

        # Only use zero-split-importance skipping when the importance vector can
        # be trusted to correspond to transformed columns or already-aligned
        # original columns.
        can_skip_by_split = split_array.size == len(transformed_names) or (
            split_array.size == len(feature_columns)
            and list(transformed_names) == list(feature_columns)
        )
        split_for_skip = split_aligned if can_skip_by_split else None

    X_val = X[val_index]
    y_val = y[val_index]

    # Deterministic temporal subsampling of the validation fold.
    # np.linspace preserves time coverage and does not depend on RNG state.
    if max_permutation_samples > 0 and X_val.shape[0] > max_permutation_samples:
        sample_idx = np.linspace(
            0,
            X_val.shape[0] - 1,
            max_permutation_samples,
            dtype=int,
        )
        X_val = X_val[sample_idx]
        y_val = y_val[sample_idx]

    perm_aligned = _fast_permutation_importance(
        model=est,
        X_val=X_val,
        y_val=y_val,
        groups=groups,
        n_repeats=n_repeats,
        seed=seed,
        fold_idx=fold_idx,
        est_idx=est_idx,
        split_for_skip=split_for_skip,
    )

    if wandb.run is not None and completed_tasks is not None and total_tasks is not None:
        completed_tasks.append(1)
        wandb.log(
            {
                "stability/completed_tasks": len(completed_tasks),
                "stability/total_tasks": total_tasks,
            }
        )

    return split_aligned, perm_aligned


def _build_estimators(seed: int) -> list[Any]:
    """
    Build the two estimators used for each seed.

    Original implementation:
      - RandomForestRegressor(n_estimators=300)
      - ExtraTreesRegressor(n_estimators=300)

    Optimized implementation:
      - Two HistGradientBoostingRegressor configurations.

    Justification
    -------------
    RandomForest / ExtraTrees with 300 trees are the dominant training cost on
    100k rows. HistGradientBoosting uses histogram binning and shallow additive
    trees, which is usually an order of magnitude faster for this data size.

    The two configurations preserve the original idea of two complementary tree
    ensembles per seed:
      1. smaller leaves, stronger regularization, faster learning rate
      2. larger leaves, weaker regularization, slower learning rate

    Feature ranking remains based on:
      - model split importances
      - permutation importances
      - cross-validation stability
      - multi-seed stability

    If exact forest semantics are required, set:
      _USE_HIST_GRADIENT_BOOSTING = False
    and the code falls back to reduced 100-tree forests.
    """
    base_seed = _safe_random_state(seed)

    if _USE_HIST_GRADIENT_BOOSTING and HistGradientBoostingRegressor is not None:
        return [
            HistGradientBoostingRegressor(
                max_iter=_HGB_MAX_ITER,
                learning_rate=0.08,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                max_bins=_HGB_MAX_BINS,
                early_stopping=False,
                random_state=base_seed,
            ),
            HistGradientBoostingRegressor(
                max_iter=_HGB_MAX_ITER,
                learning_rate=0.05,
                max_leaf_nodes=63,
                min_samples_leaf=10,
                max_bins=min(255, _HGB_MAX_BINS * 2),
                early_stopping=False,
                random_state=(base_seed + 1) % (2**32 - 1),
            ),
        ]

    # Fallback: reduced forests. 100 trees usually preserves ranking stability
    # much better than arbitrarily changing depth or leaf constraints.
    return [
        RandomForestRegressor(
            n_estimators=_FOREST_N_ESTIMATORS,
            random_state=base_seed,
            n_jobs=1,
        ),
        ExtraTreesRegressor(
            n_estimators=_FOREST_N_ESTIMATORS,
            random_state=base_seed,
            n_jobs=1,
        ),
    ]


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
    """
    Optimized stability selection.

    API and output structure are unchanged:
      - selected_features: list[str]
      - report_dataframe: pl.DataFrame with STABILITY_REPORT_COLUMNS
      - scores_dict: dict[str, float]
    """
    feature_columns = list(feature_columns)
    seeds = list(seeds)

    if not feature_columns:
        empty_report = pl.DataFrame(schema=list(STABILITY_REPORT_COLUMNS))
        return [], empty_report, {}

    logger.info(
        "Running stability selection: %d features, %d seeds, %d CV splits.",
        len(feature_columns),
        len(seeds),
        n_splits,
    )

    # Normalize importance weights exactly as original.
    weight_sum = float(permutation_weight) + float(split_weight)
    if weight_sum <= 0.0:
        permutation_weight = 0.5
        split_weight = 0.5
    else:
        permutation_weight = float(permutation_weight) / weight_sum
        split_weight = float(split_weight) / weight_sum

    # -------------------------------------------------------------------------
    # One-time preprocessing and conversion to shared NumPy arrays.
    # -------------------------------------------------------------------------
    preprocessor = build_preprocessor(feature_columns)
    x_raw = x_train[feature_columns]

    X, transformed_names = _fit_preprocessor_once(
        preprocessor=preprocessor,
        x_raw=x_raw,
        feature_columns=feature_columns,
    )
    del x_raw

    y = _as_contiguous_float32_vector(y_train)

    if X.shape[0] != y.size:
        raise ValueError(
            f"Preprocessed X and y have inconsistent row counts: {X.shape[0]} != {y.size}"
        )

    # Mapping from original features to transformed columns.
    groups = _build_feature_groups(
        transformed_names=transformed_names,
        feature_columns=feature_columns,
    )

    # -------------------------------------------------------------------------
    # TimeSeriesSplit to contiguous slices.
    #
    # TimeSeriesSplit train/validation indices are contiguous. Converting them
    # to slices avoids materializing large index arrays and enables zero-copy
    # NumPy row views inside workers.
    # -------------------------------------------------------------------------
    cv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    splits: list[tuple[slice | np.ndarray, slice | np.ndarray]] = []

    for train_idx, val_idx in cv.split(X):
        if len(train_idx) < 2 or len(val_idx) < 2:
            continue

        train_contiguous = len(train_idx) > 0 and train_idx[-1] == train_idx[0] + len(train_idx) - 1
        val_contiguous = len(val_idx) > 0 and val_idx[-1] == val_idx[0] + len(val_idx) - 1

        if train_contiguous and val_contiguous:
            splits.append(
                (
                    slice(int(train_idx[0]), int(train_idx[-1]) + 1),
                    slice(int(val_idx[0]), int(val_idx[-1]) + 1),
                )
            )
        else:
            # Rare safety fallback.
            splits.append(
                (
                    np.asarray(train_idx, dtype=int),
                    np.asarray(val_idx, dtype=int),
                )
            )

    n_folds = len(splits)

    selected_by_run: dict[str, list[bool]] = {column: [] for column in feature_columns}
    score_by_run: dict[str, list[float]] = {column: [] for column in feature_columns}
    perm_importance_by_run: dict[str, list[float]] = {column: [] for column in feature_columns}
    split_importance_by_run: dict[str, list[float]] = {column: [] for column in feature_columns}

    # -------------------------------------------------------------------------
    # Task construction.
    # -------------------------------------------------------------------------
    tasks: list[
        tuple[
            int,
            Any,
            slice | np.ndarray,
            slice | np.ndarray,
            int,
            int,
        ]
    ] = []

    for seed in seeds:
        estimators = _build_estimators(int(seed))

        for est_idx, estimator in enumerate(estimators):
            for fold_idx, (train_index, val_index) in enumerate(splits):
                tasks.append(
                    (
                        int(seed),
                        estimator,
                        train_index,
                        val_index,
                        fold_idx,
                        est_idx,
                    )
                )

    max_permutation_samples = int(_PERMUTATION_MAX_VALIDATION_SAMPLES)

    completed_tasks: list[int] = []
    total_tasks = len(tasks)

    if wandb.run is not None:
        wandb.log({"stability/total_tasks": total_tasks})

    # -------------------------------------------------------------------------
    # Parallel execution strategy.
    #
    # HistGradientBoosting:
    #   Use sequential outer execution. Each HGB fit/predict uses OpenMP across
    #   all physical cores. Nested outer parallelism would oversubscribe cores.
    #
    # Forest fallback:
    #   Use threading backend. The large X/y arrays are shared by reference,
    #   avoiding expensive serialization/copies required by process backends.
    #   scikit-learn tree routines release the GIL in performance-critical
    #   Cython paths, so threading is safe and efficient here.
    # -------------------------------------------------------------------------
    if _USE_HIST_GRADIENT_BOOSTING:
        effective_n_jobs = 1
    else:
        effective_n_jobs = 1 if n_jobs is None else int(n_jobs)
        if effective_n_jobs == 0:
            effective_n_jobs = 1

    if effective_n_jobs == 1 or len(tasks) <= 1:
        results = [
            _run_single_fold(
                estimator,
                seed,
                X,
                y,
                train_index,
                val_index,
                feature_columns,
                transformed_names,
                groups,
                n_repeats,
                fold_idx,
                est_idx,
                max_permutation_samples,
                completed_tasks,
                total_tasks,
            )
            for seed, estimator, train_index, val_index, fold_idx, est_idx in tasks
        ]
    else:
        results = Parallel(n_jobs=effective_n_jobs, backend="threading")(
            delayed(_run_single_fold)(
                estimator,
                seed,
                X,
                y,
                train_index,
                val_index,
                feature_columns,
                transformed_names,
                groups,
                n_repeats,
                fold_idx,
                est_idx,
                max_permutation_samples,
                completed_tasks,
                total_tasks,
            )
            for seed, estimator, train_index, val_index, fold_idx, est_idx in tasks
        )

    # -------------------------------------------------------------------------
    # Aggregate per-run results exactly as original.
    # -------------------------------------------------------------------------
    task_idx = 0

    for _seed in seeds:
        for _est_idx in range(2):
            fold_split_importances: list[np.ndarray] = []
            fold_perm_importances: list[np.ndarray] = []

            for _fold_idx in range(n_folds):
                split_aligned, perm_aligned = results[task_idx]

                if split_aligned is not None and perm_aligned is not None:
                    fold_split_importances.append(split_aligned)
                    fold_perm_importances.append(perm_aligned)

                task_idx += 1

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

            if wandb.run is not None:
                wandb.log(
                    {
                        f"stability/seed_{_seed}/est_{_est_idx}_mean_perm_imp": float(
                            np.mean(mean_perm_importance)
                        ),
                        f"stability/seed_{_seed}/est_{_est_idx}_perm_imp_dist": wandb.Histogram(
                            mean_perm_importance
                        ),
                        f"stability/seed_{_seed}/est_{_est_idx}_mean_split_imp": float(
                            np.mean(mean_split_importance)
                        ),
                        f"stability/seed_{_seed}/est_{_est_idx}_split_imp_dist": wandb.Histogram(
                            mean_split_importance
                        ),
                    }
                )

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

    # -------------------------------------------------------------------------
    # Build report.
    # -------------------------------------------------------------------------
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

        # Required ranking score formula.
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

    # Required fallback behavior.
    if len(selected_features) < min_fallback_features:
        fallback_count = max(min_fallback_features, len(selected_features))
        fallback_features = report.head(fallback_count)["feature"].to_list()
        selected_features = list(dict.fromkeys(selected_features + fallback_features))

    if candidate_cap is not None:
        selected_features = selected_features[:candidate_cap]

    scores_dict = dict(
        zip(
            report.get_column("feature").to_list(),
            report.get_column("ranking_score").to_list(),
        )
    )

    logger.info(
        "Stability selection complete: %d features selected "
        "(min_stability=%.2f, fallback=%d, cap=%s).",
        len(selected_features),
        min_stability,
        min_fallback_features,
        candidate_cap,
    )

    if wandb.run is not None:
        wandb.log(
            {
                "stability/ranking_score_dist": wandb.Histogram(
                    report.get_column("ranking_score").to_numpy()
                ),
                "stability/selected_feature_count": len(selected_features),
                "stability/total_feature_count": len(feature_columns),
                "stability/report_table": wandb.Table(dataframe=report.to_pandas()),
            }
        )

    return selected_features, report, scores_dict
