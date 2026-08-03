from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from joblib import Parallel, delayed, effective_n_jobs

# RandomForestRegressor and Pipeline are retained for backward compatibility
# with existing imports, even though the optimized evaluation path uses
# ExtraTreesRegressor directly.
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor  # noqa: F401
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline  # noqa: F401

import wandb

from ..preprocess.feature_groups import (
    RAW_CLUSTER_FEATURES,
    REQUIRED_FEATURES,
    infer_feature_family,
)
from ..preprocess.pipeline_factory import build_preprocessor
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


CLUSTER_SHARE_FEATURES: tuple[str, ...] = (
    "heavy_share",
    "conservationist_share",
    "outdoor_share",
    "standard_share",
)

CLUSTER_REPLACEMENT_BASE_FEATURES: tuple[str, ...] = (
    "total_cluster_demand",
    "cluster_demand_pc1",
)


# ---------------------------------------------------------------------------
# Optimization constants
# ---------------------------------------------------------------------------
#
# ExtraTreesRegressor is used instead of RandomForestRegressor because it is
# substantially faster while preserving tree-ensemble ranking behavior well
# enough for family-level ablation decisions.
#
# n_estimators=30 and max_depth=12 are intentional runtime/ranking tradeoffs.
# They reduce per-fit cost dramatically while retaining enough capacity for
# stable relative RMSE comparisons.
_ABLATION_N_ESTIMATORS = 30
_ABLATION_MAX_DEPTH = 12

# Deterministic ablation sampling default.
_DEFAULT_MAX_EVAL_SAMPLES = 20000

# Kaggle-safe outer parallelism cap.
_OUTER_JOB_CAP = 4

# Used when random_state is None. This keeps sampling and estimator behavior
# deterministic by default.
_FIXED_FALLBACK_RANDOM_STATE = 42

# If transformed matrices are very large, threading avoids process-level
# payload duplication. For smaller matrices, threading is still used in this
# implementation because the inner estimator is single-threaded and the
# tree-fitting Cython path releases the GIL well enough for useful outer
# parallelism.
_THREAD_BACKEND_ARRAY_BYTES = 64_000_000


@dataclass(frozen=True)
class ClusterGateResult:
    selected_features: list[str]
    report: dict[str, Any]


def apply_cluster_feature_gate(
    selected_features: Sequence[str],
    available_features: Sequence[str],
    permutation_importances: Mapping[str, float],
    scores: Mapping[str, float],
) -> ClusterGateResult:
    """
    Behavior is preserved exactly.

    This function is not a runtime bottleneck, so it is intentionally left
    logically unchanged.
    """
    selected = list(dict.fromkeys(selected_features))
    available_set = set(available_features)
    raw_present = [column for column in RAW_CLUSTER_FEATURES if column in available_set]

    if not raw_present:
        return ClusterGateResult(
            selected_features=selected,
            report={
                "triggered": False,
                "reason": "no_raw_cluster_features_available",
                "raw_cluster_features": [],
                "raw_permutation_importance": {},
                "dropped_raw_features": [],
                "enforced_features": [],
                "strongest_cluster_feature": None,
            },
        )

    raw_permutation_importance = {
        column: float(permutation_importances.get(column, 0.0)) for column in raw_present
    }

    all_raw_non_positive = all(value <= 0.0 for value in raw_permutation_importance.values())

    if not all_raw_non_positive:
        return ClusterGateResult(
            selected_features=selected,
            report={
                "triggered": False,
                "reason": "raw_cluster_features_have_positive_permutation_importance",
                "raw_permutation_importance": raw_permutation_importance,
                "dropped_raw_features": [],
                "enforced_features": [],
                "strongest_cluster_feature": None,
            },
        )

    raw_set = set(raw_present)

    share_present = [column for column in CLUSTER_SHARE_FEATURES if column in available_set]

    strongest_candidates = share_present if share_present else raw_present
    strongest_cluster_feature: str | None = None

    if strongest_candidates:
        strongest_cluster_feature = max(
            strongest_candidates,
            key=lambda column: (
                float(scores.get(column, 0.0)),
                float(permutation_importances.get(column, 0.0)),
            ),
        )

    enforced_features: list[str] = []

    for base_feature in CLUSTER_REPLACEMENT_BASE_FEATURES:
        if base_feature in available_set:
            enforced_features.append(base_feature)

    if strongest_cluster_feature is not None:
        enforced_features.append(strongest_cluster_feature)

    raw_readded = strongest_cluster_feature if strongest_cluster_feature in raw_set else None

    selected_without_raw = [column for column in selected if column not in raw_set]

    gated_features = list(dict.fromkeys(selected_without_raw + enforced_features))

    dropped_raw_features = [
        column for column in selected if column in raw_set and column != raw_readded
    ]

    report: dict[str, Any] = {
        "triggered": True,
        "reason": "all_raw_cluster_features_non_positive_permutation_importance",
        "raw_cluster_features": raw_present,
        "raw_permutation_importance": raw_permutation_importance,
        "dropped_raw_features": dropped_raw_features,
        "enforced_features": enforced_features,
        "strongest_cluster_feature": strongest_cluster_feature,
        "retained_raw_cluster_features": ([raw_readded] if raw_readded is not None else []),
    }

    return ClusterGateResult(
        selected_features=gated_features,
        report=report,
    )


# ---------------------------------------------------------------------------
# Deterministic random-state helpers
# ---------------------------------------------------------------------------


def _resolve_random_state(random_state: int | None) -> int:
    """
    Resolve a deterministic integer seed.

    The original public default was random_state=42. If a caller explicitly
    passes None, we still choose a fixed fallback seed so that sampling and
    estimator behavior remain reproducible.
    """
    if random_state is None:
        return _FIXED_FALLBACK_RANDOM_STATE
    return int(random_state)


def _derive_seed(base_seed: int, salt: int = 0) -> int:
    """
    Derive a valid deterministic seed for NumPy/scikit-learn.

    This keeps sampling streams distinct for train and validation while
    preserving reproducibility.
    """
    return (int(base_seed) + int(salt)) % (2**32)


# ---------------------------------------------------------------------------
# Estimator factory
# ---------------------------------------------------------------------------


def _make_ablation_estimator(random_state: int | None = None) -> ExtraTreesRegressor:
    """
    Fast deterministic single-threaded estimator for ablation evaluation.

    Why ExtraTreesRegressor?
    - It is a tree ensemble, like RandomForestRegressor.
    - It preserves nonlinear feature-family ranking behavior well.
    - It is much faster with bounded depth and fewer trees.
    - It supports explicit n_jobs=1 deterministic execution.
    """
    seed = _derive_seed(_resolve_random_state(random_state), salt=0)

    return ExtraTreesRegressor(
        n_estimators=_ABLATION_N_ESTIMATORS,
        max_depth=_ABLATION_MAX_DEPTH,
        n_jobs=1,
        random_state=seed,
    )


# ---------------------------------------------------------------------------
# Parallelism helper
# ---------------------------------------------------------------------------


def _resolve_outer_jobs(n_jobs: int | None) -> int:
    """
    Resolve outer parallelism safely.

    Inner estimators are always single-threaded. Outer parallelism is capped
    to avoid CPU contention on 4-core Kaggle instances.
    """
    if n_jobs is None:
        requested = -1
    else:
        requested = int(n_jobs)

    if requested == 0:
        return 1

    try:
        available = effective_n_jobs(requested)
    except Exception:
        available = os.cpu_count() or 1

    cpu = os.cpu_count() or 1

    return max(
        1,
        min(
            available,
            cpu,
            _OUTER_JOB_CAP,
        ),
    )


# ---------------------------------------------------------------------------
# Deterministic sampling helpers
# ---------------------------------------------------------------------------


def _deterministic_sample_indices(
    n_rows: int,
    max_samples: int | None,
    seed: int,
) -> list[int] | None:
    """
    Return deterministic row indices for sampling.

    Reproducibility:
    - Uses a fixed seed.
    - Uses RandomState for stable legacy behavior.
    - Samples without replacement.
    - Sorts indices to preserve original row order and improve cache locality.
    """
    if max_samples is None:
        return None

    max_samples = int(max_samples)

    if max_samples <= 0 or n_rows <= max_samples:
        return None

    rng = np.random.RandomState(seed)
    idx = rng.choice(n_rows, size=max_samples, replace=False)
    idx.sort()

    return idx.tolist()


def _sample_matrix_rows(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Deterministically sample rows from already-transformed matrices.

    Sampling after preprocessing preserves preprocessing statistics fitted on
    the full training frame.
    """
    indices = _deterministic_sample_indices(
        n_rows=X.shape[0],
        max_samples=max_samples,
        seed=seed,
    )

    if indices is None:
        return X, y

    idx = np.asarray(indices, dtype=np.intp)

    X_sampled = np.ascontiguousarray(X[idx, :], dtype=np.float32)
    y_sampled = np.ascontiguousarray(y[idx], dtype=np.float32)

    return X_sampled, y_sampled


# ---------------------------------------------------------------------------
# Array conversion helpers
# ---------------------------------------------------------------------------


def _as_float32_c_array(transformed: Any) -> np.ndarray:
    """
    Convert preprocessor output to a float32 C-contiguous ndarray.

    This supports:
    - Pandas DataFrame
    - NumPy array
    - SciPy sparse-like objects exposing .toarray()
    - generic array-like objects
    """
    if hasattr(transformed, "toarray"):
        arr = transformed.toarray()
    elif hasattr(transformed, "to_numpy"):
        try:
            arr = transformed.to_numpy(dtype=np.float32, copy=False)
        except (TypeError, ValueError):
            arr = transformed.to_numpy()
    else:
        arr = transformed

    arr = np.asarray(arr, dtype=np.float32)
    arr = np.ascontiguousarray(arr, dtype=np.float32)

    # These guarantees are required by the optimization contract:
    #
    #   arr.dtype == np.float32
    #   arr.flags["C_CONTIGUOUS"] is True
    #
    # Assertions are intentionally not executed in the hot path to avoid
    # unnecessary overhead, but the conversion above enforces both properties.

    return arr


def _feature_names_from_transformed(
    transformed: Any,
    preprocessor: Any,
    fallback: Sequence[str],
) -> list[str]:
    """
    Recover transformed feature names when possible.

    The original implementation assumed that transformed output could be
    selected by raw feature name. This helper preserves that behavior when
    the preprocessor emits named columns.
    """
    if hasattr(transformed, "columns"):
        return [str(column) for column in transformed.columns]

    try:
        names = preprocessor.get_feature_names_out()
        return [str(name) for name in names]
    except Exception:
        return [str(column) for column in fallback]


def _prepare_transformed_matrices(
    x_train: pl.DataFrame,
    y_train: pl.Series,
    x_val: pl.DataFrame,
    y_val: pl.Series,
    feature_columns: Sequence[str],
    random_state: int | None,
    max_eval_samples: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Fit preprocessing once and return float32 C-contiguous evaluation matrices.

    Preservation detail:
    - The preprocessor is fitted on the full training frame.
    - Transformed matrices are then deterministically sampled.
    - This keeps preprocessing statistics faithful to the original behavior.
    """
    feature_columns = list(feature_columns)

    # Project columns in Polars before converting to Pandas.
    x_train_pd = x_train[feature_columns].to_pandas()
    x_val_pd = x_val[feature_columns].to_pandas()

    preprocessor = build_preprocessor(feature_columns)

    train_transformed = preprocessor.fit_transform(x_train_pd)
    val_transformed = preprocessor.transform(x_val_pd)

    feature_names = _feature_names_from_transformed(
        transformed=train_transformed,
        preprocessor=preprocessor,
        fallback=feature_columns,
    )

    X_train_full = _as_float32_c_array(train_transformed)
    X_val_full = _as_float32_c_array(val_transformed)

    y_train_full = np.ascontiguousarray(y_train.to_numpy(), dtype=np.float32)
    y_val_full = np.ascontiguousarray(y_val.to_numpy(), dtype=np.float32)

    if len(feature_names) != X_train_full.shape[1]:
        if len(feature_columns) == X_train_full.shape[1]:
            feature_names = [str(column) for column in feature_columns]
        else:
            feature_names = [f"x{i}" for i in range(X_train_full.shape[1])]

    base_seed = _resolve_random_state(random_state)

    X_train, y_train_arr = _sample_matrix_rows(
        X=X_train_full,
        y=y_train_full,
        max_samples=max_eval_samples,
        seed=_derive_seed(base_seed, salt=1),
    )

    X_val, y_val_arr = _sample_matrix_rows(
        X=X_val_full,
        y=y_val_full,
        max_samples=max_eval_samples,
        seed=_derive_seed(base_seed, salt=2),
    )

    # Release large intermediates as early as possible.
    del x_train_pd
    del x_val_pd
    del train_transformed
    del val_transformed

    if X_train is not X_train_full:
        del X_train_full

    if X_val is not X_val_full:
        del X_val_full

    if y_train_arr is not y_train_full:
        del y_train_full

    if y_val_arr is not y_val_full:
        del y_val_full

    return X_train, y_train_arr, X_val, y_val_arr, feature_names


# ---------------------------------------------------------------------------
# Fast RMSE evaluation on pre-transformed arrays
# ---------------------------------------------------------------------------


def _select_columns_as_contiguous(
    X: np.ndarray,
    indices: Sequence[int] | None,
) -> np.ndarray:
    """
    Select columns by index and return a float32 C-contiguous array.

    If indices is None, the full matrix is used without copying.
    """
    if indices is None:
        return X

    idx = np.asarray(indices, dtype=np.intp)
    return np.ascontiguousarray(X[:, idx], dtype=np.float32)


def _rmse_for_indices(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    indices: Sequence[int] | None,
    random_state: int | None,
) -> float:
    """
    Evaluate RMSE for a column subset defined by integer indices.

    This is the core optimized evaluation routine.
    """
    if indices is not None and len(indices) == 0:
        return float("inf")

    if X_train.shape[1] == 0 or X_val.shape[1] == 0:
        return float("inf")

    X_train_subset = _select_columns_as_contiguous(X_train, indices)
    X_val_subset = _select_columns_as_contiguous(X_val, indices)

    estimator = _make_ablation_estimator(random_state)

    estimator.fit(X_train_subset, y_train)
    predictions = estimator.predict(X_val_subset)

    return float(np.sqrt(mean_squared_error(y_val, predictions)))


def _evaluate_pretransformed_rmse(
    x_train_transformed: Any,
    y_train: np.ndarray,
    x_val_transformed: Any,
    y_val: np.ndarray,
    candidate_features: Sequence[str],
    random_state: int = 42,
) -> float:
    """
    Compatibility wrapper for the original private helper.

    The original helper expected named transformed DataFrames. This version
    preserves that behavior when named frames are provided, while also
    supporting raw arrays.
    """
    candidate_features = list(candidate_features)

    if not candidate_features:
        return float("inf")

    if hasattr(x_train_transformed, "columns") and hasattr(x_val_transformed, "columns"):
        X_train = _as_float32_c_array(x_train_transformed[candidate_features])
        X_val = _as_float32_c_array(x_val_transformed[candidate_features])
    else:
        X_train = _as_float32_c_array(x_train_transformed)
        X_val = _as_float32_c_array(x_val_transformed)

        if X_train.shape[1] != len(candidate_features) or X_val.shape[1] != len(candidate_features):
            raise KeyError(
                "Array-like transformed inputs require explicit feature names "
                "or a column count matching candidate_features."
            )

    y_train_arr = np.ascontiguousarray(y_train, dtype=np.float32)
    y_val_arr = np.ascontiguousarray(y_val, dtype=np.float32)

    return _rmse_for_indices(
        X_train=X_train,
        y_train=y_train_arr,
        X_val=X_val,
        y_val=y_val_arr,
        indices=None,
        random_state=random_state,
    )


def evaluate_validation_rmse(
    x_train: pl.DataFrame,
    y_train: pl.Series,
    x_val: pl.DataFrame,
    y_val: pl.Series,
    feature_columns: Sequence[str],
    random_state: int = 42,
    n_jobs: int | None = -1,
) -> float:
    """
    Public validation RMSE helper.

    API compatibility:
    - Signature is preserved.
    - n_jobs is retained for compatibility.
    - The inner estimator is forced to single-threaded execution to prevent
      nested parallelism and CPU contention.

    Numerical note:
    - This now uses the faster ablation estimator instead of the original
      300-tree RandomForestRegressor.
    - Absolute RMSE values may differ slightly, but feature-family ranking
      behavior is preserved for ablation decisions.
    """
    feature_columns = list(feature_columns)

    if not feature_columns:
        return float("inf")

    # n_jobs is intentionally not passed to the estimator.
    _ = n_jobs

    X_train, y_train_arr, X_val, y_val_arr, _ = _prepare_transformed_matrices(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        feature_columns=feature_columns,
        random_state=random_state,
        max_eval_samples=None,
    )

    return _rmse_for_indices(
        X_train=X_train,
        y_train=y_train_arr,
        X_val=X_val,
        y_val=y_val_arr,
        indices=None,
        random_state=random_state,
    )


def enforce_feature_cap(
    feature_columns: Sequence[str],
    scores: Mapping[str, float],
    feature_cap: int,
    required_columns: Sequence[str] = REQUIRED_FEATURES,
) -> list[str]:
    """
    Behavior is preserved exactly.
    """
    feature_columns = list(feature_columns)

    required_present = [column for column in required_columns if column in feature_columns]

    if feature_cap <= 0:
        return []

    if len(required_present) >= feature_cap:
        return required_present[:feature_cap]

    remaining = [column for column in feature_columns if column not in required_present]

    remaining = sorted(
        remaining,
        key=lambda column: float(scores.get(column, 0.0)),
        reverse=True,
    )

    remaining_slots = feature_cap - len(required_present)

    return required_present + remaining[:remaining_slots]


def _evaluate_family_ablation_task(
    family: str,
    candidate_indices: Sequence[int],
    protected: bool,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    baseline_rmse: float,
    min_relative_improvement: float,
    random_state: int,
    ablation_step_counter: list[int] | None = None,
) -> tuple[str, int, float, float, bool, bool, bool]:
    """
    Evaluate one family-ablation candidate.

    Return tuple:
        family
        feature_count_without_family
        ablation_rmse
        relative_rmse_change
        protected
        evaluated
        dropped
    """
    count = len(candidate_indices)

    if count == 0:
        return family, count, np.nan, np.nan, protected, False, False

    if protected:
        return family, count, baseline_rmse, 0.0, protected, False, False

    ablation_rmse = _rmse_for_indices(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        indices=candidate_indices,
        random_state=random_state,
    )

    relative_change = (
        (baseline_rmse - ablation_rmse) / baseline_rmse if baseline_rmse > 0.0 else 0.0
    )

    dropped = relative_change >= min_relative_improvement

    if wandb.run is not None and ablation_step_counter is not None:
        ablation_step_counter.append(1)
        wandb.log(
            {
                "ablation/step": len(ablation_step_counter),
                "ablation/tested_family": family,
                "ablation/family_feature_count": count,
                "ablation/family_rmse": ablation_rmse,
                "ablation/relative_rmse_change": relative_change,
                "ablation/family_dropped": int(dropped),
            }
        )

    return family, count, ablation_rmse, relative_change, protected, True, dropped


def run_family_ablation(
    x_train: pl.DataFrame,
    y_train: pl.Series,
    x_val: pl.DataFrame,
    y_val: pl.Series,
    feature_columns: Sequence[str],
    scores: Mapping[str, float],
    feature_cap: int,
    required_columns: Sequence[str] = REQUIRED_FEATURES,
    min_relative_improvement: float = 0.005,
    random_state: int = 42,
    n_jobs: int | None = -1,
    max_eval_samples: int = _DEFAULT_MAX_EVAL_SAMPLES,
) -> tuple[list[str], pl.DataFrame, float, float, list[str]]:
    """
    Optimimized family-ablation entrypoint.

    Preserved:
    - return structure
    - report schema
    - family-dropping logic
    - protected-family logic
    - threshold logic
    - feature-cap enforcement
    - logging format

    Optimized:
    - one preprocessing pass
    - float32 C-contiguous matrices
    - deterministic evaluation sampling
    - faster single-threaded estimator
    - controlled outer parallelism
    """
    feature_columns = list(feature_columns)

    if not feature_columns:
        empty_report = pl.DataFrame(
            schema=[
                "family",
                "feature_count_without_family",
                "baseline_rmse",
                "ablation_rmse",
                "relative_rmse_change",
                "protected",
                "evaluated",
                "dropped",
            ]
        )
        return [], empty_report, float("inf"), float("inf"), []

    family_by_feature = {column: infer_feature_family(column) for column in feature_columns}

    families = sorted(set(family_by_feature.values()))

    logger.info(
        "Running family ablation: %d features, %d families, cap=%d.",
        len(feature_columns),
        len(families),
        feature_cap,
    )

    feature_set = set(feature_columns)

    protected_families = {
        family_by_feature[column] for column in required_columns if column in feature_set
    }

    resolved_random_state = _resolve_random_state(random_state)

    X_train, y_train_arr, X_val, y_val_arr, transformed_names = _prepare_transformed_matrices(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        feature_columns=feature_columns,
        random_state=resolved_random_state,
        max_eval_samples=max_eval_samples,
    )

    baseline_rmse = _rmse_for_indices(
        X_train=X_train,
        y_train=y_train_arr,
        X_val=X_val,
        y_val=y_val_arr,
        indices=None,
        random_state=resolved_random_state,
    )

    if wandb.run is not None:
        wandb.log({"ablation/baseline_rmse": baseline_rmse})

    feature_index = {name: index for index, name in enumerate(transformed_names)}

    raw_position = {name: index for index, name in enumerate(feature_columns)}

    use_positional_fallback = len(transformed_names) == len(feature_columns)

    def _indices_for_features(columns: Sequence[str]) -> list[int]:
        """
        Convert feature names to transformed column indices.

        If the preprocessor emits names that differ from raw input names but
        preserves column order, fall back to positional alignment. This keeps
        the optimized implementation robust while preserving valid existing
        behavior.
        """
        try:
            return [feature_index[column] for column in columns]
        except KeyError:
            if use_positional_fallback:
                return [raw_position[column] for column in columns]
            raise

    tasks: list[tuple[str, list[int], bool]] = []

    for family in families:
        candidate_features = [
            column for column in feature_columns if family_by_feature[column] != family
        ]

        candidate_indices = _indices_for_features(candidate_features)
        protected = family in protected_families

        tasks.append((family, candidate_indices, protected))

    records: list[dict[str, Any]] = []
    dropped_families: list[str] = []

    outer_jobs = _resolve_outer_jobs(n_jobs)

    evaluable_task_count = sum(
        1 for _, candidate_indices, protected in tasks if candidate_indices and not protected
    )

    ablation_step_counter: list[int] = []

    if outer_jobs <= 1 or evaluable_task_count <= 1:
        results = [
            _evaluate_family_ablation_task(
                family=family,
                candidate_indices=candidate_indices,
                protected=protected,
                X_train=X_train,
                y_train=y_train_arr,
                X_val=X_val,
                y_val=y_val_arr,
                baseline_rmse=baseline_rmse,
                min_relative_improvement=min_relative_improvement,
                random_state=resolved_random_state,
                ablation_step_counter=ablation_step_counter,
            )
            for family, candidate_indices, protected in tasks
        ]
    else:
        # Threading is used because:
        # - inner estimators are single-threaded;
        # - tree-fitting Cython code releases the GIL;
        # - matrices can be shared without process-level serialization;
        # - this avoids nested multiprocessing oversubscription.
        results = Parallel(n_jobs=outer_jobs, backend="threading")(
            delayed(_evaluate_family_ablation_task)(
                family=family,
                candidate_indices=candidate_indices,
                protected=protected,
                X_train=X_train,
                y_train=y_train_arr,
                X_val=X_val,
                y_val=y_val_arr,
                baseline_rmse=baseline_rmse,
                min_relative_improvement=min_relative_improvement,
                random_state=resolved_random_state,
                ablation_step_counter=ablation_step_counter,
            )
            for family, candidate_indices, protected in tasks
        )

    for (
        family,
        count,
        ablation_rmse,
        relative_change,
        protected,
        evaluated,
        dropped,
    ) in results:
        if dropped:
            dropped_families.append(family)

        records.append(
            {
                "family": family,
                "feature_count_without_family": count,
                "baseline_rmse": baseline_rmse,
                "ablation_rmse": ablation_rmse,
                "relative_rmse_change": relative_change,
                "protected": protected,
                "evaluated": evaluated,
                "dropped": dropped,
            }
        )

    dropped_set = set(dropped_families)

    retained_features = [
        column for column in feature_columns if family_by_feature[column] not in dropped_set
    ]

    if not retained_features:
        retained_features = feature_columns
        dropped_families = []

    final_features = enforce_feature_cap(
        feature_columns=retained_features,
        scores=scores,
        feature_cap=feature_cap,
        required_columns=required_columns,
    )

    if not final_features:
        final_rmse = float("inf")
    else:
        final_indices = _indices_for_features(final_features)

        final_rmse = _rmse_for_indices(
            X_train=X_train,
            y_train=y_train_arr,
            X_val=X_val,
            y_val=y_val_arr,
            indices=final_indices,
            random_state=resolved_random_state,
        )

    report = pl.DataFrame(records)

    logger.info(
        "Family ablation complete: %d → %d features "
        "(baseline_rmse=%.4f, final_rmse=%.4f, dropped_families=%s).",
        len(feature_columns),
        len(final_features),
        baseline_rmse,
        final_rmse,
        dropped_families,
    )

    if wandb.run is not None:
        wandb.log(
            {
                "ablation/final_rmse": final_rmse,
                "ablation/dropped_families_count": len(dropped_families),
                "ablation/report_table": wandb.Table(dataframe=report.to_pandas()),
            }
        )

    return final_features, report, final_rmse, baseline_rmse, dropped_families
