from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.optimize import minimize

from .types import (
    EnsembleStrategy,
    WeightOptimizationError,
)

WEIGHT_TOLERANCE: float = 1e-8
SUM_TOLERANCE: float = 1e-6


def fit_constrained_weights(
    *,
    z_oof: np.ndarray,
    y_oof: np.ndarray,
    columns: Sequence[str],
    ftol: float = 1e-12,
    maxiter: int = 1000,
) -> dict[str, Any]:
    z_arr = np.asarray(z_oof, dtype=np.float64)
    y_arr = np.asarray(y_oof, dtype=np.float64).ravel()

    if z_arr.ndim != 2 or z_arr.shape[0] != y_arr.shape[0]:
        raise WeightOptimizationError(
            f"Dimension mismatch: Z shape {z_arr.shape}, y shape {y_arr.shape}"
        )

    n_samples, n_models = z_arr.shape
    if n_models == 0 or n_samples == 0:
        raise WeightOptimizationError("Cannot fit weights on empty input arrays.")

    # Remove zero-variance or constant duplicate columns to prevent singular Hessians
    col_stds = np.std(z_arr, axis=0)
    valid_col_mask = col_stds > 1e-12
    if not np.any(valid_col_mask):
        # Fallback to equal weighting if all columns are degenerate
        equal_w = {col: 1.0 / n_models for col in columns}
        return {
            "strategy": EnsembleStrategy.CONSTRAINED_WEIGHTED.value,
            "weights": equal_w,
            "active_models_count": n_models,
            "objective_value": 0.0,
            "converged": False,
            "fallback": True,
        }

    # Initial equal weights
    w0 = np.full(n_models, 1.0 / n_models, dtype=np.float64)

    # Objective function: Mean Squared Error (or RSS)
    def objective(w: np.ndarray) -> float:
        pred = np.dot(z_arr, w)
        diff = pred - y_arr
        return float(np.mean(np.square(diff)))

    # Gradient of MSE: (2/N) * Z.T * (Z*w - y)
    def gradient(w: np.ndarray) -> np.ndarray:
        pred = np.dot(z_arr, w)
        diff = pred - y_arr
        return (2.0 / n_samples) * np.dot(z_arr.T, diff)

    # Constraints: sum(w) == 1.0
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    # Bounds: w_i in [0.0, 1.0]
    bounds = [(0.0, 1.0) for _ in range(n_models)]

    res = minimize(
        fun=objective,
        x0=w0,
        jac=gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": ftol, "maxiter": maxiter, "disp": False},
    )

    raw_weights = res.x
    # Post-process weights: enforce non-negativity and exact sum to 1.0
    cleaned_weights = np.maximum(0.0, raw_weights)
    sum_w = np.sum(cleaned_weights)
    if sum_w <= 0.0:
        raise WeightOptimizationError("Sum of optimized weights is non-positive.")
    normalized_weights = cleaned_weights / sum_w

    # Invariant assertions
    if np.any(normalized_weights < -WEIGHT_TOLERANCE):
        raise WeightOptimizationError(
            "Negative weight detected violating non-negativity invariant."
        )
    if abs(np.sum(normalized_weights) - 1.0) > SUM_TOLERANCE:
        raise WeightOptimizationError("Weights do not sum to 1.0 within tolerance.")

    weight_map = {col: float(normalized_weights[i]) for i, col in enumerate(columns)}

    return {
        "strategy": EnsembleStrategy.CONSTRAINED_WEIGHTED.value,
        "weights": weight_map,
        "active_models_count": int(np.sum(normalized_weights > 1e-4)),
        "objective_value": float(res.fun),
        "converged": bool(res.success),
        "message": str(res.message),
    }


def predict_constrained_weights(
    z_matrix: np.ndarray,
    weights_map: Mapping[str, float],
    columns: Sequence[str],
) -> np.ndarray:
    w_vec = np.array([weights_map.get(col, 0.0) for col in columns], dtype=np.float64)
    return np.dot(z_matrix, w_vec)
