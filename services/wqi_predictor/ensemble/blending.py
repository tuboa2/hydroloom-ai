from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def arithmetic_mean_predictions(z: np.ndarray) -> np.ndarray:
    """Row-wise arithmetic mean: y_hat_i = (1/M) * sum(z_ij)."""
    return np.mean(np.asarray(z, dtype=np.float64), axis=1)


def trimmed_mean_predictions(
    z: np.ndarray,
    *,
    trim_fraction: float = 0.10,
) -> np.ndarray:
    z_arr = np.asarray(z, dtype=np.float64)
    n_samples, m_models = z_arr.shape

    k = int(np.floor(trim_fraction * m_models))
    if k == 0 or m_models < 5:
        # Fallback to arithmetic mean if fewer than 5 models or k=0
        return np.mean(z_arr, axis=1)

    # Sort each row independently along axis 1
    sorted_z = np.sort(z_arr, axis=1)
    trimmed = sorted_z[:, k : m_models - k]
    return np.mean(trimmed, axis=1)


def mean_median_blend_predictions(
    z: np.ndarray,
    *,
    lam: float,
) -> np.ndarray:
    z_arr = np.asarray(z, dtype=np.float64)
    mean_val = np.mean(z_arr, axis=1)
    median_val = np.median(z_arr, axis=1)
    return lam * mean_val + (1.0 - lam) * median_val


def select_mean_median_lambda(
    *,
    z_val: np.ndarray,
    y_val: np.ndarray,
    lambda_grid: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> tuple[float, float]:
    y_true = np.asarray(y_val, dtype=np.float64).ravel()
    best_lam = 0.5
    best_rmse = float("inf")

    for lam in lambda_grid:
        preds = mean_median_blend_predictions(z_val, lam=lam)
        rmse = float(np.sqrt(np.mean(np.square(preds - y_true))))
        if rmse < best_rmse - 1e-12:
            best_rmse = rmse
            best_lam = lam
        elif abs(rmse - best_rmse) <= 1e-12 and lam == 0.5:
            # Preference tie-break rule: select 0.5
            best_lam = 0.5

    return best_lam, best_rmse
