from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..models.common import clip_predictions

logger = logging.getLogger(__name__)

_EPSILON: float = 1e-12


def _as_arrays(
    y_true_arr: Sequence[float], y_pred_arr: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    y_true_np = np.asarray(y_true_arr, dtype=np.float64)
    y_pred_np = np.asarray(y_pred_arr, dtype=np.float64)

    if y_true_np.shape != y_pred_np.shape:
        raise ValueError(
            "y_true_arr and y_pred_arr must have the same shape."
            f"Got {y_true_np.shape} and {y_pred_np.shape}."
        )

    if y_true_np.size == 0:
        return y_true_np, y_pred_np

    return y_true_np, y_pred_np


def rmse(y_true_arr: np.ndarray, y_pred_arr: np.ndarray) -> float:
    if y_true_arr.size == 0:
        return 0.0

    return float(np.sqrt(np.mean(np.square(y_pred_arr - y_true_arr))))


def nse(y_true_arr: np.ndarray, y_pred_arr: np.ndarray) -> float:
    if y_true_arr.size == 0:
        return 0.0

    numerator = float(np.sum(np.square(y_true_arr - y_pred_arr)))
    denominator = float(np.sum(np.square(y_true_arr - np.mean(y_true_arr))))

    if denominator <= _EPSILON:
        return 0.0

    return float(1.0 - (numerator / denominator))


def mae(y_true_arr: np.ndarray, y_pred_arr: np.ndarray) -> float:
    if y_true_arr.size == 0:
        return 0.0

    return float(np.mean(np.abs(y_true_arr - y_pred_arr)))


def rmsle(y_true_arr: np.ndarray, y_pred_arr: np.ndarray) -> float:
    if y_true_arr.size == 0:
        return 0.0

    y_true_arr_safe = np.clip(y_true_arr, 0.0, 100.0)
    y_pred_arr_safe = np.clip(y_pred_arr, 0.0, 100.0)

    return float(np.sqrt(np.mean(np.square(np.log1p(y_true_arr_safe) - np.log1p(y_pred_arr_safe)))))


def r2(y_true_arr: np.ndarray, y_pred_arr: np.ndarray) -> float:
    if y_true_arr.size == 0:
        return 0.0

    ss_res = float(np.sum(np.square(y_true_arr - y_pred_arr)))
    ss_tot = float(np.sum(np.square(y_true_arr - np.mean(y_true_arr))))

    if ss_tot <= _EPSILON:
        return 0.0

    return float(1.0 - ss_res / ss_tot)


def explained_variance(y_true_arr: np.ndarray, y_pred_arr: np.ndarray) -> float:
    if y_true_arr.size == 0:
        return 0.0

    residual = y_true_arr - y_pred_arr
    variance_true = float(np.var(y_true_arr))
    variance_residual = float(np.var(residual))

    if variance_true <= _EPSILON:
        return 0.0

    return float(1.0 - (variance_residual / variance_true))


def max_absolute_error(y_true_arr: np.ndarray, y_pred_arr: np.ndarray) -> float:
    if y_true_arr.size == 0:
        return 0.0

    return float(np.max(np.abs(y_true_arr - y_pred_arr)))


def zone_stratified_metrics(y_true_arr: np.ndarray, y_pred_arr: np.ndarray) -> dict[str, Any]:
    zones = {
        "critical": (0.0, 25.0),
        "poor": (25.0, 50.0),
        "marginal": (50.0, 75.0),
        "good": (70.0, 85.0),
        "excellent": (85.0, 100.0),
    }

    result: dict[str, Any] = {}

    for zone_name, (lower, upper) in zones.items():
        if zone_name == "excellent":
            mask = (y_true_arr >= lower) & (y_true_arr <= upper)
        else:
            mask = (y_true_arr >= lower) & (y_true_arr < upper)

        count = int(np.sum(mask))

        if count == 0:
            zone_mae = 0.0
        else:
            zone_mae = float(np.mean(np.abs(y_true_arr[mask] - y_pred_arr[mask])))

        result[zone_name] = {
            "mae": zone_mae,
            "count": count,
        }

    return result


def zero_event_metrics(y_true_arr: np.ndarray, y_pred_arr: np.ndarray) -> dict[str, Any]:
    mask = np.isclose(y_true_arr, 0.0, rtol=0.0, atol=1e-12)
    count = int(np.sum(mask))

    if count == 0:
        return {
            "zero_event_mae": 0.0,
            "zero_event_count": 0,
        }

    return {
        "zero_event_mae": float(np.mean(np.abs(y_true_arr[mask] - y_pred_arr[mask]))),
        "zero_event_count": count,
    }


def cv_stability_metrics(fold_rmse: Sequence[float] | None) -> dict[str, float]:
    if not fold_rmse:
        return {
            "cv_rmse_mean": 0.0,
            "cv_rmse_std": 0.0,
            "cv_rmse_coefficient_of_variation": 0.0,
        }

    values = np.asarray(fold_rmse, dtype=np.float64)
    mean_value = float(np.mean(values))
    std_value = float(np.std(values, ddof=0))

    if mean_value <= _EPSILON:
        coefficient = 0.0
    else:
        coefficient = float(std_value / mean_value)

    return {
        "cv_rmse_mean": mean_value,
        "cv_rmse_std": std_value,
        "cv_rmse_coefficient_of_variation": coefficient,
    }


def compute_all_metrics(
    y_true_arr: Sequence[float] | None = None,
    y_pred_arr: Sequence[float] | None = None,
    fold_rmse: Sequence[float] | None = None,
    *,
    y_true: Sequence[float] | None = None,
    y_pred: Sequence[float] | None = None,
) -> dict[str, Any]:
    actual_true = y_true if y_true is not None else y_true_arr
    actual_pred = y_pred if y_pred is not None else y_pred_arr

    if actual_true is None or actual_pred is None:
        raise ValueError("Both target and prediction sequences must be provided.")

    y_true_arr, y_pred_arr_raw = _as_arrays(actual_true, actual_pred)
    y_pred_arr = clip_predictions(y_pred_arr_raw)

    metrics: dict[str, Any] = {
        "rmse": rmse(y_true_arr, y_pred_arr),
        "nse": nse(y_true_arr, y_pred_arr),
        "mae": mae(y_true_arr, y_pred_arr),
        "rmsle": rmsle(y_true_arr, y_pred_arr),
        "r2": r2(y_true_arr, y_pred_arr),
        "explained_variance": explained_variance(y_true_arr, y_pred_arr),
        "max_absolute_error": max_absolute_error(y_true_arr, y_pred_arr),
    }

    zone_metrics = zone_stratified_metrics(y_true_arr, y_pred_arr)

    metrics["zone_metrics"] = zone_metrics
    metrics["critical_zone_mae"] = zone_metrics["critical"]["mae"]
    metrics["critical_zone_count"] = zone_metrics["critical"]["count"]
    metrics["poor_zone_mae"] = zone_metrics["poor"]["mae"]
    metrics["poor_zone_count"] = zone_metrics["poor"]["count"]
    metrics["marginal_zone_mae"] = zone_metrics["marginal"]["mae"]
    metrics["marginal_zone_count"] = zone_metrics["marginal"]["count"]
    metrics["good_zone_mae"] = zone_metrics["good"]["mae"]
    metrics["good_zone_count"] = zone_metrics["good"]["count"]
    metrics["excellent_zone_mae"] = zone_metrics["excellent"]["mae"]
    metrics["excellent_zone_count"] = zone_metrics["excellent"]["count"]

    metrics.update(zero_event_metrics(y_true_arr, y_pred_arr))
    metrics.update(cv_stability_metrics(fold_rmse))

    return metrics
