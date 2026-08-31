from __future__ import annotations

import numpy as np

from .types import PostprocessingConfig


def select_bounds_calibration(
    *,
    y_val: np.ndarray,
    val_preds: np.ndarray,
    train_wqi: np.ndarray,
) -> PostprocessingConfig:
    y_true = np.asarray(y_val, dtype=np.float64).ravel()
    raw_preds = np.asarray(val_preds, dtype=np.float64).ravel()
    max_tr = float(np.max(train_wqi))

    # Candidate 1: Full range [0.0, 100.0]
    preds_full = np.clip(raw_preds, 0.0, 100.0)
    rmse_full = float(np.sqrt(np.mean(np.square(preds_full - y_true))))

    # Candidate 2: Dynamic range [0.0, max_tr + 5.0]
    dyn_max = max_tr + 5.0
    preds_dyn = np.clip(raw_preds, 0.0, dyn_max)
    rmse_dyn = float(np.sqrt(np.mean(np.square(preds_dyn - y_true))))

    if rmse_dyn < rmse_full - 1e-12:
        return PostprocessingConfig(
            clip_min=0.0,
            clip_max=dyn_max,
            bounds_rule="dynamic",
            max_train_wqi=max_tr,
            dynamic_candidate_max=dyn_max,
            selected_by=f"Validation RMSE improvement: {rmse_dyn:.4f} vs {rmse_full:.4f}",
        )
    else:
        return PostprocessingConfig(
            clip_min=0.0,
            clip_max=100.0,
            bounds_rule="full",
            max_train_wqi=max_tr,
            dynamic_candidate_max=dyn_max,
            selected_by="Standard full range bounds selected",
        )


def apply_postprocessing(
    predictions: np.ndarray,
    config: PostprocessingConfig,
) -> np.ndarray:
    arr = np.asarray(predictions, dtype=np.float64)
    return np.clip(arr, config.clip_min, config.clip_max)
