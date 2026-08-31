from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.model_selection import TimeSeriesSplit

from .model_builder import instantiate_model
from .types import (
    Hemisphere,
    Level0ModelSpec,
    OOFGenerationError,
    OOFResult,
    ValidationPredictionResult,
)


def generate_oof_matrix(
    *,
    hemisphere: Hemisphere,
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    target_col: str = "water_quality_index",
    specs: Sequence[Level0ModelSpec],
    selected_features: Sequence[str],
    n_splits: int = 5,
    output_dir: Path,
) -> tuple[OOFResult, ValidationPredictionResult]:
    n_train = train_df.height
    n_val = val_df.height
    n_models = len(specs)

    if n_train == 0 or n_val == 0 or n_models == 0:
        raise OOFGenerationError("Input dataframes and model specs must be non-empty.")

    x_train_np = train_df.select(selected_features).to_numpy().astype(np.float64, order="C")
    y_train_np = train_df[target_col].to_numpy().astype(np.float64)

    x_val_np = val_df.select(selected_features).to_numpy().astype(np.float64, order="C")

    oof_matrix = np.full((n_train, n_models), fill_value=np.nan, dtype=np.float64, order="C")
    val_preds_matrix = np.zeros((n_val, n_models), dtype=np.float64, order="C")
    fold_id_arr = np.full(n_train, fill_value=-1, dtype=np.int32)
    valid_oof_mask = np.zeros(n_train, dtype=bool)

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=0)
    splits = list(tscv.split(x_train_np))

    # Identify initial burn-in window (indices prior to fold 0 test split)
    first_test_start = splits[0][1][0]
    burn_in_indices = np.arange(0, first_test_start, dtype=np.int64)

    model_columns = tuple(spec.model_id for spec in specs)

    # 1. Generate OOF Predictions via Temporal CV
    for fold_idx, (tr_idx, oof_idx) in enumerate(splits):
        fold_id_arr[oof_idx] = fold_idx
        valid_oof_mask[oof_idx] = True

        x_tr_fold = x_train_np[tr_idx]
        y_tr_fold = y_train_np[tr_idx]
        x_oof_fold = x_train_np[oof_idx]

        for m_idx, spec in enumerate(specs):
            model = instantiate_model(spec)
            model.fit(x_tr_fold, y_tr_fold)
            preds = np.asarray(model.predict(x_oof_fold), dtype=np.float64).ravel()

            if not np.all(np.isfinite(preds)):
                raise OOFGenerationError(
                    f"Non-finite OOF prediction encountered in model {spec.model_id}, fold {fold_idx}"
                )
            oof_matrix[oof_idx, m_idx] = preds

    # 2. Fit Level-0 Models on Full Train (0..1094) and Predict on Validation (1095..1459)
    for m_idx, spec in enumerate(specs):
        model_full = instantiate_model(spec)
        model_full.fit(x_train_np, y_train_np)
        val_preds = np.asarray(model_full.predict(x_val_np), dtype=np.float64).ravel()

        if not np.all(np.isfinite(val_preds)):
            raise OOFGenerationError(
                f"Non-finite Validation prediction encountered in model {spec.model_id}"
            )
        val_preds_matrix[:, m_idx] = val_preds

    # Ensure output directories exist
    output_dir.mkdir(parents=True, exist_ok=True)

    oof_result = OOFResult(
        oof_matrix=oof_matrix,
        columns=model_columns,
        train_row_index=np.arange(n_train, dtype=np.int64),
        fold_id=fold_id_arr,
        valid_oof_mask=valid_oof_mask,
        burn_in_indices=burn_in_indices,
        metadata={
            "hemisphere": hemisphere.value,
            "n_train_rows": n_train,
            "n_models": n_models,
            "n_splits": n_splits,
            "burn_in_count": len(burn_in_indices),
            "valid_oof_count": int(np.sum(valid_oof_mask)),
        },
    )

    val_result = ValidationPredictionResult(
        columns=model_columns,
        predictions=val_preds_matrix,
        row_index=np.arange(n_train, n_train + n_val, dtype=np.int64),
        metadata={
            "hemisphere": hemisphere.value,
            "n_val_rows": n_val,
            "n_models": n_models,
        },
    )

    return oof_result, val_result
