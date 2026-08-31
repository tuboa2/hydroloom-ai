from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl

from .level0_builder import instantiate_model
from .residual_corrector import ResidualAutoregressiveCorrector
from .types import (
    FinalFrozenState,
    Hemisphere,
)


def execute_production_refit(
    *,
    hemisphere: Hemisphere,
    full_historical_df: pl.DataFrame,  # Days 0..1459 (1460 rows)
    target_col: str = "water_quality_index",
    frozen_state: FinalFrozenState,
    output_dir: Path,
) -> dict[str, Any]:
    """Train all frozen Level-0 model specs and residual corrector on full Train+Val."""
    n_rows = full_historical_df.height
    if n_rows != 1460:
        raise ValueError(f"Production refit expects exactly 1460 rows (Years 0-3). Got {n_rows}.")

    features = list(frozen_state.selected_features)
    x_np = full_historical_df.select(features).to_numpy().astype(np.float64, order="C")
    y_np = full_historical_df[target_col].to_numpy().astype(np.float64)

    refit_models_dir = output_dir / "level0_models"
    refit_models_dir.mkdir(parents=True, exist_ok=True)

    fitted_level0_paths: dict[str, str] = {}
    level0_preds_matrix = np.zeros((n_rows, len(frozen_state.level0_specs)), dtype=np.float64)

    # 1. Refit all Level-0 models
    for idx, spec in enumerate(frozen_state.level0_specs):
        model = instantiate_model(spec)
        model.fit(x_np, y_np)

        model_save_path = refit_models_dir / f"{spec.model_id}.joblib"
        joblib.dump(model, model_save_path)
        fitted_level0_paths[spec.model_id] = str(model_save_path)

        level0_preds_matrix[:, idx] = model.predict(x_np)

    # 2. Refit Residual AR Corrector if enabled
    residual_path = None
    if frozen_state.residual_enabled:
        res_dir = output_dir / "residual"
        res_dir.mkdir(parents=True, exist_ok=True)

        corrector = ResidualAutoregressiveCorrector()
        valid_mask = np.ones(n_rows, dtype=bool)
        # Strategy blend on full history
        w_map = frozen_state.strategy_parameters.get("weights", {})
        base_preds = np.dot(
            level0_preds_matrix,
            np.array(
                [
                    w_map.get(s.model_id, 1.0 / len(frozen_state.level0_specs))
                    for s in frozen_state.level0_specs
                ]
            ),
        )
        corrector.fit(
            y_train=y_np,
            base_oof_preds=base_preds,
            valid_mask=valid_mask,
        )
        residual_save_path = res_dir / "residual_corrector.joblib"
        joblib.dump(corrector, residual_save_path)
        residual_path = str(residual_save_path)

    # 3. Create pipeline bundle metadata
    manifest = {
        "hemisphere": hemisphere.value,
        "refit_sample_count": n_rows,
        "feature_set_hash": frozen_state.feature_set_hash,
        "frozen_state_hash": frozen_state.frozen_state_hash,
        "fitted_level0_models": fitted_level0_paths,
        "residual_corrector_path": residual_path,
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest
