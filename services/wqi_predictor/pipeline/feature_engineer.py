from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import polars as pl

from .. import config
from ..data.ingestion import load_hemisphere
from ..features.engineer import EngineeredDataset, build_engineered_dataset
from ..seeding import env_seed
from ..tracking import ExperimentTracker, TrackingConfig


def _save_parquet(dataframe: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.write_parquet(path)


def _save_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=str,
        )
    )


def _series_summary(series: pl.Series) -> dict[str, float | int]:
    mean_val = series.mean()
    std_val = series.std(ddof=0)
    min_val = series.min()
    max_val = series.max()
    zero_cnt = (series == 0).sum()

    return {
        "mean": float(mean_val) if mean_val is not None else 0.0,
        "std": float(std_val) if std_val is not None else 0.0,
        "min": float(min_val) if min_val is not None else 0.0,
        "max": float(max_val) if max_val is not None else 0.0,
        "zero_count": int(zero_cnt) if zero_cnt is not None else 0,
    }


def _split_engineered_dataset(
    engineered: EngineeredDataset,
) -> dict[str, pl.DataFrame | pl.Series]:
    split_frames: dict[str, pl.DataFrame | pl.Series] = {}

    for split_name, mask in engineered.split_masks.items():
        split_frames[f"x_{split_name}"] = engineered.feature_frame.filter(mask)
        split_frames[f"y_{split_name}"] = engineered.target.filter(mask)

    return split_frames


def _log_feature_engineer_metrics(
    tracker: ExperimentTracker,
    engineered: EngineeredDataset,
    split_frames: dict[str, pl.DataFrame | pl.Series],
) -> None:
    total_nulls = int(engineered.feature_frame.null_count().sum_horizontal().item())

    target_summaries = {}
    for key, frame_or_series in split_frames.items():
        if key.startswith("y_"):
            split_name = key.replace("y_", "")
            series = (
                frame_or_series.to_series()
                if isinstance(frame_or_series, pl.DataFrame)
                else frame_or_series
            )
            target_summaries[split_name] = _series_summary(series)

    metrics = {
        engineered.hemisphere: {
            "feature_count": engineered.metadata["feature_count"],
            "numeric_feature_count": engineered.metadata["numeric_feature_count"],
            "categorical_feature_count": engineered.metadata["categorical_feature_count"],
            "split": engineered.metadata["split_sizes"],
            "target": target_summaries,
            "null_count": total_nulls,
        }
    }

    tracker.log_metrics(metrics)


def run(tracking_enabled: bool = True) -> dict[str, Any]:
    env_seed()

    base_dir = config.ARTIFACT_DIR / "feature-engineer"

    base_dir.mkdir(parents=True, exist_ok=True)
    config.MLFLOW_DIR.mkdir(parents=True, exist_ok=True)

    mlflow_tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        config.MLFLOW_DIR.as_uri(),
    )

    tracking_config = TrackingConfig(
        experiment="temporal-feature-engineering",
        run_name="feature-engineering",
        mlflow_tracking_uri=mlflow_tracking_uri,
        enabled=tracking_enabled,
    )

    summary: dict[str, Any] = {
        "phase": "feature_engineer",
        "random_state": config.RANDOM_STATE,
        "base_dir": str(base_dir),
        "hemispheres": {},
    }

    with ExperimentTracker(tracking_config) as tracker:
        tracker.log_params(
            {
                "phase": "feature_engineer",
                "random_state": config.RANDOM_STATE,
                "data_dir": str(config.DATA_DIR),
                "base_dir": str(base_dir),
                "hemispheres": list(config.HEMISPHERES),
                "cold_start_target": 50.0,
                "seasonal_smoothing_constant": 10.0,
            }
        )

        for hemisphere in config.HEMISPHERES:
            ingested = load_hemisphere(hemisphere)
            engineered = build_engineered_dataset(
                ingested=ingested,
                cold_start_target=50.0,
            )

            split_frames = _split_engineered_dataset(engineered)

            hemi_dir = base_dir / hemisphere
            split_dir = hemi_dir / "splits"
            split_dir.mkdir(parents=True, exist_ok=True)

            # Dynamic & safe parquet exporting for all splits
            for key, frame_or_series in split_frames.items():
                if key.startswith("x_"):
                    split_name = key.replace("x_", "")
                    _save_parquet(frame_or_series, split_dir / f"X_{split_name}.parquet")
                elif key.startswith("y_"):
                    split_name = key.replace("y_", "")
                    # Convert Polars Series safely with target column name
                    target_df = (
                        frame_or_series.alias(config.TARGET_COLUMN).to_frame()
                        if isinstance(frame_or_series, pl.Series)
                        else frame_or_series
                    )
                    _save_parquet(target_df, split_dir / f"y_{split_name}.parquet")

            _save_json(
                engineered.metadata,
                base_dir / "feature_engineer_metadata.json",
            )

            _save_json(
                engineered.metadata["feature_columns"],
                base_dir / "feature_columns.json",
            )

            _log_feature_engineer_metrics(
                tracker=tracker,
                engineered=engineered,
                split_frames=split_frames,
            )

            # Log directory containing artifacts
            tracker.log_artifact(base_dir)

            summary["hemispheres"][hemisphere] = {
                "base_dir": str(base_dir),
                "feature_count": engineered.metadata["feature_count"],
                "split_sizes": engineered.metadata["split_sizes"],
            }

        tracker.log_params({"status": "complete"})

    return summary


if __name__ == "__main__":
    run()
