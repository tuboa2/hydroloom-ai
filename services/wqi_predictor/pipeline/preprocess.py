from __future__ import annotations
import json
import os
import math
import polars as pl
from pathlib import Path
from typing import Any
from .. import config
from ..data import drift
from ..data.ingestion import load_hemisphere
from ..data.splitter import split_chronologically
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

def _drift_frames(split) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    train_df = pl.concat([split.x_train, split.y_train.to_frame()], how="horizontal")
    val_df = pl.concat([split.x_val, split.y_val.to_frame()], how="horizontal")
    test_df = pl.concat([split.x_test, split.y_test.to_frame()], how="horizontal")

    return train_df, val_df, test_df

def _extract_column_metric(
    dataframe: pl.DataFrame,
    column: str,
    comparison: str,
    metric: str,
) -> float | None:
    rows = dataframe.filter(
        (pl.col("column") == column) & 
        (pl.col("comparison") == comparison)
    )

    if rows.is_empty():
        return None

    value = rows.get_column(metric).item(0)

    if value is None or math.isnan(value):
        return None

    return float(value)

def _extract_comparison_metric(
    dataframe: pl.DataFrame,
    comparison: str,
    metric: str,
) -> float | None:
    rows = dataframe.filter(
        pl.col("comparison") == comparison
    )

    if rows.is_empty():
        return None

    value = rows.get_column(metric).item(0)

    if value is None or math.isnan(value):
        return None

    return float(value)

def _mean_psi(
    dataframe: pl.DataFrame,
    comparison: str
) -> float | None:
    values = (
        dataframe
        .filter(pl.col("comparison") == comparison)
        .get_column("psi")
        .drop_nulls()
    )

    if values.is_empty():
        return None

    mean_val = values.mean()

    return float(mean_val) if mean_val is not None else None

def _log_hemisphere_metrics(
    tracker: ExperimentTracker,
    hemisphere: str,
    ks: pl.DataFrame,
    psi: pl.DataFrame,
    adversarial: pl.DataFrame,
    metadata: dict[str, Any],
) -> None:
    metrics = {
        hemisphere: {
            "rows": metadata["ingestion"]["shape"]["rows"],
            "features": metadata["ingestion"]["feature_count"],
            "null_count": metadata["ingestion"]["null_count"],
            "split": {
                "train_rows": metadata["split"]["sizes"]["train"],
                "val_rows": metadata["split"]["sizes"]["validation"],
                "test_rows": metadata["split"]["sizes"]["test"],
            },
            "target": {
                "train_mean": metadata["split"]["target"]["train"]["mean"],
                "val_mean": metadata["split"]["target"]["validation"]["mean"],
                "test_mean": metadata["split"]["target"]["test"]["mean"],
                "train_zero_count": metadata["split"]["target"]["train"]["zero_count"],
                "val_zero_count": metadata["split"]["target"]["validation"]["zero_count"],
                "test_zero_count": metadata["split"]["target"]["validation"]["zero_count"],
            },
            "drift": {
                "ks_wqi_train_vs_val": _extract_column_metric(
                    ks,
                    config.TARGET_COLUMN,
                    "train_vs_val",
                    "ks_statistic"
                ),
                "ks_wqi_train_vs_test": _extract_column_metric(
                    ks,
                    config.TARGET_COLUMN,
                    "train_vs_test",
                    "ks_statistic",
                ),
                "psi_wqi_train_as_base_vs_validation": _extract_column_metric(
                    psi,
                    config.TARGET_COLUMN,
                    "train_as_base_vs_validation",
                    "psi",
                ),
                "psi_wqi_train_as_base_vs_test": _extract_column_metric(
                    psi,
                    config.TARGET_COLUMN,
                    "train_as_base_vs_test",
                    "psi",
                ),
                "psi_mean_train_as_base_vs_validation": _mean_psi(
                    psi,
                    "train_as_base_vs_validation",
                ),
                "psi_mean_train_as_base_vs_test": _mean_psi(
                    psi,
                    "train_as_base_vs_test",
                ),
                "adversarial_auc_train_vs_validation": _extract_comparison_metric(
                    adversarial,
                    "train_vs_validation",
                    "auc_mean",
                ),
                "adversarial_auc_train_vs_test": _extract_comparison_metric(
                    adversarial,
                    "train_vs_test",
                    "auc_mean",
                ),
                "adversarial_auc_validation_vs_test": _extract_comparison_metric(
                    adversarial,
                    "validation_vs_test",
                    "auc_mean",
                ),
            },
        }
    }

    tracker.log_metrics(metrics)

def run(tracking_enabled: bool = True) -> dict[str, Any]:
    env_seed()

    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    config.MLFLOW_DIR.mkdir(parents=True, exist_ok=True)

    mlflow_tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        config.MLFLOW_DIR.as_uri()
    )

    tracking_config = TrackingConfig(
        run_name="preprocess-data-governance",
        mlflow_tracking_uri=mlflow_tracking_uri,
        enabled=tracking_enabled
    )

    summary: dict[str, Any] = {
        "phase": "preprocess",
        "random_state": config.RANDOM_STATE,
        "artifact_dir": str(config.ARTIFACT_DIR),
        "hemispheres": {}
    }

    with ExperimentTracker(tracking_config) as tracker:
        tracker.log_params(
            {
                "phase": "preprocess",
                "random_state": config.RANDOM_STATE,
                "data_dir": str(config.DATA_DIR),
                "artifact_dir": str(config.ARTIFACT_DIR),
                "hemispheres": list(config.HEMISPHERES),
                "leakage_blacklist": sorted(config.LEAKAGE_BLACKLIST),
                "evidence_exclusions": sorted(config.EVIDENCE_EXCLUSIONS),
                "train_years": sorted(config.TRAIN_YEARS),
                "validation_year": config.VALIDATION_YEAR,
                "test_year": config.TEST_YEAR,
            }
        )

        for hemisphere in config.HEMISPHERES:
            ingested = load_hemisphere(hemisphere)
            split = split_chronologically(ingested)

            artifact_dir = config.ARTIFACT_DIR / hemisphere
            split_dir = artifact_dir / "splits"
            drift_dir = artifact_dir / "drift"

            split_dir.mkdir(parents=True, exist_ok=True)
            drift_dir.mkdir(parents=True, exist_ok=True)

            _save_parquet(split.x_train, split_dir / "X_train.parquet")
            _save_parquet(split.y_train.to_frame(config.TARGET_COLUMN), split_dir / "y_train.parquet")

            _save_parquet(split.x_val, split_dir / "X_val.parquet")
            _save_parquet(
                split.y_val.to_frame(config.TARGET_COLUMN),
                split_dir / "y_val.parquet",
            )

            _save_parquet(split.x_test, split_dir / "X_test.parquet")
            _save_parquet(
                split.y_test.to_frame(config.TARGET_COLUMN),
                split_dir / "y_test.parquet",
            )

            train_df, val_df, test_df = _drift_frames(split)
            drift_columns = list(split.x_train.columns) + [config.TARGET_COLUMN]

            ks_metrics = drift.compute_ks_metrics(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                columns=drift_columns
            )

            psi_metrics = drift.compute_psi_metrics(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                columns=drift_columns,
            )

            adversarial_metrics = drift.compute_adversarial_validation(
                train_features=split.x_train,
                val_features=split.x_val,
                test_features=split.x_test,
            )

            ks_metrics.write_csv(drift_dir / "ks_metrics.csv")
            psi_metrics.write_csv(drift_dir / "psi_metrics.csv")
            adversarial_metrics.write_csv(
                drift_dir / "adversarial_validation.csv"
            )

            metadata = {
                "ingestion": ingested.metadata,
                "split": split.metadata,
                "governance": {
                    "leakage_blacklist": sorted(config.LEAKAGE_BLACKLIST),
                    "evidence_exclusions": sorted(config.EVIDENCE_EXCLUSIONS),
                    "feature_exclusions": sorted(config.FEATURE_EXCLUSIONS),
                    "forbidden_drop_patterns": list(config.FORBIDDEN_DROP_PATTERNS),
                    "forbidden_raise_patterns": list(config.FORBIDDEN_RAISE_PATTERNS),
                },
            }

            _save_json(metadata, artifact_dir / "preprocess_metadata.json")

            _log_hemisphere_metrics(
                tracker=tracker,
                hemisphere=hemisphere,
                ks=ks_metrics,
                psi=psi_metrics,
                adversarial=adversarial_metrics,
                metadata=metadata,
            )

            tracker.log_artifact(artifact_dir)

            summary["hemispheres"][hemisphere] = {
                "artifact_dir": str(artifact_dir),
                "data_sha256": ingested.data_hash,
                "feature_count": ingested.metadata["feature_count"],
                "split_sizes": split.metadata["sizes"],
            }

        tracker.log_params({"status": "complete"})

    return summary

if __name__ == "__main__":
    run()
            