from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import polars as pl

from .. import config
from ..preprocess.feature_groups import (
    FEATURE_CAPS,
    REQUIRED_FEATURES,
    infer_feature_family,
)
from ..preprocess.pipeline_factory import build_preprocessor
from ..seeding import env_seed
from ..selection.ablation import (
    apply_cluster_feature_gate,
    run_family_ablation,
)
from ..selection.cv import cv_fold_boundaries, gap_robustness_check
from ..selection.screening import run_screening
from ..selection.stability import run_stability_selection
from ..tracking import ExperimentTracker, TrackingConfig

ARTIFACT_DIR = config.ARTIFACT_DIR / "selection"


def apply_south_upfront_exclusions(
    feature_columns: Sequence[str],
    hemisphere: str,
    enabled_features: Sequence[str] = (),
) -> tuple[list[str], list[str]]:
    feature_columns = list(feature_columns)

    if hemisphere.strip().lower() != "south":
        return feature_columns, []

    enabled = set(enabled_features)

    exclusions = config.SOUTH_DEFAULT_EXCLUSIONS - enabled

    retained = [column for column in feature_columns if column not in exclusions]

    dropped = [column for column in feature_columns if column in exclusions]

    return retained, dropped


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


def _save_csv(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_csv(path)


def _load_splits(hemisphere: str) -> dict[str, pl.DataFrame | pl.Series]:
    base_dir = config.ARTIFACT_DIR / "preprocess" / hemisphere / "splits"

    if not base_dir.exists():
        raise FileNotFoundError(
            f"Missing Preprocess Artifacts for {hemisphere}."
            "Run Preprocessing before executing the Selection phase."
        )

    x_train = pl.read_parquet(base_dir / "X_train.parquet")
    y_train = pl.read_parquet(base_dir / "y_train.parquet")[config.TARGET_COLUMN]

    x_val = pl.read_parquet(base_dir / "X_val.parquet")
    y_val = pl.read_parquet(base_dir / "y_val.parquet")[config.TARGET_COLUMN]

    x_test = pl.read_parquet(base_dir / "X_test.parquet")
    y_test = pl.read_parquet(base_dir / "y_test.parquet")

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
    }


def run_selection(
    tracking_enabled: bool = True,
    stability_seeds: tuple[int, ...] = (42, 1337, 2024, 2025, 2026),
    n_splits: int = 5,
    stability_n_repeats: int = 5,
    ablation_min_relative_improvement: float = 0.005,
    n_jobs: int | None = -1,
    south_enabled_features: Sequence[str] = (),
) -> dict[str, Any]:
    env_seed()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    config.MLFLOW_DIR.mkdir(parents=True, exist_ok=True)

    mlflow_tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        config.MLFLOW_DIR.as_uri(),
    )

    tracking_config = TrackingConfig(
        experiment="selection",
        run_name="selection",
        mlflow_tracking_uri=mlflow_tracking_uri,
        enabled=tracking_enabled,
    )

    summary: dict[str, Any] = {
        "phase": "selection",
        "random_state": config.RANDOM_STATE,
        "artifact_dir": str(ARTIFACT_DIR),
        "hemispheres": {},
    }

    with ExperimentTracker(tracking_config) as tracker:
        tracker.log_params(
            {
                "phase": "phase3",
                "random_state": config.RANDOM_STATE,
                "artifact_dir": str(ARTIFACT_DIR),
                "hemispheres": list(config.HEMISPHERES),
                "stability_seeds": list(stability_seeds),
                "n_splits": n_splits,
                "stability_n_repeats": stability_n_repeats,
                "ablation_min_relative_improvement": ablation_min_relative_improvement,
                "required_features": list(REQUIRED_FEATURES),
                "north_feature_cap": FEATURE_CAPS["north"],
                "south_feature_cap": FEATURE_CAPS["south"],
                "south_default_exclusions": sorted(config.SOUTH_DEFAULT_EXCLUSIONS),
                "south_enabled_features": list(south_enabled_features),
            }
        )

        for hemisphere in config.HEMISPHERES:
            feature_cap = FEATURE_CAPS[hemisphere]

            splits = _load_splits(hemisphere)

            x_train = splits["x_train"]
            y_train = splits["y_train"]
            x_val = splits["x_val"]
            y_val = splits["y_val"]
            x_test = splits["x_test"]

            initial_features = list(x_train.columns)
            south_upfront_exclusions: list[str] = []

            if hemisphere.strip().lower() == "south":
                initial_features, south_upfront_exclusions = apply_south_upfront_exclusions(
                    feature_columns=initial_features,
                    hemisphere=hemisphere,
                    enabled_features=south_enabled_features,
                )

                x_train = x_train[initial_features]
                x_val = x_val[initial_features]
                x_test = x_test[initial_features]

            screened_features, screening_report = run_screening(
                x_train=x_train,
                feature_columns=initial_features,
            )

            candidate_cap = int(feature_cap * 1.5)

            stability_selected, stability_report, stability_scores = run_stability_selection(
                x_train=x_train,
                y_train=y_train,
                feature_columns=screened_features,
                n_splits=n_splits,
                gap=0,
                seeds=stability_seeds,
                min_stability=0.70,
                run_score_threshold=0.02,
                candidate_cap=candidate_cap,
                min_fallback_features=10,
                n_repeats=stability_n_repeats,
                n_jobs=n_jobs,
            )

            if stability_report.is_empty():
                permutation_importances = {}
            else:
                permutation_importances = dict(
                    zip(
                        stability_report["feature"],
                        stability_report["mean_perm_importance"].fill_null(0.0),
                    )
                )

            cluster_gate_result = apply_cluster_feature_gate(
                selected_features=stability_selected,
                available_features=screened_features,
                permutation_importances=permutation_importances,
                scores=stability_scores,
            )

            stability_selected = cluster_gate_result.selected_features
            cluster_gate_report = cluster_gate_result.report

            required_present = [
                feature
                for feature in REQUIRED_FEATURES
                if feature in screened_features and feature not in stability_selected
            ]

            stability_selected = list(dict.fromkeys(stability_selected + required_present))

            (
                final_features,
                ablation_report,
                final_validation_rmse,
                baseline_validation_rmse,
                dropped_families,
            ) = run_family_ablation(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                feature_columns=stability_selected,
                scores=stability_scores,
                feature_cap=feature_cap,
                required_columns=REQUIRED_FEATURES,
                min_relative_improvement=ablation_min_relative_improvement,
                random_state=config.RANDOM_STATE,
                n_jobs=n_jobs,
            )

            preprocessor = build_preprocessor(final_features)
            preprocessor.fit(x_train[final_features])

            validation_transformed = preprocessor.transform(x_val[final_features])

            if validation_transformed.shape[0] != x_val.shape[0]:
                raise ValueError(
                    f"{hemisphere}: preprocessor validation transform row count mismatch."
                )

            gap_robustness = gap_robustness_check(
                x_train=x_train,
                y_train=y_train,
                feature_columns=final_features,
                n_splits=n_splits,
                random_state=config.RANDOM_STATE,
                n_jobs=n_jobs,
            )

            artifact_dir = ARTIFACT_DIR / hemisphere
            reports_dir = artifact_dir / "reports"

            artifact_dir.mkdir(parents=True, exist_ok=True)
            reports_dir.mkdir(parents=True, exist_ok=True)

            _save_json(
                screening_report,
                reports_dir / "stage1_screening_report.json",
            )

            _save_csv(
                stability_report,
                reports_dir / "stability_selection_report.csv",
            )

            _save_csv(
                ablation_report,
                reports_dir / "family_ablation_report.csv",
            )

            _save_json(
                cluster_gate_report,
                reports_dir / "cluster_feature_gate_report.json",
            )

            _save_json(
                {
                    "gap_0": cv_fold_boundaries(
                        x=x_train,
                        n_splits=n_splits,
                        gap=0,
                    ),
                    "gap_7": cv_fold_boundaries(
                        x=x_train,
                        n_splits=n_splits,
                        gap=7,
                    ),
                },
                reports_dir / "cv_fold_boundaries.json",
            )

            _save_json(
                gap_robustness,
                reports_dir / "gap_robustness_check.json",
            )

            final_feature_families = {
                feature: infer_feature_family(feature) for feature in final_features
            }

            selected_payload = {
                "hemisphere": hemisphere,
                "feature_cap": feature_cap,
                "required_features": list(REQUIRED_FEATURES),
                "south_upfront_exclusions": south_upfront_exclusions,
                "cluster_feature_gate": cluster_gate_report,
                "final_feature_count": len(final_features),
                "final_features": final_features,
                "final_feature_families": final_feature_families,
                "dropped_families": dropped_families,
            }

            _save_json(
                selected_payload,
                artifact_dir / "selected_features.json",
            )

            joblib.dump(
                preprocessor,
                artifact_dir / "preprocessor.joblib",
            )

            metadata = {
                "hemisphere": hemisphere,
                "feature_cap": feature_cap,
                "initial_feature_count": len(initial_features),
                "south_upfront_exclusions": south_upfront_exclusions,
                "screened_feature_count": len(screened_features),
                "stability_selected_count": len(stability_selected),
                "final_feature_count": len(final_features),
                "baseline_validation_rmse": baseline_validation_rmse,
                "final_validation_rmse": final_validation_rmse,
                "cluster_feature_gate": cluster_gate_report,
                "gap_robustness": gap_robustness,
                "dropped_families": dropped_families,
                "final_features": final_features,
                "final_feature_families": final_feature_families,
            }

            _save_json(
                metadata,
                artifact_dir / "phase3_metadata.json",
            )

            tracker.log_metrics(
                {
                    hemisphere: {
                        "initial_feature_count": len(initial_features),
                        "south_upfront_exclusion_count": len(south_upfront_exclusions),
                        "screened_feature_count": len(screened_features),
                        "stability_selected_count": len(stability_selected),
                        "final_feature_count": len(final_features),
                        "baseline_validation_rmse": baseline_validation_rmse,
                        "final_validation_rmse": final_validation_rmse,
                        "cluster_gate_triggered": bool(cluster_gate_report.get("triggered", False)),
                        "gap0_cv_mean_rmse": gap_robustness["gap_0"]["mean_rmse"],
                        "gap7_cv_mean_rmse": gap_robustness["gap_7"]["mean_rmse"],
                    }
                }
            )

            tracker.log_artifact(artifact_dir)

            summary["hemispheres"][hemisphere] = {
                "artifact_dir": str(artifact_dir),
                "initial_feature_count": len(initial_features),
                "south_upfront_exclusions": south_upfront_exclusions,
                "screened_feature_count": len(screened_features),
                "stability_selected_count": len(stability_selected),
                "final_feature_count": len(final_features),
                "baseline_validation_rmse": baseline_validation_rmse,
                "final_validation_rmse": final_validation_rmse,
                "cluster_gate_triggered": bool(cluster_gate_report.get("triggered", False)),
            }

        tracker.log_params({"status": "complete"})

    return summary

if __name__ == "__main__":
    run_selection()
