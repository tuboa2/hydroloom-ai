from __future__ import annotations

import argparse
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.model_selection import TimeSeriesSplit

from ..config import (
    CV_CONFIG,
    MODEL_CONFIG,
)
from ..evaluation import write_report_card
from ..evaluation.scorer import compute_all_metrics, rmse
from ..explainability.shap_analyzer import generate_shap_artifacts
from ..models import (
    catboost_regressor,
    lightgbm_regressor,
    linear_regressor,
    xgboost_regressor,
)
from ..models.common import (
    ArtifactError,
    ModelError,
    build_fold_preprocessor,
    clip_predictions,
    ensure_dir,
    hash_features,
    make_candidate_dirname,
    read_json,
    save_candidate_artifacts,
    validate_feature_contract,
    write_json,
)
from ..tuning.optuna_orchestrator import (
    BestTrialResult,
    run,
)
from ..utils.logging_config import get_logger

logger = get_logger(__name__)

_DEFAULT_BASELINES: dict[str, dict[str, float]] = {
    "north": {
        "inner_cv_gap0_rmse": 5.961611405161389,
        "val_rmse": 5.53710849854081,
    },
    "south": {
        "inner_cv_gap0_rmse": 4.8967459943013,
        "val_rmse": 4.999028005881192,
    },
}


@dataclass(frozen=True)
class StudySpec:
    model_family: str
    loss_name: str
    n_trials: int


@dataclass(frozen=True)
class HemisphereData:
    hemisphere: str
    x_train: pl.DataFrame
    y_train: pl.Series
    x_val: pl.DataFrame
    y_val: pl.Series
    selected_features: list[str]
    baseline: dict[str, float]
    feature_set_hash: str


_DEFAULT_SPECS = (
    [
        StudySpec(model_family=f, loss_name=l, n_trials=200)
        for f in ("xgboost", "lightgbm")
        for l in ("rmse", "logcosh", "huber", "quantile")
    ]
    + [
        StudySpec(model_family="catboost", loss_name=l, n_trials=200)
        for l in ("rmse", "logcosh", "mae")
    ]
    + [StudySpec(model_family="linear", loss_name="diversity", n_trials=200)]
)


def default_study_specs() -> list[StudySpec]:
    return _DEFAULT_SPECS[:]


def _extract_selected_features(payload: Any) -> list[str]:
    payload_type = type(payload)

    if payload_type is list:
        return list(map(str, payload))

    if payload_type is dict:
        for key in ("final_features", "features", "selected_features", "feature_columns"):
            val = payload.get(key)
            if type(val) is list:
                return list(map(str, val))

    raise ArtifactError("Could not extract selected features from Feature Selection artifacts.")


def _load_selected_features(path: Path) -> list[str]:
    payload = read_json(path)
    features = _extract_selected_features(payload)

    if not features:
        raise ArtifactError(f"Selected feature list is empty in {path}")

    return features


def _load_target(path: Path) -> pl.Series:
    schema = pl.read_parquet_schema(path)
    target_col = MODEL_CONFIG["target_column"]

    if target_col in schema:
        col_to_read = target_col
    elif len(schema) == 1:
        col_to_read = next(iter(schema))
    else:
        raise ArtifactError(f"Target '{target_col}' not found, and file has multiple columns.")

    return pl.read_parquet(path, columns=[col_to_read]).to_series().cast(pl.Float32)


def _load_baseline(hemisphere: str, selection_dir: Path) -> dict[str, float]:
    metadata_path = selection_dir / "selection_metadata.json"

    if not metadata_path.exists():
        raise ArtifactError(f"Missing metadata for hemisphere={hemisphere}")

    target = read_json(metadata_path)

    try:
        target = target["phase3_baseline"]
    except (KeyError, TypeError):
        try:
            target = target["baseline"]
        except (KeyError, TypeError):
            pass

    val = (
        target.get("final_val_rmse")
        or target.get("val_rmse")
        or target.get("final_validation_rmse")
        or target.get("baseline_validation_rmse")
    )

    try:
        inner = target["gap_robustness"]["gap_0"]["mean_rmse"]
    except (KeyError, TypeError):
        inner = target.get("inner_cv_gap0_rmse") or target.get("gap_0_mean_rmse")

    if val is not None and inner is not None:
        baseline = dict(_DEFAULT_BASELINES.get(hemisphere.lower(), {}))
        baseline["val_rmse"] = float(val)
        baseline["validation_rmse"] = float(val)
        baseline["inner_cv_gap0_rmse"] = float(inner)
        return baseline

    raise ArtifactError(f"Unable to resolve Phase 3 baselines for hemisphere={hemisphere}")


def load_hemisphere_data(hemisphere: str, artifacts_root: Path) -> HemisphereData:
    hemisphere_lower = hemisphere.lower()

    splits_dir = artifacts_root / "feature-engineer" / hemisphere_lower / "splits"
    selection_dir = artifacts_root / "selection" / hemisphere_lower

    x_train_path = (
        splits_dir / "X_train.parquet"
        if (splits_dir / "X_train.parquet").exists()
        else splits_dir / "x_train.parquet"
    )
    y_train_path = splits_dir / "y_train.parquet"
    x_val_path = (
        splits_dir / "X_val.parquet"
        if (splits_dir / "X_val.parquet").exists()
        else splits_dir / "x_val.parquet"
    )
    y_val_path = splits_dir / "y_val.parquet"
    features_path = selection_dir / "selected_features.json"

    required_paths = (x_train_path, y_train_path, x_val_path, y_val_path, features_path)

    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise ArtifactError(f"Missing required Phase 2/3 artifacts: {missing}")

    selected_features = _load_selected_features(features_path)

    x_train = pl.read_parquet(x_train_path, columns=selected_features)
    x_val = pl.read_parquet(x_val_path, columns=selected_features)

    y_train = _load_target(y_train_path)
    y_val = _load_target(y_val_path)

    validate_feature_contract(x_train, x_val, selected_features)
    baseline = _load_baseline(hemisphere_lower, selection_dir)
    feature_set_hash = hash_features(selected_features)

    logger.info(
        "Loaded hemisphere=%s train_rows=%d val_rows=%d features=%d",
        hemisphere_lower,
        x_train.height,
        x_val.height,
        len(selected_features),
    )

    return HemisphereData(
        hemisphere=hemisphere_lower,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        selected_features=selected_features,
        baseline=baseline,
        feature_set_hash=feature_set_hash,
    )


def _train_model_full(
    model_family: str,
    loss_name: str,
    best_params: Mapping[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    best_iteration_count: int,
) -> Any:
    family = model_family.lower()
    params = dict(best_params)

    if family == "xgboost":
        return xgboost_regressor.train_xgboost_full(
            params=params,
            loss_name=loss_name,
            x_train=x_train,
            y_train=y_train,
            num_boost_round=best_iteration_count,
        )

    if family == "lightgbm":
        return lightgbm_regressor.train_lightgbm_full(
            params=params,
            loss_name=loss_name,
            x_train=x_train,
            y_train=y_train,
            num_boost_round=best_iteration_count,
        )

    if family == "catboost":
        return catboost_regressor.train_catboost_full(
            params=params,
            loss_name=loss_name,
            x_train=x_train,
            y_train=y_train,
            num_iterations=best_iteration_count,
        )

    if family == "linear":
        model, _ = linear_regressor.train_linear_model(
            params=params,
            x_train=x_train,
            y_train=y_train,
        )
        return model

    raise ModelError(f"Unsupported model family: {model_family}")


def _predict_model(
    model_family: str,
    model: Any,
    x: np.ndarray,
    best_iteration_count: int,
) -> np.ndarray:
    family = model_family.lower()

    if family == "xgboost":
        return xgboost_regressor.predict_xgboost(
            model,
            x,
            best_iteration_count=best_iteration_count,
        )

    if family == "lightgbm":
        return lightgbm_regressor.predict_lightgbm(
            model,
            x,
            best_iteration_count=best_iteration_count,
        )

    if family == "catboost":
        return catboost_regressor.predict_catboost(model, x)

    if family == "linear":
        return linear_regressor.predict_linear(model, x)

    raise ModelError(f"Unsupported model family: {model_family}")


def _refit_and_evaluate(
    data: HemisphereData,
    result: BestTrialResult,
) -> tuple[Any, Any, np.ndarray, dict[str, Any]]:
    preprocessor = build_fold_preprocessor(data.selected_features)

    x_train_processed = np.asarray(
        preprocessor.fit_transform(data.x_train),
        dtype=np.float32,
    )
    x_val_processed = np.asarray(
        preprocessor.transform(data.x_val),
        dtype=np.float32,
    )

    y_train = data.y_train.to_numpy()
    y_val = data.y_val.to_numpy()

    model = _train_model_full(
        model_family=result.model_family,
        loss_name=result.loss_name,
        best_params=result.best_params,
        x_train=x_train_processed,
        y_train=y_train,
        best_iteration_count=result.best_iteration_count,
    )

    y_pred_raw = _predict_model(
        model_family=result.model_family,
        model=model,
        x=x_val_processed,
        best_iteration_count=result.best_iteration_count,
    )

    y_pred = clip_predictions(y_pred_raw)

    metrics = compute_all_metrics(
        y_true_arr=y_val,
        y_pred_arr=y_pred,
        fold_rmse=result.fold_rmse_values,
    )

    metrics["validation_rmse"] = metrics.get("rmse")
    metrics["validation_nse"] = metrics.get("nse")

    return model, preprocessor, y_pred, metrics


def _finite_gate_value(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _candidate_gate(
    result: BestTrialResult,
    validation_metrics: Mapping[str, Any],
    baseline: Mapping[str, float],
    shap_summary: Mapping[str, Any] | None,
) -> tuple[bool, bool]:
    leakage_detected = False

    if shap_summary is not None:
        leakage_detected = bool(shap_summary.get("leakage_detected"))

    if leakage_detected:
        return False, True

    cv_rmse = result.inner_cv_rmse_mean
    val_rmse = validation_metrics.get("validation_rmse")
    val_nse = validation_metrics.get("validation_nse")
    cz_mae = validation_metrics.get("critical_zone_mae")

    try:
        if not (
            math.isfinite(cv_rmse)
            and math.isfinite(val_rmse)
            and math.isfinite(val_nse)
            and math.isfinite(cz_mae)
        ):
            return False, False

        baseline_val_rmse = baseline.get("validation_rmse") or baseline.get("val_rmse")
        passed = cv_rmse < baseline["inner_cv_gap0_rmse"] and val_rmse < baseline_val_rmse
        return passed, False

    except TypeError:
        return False, False


def compute_gap7_diagnostic(
    data: HemisphereData,
    result: BestTrialResult,
) -> dict[str, float]:
    cv = TimeSeriesSplit(
        n_splits=int(CV_CONFIG["n_splits"]),
        gap=int(CV_CONFIG["gap_robustness"]),
    )

    x_all = data.x_train
    y_all = data.y_train.to_numpy()

    fold_scores: list[float] = []

    for train_idx, val_idx in cv.split(x_all):
        x_fold_train = x_all[train_idx]
        x_fold_val = x_all[val_idx]

        y_fold_train = y_all[train_idx]
        y_fold_val = y_all[val_idx]

        preprocessor = build_fold_preprocessor(data.selected_features)

        x_fold_train_processed = np.asarray(
            preprocessor.fit_transform(x_fold_train),
            dtype=np.float32,
        )
        x_fold_val_processed = np.asarray(
            preprocessor.transform(x_fold_val),
            dtype=np.float32,
        )

        model = _train_model_full(
            model_family=result.model_family,
            loss_name=result.loss_name,
            best_params=result.best_params,
            x_train=x_fold_train_processed,
            y_train=y_fold_train,
            best_iteration_count=result.best_iteration_count,
        )

        y_pred_raw = _predict_model(
            model_family=result.model_family,
            model=model,
            x=x_fold_val_processed,
            best_iteration_count=result.best_iteration_count,
        )

        y_pred = clip_predictions(y_pred_raw)
        fold_scores.append(float(rmse(y_fold_val, y_pred)))

    if not fold_scores:
        return {
            "gap7_mean_rmse": float("nan"),
            "gap7_std_rmse": float("nan"),
        }

    return {
        "gap7_mean_rmse": float(np.mean(fold_scores)),
        "gap7_std_rmse": float(np.std(fold_scores, ddof=0)),
    }


def process_study(
    spec: StudySpec,
    data: HemisphereData,
    model_dir: Path,
    storage: str,
    run_gap7_diagnostics: bool,
    top_k: int = 5,
) -> dict[str, Any]:

    logger.info(
        "Starting study hemisphere=%s family=%s loss=%s trials=%d",
        data.hemisphere,
        spec.model_family,
        spec.loss_name,
        spec.n_trials,
    )

    candidate_results = run(
        hemisphere=data.hemisphere,
        model_family=spec.model_family,
        loss_name=spec.loss_name,
        x_train=data.x_train,
        y_train=data.y_train,
        feature_columns=data.selected_features,
        n_trials=spec.n_trials,
        storage=storage,
        top_k=top_k,
    )

    if isinstance(candidate_results, BestTrialResult):
        candidate_results = [candidate_results]

    candidate_dirname = make_candidate_dirname(spec.model_family, spec.loss_name)
    candidate_dir = model_dir / "candidates" / candidate_dirname
    ensure_dir(candidate_dir)

    evaluated_candidates = []
    for cand in candidate_results:
        m, prep, y_val_p, val_metrics = _refit_and_evaluate(
            data=data,
            result=cand,
        )
        passed_prelim, _ = _candidate_gate(
            result=cand,
            validation_metrics=val_metrics,
            baseline=data.baseline,
            shap_summary=None,
        )
        evaluated_candidates.append(
            {
                "result": cand,
                "model": m,
                "preprocessor": prep,
                "y_val_pred": y_val_p,
                "validation_metrics": val_metrics,
                "passed_prelim": passed_prelim,
            }
        )

    passing_candidates = [c for c in evaluated_candidates if c["passed_prelim"]]
    if passing_candidates:
        passing_candidates.sort(
            key=lambda c: c["validation_metrics"].get("validation_rmse") or math.inf
        )
        chosen = passing_candidates[0]
    else:
        evaluated_candidates.sort(key=lambda c: c["result"].inner_cv_rmse_mean)
        chosen = evaluated_candidates[0]

    result = chosen["result"]
    model = chosen["model"]
    preprocessor = chosen["preprocessor"]
    y_val_pred = chosen["y_val_pred"]
    validation_metrics = chosen["validation_metrics"]

    if run_gap7_diagnostics:
        gap7_metrics = compute_gap7_diagnostic(data=data, result=result)
        validation_metrics.update(gap7_metrics)

    candidate_meta = {
        "hemisphere": data.hemisphere,
        "model_family": result.model_family,
        "loss": result.loss_name,
        "study_name": result.study_name,
        "best_trial_number": result.best_trial_number,
        "best_iteration_count": result.best_iteration_count,
        "feature_count": len(data.selected_features),
        "feature_set_hash": data.feature_set_hash,
        "artifact_dir": str(candidate_dir),
    }

    shap_summary: dict[str, Any] | None = None

    if result.model_family.lower() in {"xgboost", "lightgbm", "catboost"}:
        x_val_processed = np.asarray(
            preprocessor.transform(data.x_val),
            dtype=np.float32,
        )

        shap_summary = generate_shap_artifacts(
            model=model,
            x=x_val_processed,
            feature_names=data.selected_features,
            output_dir=model_dir / "shap" / candidate_dirname,
            model_family=result.model_family,
            y_true=data.y_val.to_numpy(),
            y_pred=y_val_pred,
        )

    passed_baseline_gate, leakage_detected = _candidate_gate(
        result=result,
        validation_metrics=validation_metrics,
        baseline=data.baseline,
        shap_summary=shap_summary,
    )

    report_card = write_report_card(
        output_dir=candidate_dir,
        candidate_meta=candidate_meta,
        metrics=validation_metrics,
        baseline=data.baseline,
        feature_columns=data.selected_features,
        passed_baseline_gate=passed_baseline_gate,
        leakage_detected=leakage_detected,
    )

    shap_path = (
        str(model_dir / "shap" / candidate_dirname / "shap_summary.json") if shap_summary else None
    )

    save_candidate_artifacts(
        candidate_dir=candidate_dir,
        model=model,
        preprocessor=preprocessor,
        best_params=result.best_params,
        cv_metrics=result.cv_metrics_df,
        val_metrics=validation_metrics,
        report_card=report_card,
        extra_metadata={
            **candidate_meta,
            "inner_cv_rmse_mean": result.inner_cv_rmse_mean,
            "inner_cv_rmse_std": result.inner_cv_rmse_std,
            "passed_baseline_gate": passed_baseline_gate,
            "leakage_detected": leakage_detected,
            "shap_summary_path": shap_path,
        },
    )

    val_rmse = validation_metrics.get("validation_rmse")
    cv_rmse_baseline = data.baseline["inner_cv_gap0_rmse"]
    val_rmse_baseline = data.baseline["validation_rmse"]

    cv_improvement = cv_rmse_baseline - result.inner_cv_rmse_mean
    val_improvement = (val_rmse_baseline - val_rmse) if val_rmse is not None else math.nan

    candidate_entry: dict[str, Any] = {
        "model_family": result.model_family,
        "loss": result.loss_name,
        "study_name": result.study_name,
        "best_trial_number": result.best_trial_number,
        "inner_cv_rmse_mean": result.inner_cv_rmse_mean,
        "inner_cv_rmse_std": result.inner_cv_rmse_std,
        "validation_rmse": val_rmse,
        "validation_nse": validation_metrics.get("validation_nse"),
        "critical_zone_mae": validation_metrics.get("critical_zone_mae"),
        "zero_event_mae": validation_metrics.get("zero_event_mae"),
        "max_absolute_error": validation_metrics.get("max_absolute_error"),
        "best_iteration_count": result.best_iteration_count,
        "passed_baseline_gate": passed_baseline_gate,
        "leakage_detected": leakage_detected,
        "artifact_dir": str(candidate_dir),
        "shap_dir": str(model_dir / "shap" / candidate_dirname) if shap_summary else None,
        "baseline_comparison": {
            "phase3_inner_cv_gap0_rmse": cv_rmse_baseline,
            "phase3_validation_rmse": val_rmse_baseline,
            "inner_cv_improvement": cv_improvement,
            "validation_improvement": val_improvement,
        },
    }

    if run_gap7_diagnostics and "gap7_mean_rmse" in validation_metrics:
        candidate_entry["gap7_mean_rmse"] = validation_metrics["gap7_mean_rmse"]
        candidate_entry["gap7_std_rmse"] = validation_metrics["gap7_std_rmse"]

    write_json(candidate_dir / "candidate_entry.json", candidate_entry)

    logger.info(
        "Completed study=%s passed_gate=%s validation_rmse=%.6f",
        result.study_name,
        passed_baseline_gate,
        val_rmse if val_rmse is not None else math.nan,
    )

    return candidate_entry


def build_candidate_registry(
    data: HemisphereData,
    model_dir: Path,
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:

    passed_candidates = [c for c in candidates if c.get("passed_baseline_gate")]

    passed_candidates.sort(key=lambda c: c.get("validation_rmse") or math.inf)

    ranking = []
    recommendation = None

    if passed_candidates:
        best = passed_candidates[0]

        recommendation = {
            "study_name": best.get("study_name"),
            "model_family": best.get("model_family"),
            "loss": best.get("loss"),
            "artifact_dir": best.get("artifact_dir"),
            "reason": "Lowest Validation RMSE among candidates passing all model gates.",
        }

        ranking = [
            {
                "rank": idx + 1,
                "study_name": c.get("study_name"),
                "model_family": c.get("model_family"),
                "loss": c.get("loss"),
                "validation_rmse": c.get("validation_rmse"),
                "validation_nse": c.get("validation_nse"),
            }
            for idx, c in enumerate(passed_candidates)
        ]

    registry = {
        "hemisphere": data.hemisphere,
        "phase": "model",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "baseline": data.baseline,
        "feature_count": len(data.selected_features),
        "feature_set_hash": data.feature_set_hash,
        "passed_candidates": passed_candidates,
        "ranking": ranking,
        "recommendation": recommendation,
        "phase5_consumption": {
            "level0_base_models": True,
            "stacking_features": "out-of-fold predictions must be generated in Phase 5",
            "test_set_policy": "untouched",
        },
    }

    registry_path = model_dir / "candidate_registry.json"
    write_json(registry_path, registry)

    logger.info("Candidate registry written to %s", registry_path)

    return registry


def run_hemisphere(
    hemisphere: str,
    artifacts_root: Path,
    run_gap7_diagnostics: bool,
) -> dict[str, Any]:

    data = load_hemisphere_data(hemisphere, artifacts_root)
    model_dir = artifacts_root / "model" / data.hemisphere

    for sub_dir in ("candidates", "reports", "diagnostics", "shap", "optuna"):
        (model_dir / sub_dir).mkdir(parents=True, exist_ok=True)

    storage = f"sqlite:///{model_dir}/optuna/hydromind_model.db"

    candidates: list[dict[str, Any]] = []
    study_failures: list[dict[str, Any]] = []

    for spec in default_study_specs():
        try:
            candidate_entry = process_study(
                spec=spec,
                data=data,
                model_dir=model_dir,
                storage=storage,
                run_gap7_diagnostics=run_gap7_diagnostics,
            )
            candidates.append(candidate_entry)
        except Exception as exc:
            logger.exception(
                "Study failed hemisphere=%s family=%s loss=%s",
                data.hemisphere,
                spec.model_family,
                spec.loss_name,
            )
            study_failures.append(
                {
                    "model_family": spec.model_family,
                    "loss": spec.loss_name,
                    "error": str(exc),
                }
            )

    registry = build_candidate_registry(data, model_dir, candidates)

    passed_candidates = registry.get("passed_candidates", [])

    status = "accepted" if passed_candidates else "failed"

    if status == "failed":
        logger.error(
            "model failed acceptance gate for hemisphere=%s: no candidate beat Phase 3 baseline.",
            data.hemisphere,
        )
    else:
        logger.info(
            "model accepted for hemisphere=%s with %d passed candidates.",
            data.hemisphere,
            len(passed_candidates),
        )

    metadata = {
        "hemisphere": data.hemisphere,
        "phase": "model",
        "status": status,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "feature_count": data.x_train.width,
        "selected_features": data.selected_features,
        "feature_set_hash": data.feature_set_hash,
        "phase3_baseline": data.baseline,
        "studies": [
            {
                "study_name": c.get("study_name"),
                "model_family": c.get("model_family"),
                "loss": c.get("loss"),
                "best_trial_number": c.get("best_trial_number"),
                "inner_cv_rmse_mean": c.get("inner_cv_rmse_mean"),
                "inner_cv_rmse_std": c.get("inner_cv_rmse_std"),
                "validation_rmse": c.get("validation_rmse"),
                "validation_nse": c.get("validation_nse"),
                "passed_baseline_gate": c.get("passed_baseline_gate"),
                "leakage_detected": c.get("leakage_detected"),
            }
            for c in candidates
        ],
        "candidates": candidates,
        "passed_candidates": [c.get("study_name") for c in passed_candidates],
        "best_candidate": passed_candidates[0] if passed_candidates else None,
        "study_failures": study_failures,
        "test_set_used": False,
    }

    metadata_path = model_dir / "model_metadata.json"
    write_json(metadata_path, metadata)

    return metadata


def run_model(
    hemispheres: Sequence[str] = ("north", "south"),
    artifacts_root: str | Path = "artifacts",
    run_gap7_diagnostics: bool = True,
) -> dict[str, Any]:

    resolved_root = Path(artifacts_root)

    model_root = resolved_root / "model"
    model_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "started_at_utc": datetime.now(UTC).isoformat(),
        "artifacts_root": str(resolved_root),
        "hemispheres": {},
    }

    t0_hardware = time.perf_counter()

    for hemisphere in hemispheres:
        hemisphere_lower = hemisphere.lower()

        try:
            summary["hemispheres"][hemisphere_lower] = run_hemisphere(
                hemisphere=hemisphere_lower,
                artifacts_root=resolved_root,
                run_gap7_diagnostics=run_gap7_diagnostics,
            )
        except Exception as exc:
            logger.exception("model execution failed for hemisphere=%s", hemisphere_lower)
            summary["hemispheres"][hemisphere_lower] = {
                "hemisphere": hemisphere_lower,
                "status": "failed",
                "error": str(exc),
                "test_set_used": False,
            }

    summary["completed_at_utc"] = datetime.now(UTC).isoformat()
    summary["execution_duration_seconds"] = round(time.perf_counter() - t0_hardware, 3)

    summary_path = model_root / "model_execution_summary.json"
    write_json(summary_path, summary)

    logger.info(
        "Model execution summary written to %s in %.3f seconds",
        summary_path,
        summary["execution_duration_seconds"],
    )

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run HydroMind Service A model Stage 2 model engineering."
    )
    parser.add_argument(
        "--hemispheres",
        nargs="+",
        default=["north", "south"],
        help="Hemispheres to process. Defaults to north and south.",
    )
    parser.add_argument(
        "--artifacts-root",
        default="artifacts",
        help="Root artifact directory. Defaults to artifacts.",
    )
    parser.add_argument(
        "--disable-gap7",
        action="store_true",
        help="Disable secondary gap=7 robustness diagnostics.",
    )

    args = parser.parse_args()

    run_model(
        hemispheres=args.hemispheres,
        artifacts_root=args.artifacts_root,
        run_gap7_diagnostics=not args.disable_gap7,
    )
