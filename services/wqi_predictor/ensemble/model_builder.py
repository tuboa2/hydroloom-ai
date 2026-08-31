from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from services.wqi_predictor.ensemble.types import Hemisphere, ModelSpec

MANDATORY_SEEDS: tuple[int, ...] = (42, 1337, 2024, 2025, 2026)


def build_model_specs(
    *,
    hemisphere: Hemisphere,
    integrity_report: Mapping[str, Any],
    seeds: Sequence[int] = MANDATORY_SEEDS,
    artifacts_root: Path,
) -> tuple[ModelSpec, ...]:
    accepted = integrity_report.get("accepted_candidates", [])
    specs: list[ModelSpec] = []

    for cand in accepted:
        cand_name = cand["name"]
        study_name = cand.get("study_name", cand_name)
        model_family = cand["model_family"]
        loss_name = cand.get("loss_name", "rmse")
        base_params = cand.get("hyperparameters", {})
        best_iter = int(cand.get("best_iteration_count", 100))
        feat_hash = cand.get("feature_set_hash", "frozen")

        for seed in seeds:
            model_id = f"{cand_name}::seed={seed}"
            seed_params = build_seed_params(
                model_family=model_family,
                base_params=base_params,
                seed=seed,
                best_iteration_count=best_iter,
            )
            artifact_dir = artifacts_root / "ensemble" / hemisphere.value / "model" / model_id

            spec = ModelSpec(
                candidate_name=cand_name,
                study_name=study_name,
                model_family=model_family,
                loss_name=loss_name,
                seed=seed,
                feature_set_hash=feat_hash,
                hyperparameters=seed_params,
                best_iteration_count=best_iter,
                artifact_dir=artifact_dir,
                model_id=model_id,
            )
            specs.append(spec)

    return tuple(specs)


build_level0_specs = build_model_specs


def build_seed_params(
    *,
    model_family: str,
    base_params: Mapping[str, Any],
    seed: int,
    best_iteration_count: int,
) -> dict[str, Any]:
    params = dict(base_params)
    family = model_family.lower()

    if family == "lightgbm":
        params["random_state"] = seed
        params["n_estimators"] = best_iteration_count
        params["early_stopping_round"] = None
        params["verbosity"] = -1
        params["n_jobs"] = -1
    elif family == "xgboost":
        params["random_state"] = seed
        params["n_estimators"] = best_iteration_count
        params["early_stopping_rounds"] = None
        params["verbosity"] = 0
        params["n_jobs"] = -1
    elif family == "catboost":
        params["random_seed"] = seed
        params["iterations"] = best_iteration_count
        params["early_stopping_rounds"] = None
        params["verbose"] = False
        params["thread_count"] = -1
    elif family in {"ridge", "linear", "elasticnet"}:
        params["random_state"] = seed

    return params


def instantiate_model(spec: ModelSpec) -> Any:
    family = spec.model_family.lower()
    params = dict(spec.hyperparameters)

    if family == "lightgbm":
        import lightgbm as lgb

        return lgb.LGBMRegressor(**params)
    elif family == "xgboost":
        import xgboost as xgb

        return xgb.XGBRegressor(**params)
    elif family == "catboost":
        import catboost as cb

        return cb.CatBoostRegressor(**params)
    elif family == "ridge":
        from sklearn.linear_model import Ridge

        return Ridge(**params)
    elif family == "elasticnet":
        from sklearn.linear_model import ElasticNet

        return ElasticNet(**params)
    else:
        from sklearn.linear_model import LinearRegression

        return LinearRegression()
