from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import polars as pl
from optuna.exceptions import TrialPruned
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from sklearn.model_selection import TimeSeriesSplit

from ..config import CV_CONFIG, OPTUNA_CONFIG
from ..evaluation.scorer import rmse
from ..models import (
    catboost_regressor,
    lightgbm_regressor,
    linear_regressor,
    xgboost_regressor,
)
from ..models.common import (
    ModelError,
    build_fold_preprocessor,
    clip_predictions,
    ensure_dir,
    make_study_name,
)
from . import search_spaces

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BestTrialResult:
    study_name: str
    hemisphere: str
    model_family: str
    loss_name: str
    best_trial_number: int
    best_params: dict[str, Any]
    inner_cv_rmse_mean: float
    inner_cv_rmse_std: float
    fold_rmse_values: list[float]
    best_iteration_count: int
    cv_metrics_df: pl.DataFrame = field(repr=False)


def _suggest_params(trial: optuna.trial.Trial, model_family: str, loss_name: str) -> dict[str, Any]:
    family = model_family.lower()
    if family == "xgboost":
        return search_spaces.suggest_xgboost(trial, loss_name)
    if family == "lightgbm":
        return search_spaces.suggest_lightgbm(trial, loss_name)
    if family == "catboost":
        return search_spaces.suggest_catboost(trial, loss_name)
    if family == "linear":
        return search_spaces.suggest_linear(trial, loss_name)
    raise ModelError(f"Unsupported model family: {model_family}")


def _mean_best_iteration(model_family: str, best_iterations: Sequence[int]) -> int:
    if model_family.lower() == "linear":
        return 0
    if not best_iterations:
        return 1
    return max(1, int(np.round(float(np.mean(best_iterations)))))


def _prepare_storage(storage: str | None) -> str:
    resolved_storage = storage or OPTUNA_CONFIG["storage"]
    if resolved_storage.startswith("sqlite:///"):
        ensure_dir(Path(resolved_storage.replace("sqlite:///", "", 1)).parent)
    return resolved_storage


def trial_to_result(
    trial: optuna.trial.FrozenTrial,
    study_name: str,
    hemisphere: str,
    model_family: str,
    loss_name: str,
) -> BestTrialResult:
    records = trial.user_attrs.get("fold_records") or [
        {"fold": idx, "rmse": val} for idx, val in enumerate(trial.user_attrs.get("fold_rmse", []))
    ]
    return BestTrialResult(
        study_name=study_name,
        hemisphere=hemisphere,
        model_family=model_family,
        loss_name=loss_name,
        best_trial_number=int(trial.number),
        best_params=dict(trial.user_attrs.get("params", dict(trial.params))),
        inner_cv_rmse_mean=float(trial.user_attrs.get("cv_rmse_mean", trial.value or 0.0)),
        inner_cv_rmse_std=float(trial.user_attrs.get("cv_rmse_std", 0.0)),
        fold_rmse_values=list(trial.user_attrs.get("fold_rmse", [])),
        best_iteration_count=int(trial.user_attrs.get("best_iteration_count", 0)),
        cv_metrics_df=pl.DataFrame(records),
    )


def get_top_trials(
    study: optuna.Study,
    study_name: str,
    hemisphere: str,
    model_family: str,
    loss_name: str,
    top_k: int = 15,
) -> list[BestTrialResult]:
    completed = [t for t in study.get_trials(states=(TrialState.COMPLETE,)) if t.value is not None]
    if not completed:
        return []
    completed.sort(key=lambda t: t.value)
    return [
        trial_to_result(t, study_name, hemisphere, model_family, loss_name)
        for t in completed[:top_k]
    ]


def run(
    hemisphere: str,
    model_family: str,
    loss_name: str,
    x_train: pl.DataFrame,
    y_train: pl.Series,
    feature_columns: Sequence[str],
    n_trials: int,
    storage: str | None = None,
    top_k: int = 1,
) -> BestTrialResult | list[BestTrialResult]:
    study_name = make_study_name(hemisphere, model_family, loss_name)
    resolved_storage = _prepare_storage(storage)

    logger.info(
        "Creating/loading Optuna study. Study: %s | Family: %s | Loss: %s | Trials Budget: %d",
        study_name,
        model_family,
        loss_name,
        n_trials,
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=resolved_storage,
        sampler=TPESampler(seed=int(OPTUNA_CONFIG["sampler_seed"])),
        pruner=MedianPruner(
            n_startup_trials=int(OPTUNA_CONFIG["n_startup_trials"]),
            n_warmup_steps=int(OPTUNA_CONFIG["n_warmup_steps"]),
        ),
        direction=str(OPTUNA_CONFIG["direction"]),
        load_if_exists=True,
    )

    study.set_user_attr("hemisphere", hemisphere)
    study.set_user_attr("model_family", model_family)
    study.set_user_attr("loss_name", loss_name)

    selected_features = list(feature_columns)

    x_all_df = x_train.select(selected_features).to_pandas()
    y_all_np = np.asarray(y_train, dtype=np.float64).ravel()

    cv = TimeSeriesSplit(
        n_splits=int(CV_CONFIG["n_splits"]),
        gap=int(CV_CONFIG["gap_primary"]),
    )

    precomputed_folds = []
    for train_idx, val_idx in cv.split(x_all_df):
        x_fold_train = x_all_df.iloc[train_idx]
        x_fold_val = x_all_df.iloc[val_idx]

        preprocessor = build_fold_preprocessor(selected_features)

        x_train_processed = preprocessor.fit_transform(x_fold_train)
        x_val_processed = preprocessor.transform(x_fold_val)

        precomputed_folds.append(
            (
                np.ascontiguousarray(x_train_processed, dtype=np.float64),
                np.ascontiguousarray(y_all_np[train_idx], dtype=np.float64),
                np.ascontiguousarray(x_val_processed, dtype=np.float64),
                np.ascontiguousarray(y_all_np[val_idx], dtype=np.float64),
            )
        )

    family_lower = model_family.lower()
    if family_lower == "xgboost":

        def train_fn(p, xt, yt, xv, yv):
            return xgboost_regressor.train_xgboost_fold(
                params=p, loss_name=loss_name, x_train=xt, y_train=yt, x_val=xv, y_val=yv
            )

        predict_fn = xgboost_regressor.predict_xgboost
    elif family_lower == "lightgbm":

        def train_fn(p, xt, yt, xv, yv):
            return lightgbm_regressor.train_lightgbm_fold(
                params=p, loss_name=loss_name, x_train=xt, y_train=yt, x_val=xv, y_val=yv
            )

        predict_fn = lightgbm_regressor.predict_lightgbm
    elif family_lower == "catboost":

        def train_fn(p, xt, yt, xv, yv):
            return catboost_regressor.train_catboost_fold(
                params=p, loss_name=loss_name, x_train=xt, y_train=yt, x_val=xv, y_val=yv
            )

        def predict_fn(model, xv, bi):
            return catboost_regressor.predict_catboost(model, xv)

    elif family_lower == "linear":

        def train_fn(p, xt, yt, xv, yv):
            return linear_regressor.train_linear_model(params=p, x_train=xt, y_train=yt)

        def predict_fn(model, xv, bi):
            return linear_regressor.predict_linear(model, xv)

    else:
        raise ModelError(f"Unsupported model family: {model_family}")

    def objective(trial: optuna.trial.Trial) -> float:
        params = _suggest_params(trial, model_family, loss_name)
        trial.set_user_attr("params", params)

        fold_scores = []
        best_iterations = []
        fold_records = []

        for fold_idx, (x_t, y_t, x_v, y_v) in enumerate(precomputed_folds):
            try:
                model, best_iter = train_fn(params, x_t, y_t, x_v, y_v)
                y_pred_raw = predict_fn(model, x_v, best_iter)
            except lgb.basic.LightGBMError as exc:
                logger.warning(
                    "LightGBMError encountered in fold %d for trial %d: %s. Pruning trial.",
                    fold_idx,
                    trial.number,
                    exc,
                )
                raise TrialPruned() from exc

            y_pred = clip_predictions(y_pred_raw)
            fold_rmse = float(rmse(y_v, y_pred))

            fold_scores.append(fold_rmse)
            best_iterations.append(int(best_iter))
            fold_records.append(
                {
                    "fold": fold_idx,
                    "rmse": fold_rmse,
                    "best_iteration_count": int(best_iter),
                }
            )

            trial.report(fold_rmse, step=fold_idx)
            if trial.should_prune():
                raise TrialPruned()

        mean_rmse = float(np.mean(fold_scores))

        trial.set_user_attr("fold_rmse", fold_scores)
        trial.set_user_attr("cv_rmse_mean", mean_rmse)
        trial.set_user_attr("cv_rmse_std", float(np.std(fold_scores, ddof=0)))
        trial.set_user_attr(
            "best_iteration_count", _mean_best_iteration(model_family, best_iterations)
        )
        trial.set_user_attr("fold_records", fold_records)

        return mean_rmse

    completed_trials_count = len(study.trials)
    remaining_trials = max(0, int(n_trials) - completed_trials_count)

    if remaining_trials > 0:
        logger.info(
            "Running %d new trials... Existing: %d", remaining_trials, completed_trials_count
        )

        study.optimize(objective, n_trials=remaining_trials, gc_after_trial=False)
    else:
        logger.info("Trial budget satisfied. Skipping optimization.")

    if top_k > 1:
        top_trials = get_top_trials(
            study, study_name, hemisphere, model_family, loss_name, top_k=top_k
        )
        if not top_trials:
            raise ModelError(f"No completed Optuna trials for study: {study_name}")
        return top_trials

    try:
        best_trial = study.best_trial
    except ValueError:
        raise ModelError(f"No completed Optuna trials for study: {study_name}")

    return trial_to_result(best_trial, study_name, hemisphere, model_family, loss_name)
