from __future__ import annotations

import logging
import os
from typing import Any

import lightgbm as lgb
import losses
import numpy as np
import tl2cgen

from ..config import ARTIFACT_DIR, EARLY_STOPPING_ROUNDS, RANDOM_STATE

logger = logging.getLogger(__name__)


def _lgbm_train_params(params: dict[str, Any], loss_name: str) -> dict[str, Any]:
    lgbm_params: dict[str, Any] = {
        "learning_rate": float(params["learning_rate"]),
        "num_leaves": int(params["num_leaves"]),
        "max_depth": int(params["max_depth"]),
        "subsample": float(params["subsample"]),
        "bagging_freq": int(params.get("subsample_freq", 0)),
        "feature_fraction": float(params["colsample_bytree"]),
        "feature_fraction_bynode": float(params["feature_fraction_bynode"]),
        "min_child_samples": int(params["min_child_samples"]),
        "lambda_l1": float(params["reg_alpha"]),
        "lambda_l2": float(params["reg_lambda"]),
        "min_gain_to_split": float(params["min_gain_to_split"]),
        "path_smooth": float(params["path_smooth"]),
        "boosting_type": str(params.get("boosting_type", "gbdt")),
        "num_threads": int(params.get("n_jobs", -1)),
        "seed": int(params.get("random_state", RANDOM_STATE)),
        "verbose": -1,
    }

    loss = loss_name.lower()

    if loss == "rmse":
        lgbm_params["objective"] = "regression"
        lgbm_params["metric"] = "rmse"
    elif loss == "quantile":
        lgbm_params["objective"] = "quantile"
        lgbm_params["alpha"] = float(params.get("quantile_alpha", 0.5))
        lgbm_params["metric"] = "rmse"
    elif loss in {"logcosh", "huber"}:
        lgbm_params["metric"] = "None"
    else:
        raise ValueError(f"Unsupported LightGBM loss name: {loss_name}")

    return lgbm_params


def _objective_and_eval(loss_name: str, params: dict[str, Any]) -> tuple[Any | None, Any | None]:
    loss = loss_name.lower()

    if loss in {"rmse", "quantile"}:
        return None, None
    if loss == "logcosh":
        return losses.lgbm_log_cosh_objective, losses.lgbm_rmse_eval
    if loss == "huber":
        delta = float(params.get("huber_delta", 1.0))
        return losses.lgbm_huber_objective(delta), losses.lgbm_rmse_eval

    raise ValueError(f"Unsupported LightGBM loss name: {loss_name}")


def train_lightgbm_fold(
    params: dict[str, Any],
    loss_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
) -> tuple[lgb.Booster, int]:
    x_train = np.ascontiguousarray(x_train, dtype=np.float32)
    x_val = np.ascontiguousarray(x_val, dtype=np.float32)

    y_train = np.ascontiguousarray(y_train, dtype=np.float64)
    y_val = np.ascontiguousarray(y_val, dtype=np.float64)

    train_set = lgb.Dataset(x_train, label=y_train, free_raw_data=True)
    val_set = lgb.Dataset(x_val, label=y_val, reference=train_set, free_raw_data=True)

    lgbm_params = _lgbm_train_params(params, loss_name)
    objective, eval_metric = _objective_and_eval(loss_name, params)
    num_boost_round = int(params.get("n_estimators", 1000))

    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=0),
    ]

    booster = lgb.train(
        params=lgbm_params,
        train_set=train_set,
        num_boost_round=num_boost_round,
        valid_sets=[train_set, val_set],
        valid_names=["train", "valid"],
        fobj=objective,
        feval=eval_metric,
        callbacks=callbacks,
    )

    best_iteration = getattr(booster, "best_iteration", None)
    best_iteration_count = (
        num_boost_round
        if best_iteration is None or int(best_iteration) <= 0
        else int(best_iteration)
    )

    logger.debug(
        "LightGBM fold training completed. Loss = %s Best Iteration Count = %d",
        loss_name,
        best_iteration_count,
    )
    return booster, best_iteration_count


def train_lightgbm_full(
    params: dict[str, Any],
    loss_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    num_boost_round: int,
) -> lgb.Booster:

    x_train = np.ascontiguousarray(x_train, dtype=np.float32)
    y_train = np.ascontiguousarray(y_train, dtype=np.float64)
        
    train_set = lgb.Dataset(x_train, label=y_train, free_raw_data=True)

    lgbm_params = _lgbm_train_params(params, loss_name)
    objective, _ = _objective_and_eval(loss_name, params)

    booster = lgb.train(
        params=lgbm_params,
        train_set=train_set,
        num_boost_round=max(1, int(num_boost_round)),
        fobj=objective,
        callbacks=[lgb.log_evaluation(period=0)],
    )

    logger.info(
        "LightGBM full Train refit completed. Loss = %s Num Boost Round = %d",
        loss_name,
        max(1, int(num_boost_round)),
    )
    return booster


def compile_nanosecond_model(model: lgb.Booster) -> str:
    tl_model = tl2cgen.Model.from_lightgbm(model)

    artifact_path = f"{ARTIFACT_DIR}/models"
    os.makedirs(artifact_path, exist_ok=True)
    libpath = f"{artifact_path}/lightgbm_optimized.so"

    logger.info("Compiling LightGBM to native C library at %s", libpath)
    tl_model.export_lib(toolchain="gcc", libpath=libpath, params={"parallel_comp": 4})
    return libpath


def predict_nanosecond(predictor: tl2cgen.Predictor, x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    dmat = tl2cgen.DMatrix(x)
    return predictor.predict(dmat)
