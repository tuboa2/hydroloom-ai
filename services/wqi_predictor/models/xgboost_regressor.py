from __future__ import annotations

import logging
import os
from typing import Any

import losses
import numpy as np
import tl2cgen
import xgboost as xgb

from ..config import ARTIFACT_DIR, EARLY_STOPPING_ROUNDS, RANDOM_STATE

logger = logging.getLogger(__name__)

SupportedLoss = str


def _xgb_train_params(params: dict[str, Any], loss_name: str) -> dict[str, Any]:
    xgb_params: dict[str, Any] = {
        "max_depth": int(params["max_depth"]),
        "eta": float(params["learning_rate"]),
        "subsample": float(params["subsample"]),
        "colsample_bytree": float(params["colsample_bytree"]),
        "colsample_bylevel": float(params["colsample_bylevel"]),
        "colsample_bynode": float(params["colsample_bynode"]),
        "min_child_weight": int(params["min_child_weight"]),
        "alpha": float(params["reg_alpha"]),
        "lambda": float(params["reg_lambda"]),
        "gamma": float(params["gamma"]),
        "max_bin": int(params["max_bin"]),
        "tree_method": str(params.get("tree_method", "hist")),
        "nthread": int(params.get("n_jobs", -1)),
        "seed": int(params.get("random_state", RANDOM_STATE)),
    }

    loss = loss_name.lower()

    if loss == "rmse":
        xgb_params["objective"] = "reg:squarederror"
        xgb_params["eval_metric"] = "rmse"
    elif loss in {"logcosh", "huber", "quantile"}:
        xgb_params["eval_metric"] = "rmse"
    else:
        raise ValueError(f"Unsupported XGBoost loss name: {loss_name}")

    return xgb_params


def _objective_and_eval(
    loss_name: str,
    params: dict[str, Any],
) -> tuple[Any | None, Any | None]:
    loss = loss_name.lower()

    if loss == "rmse":
        return None, None

    if loss == "logcosh":
        return losses.xgb_log_cosh_objective, losses.xgb_rmse_eval

    if loss == "huber":
        delta = float(params.get("huber_delta", 1.0))
        return losses.xgb_huber_objective(delta), losses.xgb_rmse_eval

    if loss == "quantile":
        alpha = float(params.get("quantile_alpha", 0.5))
        return losses.xgb_quantile_objective(alpha), losses.xgb_rmse_eval

    raise ValueError(f"Unsupported XGBoost loss name: {loss_name}.")


def train_xgboost_fold(
    params: dict[str, Any],
    loss_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
) -> tuple[xgb.Booster, int]:

    x_train = np.ascontiguousarray(x_train, dtype=np.float32)
    x_val = np.ascontiguousarray(x_val, dtype=np.float32)

    y_train = np.ascontiguousarray(y_train, dtype=np.float64)
    y_val = np.ascontiguousarray(y_val, dtype=np.float64)

    dtrain = xgb.DMatrix(x_train, label=y_train, nthread=-1)
    dval = xgb.DMatrix(x_val, label=y_val, ref=dtrain, nthread=-1)

    xgb_params = _xgb_train_params(params, loss_name)
    objective, eval_metric = _objective_and_eval(loss_name, params)
    num_boost_round = int(params.get("n_estimators", 1000))

    booster = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dval, "valid")],
        obj=objective,
        feval=eval_metric,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )

    best_iteration = getattr(booster, "best_iteration", None)
    best_iteration_count = num_boost_round if best_iteration is None else int(best_iteration) + 1

    logger.debug(
        "XGBoost fold training completed. Loss = %s, Best Iteration Count = %d",
        loss_name,
        best_iteration_count,
    )

    return booster, best_iteration_count


def train_xgboost_full(
    params: dict[str, Any],
    loss_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    num_boost_round: int,
) -> xgb.Booster:
    x_train = np.ascontiguousarray(x_train, dtype=np.float32)
    y_train = np.ascontiguousarray(y_train, dtype=np.float64)

    dtrain = xgb.DMatrix(x_train, label=y_train, nthread=-1)

    xgb_params = _xgb_train_params(params, loss_name)
    objective, eval_metric = _objective_and_eval(loss_name, params)

    booster = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=max(1, int(num_boost_round)),
        obj=objective,
        feval=eval_metric,
        verbose_eval=False,
    )

    logger.info(
        "XGBoost full Train refit completed. Loss = %s Num Boost Round = %d",
        loss_name,
        max(1, int(num_boost_round)),
    )

    return booster


def predict_xgboost(
    model: xgb.Booster,
    x: np.ndarray,
    best_iteration_count: int | None = None,
) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)

    if best_iteration_count is not None and int(best_iteration_count) > 0:
        return model.inplace_predict(x, iteration_range=(0, int(best_iteration_count)))

    return model.inplace_predict(x)


def compile_nanosecond_model(model: xgb.Booster) -> str:
    tl_model = tl2cgen.Model.from_xgboost(model)

    artifact_path = f"{ARTIFACT_DIR}/models"
    os.makedirs(artifact_path, exist_ok=True)

    libpath = f"{artifact_path}/xgboost.so"

    logger.info("Compiling XGBoost model to native C library at %s", libpath)
    tl_model.export_lib(toolchain="gcc", libpath=libpath, params={"parallel_comp": 4})

    return libpath


def predict_nanosecond(predictor: tl2cgen.Predictor, x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    dmat = tl2cgen.DMatrix(x)
    return predictor.predict(dmat)
