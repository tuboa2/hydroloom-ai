from __future__ import annotations

import logging
from typing import Any

import numpy as np
from catboost import CatBoostRegressor, Pool

from ..config import EARLY_STOPPING_ROUNDS, RANDOM_STATE

logger = logging.getLogger(__name__)


def _catboost_train_params(
    params: dict[str, Any],
    loss_name: str,
    iterations_override: int | None = None,
) -> dict[str, Any]:
    loss = loss_name.lower()

    if loss == "rmse":
        loss_function = "RMSE"
    elif loss == "quantile":
        loss_function = "Quantile:alpha=0.5"
    elif loss == "mae":
        loss_function = "MAE"
    else:
        raise ValueError(f"Unsupported CatBoost loss name: {loss_name}")

    iterations = int(
        iterations_override if iterations_override is not None else params["iterations"]
    )

    catboost_params: dict[str, Any] = {
        "iterations": max(1, iterations),
        "depth": int(params["depth"]),
        "learning_rate": float(params["learning_rate"]),
        "l2_leaf_reg": float(params["l2_leaf_reg"]),
        "subsample": float(params["subsample"]),
        "colsample_bylevel": float(params["colsample_bylevel"]),
        "min_data_in_leaf": float(params["min_data_in_leaf"]),
        "random_strength": float(params["random_strength"]),
        "bagging_temperature": float(params["bagging_temperature"]),
        "border_count": int(params["border_count"]),
        "grow_policy": str(params["grow_policy"]),
        "loss_function": loss_function,
        "random_seed": int(params.get("random_seed", RANDOM_STATE)),
        "verbose": False,
        "allow_writing_files": bool(params.get("allow_writing_files", False)),
        "thread_count": -1,
        "task_type": "CPU",
    }

    return catboost_params


def _tree_count(model: CatBoostRegressor, fallback: int) -> int:
    tree_count = getattr(model, "tree_count_", None)

    if tree_count is not None and int(tree_count) > 0:
        return int(tree_count)

    getter = getattr(model, "get_tree_count", None)

    if callable(getter):
        try:
            value = getter()
            if value is not None and int(value) > 0:
                return int(value)
        except Exception:
            logger.debug("CatBoost get_tree_count failed; using fallback.", exc_info=True)

    return max(1, int(fallback))


def train_catboost_fold(
    params: dict[str, Any],
    loss_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
) -> tuple[CatBoostRegressor, int]:

    x_train = np.ascontiguousarray(x_train, dtype=np.float32)
    y_train = np.ascontiguousarray(y_train, dtype=np.float64)
    x_val = np.ascontiguousarray(x_val, dtype=np.float32)
    y_val = np.ascontiguousarray(y_val, dtype=np.float64)

    train_pool = Pool(x_train, label=y_train)
    val_pool = Pool(x_val, label=y_val)

    catboost_params = _catboost_train_params(params, loss_name)

    model = CatBoostRegressor(**catboost_params)

    model.fit(
        train_pool,
        eval_set=val_pool,
        early_stopping_rounds=early_stopping_rounds,
        verbose=False,
    )

    best_iteration_count = _tree_count(model, fallback=catboost_params["iterations"])

    logger.debug(
        "CatBoost fold training completed. Loss = %s Best Iteration Count = %d",
        loss_name,
        best_iteration_count,
    )

    return model, best_iteration_count


def train_catboost_full(
    params: dict[str, Any],
    loss_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    num_iterations: int,
) -> CatBoostRegressor:

    x_train = np.ascontiguousarray(x_train, dtype=np.float32)
    y_train = np.ascontiguousarray(y_train, dtype=np.float64)

    train_pool = Pool(x_train, label=y_train)

    catboost_params = _catboost_train_params(
        params,
        loss_name,
        iterations_override=num_iterations,
    )

    model = CatBoostRegressor(**catboost_params)

    model.fit(train_pool, verbose=False)

    logger.info(
        "CatBoost full train refit completed. Loss = %s, Iterations = %d",
        loss_name,
        catboost_params["iterations"],
    )

    return model


def predict_catboost(model: CatBoostRegressor, x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)

    return np.asarray(model.predict(x, thread_count=-1), dtype=np.float64)
