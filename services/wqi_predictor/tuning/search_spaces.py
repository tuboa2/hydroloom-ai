from __future__ import annotations

import os
from typing import Any

import optuna
import psutil

from ..config import RANDOM_STATE

PHYSICAL_CORES = psutil.cpu_count(logical=False) or os.cpu_count() or 4
os.environ["OMP_NUM_THREADS"] = str(PHYSICAL_CORES)
os.environ["MKL_NUM_THREADS"] = str(PHYSICAL_CORES)


def suggest_xgboost(trial: optuna.trial.Trial, loss_name: str) -> dict[str, Any]:
    params: dict[str, Any] = {
        "tree_method": "hist",
        "device": "cpu",
        "n_jobs": PHYSICAL_CORES,
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "max_bin": trial.suggest_int("max_bin", 256, 512),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.3, 1.0),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.3, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "random_state": RANDOM_STATE,
    }

    if loss_name.lower() == "huber":
        params["objective"] = "reg:pseudohubererror"
        params["huber_delta"] = trial.suggest_float("huber_delta", 0.5, 10.0)
    elif loss_name.lower() == "quantile":
        params["objective"] = "reg:quantileerror"
        params["quantile_alpha"] = 0.5
    else:
        params["objective"] = "reg:squarederror"

    return params


def suggest_lightgbm(trial: optuna.trial.Trial, loss_name: str) -> dict[str, Any]:
    max_depth = trial.suggest_int("max_depth", 3, 10)
    max_leaves_for_depth = min(255, 2**max_depth)

    params: dict[str, Any] = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "n_jobs": PHYSICAL_CORES,
        "verbose": -1,
        "max_depth": max_depth,
        "num_leaves": trial.suggest_int("num_leaves", 15, max_leaves_for_depth),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "feature_fraction_bynode": trial.suggest_float("feature_fraction_bynode", 0.3, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 5.0),
        "path_smooth": trial.suggest_float("path_smooth", 0.0, 10.0),
        "random_state": RANDOM_STATE,
    }

    if loss_name.lower() == "huber":
        params["objective"] = "huber"
        params["alpha"] = trial.suggest_float("huber_delta", 0.5, 10.0)
    elif loss_name.lower() == "quantile":
        params["objective"] = "quantile"
        params["alpha"] = 0.5

    return params


def suggest_catboost(trial: optuna.trial.Trial, loss_name: str) -> dict[str, Any]:
    grow_policy = trial.suggest_categorical(
        "grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]
    )
    bootstrap_type = trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"])

    params: dict[str, Any] = {
        "thread_count": PHYSICAL_CORES,
        "verbose": False,
        "allow_writing_files": False,
        "random_seed": RANDOM_STATE,
        "grow_policy": grow_policy,
        "depth": trial.suggest_int("depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.3, 1.0),
        "bootstrap_type": bootstrap_type,
    }

    if bootstrap_type == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 10.0)
    elif bootstrap_type in ["Bernoulli", "MVS"]:
        params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)

    if grow_policy in ["Depthwise", "Lossguide"]:
        params["min_data_in_leaf"] = trial.suggest_int("min_data_in_leaf", 5, 100)

    if loss_name.lower() == "quantile":
        params["loss_function"] = "Quantile:alpha=0.5"
        params["eval_metric"] = "Quantile:alpha=0.5"
    elif loss_name.lower() == "huber":
        huber_delta = trial.suggest_float("huber_delta", 0.5, 10.0)
        params["loss_function"] = f"Huber:delta={huber_delta}"
        params["eval_metric"] = "RMSE"
    else:
        params["loss_function"] = "RMSE"
        params["eval_metric"] = "RMSE"

    return params


def suggest_linear(trial: optuna.trial.Trial, loss_name: str = "mse") -> dict[str, Any]:
    model_type = trial.suggest_categorical("model_type", ["ridge", "elasticnet"])

    params: dict[str, Any] = {
        "model_type": model_type,
        "alpha": trial.suggest_float("alpha", 1e-4, 1e4, log=True),
        "target_transform": trial.suggest_categorical(
            "target_transform", ["none", "quantile_normal"]
        ),
    }

    if model_type == "elasticnet":
        params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)
    else:
        params["l1_ratio"] = 0.0

    return params
