from __future__ import annotations

from collections.abc import Callable

import numpy as np

EPSILON: float = 1e-6

Array = np.ndarray
ObjectiveOutput = tuple[Array, Array]


def _prepare_arrays(
    y_true: Array,
    y_pred: Array,
) -> tuple[Array, Array]:
    # flatten and validate prediction/target arrays
    y_true_arr = np.asarray(y_true, dtype=np.float32).ravel()
    y_pred_arr = np.asarray(y_pred, dtype=np.float32).ravel()

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape. "
            f"Got {y_true_arr.shape} and {y_pred_arr.shape}."
        )

    return y_true_arr, y_pred_arr


def _rmse_value(
    y_true: Array,
    y_pred: Array,
) -> float:
    # compute rmse between two arrays
    y_true_arr, y_pred_arr = _prepare_arrays(y_true, y_pred)

    if y_true_arr.size == 0:
        return 0.0

    diff = y_pred_arr - y_true_arr

    return float(np.sqrt(np.mean(np.square(diff))))


def log_cosh_loss(
    y_true: Array,
    y_pred: Array,
) -> float:
    # compute mean log-cosh loss
    y_true_arr, y_pred_arr = _prepare_arrays(y_true, y_pred)

    if y_true_arr.size == 0:
        return 0.0

    diff = y_pred_arr - y_true_arr

    # stable log(cosh(x)) = log1p(exp(-2|x|)) + |x| - log(2)
    abs_diff = np.abs(diff)
    loss = np.log1p(np.exp(-2.0 * abs_diff)) + abs_diff - np.log(2.0)

    return float(np.mean(loss))


def log_cosh_objective_xgb(
    y_true: Array,
    y_pred: Array,
) -> ObjectiveOutput:
    # log-cosh gradient/hessian objective
    y_true_arr, y_pred_arr = _prepare_arrays(y_true, y_pred)

    diff = y_pred_arr - y_true_arr
    gradient = np.tanh(diff)
    hessian = np.maximum(1.0 - np.square(gradient), EPSILON)

    return gradient, hessian


def log_cosh_objective_lgbm(
    y_true: Array,
    y_pred: Array,
) -> ObjectiveOutput:
    # log-cosh gradient/hessian objective for light gbm
    return log_cosh_objective_xgb(y_true, y_pred)


def huber_objective_xgb(
    y_true: Array,
    y_pred: Array,
    delta: float,
) -> ObjectiveOutput:
    # huber gradient/hessian objective with tunable delta
    if delta <= 0.0:
        raise ValueError("Huber delta must be positive.")

    y_true_arr, y_pred_arr = _prepare_arrays(y_true, y_pred)

    diff = y_pred_arr - y_true_arr
    abs_diff = np.abs(diff)

    gradient = np.where(
        abs_diff <= delta,
        diff,
        delta * np.sign(diff),
    )
    hessian = np.where(
        abs_diff <= delta,
        1.0,
        EPSILON,
    )

    return gradient.astype(np.float32), hessian.astype(np.float32)


def huber_objective_lgbm(
    y_true: Array,
    y_pred: Array,
    delta: float,
) -> ObjectiveOutput:
    # huber gradient/hessian objective for light gbm
    return huber_objective_xgb(y_true, y_pred, delta)


def quantile_objective_xgb(
    y_true: Array,
    y_pred: Array,
    alpha: float = 0.5,
) -> ObjectiveOutput:
    # quantile gradient/hessian objective for median regression
    if not 0.0 < alpha < 1.0:
        raise ValueError("Quantile alpha must be in the open interval (0, 1).")

    y_true_arr, y_pred_arr = _prepare_arrays(y_true, y_pred)

    gradient = np.where(
        y_pred_arr < y_true_arr,
        -alpha,
        1.0 - alpha,
    )

    hessian = np.full_like(gradient, fill_value=EPSILON, dtype=np.float32)

    return gradient.astype(np.float32), hessian


def rmse_eval_lgbm(
    y_true: Array,
    y_pred: Array,
) -> float:
    # rmse evaluation helper
    return _rmse_value(y_true, y_pred)


def xgb_log_cosh_objective(
    y_pred: Array,
    dtrain: object,
) -> ObjectiveOutput:
    # xgboost-native log-cosh objective wrapper
    y_true = dtrain.get_label()
    return log_cosh_objective_xgb(y_true, y_pred)


def xgb_huber_objective(delta: float) -> Callable[[Array, object], ObjectiveOutput]:
    # create an xgboost-native huber objective closure
    def _objective(y_pred: Array, dtrain: object) -> ObjectiveOutput:
        y_true = dtrain.get_label()
        return huber_objective_xgb(y_true, y_pred, delta)

    return _objective


def xgb_quantile_objective(
    alpha: float = 0.5,
) -> Callable[[Array, object], ObjectiveOutput]:
    # create an xgboost-native quantile objective closure
    def _objective(y_pred: Array, dtrain: object) -> ObjectiveOutput:
        y_true = dtrain.get_label()
        return quantile_objective_xgb(y_true, y_pred, alpha)

    return _objective


def xgb_rmse_eval(y_pred: Array, dtrain: object) -> tuple[str, float]:
    # xgboost-native rmse evaluation metric
    y_true = dtrain.get_label()
    return "rmse", _rmse_value(y_true, y_pred)


def lgbm_log_cosh_objective(y_pred: Array, dataset: object) -> ObjectiveOutput:
    # lightgbm-native log-cosh objective wrapper
    y_true = dataset.get_label()
    return log_cosh_objective_lgbm(y_true, y_pred)


def lgbm_huber_objective(delta: float) -> Callable[[Array, object], ObjectiveOutput]:
    # create a lightgbm-native huber objective closure
    def _objective(y_pred: Array, dataset: object) -> ObjectiveOutput:
        y_true = dataset.get_label()
        return huber_objective_lgbm(y_true, y_pred, delta)

    return _objective


def lgbm_quantile_objective(alpha: float = 0.5) -> Callable[[Array, object], ObjectiveOutput]:
    # create a lightgbm-native quantile objective closure
    def _objective(y_pred: Array, dataset: object) -> ObjectiveOutput:
        y_true = dataset.get_label()
        return quantile_objective_xgb(y_true, y_pred, alpha)

    return _objective


def lgbm_rmse_eval(y_pred: Array, dataset: object) -> tuple[str, float, bool]:
    # lightgbm-native rmse evaluation metric
    y_true = dataset.get_label()
    return "rmse", _rmse_value(y_true, y_pred), False
