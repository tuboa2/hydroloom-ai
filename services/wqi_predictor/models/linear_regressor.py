from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import QuantileTransformer

from ..config import RANDOM_STATE

logger = logging.getLogger(__name__)

LinearModel = Ridge | ElasticNet | TransformedTargetRegressor


def train_linear_model(
    params: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[LinearModel, int]:
    x_train = np.ascontiguousarray(x_train, dtype=np.float64)
    y_train = np.ascontiguousarray(y_train, dtype=np.float64)

    model_type = str(params.get("model_type", "ridge")).lower()
    alpha = float(params.get("alpha", 1.0))
    l1_ratio = float(params.get("l1_ratio", 0.0))
    target_transform = str(params.get("target_transform", "none")).lower()

    if model_type == "ridge":
        base_model: Ridge | ElasticNet = Ridge(
            alpha=alpha,
            random_state=RANDOM_STATE,
            solver="cholesky",
            max_iter=10000,
        )
    elif model_type == "elasticnet":
        base_model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            random_state=RANDOM_STATE,
            max_iter=10000,
            selection="random",
        )
    else:
        raise ValueError(f"Unsupported linear model_type: {model_type}")

    if target_transform == "quantile_normal":
        n_quantiles = min(len(x_train), 1000)
        model: LinearModel = TransformedTargetRegressor(
            regressor=base_model,
            transformer=QuantileTransformer(
                n_quantiles=n_quantiles,
                output_distribution="normal",
                random_state=RANDOM_STATE,
            ),
        )
    elif target_transform == "none":
        model = base_model
    else:
        raise ValueError(f"Unsupported target_transform: {target_transform}")

    model.fit(x_train, y_train)

    logger.info(
        "Linear model training completed. Model Type: %s, Target Transformer: %s",
        model_type,
        target_transform,
    )

    return model, 0


def predict_linear(model: LinearModel, x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float64)

    if isinstance(model, (Ridge, ElasticNet)):
        return np.add(np.dot(x, model.coef_), model.intercept_, dtype=np.float64)

    return np.asarray(model.predict(x), dtype=np.float64)
