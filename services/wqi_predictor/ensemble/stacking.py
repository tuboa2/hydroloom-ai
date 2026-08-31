from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.model_selection import TimeSeriesSplit

from .types import (
    EnsembleStrategy,
)

MAX_COEFFICIENT_NORM: float = 10.0


def fit_ridge_stack(
    *,
    z_oof: np.ndarray,
    y_oof: np.ndarray,
    columns: Sequence[str],
    alphas: Sequence[float] | None = None,
    n_splits: int = 5,
) -> tuple[RidgeCV, dict[str, Any]]:
    if alphas is None:
        alphas = np.logspace(-4, 4, 81)

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=0)
    ridge = RidgeCV(alphas=alphas, cv=tscv, scoring="neg_root_mean_squared_error")
    ridge.fit(z_oof, y_oof)

    coefs = ridge.coef_.astype(np.float64)
    intercept = float(ridge.intercept_)

    # Coefficient sanity warning check
    max_coef = float(np.max(np.abs(coefs)))
    metadata = {
        "strategy": EnsembleStrategy.RIDGE_STACK.value,
        "selected_alpha": float(ridge.alpha_),
        "intercept": intercept,
        "coefficients": {col: float(coefs[i]) for i, col in enumerate(columns)},
        "max_coefficient_magnitude": max_coef,
        "extrapolation_warning": bool(max_coef > MAX_COEFFICIENT_NORM),
    }

    return ridge, metadata


def fit_elasticnet_stack(
    *,
    z_oof: np.ndarray,
    y_oof: np.ndarray,
    columns: Sequence[str],
    l1_ratios: Sequence[float] | None = None,
    alphas: Sequence[float] | None = None,
    n_splits: int = 5,
) -> tuple[ElasticNetCV, dict[str, Any]]:
    if l1_ratios is None:
        l1_ratios = (0.1, 0.3, 0.5, 0.7, 0.9)
    if alphas is None:
        alphas = np.logspace(-4, 2, 61)

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=0)
    enet = ElasticNetCV(
        l1_ratio=l1_ratios,
        alphas=alphas,
        cv=tscv,
        max_iter=2000,
        random_state=42,
        selection="cyclic",
    )
    enet.fit(z_oof, y_oof)

    coefs = enet.coef_.astype(np.float64)
    intercept = float(enet.intercept_)

    max_coef = float(np.max(np.abs(coefs)))
    metadata = {
        "strategy": EnsembleStrategy.ELASTICNET_STACK.value,
        "selected_alpha": float(enet.alpha_),
        "selected_l1_ratio": float(enet.l1_ratio_),
        "intercept": intercept,
        "coefficients": {col: float(coefs[i]) for i, col in enumerate(columns)},
        "max_coefficient_magnitude": max_coef,
        "extrapolation_warning": bool(max_coef > MAX_COEFFICIENT_NORM),
    }

    return enet, metadata


def predict_stack(meta_model: Any, z_matrix: np.ndarray) -> np.ndarray:
    preds = meta_model.predict(z_matrix)
    return np.asarray(preds, dtype=np.float64).ravel()
