from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

from .types import (
    ResidualCorrectionError,
    ResidualGateResult,
)


def compute_ljung_box_stat(
    residuals: np.ndarray, lags: Sequence[int] = (1, 7, 14, 28)
) -> dict[str, Any]:
    res = np.asarray(residuals, dtype=np.float64).ravel()
    n = len(res)
    valid_lags = [lag for lag in lags if lag < n]
    if not valid_lags:
        return {"q_stat_sum": 0.0, "significant_lags": [], "per_lag_q": {}}

    res_centered = res - np.mean(res)
    var = np.sum(np.square(res_centered))
    if var <= 1e-14:
        return {"q_stat_sum": 0.0, "significant_lags": [], "per_lag_q": {}}

    per_lag_q: dict[int, float] = {}
    q_sum = 0.0

    for lag in valid_lags:
        r_k = np.sum(res_centered[lag:] * res_centered[:-lag]) / var
        q_k = n * (n + 2) * (r_k**2) / (n - lag)
        per_lag_q[lag] = float(q_k)
        q_sum += q_k

    significant = [lag for lag, q in per_lag_q.items() if q > 3.841]

    return {
        "q_stat_sum": float(q_sum),
        "significant_lags": significant,
        "per_lag_q": per_lag_q,
    }


class ResidualAutoregressiveCorrector:
    def __init__(self, alpha: float = 1.0, max_clip: float = 5.0) -> None:
        self.alpha = alpha
        self.max_clip = max_clip
        self.model: Ridge | None = None
        self.y_train_std: float = 1.0
        self.effective_clip: float = 5.0

    def build_residual_dataset(
        self,
        *,
        y_true: np.ndarray,
        base_preds: np.ndarray,
        exogenous_features: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build residual autoregressive feature matrix: [e_lag1, e_lag7, base_yhat, exogenous]."""
        y_arr = np.asarray(y_true, dtype=np.float64).ravel()
        y_hat = np.asarray(base_preds, dtype=np.float64).ravel()
        n = len(y_arr)

        residuals = y_arr - y_hat
        lag1 = np.zeros(n, dtype=np.float64)
        lag7 = np.zeros(n, dtype=np.float64)

        lag1[1:] = residuals[:-1]
        if n > 7:
            lag7[7:] = residuals[:-7]

        feat_cols = [lag1.reshape(-1, 1), lag7.reshape(-1, 1), y_hat.reshape(-1, 1)]
        if exogenous_features is not None:
            feat_cols.append(np.asarray(exogenous_features, dtype=np.float64))

        x_res = np.hstack(feat_cols)
        return x_res, residuals

    def fit(
        self,
        *,
        y_train: np.ndarray,
        base_oof_preds: np.ndarray,
        valid_mask: np.ndarray,
        exogenous_train: np.ndarray | None = None,
    ) -> None:
        """Fit residual Ridge model on valid non-burn-in OOF residuals."""
        y_valid = y_train[valid_mask]
        base_valid = base_oof_preds[valid_mask]
        exo_valid = exogenous_train[valid_mask] if exogenous_train is not None else None

        self.y_train_std = float(np.std(y_train))
        self.effective_clip = max(5.0, 0.10 * self.y_train_std)

        x_res, residuals = self.build_residual_dataset(
            y_true=y_valid,
            base_preds=base_valid,
            exogenous_features=exo_valid,
        )

        self.model = Ridge(alpha=self.alpha, random_state=42)
        self.model.fit(x_res, residuals)

    def predict_sequential(
        self,
        *,
        base_test_preds: np.ndarray,
        y_train_end_residuals: Sequence[float],
        exogenous_test: np.ndarray | None = None,
    ) -> np.ndarray:
        """Sequential inference for test horizon where true residual is only revealed post-step."""
        if self.model is None:
            raise ResidualCorrectionError("Residual model is not fitted.")

        y_hat = np.asarray(base_test_preds, dtype=np.float64).ravel()
        n = len(y_hat)
        corrected_preds = np.zeros(n, dtype=np.float64)

        # Buffer containing historical residuals
        res_buffer = list(y_train_end_residuals)

        for t in range(n):
            e_lag1 = res_buffer[-1] if len(res_buffer) >= 1 else 0.0
            e_lag7 = res_buffer[-7] if len(res_buffer) >= 7 else 0.0
            base_t = y_hat[t]

            feat_row = [e_lag1, e_lag7, base_t]
            if exogenous_test is not None:
                feat_row.extend(exogenous_test[t].tolist())

            x_row = np.array(feat_row, dtype=np.float64).reshape(1, -1)
            raw_e_pred = float(self.model.predict(x_row)[0])

            # Clip correction magnitude to preserve safety bounds
            clipped_e = float(np.clip(raw_e_pred, -self.effective_clip, self.effective_clip))
            corrected_preds[t] = base_t + clipped_e

            # In simulation / sequential mode, the predicted residual acts as best proxy
            res_buffer.append(clipped_e)

        return corrected_preds


def evaluate_residual_activation_gate(
    *,
    y_val: np.ndarray,
    base_val_preds: np.ndarray,
    corrected_val_preds: np.ndarray,
    pre_max_ae: float,
) -> ResidualGateResult:
    y_true = np.asarray(y_val, dtype=np.float64).ravel()
    base_res = y_true - base_val_preds
    corr_res = y_true - corrected_val_preds

    base_rmse = float(np.sqrt(np.mean(np.square(base_res))))
    corr_rmse = float(np.sqrt(np.mean(np.square(corr_res))))
    corr_max_ae = float(np.max(np.abs(corr_res)))

    lb_pre = compute_ljung_box_stat(base_res)
    lb_post = compute_ljung_box_stat(corr_res)

    pass_rmse = corr_rmse <= (base_rmse * 0.995 + 1e-12)
    if lb_pre["q_stat_sum"] <= 1e-6:
        pass_lb = lb_post["q_stat_sum"] <= (lb_pre["q_stat_sum"] + 1e-6)
    else:
        pass_lb = (lb_post["q_stat_sum"] < lb_pre["q_stat_sum"] - 1e-6) and (
            len(lb_post["significant_lags"]) <= len(lb_pre["significant_lags"])
        )
    max_ae_limit = max(pre_max_ae * 1.01, pre_max_ae + 0.1)
    pass_max_ae = corr_max_ae <= max_ae_limit

    enabled = pass_rmse and pass_lb and pass_max_ae
    reason = (
        f"RMSE Gain >= 0.5%: {pass_rmse} ({corr_rmse:.4f} vs {base_rmse:.4f}), "
        f"LB Autocorrelation Drop: {pass_lb} ({lb_post['q_stat_sum']:.2f} < {lb_pre['q_stat_sum']:.2f}), "
        f"MaxAE Guard: {pass_max_ae} ({corr_max_ae:.4f} <= {max_ae_limit:.4f})"
    )

    return ResidualGateResult(
        enabled=enabled,
        reason=reason,
        base_metrics={"rmse": base_rmse, "max_ae": pre_max_ae},
        corrected_metrics={"rmse": corr_rmse, "max_ae": corr_max_ae},
        ljung_box_report={"pre": lb_pre, "post": lb_post},
    )
