from __future__ import annotations

import numpy as np

from .types import (
    DecileDiagnosticMetric,
    DiagnosticZoneMetric,
    MonthlyDiagnosticMetric,
    WQIZone,
)


def compute_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute the 7-metric evaluation suite with high numerical stability on Kaggle CPU."""
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_p = np.asarray(y_pred, dtype=np.float64).ravel()
    n = len(y_t)

    if n == 0:
        return {}

    diff = y_p - y_t
    abs_diff = np.abs(diff)
    sq_diff = np.square(diff)

    rmse = float(np.sqrt(np.mean(sq_diff)))
    mae = float(np.mean(abs_diff))
    max_ae = float(np.max(abs_diff))

    # NSE (Nash-Sutcliffe Efficiency)
    y_mean = np.mean(y_t)
    denom_nse = np.sum(np.square(y_t - y_mean))
    nse = float(1.0 - (np.sum(sq_diff) / denom_nse)) if denom_nse > 1e-12 else 0.0

    # RMSLE
    log_t = np.log1p(np.maximum(0.0, y_t))
    log_p = np.log1p(np.maximum(0.0, y_p))
    rmsle = float(np.sqrt(np.mean(np.square(log_p - log_t))))

    # R2 & Explained Variance
    var_t = np.var(y_t)
    r2 = float(1.0 - (np.mean(sq_diff) / var_t)) if var_t > 1e-12 else 0.0
    exp_var = float(1.0 - (np.var(diff) / var_t)) if var_t > 1e-12 else 0.0

    return {
        "rmse": rmse,
        "nse": nse,
        "mae": mae,
        "rmsle": rmsle,
        "r2": r2,
        "explained_variance": exp_var,
        "max_absolute_error": max_ae,
    }


def compute_zone_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> list[DiagnosticZoneMetric]:
    """Compute error metrics segmented across the 5 WQI operational zones."""
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_p = np.asarray(y_pred, dtype=np.float64).ravel()

    zones = [
        (WQIZone.CRITICAL.value, 0.0, 25.0),
        (WQIZone.POOR.value, 25.0, 50.0),
        (WQIZone.MARGINAL.value, 50.0, 70.0),
        (WQIZone.GOOD.value, 70.0, 85.0),
        (WQIZone.EXCELLENT.value, 85.0, 100.001),
    ]

    results: list[DiagnosticZoneMetric] = []
    for zone_name, low, high in zones:
        mask = (y_t >= low) & (y_t < high)
        count = int(np.sum(mask))

        if count == 0:
            results.append(
                DiagnosticZoneMetric(
                    zone_name=zone_name,
                    observation_count=0,
                    prediction_count=0,
                    mae=None,
                    rmse=None,
                    bias=None,
                    max_absolute_error=None,
                )
            )
            continue

        z_diff = y_p[mask] - y_t[mask]
        results.append(
            DiagnosticZoneMetric(
                zone_name=zone_name,
                observation_count=count,
                prediction_count=count,
                mae=float(np.mean(np.abs(z_diff))),
                rmse=float(np.sqrt(np.mean(np.square(z_diff)))),
                bias=float(np.mean(z_diff)),
                max_absolute_error=float(np.max(np.abs(z_diff))),
            )
        )

    return results


def compute_monthly_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> list[MonthlyDiagnosticMetric]:
    """Compute temporal error metrics across months (approximate 30-day blocks)."""
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_p = np.asarray(y_pred, dtype=np.float64).ravel()
    n = len(y_t)

    results: list[MonthlyDiagnosticMetric] = []
    for month in range(12):
        start = month * 30
        end = min(n, (month + 1) * 30) if month < 11 else n
        if start >= n:
            break

        m_t = y_t[start:end]
        m_p = y_p[start:end]
        count = len(m_t)
        if count == 0:
            continue

        diff = m_p - m_t
        rmse = float(np.sqrt(np.mean(np.square(diff))))
        mae = float(np.mean(np.abs(diff)))
        max_ae = float(np.max(np.abs(diff)))

        # NSE reported only if count >= 10
        nse = None
        if count >= 10:
            var_t = np.sum(np.square(m_t - np.mean(m_t)))
            if var_t > 1e-12:
                nse = float(1.0 - (np.sum(np.square(diff)) / var_t))

        results.append(
            MonthlyDiagnosticMetric(
                month=month + 1,
                observation_count=count,
                rmse=rmse,
                mae=mae,
                nse=nse,
                max_absolute_error=max_ae,
            )
        )

    return results


def compute_decile_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> list[DecileDiagnosticMetric]:
    """Compute error distributions across 10 deciles of true Test WQI."""
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_p = np.asarray(y_pred, dtype=np.float64).ravel()

    quantiles = np.percentile(y_t, np.linspace(0, 100, 11))
    results: list[DecileDiagnosticMetric] = []

    for d in range(10):
        low = quantiles[d]
        high = quantiles[d + 1]
        mask = (y_t >= low) & (y_t <= high) if d == 9 else (y_t >= low) & (y_t < high)
        count = int(np.sum(mask))
        if count == 0:
            continue

        diff = y_p[mask] - y_t[mask]
        results.append(
            DecileDiagnosticMetric(
                decile=d + 1,
                observation_count=count,
                rmse=float(np.sqrt(np.mean(np.square(diff)))),
                mae=float(np.mean(np.abs(diff))),
                bias=float(np.mean(diff)),
            )
        )

    return results
