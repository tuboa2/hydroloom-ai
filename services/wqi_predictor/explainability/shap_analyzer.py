from __future__ import annotations

import logging
import re
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ..models.common import ensure_dir, write_json

logger = logging.getLogger(__name__)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

try:
    import shap

    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

_EXACT_LEAKAGE = {
    "day_index",
    "year_index",
    "hemisphere",
    "water_quality_index",
}

_FUZZY_LEAKAGE = re.compile(r"future|lead\d+", re.IGNORECASE)


def _is_suspicious_feature(feature_name: str) -> bool:
    return feature_name in _EXACT_LEAKAGE or bool(_FUZZY_LEAKAGE.search(feature_name))


def _extract_shap_values(raw_shap_values: Any, class_index: int = 0) -> np.ndarray:
    if hasattr(raw_shap_values, "values"):
        raw_shap_values = raw_shap_values.values

    if isinstance(raw_shap_values, list):
        logger.warning(
            "Multiclass SHAP values detected. Extracting explanations for class index %d.",
            class_index,
        )
        if class_index >= len(raw_shap_values):
            raise IndexError(
                f"class_index {class_index} is out of bounds for {len(raw_shap_values)} classes."
            )
        raw_shap_values = raw_shap_values[class_index]

    values = np.asarray(raw_shap_values)

    if values.ndim == 1:
        values = values[np.newaxis, :]

    return values


def _save_top_features(
    output_dir: Path,
    feature_names: Sequence[str],
    mean_abs_shap: np.ndarray,
) -> list[dict[str, Any]]:
    order = np.argsort(-mean_abs_shap)
    sorted_shap = mean_abs_shap[order]

    if isinstance(feature_names, np.ndarray):
        sorted_features = feature_names[order].tolist()
    else:
        sorted_features = [feature_names[i] for i in order]

    suspicious = [_is_suspicious_feature(f) for f in sorted_features]

    df = pl.DataFrame(
        {
            "rank": np.arange(1, len(order) + 1, dtype=np.int32),
            "feature": sorted_features,
            "mean_abs_shap": sorted_shap,
            "suspicious": suspicious,
        }
    )

    df.write_csv(output_dir / "shap_feature_importance.csv")

    records = df.to_dicts()
    write_json(output_dir / "shap_feature_importance.json", records)

    return records


def _save_summary_plot(
    output_dir: Path,
    shap_values: np.ndarray,
    x: np.ndarray,
    feature_names: Sequence[str],
    max_samples: int = 2000,
    max_display: int = 20,
) -> str | None:
    if not SHAP_AVAILABLE or not MATPLOTLIB_AVAILABLE:
        return None

    n_samples = shap_values.shape[0]
    if n_samples > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_samples, size=max_samples, replace=False)
        shap_values = shap_values[idx]
        x = x[idx]

    plt.figure(figsize=(10, 6))

    try:
        shap.summary_plot(
            shap_values,
            x,
            feature_names=list(feature_names),
            max_display=max_display,
            show=False,
        )

        path = output_dir / "shap_summary.png"

        plt.savefig(path, dpi=100, bbox_inches="tight")
        return str(path)

    except Exception:
        logger.exception("Failed to save SHAP summary plot.")
        return None

    finally:
        plt.close("all")


def _save_dependence_plots(
    output_dir: Path,
    shap_values: np.ndarray,
    x: np.ndarray,
    feature_names: Sequence[str],
    top_n: int = 10,
    max_samples: int = 2000,
    interaction_index: int | str | None = None,
) -> list[str]:
    if not SHAP_AVAILABLE or not MATPLOTLIB_AVAILABLE:
        return []

    n_samples = shap_values.shape[0]
    if n_samples > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_samples, size=max_samples, replace=False)
        shap_values = shap_values[idx]
        x = x[idx]

    mean_abs = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs)[::-1][: int(top_n)]

    feature_names_list = list(feature_names)
    saved: list[str] = []

    for feature_index in top_indices:
        idx_int = int(feature_index)
        feature_name = feature_names_list[idx_int]
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", feature_name)
        path = output_dir / f"shap_dependence_{safe_name}.png"

        plt.figure(figsize=(8, 6))
        try:
            shap.dependence_plot(
                idx_int,
                shap_values,
                x,
                feature_names=feature_names_list,
                interaction_index=interaction_index,
                show=False,
            )

            plt.savefig(path, dpi=100, bbox_inches="tight")
            saved.append(str(path))
        except Exception:
            logger.exception(
                "Failed to save SHAP dependence plot for feature '%s' (index %d).",
                feature_name,
                idx_int,
            )
        finally:
            plt.close("all")

    return saved


def _save_instance_plots(
    output_dir: Path,
    explainer: Any,
    shap_values: np.ndarray,
    x: np.ndarray,
    feature_names: Sequence[str],
    y_true: np.ndarray | None,
    y_pred: np.ndarray | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"force_plots": [], "waterfall_plots": []}

    if y_true is None or y_pred is None or not SHAP_AVAILABLE or not MATPLOTLIB_AVAILABLE:
        return result

    y_true_arr = np.asarray(y_true).ravel()
    y_pred_arr = np.asarray(y_pred).ravel()

    if y_true_arr.size == 0 or y_pred_arr.size == 0:
        return result

    if y_true_arr.size != y_pred_arr.size:
        logger.error(
            "Dimension mismatch: y_true (%d) and y_pred (%d)", y_true_arr.size, y_pred_arr.size
        )
        return result

    if y_true_arr.size != x.shape[0]:
        logger.error(
            "Dimension mismatch: evaluation targets (%d) vs feature matrix rows (%d)",
            y_true_arr.size,
            x.shape[0],
        )
        return result

    feature_names_list = list(feature_names)

    abs_errors = np.abs(y_true_arr - y_pred_arr)
    worst_indices = np.argsort(abs_errors)[::-1][:20]

    expected_value = explainer.expected_value
    class_index = 0

    if isinstance(expected_value, (list, tuple, np.ndarray)):
        logger.warning(
            "Multiclass base expected values detected. Extracting for class index %d.", class_index
        )
        expected_value = float(np.ravel(expected_value)[class_index])
    else:
        expected_value = float(expected_value)

    for position, sample_index in enumerate(worst_indices):
        idx = int(sample_index)
        plt.figure(figsize=(10, 4))
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r".*identical low and high xlims.*",
                    category=UserWarning,
                )
                shap.force_plot(
                    expected_value,
                    shap_values[idx],
                    x[idx],
                    feature_names=feature_names_list,
                    matplotlib=True,
                    show=False,
                )
            path = output_dir / f"shap_force_worst_{position + 1:02d}.png"
            plt.savefig(path, dpi=100, bbox_inches="tight")
            result["force_plots"].append(str(path))
        except Exception:
            logger.exception("Failed to save SHAP force plot for sample %d.", idx)
        finally:
            plt.close("all")

    critical_indices = np.where(y_true_arr < 25.0)[0]

    if critical_indices.size > 0:
        crit_errors = abs_errors[critical_indices]
        if crit_errors.size > 10:
            top_k = np.argpartition(crit_errors, -10)[-10:]
            top_k_sorted = top_k[np.argsort(crit_errors[top_k])[::-1]]
            critical_selection = critical_indices[top_k_sorted]
        else:
            critical_selection = critical_indices[np.argsort(crit_errors)[::-1]]

        for position, sample_index in enumerate(critical_selection):
            idx = int(sample_index)
            plt.figure(figsize=(10, 6))
            try:
                explanation = shap.Explanation(
                    values=shap_values[idx],
                    base_values=expected_value,
                    data=x[idx],
                    feature_names=feature_names_list,
                )
                shap.plots.waterfall(explanation, show=False)
                path = output_dir / f"shap_waterfall_critical_{position + 1:02d}.png"
                plt.savefig(path, dpi=100, bbox_inches="tight")
                result["waterfall_plots"].append(str(path))
            except Exception:
                logger.exception("Failed to save SHAP waterfall plot for sample %d.", idx)
            finally:
                plt.close("all")

    return result


def generate_shap_artifacts(
    model: Any,
    x: np.ndarray | pl.DataFrame,
    feature_names: Sequence[str],
    output_dir: str | Path,
    model_family: str,
    y_true: Sequence[float] | None = None,
    y_pred: Sequence[float] | None = None,
) -> dict[str, Any]:
    resolved_dir = ensure_dir(output_dir)
    names = list(feature_names)

    summary: dict[str, Any] = {
        "available": False,
        "model_family": model_family,
        "feature_count": len(names),
        "output_dir": str(resolved_dir),
        "top_features": [],
        "leakage_detected": False,
        "suspicious_features": [],
        "artifacts": {},
    }

    if model_family.lower() == "linear":
        summary["reason"] = "SHAP TreeExplainer is not required for linear diversity models."
        write_json(resolved_dir / "shap_unsupported.json", summary)
        logger.info("Skipping SHAP for linear candidate.")
        return summary

    if not SHAP_AVAILABLE:
        summary["reason"] = "SHAP package is unavailable in this environment."
        write_json(resolved_dir / "shap_unavailable.json", summary)
        logger.warning("SHAP package unavailable; skipping SHAP artifacts.")
        return summary

    if isinstance(x, pl.DataFrame):
        x_arr = x.to_numpy()
    else:
        x_arr = np.asarray(x)

    try:
        explainer = shap.TreeExplainer(model)
        raw_values = explainer.shap_values(x_arr)
        shap_values = _extract_shap_values(raw_values)
    except Exception as exc:
        summary["reason"] = f"SHAP explainer failed: {exc}"
        write_json(resolved_dir / "shap_error.json", summary)
        logger.exception("SHAP explanation failed for model_family: %s", model_family)
        return summary

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_features = _save_top_features(resolved_dir, names, mean_abs_shap)

    summary["available"] = True
    summary["top_features"] = top_features[:20]

    suspicious = [record["feature"] for record in top_features if record["suspicious"]]
    summary["suspicious_features"] = suspicious
    summary["leakage_detected"] = bool(suspicious)

    summary["artifacts"]["feature_importance_csv"] = str(
        resolved_dir / "shap_feature_importance.csv"
    )
    summary["artifacts"]["feature_importance_json"] = str(
        resolved_dir / "shap_feature_importance.json"
    )

    summary_plot = _save_summary_plot(resolved_dir, shap_values, x_arr, names)
    if summary_plot:
        summary["artifacts"]["summary_plot"] = summary_plot

    dependence_plots = _save_dependence_plots(resolved_dir, shap_values, x_arr, names, top_n=10)
    if dependence_plots:
        summary["artifacts"]["dependence_plots"] = dependence_plots

    y_true_arr = None if y_true is None else np.asarray(y_true, dtype=np.float32).ravel()
    y_pred_arr = None if y_pred is None else np.asarray(y_pred, dtype=np.float32).ravel()

    instance_artifacts = _save_instance_plots(
        resolved_dir,
        explainer,
        shap_values,
        x_arr,
        names,
        y_true_arr,
        y_pred_arr,
    )

    if instance_artifacts["force_plots"]:
        summary["artifacts"]["force_plots"] = instance_artifacts["force_plots"]

    if instance_artifacts["waterfall_plots"]:
        summary["artifacts"]["waterfall_plots"] = instance_artifacts["waterfall_plots"]

    write_json(resolved_dir / "shap_summary.json", summary)

    if summary["leakage_detected"]:
        logger.error(
            "SHAP leakage diagnostics detected suspicious dominant features: %s",
            summary["suspicious_features"],
        )
    else:
        logger.info("SHAP leakage diagnostics passed for %s candidate.", model_family)

    return summary
