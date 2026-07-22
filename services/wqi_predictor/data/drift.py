from __future__ import annotations
import numpy as np
import polars as pl
from scipy.stats import ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from .. import config

EPSILON = 1e-6

def _psi_category(psi: float) -> str:
    if np.isnan(psi):
        return "unknown"
    if psi < 0.10:
        return "stable"
    if psi < 0.25:
        return "moderate_drift"
    return "significant_drift"

def _psi_numeric(base: pl.Series, compare: pl.Series, bins: int) -> float:
    base_values = base.drop_nulls().to_numpy().astype(float)
    compare_values = compare.drop_nulls().to_numpy().astype(float)

    if base_values.size == 0 or compare_values.size == 0:
        return np.nan

    if np.unique(base_values).size <= 1:
        return 0.0

    quantiles = np.quantile(base_values, np.linspace(0.0, 1.0, bins + 1))
    edges = np.unique(quantiles).astype(float)

    if edges.size < 2:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    base_counts, _ = np.histogram(base_values, bins=edges)
    compare_counts, _ = np.histogram(compare_values, bins=edges)

    base_pct = base_counts / base_counts.sum() + EPSILON
    compare_pct = compare_counts / compare_counts.sum() + EPSILON

    psi = np.sum((compare_pct - base_pct) * np.log(compare_pct / base_pct))
    return float(psi)

def _psi_categorical(base: pl.Series, compare: pl.Series) -> float:
    base_values = base.drop_nulls().cast(pl.String)
    compare_values = compare.drop_nulls().cast(pl.String)

    if base_values.is_empty() or compare_values.is_empty():
        return np.nan

    categories_df = pl.concat([base_values, compare_values]).unique().to_frame("category")

    base_counts = base_values.value_counts(normalize=True)
    compare_counts = compare_values.value_counts(normalize=True)

    aligned = (
        categories_df
        .join(base_counts, left_on="category", right_on=base_counts.columns[0], how="left")
        .join(compare_counts, left_on="category", right_on=compare_counts.columns[0], how="left")
        .select([
            (pl.col(base_counts.columns[1]).fill_null(0.0) + EPSILON).alias("base_pct"),
            (pl.col(compare_counts.columns[1]).fill_null(0.0) + EPSILON).alias("compare_pct"),
        ])
    )

    psi = aligned.select(
        ((pl.col("compare_pct") - pl.col("base_pct")) * (pl.col("compare_pct") / pl.col("base_pct")).log()).sum()
    ).item()

    return float(psi)

def compute_ks_metrics(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    columns: list[str]
) -> pl.DataFrame:
    dataframes = {
        "train": train_df,
        "validation": val_df,
        "test": test_df
    }

    comparisons = [
        ("train_vs_validation", "train", "validation"),
        ("train_vs_test", "train", "test"),
        ("validation_vs_test", "validation", "test"),
    ]

    records: list[dict[str, object]] = []

    for column in columns:
        if not train_df[column].dtype.is_numeric():
            continue
           
        for comparison_name, base_key, compare_key in comparisons:
            base = dataframes[base_key][column].drop_nulls().to_numpy().astype(float)
            compare = dataframes[compare_key][column].drop_nulls().to_numpy().astype(float)

            if base.size == 0 or compare.size == 0:
                statistic = np.nan
                p_value = np.nan
            else:
                result = ks_2samp(base, compare)
                statistic = float(result.statistic)
                p_value = float(result.pvalue)

            records.append(
                {
                    "column": column,
                    "comparison": comparison_name,
                    "ks_statistic": statistic,
                    "ks_p_value": p_value
                }
            )

    return pl.DataFrame(records)

# TODO: Compute PSI Metrics
# TODO: Adverserial AUC
# TODO: Compute Adverserial Validation
# TODO: Preprocessing Pipeline Orchestrator
# TODO: PyTest Verification
