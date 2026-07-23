from __future__ import annotations
import numpy as np
import polars as pl
import polars.selectors as cs
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

def compute_psi_metrics(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    columns: list[str],
) -> pl.DataFrame:
    comparisons = [
        ("train_as_base_vs_validation", train_df, val_df),
        ("train_as_base_vs_test", train_df, test_df),
        ("validation_as_base_vs_test", val_df, test_df)
    ]

    records: list[dict[str, object]] = []

    for column in columns:
        is_numeric = train_df.schema[column].is_numeric()
        for comparison_name, base_df, compare_df in comparisons:
            if is_numeric:
                psi = _psi_numeric(
                    base_df[column],
                    compare_df[column],
                    config.PSI_BINS
                )
            else:
                psi = _psi_categorical(base_df[column], compare_df[column])

            records.append(
                {
                    "column": column,
                    "comparison": comparison_name,
                    "psi": psi,
                    "psi_category": _psi_category(psi),
                }
            )

    return pl.DataFrame(records)

def _adversarial_auc(
    base_features: pl.DataFrame,
    compare_features: pl.DataFrame,
    random_state: int
) -> dict[str, float | int]:
    x = pl.concat([base_features, compare_features])
    y = np.concatenate(
        [
            np.zeros(len(base_features), dtype=int),
            np.ones(len(compare_features), dtype=int)
        ]
    )

    if x.is_empty() or len(np.unique(y)) < 2:
        return {
            "auc_mean": np.nan,
            "auc_std": np.nan,
            "n_base": int(len(base_features)),
            "n_compare": int(len(compare_features))
        }

    numeric_features = x.select(cs.numeric()).columns
    categorical_features = x.select(cs.categorical()).columns

    if not numeric_features and not categorical_features:
        return {
            "auc_mean": np.nan,
            "auc_std": np.nan,
            "n_base": int(len(base_features)),
            "n_compare": int(len(compare_features)),
        }

    transformers = []

    if numeric_features:
        transformers.append(
            (
                "numeric",
                SimpleImputer(strategy="median"),
                numeric_features,
            )
        )

    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1
                            ),
                        ),
                    ]
                ),
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers)
    x_processed = preprocessor.fit_transform(x)

    model = HistGradientBoostingClassifier(
        random_state=random_state,
        early_stopping=False,
        max_iter=100,
        learning_rate=0.1
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state
    )

    scores = cross_val_score(
        estimator=model,
        X=x_processed,
        y=y,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )

    return {
        "auc_mean": float(np.mean(scores)),
        "auc_std": float(np.std(scores)),
        "n_base": int(len(base_features)),
        "n_compare": int(len(compare_features)),
    }
    
def compute_adversarial_validation(
    train_features: pl.DataFrame,
    val_features: pl.DataFrame,
    test_features: pl.DataFrame,
    random_state: int = config.RANDOM_STATE,
) -> pl.DataFrame:
    comparisons = [
        ("train_vs_validation", train_features, val_features),
        ("train_vs_test", train_features, test_features),
        ("validation_vs_test", val_features, test_features),
    ]

    records: list[dict[str, object]] = []

    for comparison_name, base_df, compare_df in comparisons:
        metrics = _adversarial_auc(base_df, compare_df, random_state)
        records.append({ "comparison": comparison_name, **metrics })

    return pl.DataFrame(records)
