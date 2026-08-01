from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

from ..preprocess.feature_groups import (
    RAW_CLUSTER_FEATURES,
    REQUIRED_FEATURES,
    infer_feature_family,
)
from ..preprocess.pipeline_factory import build_preprocessor

CLUSTER_SHARE_FEATURES: tuple[str, ...] = (
    "heavy_share",
    "conservationist_share",
    "outdoor_share",
    "standard_share",
)

CLUSTER_REPLACEMENT_BASE_FEATURES: tuple[str, ...] = (
    "total_cluster_demand",
    "cluster_demand_pc1",
)

@dataclass(frozen=True)
class ClusterGateResult:
    selected_features: list[str]
    report: dict[str, Any]

def apply_cluster_feature_gate(
    selected_features: Sequence[str],
    available_features: Sequence[str],
    permutation_importances: Mapping[str, float],
    scores: Mapping[str, float],
) -> ClusterGateResult:
    selected = list(dict.fromkeys(selected_features))
    available_set = set(available_features)

    raw_present = [
        column for column in RAW_CLUSTER_FEATURES if column in available_set
    ]

    if not raw_present:
        return ClusterGateResult(
            selected_features=selected,
            report={
                "triggered": False,
                "reason": "no_raw_cluster_features_available",
                "raw_cluster_features": [],
                "raw_permutation_importance": {},
                "dropped_raw_features": [],
                "enforced_features": [],
                "strongest_cluster_feature": None,
            },
        )

    raw_permutation_importance = {
        column: float(permutation_importances.get(column, 0.0))
        for column in raw_present
    }

    all_raw_non_positive = all(
        value <= 0.0 for value in raw_permutation_importance.values()
    )

    if not all_raw_non_positive:
        return ClusterGateResult(
            selected_features=selected,
            report={
                "triggered": False,
                "reason": "raw_cluster_features_have_positive_permutation_importance",
                "raw_permutation_importance": raw_permutation_importance,
                "dropped_raw_features": [],
                "enforced_features": [],
                "strongest_cluster_feature": None,
            },
        )

    raw_set = set(raw_present)

    share_present = [
        column for column in CLUSTER_SHARE_FEATURES if column in available_set
    ]

    strongest_candidates = share_present if share_present else raw_present
    strongest_cluster_feature: str | None = None

    if strongest_candidates:
        strongest_cluster_feature = max(
            strongest_candidates,
            key=lambda column: (
                float(scores.get(column, 0.0)),
                float(permutation_importances.get(column, 0.0)),
            ),
        )

    enforced_features: list[str] = []

    for base_feature in CLUSTER_REPLACEMENT_BASE_FEATURES:
        if base_feature in available_set:
            enforced_features.append(base_feature)

    if strongest_cluster_feature is not None:
        enforced_features.append(strongest_cluster_feature)

    raw_readded = (
        strongest_cluster_feature
        if strongest_cluster_feature in raw_set
        else None
    )

    selected_without_raw = [
        column for column in selected if column not in raw_set
    ]

    gated_features = list(dict.fromkeys(selected_without_raw + enforced_features))

    dropped_raw_features = [
        column
        for column in selected
        if column in raw_set and column != raw_readded
    ]

    report: dict[str, Any] = {
        "triggered": True,
        "reason": "all_raw_cluster_features_non_positive_permutation_importance",
        "raw_cluster_features": raw_present,
        "raw_permutation_importance": raw_permutation_importance,
        "dropped_raw_features": dropped_raw_features,
        "enforced_features": enforced_features,
        "strongest_cluster_feature": strongest_cluster_feature,
        "retained_raw_cluster_features": (
            [raw_readded] if raw_readded is not None else []
        ),
    }

    return ClusterGateResult(
        selected_features=gated_features,
        report=report,
    )

def evaluate_validation_rmse(
    x_train: pl.DataFrame,
    y_train: pl.Series,
    x_val: pl.DataFrame,
    y_val: pl.Series,
    feature_columns: Sequence[str],
    random_state: int = 42,
    n_jobs: int | None = -1,
) -> float:
    feature_columns = list(feature_columns)

    if not feature_columns:
        return float("inf")

    preprocessor = build_preprocessor(feature_columns)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=random_state,
                    n_jobs=n_jobs,
                ),
            ),
        ]
    )

    pipeline.fit(x_train[feature_columns], y_train)

    predictions = pipeline.predict(x_val[feature_columns])

    return float(np.sqrt(mean_squared_error(y_val, predictions)))

def enforce_feature_cap(
    feature_columns: Sequence[str],
    scores: Mapping[str, float],
    feature_cap: int,
    required_columns: Sequence[str] = REQUIRED_FEATURES,
) -> list[str]:
    feature_columns = list(feature_columns)

    required_present = [
        column for column in required_columns if column in feature_columns
    ]

    if feature_cap <= 0:
        return []

    if len(required_present) >= feature_cap:
        return required_present[:feature_cap]

    remaining = [
        column for column in feature_columns if column not in required_present
    ]

    
    remaining = sorted(
        remaining,
        key=lambda column: float(scores.get(column, 0.0)),
        reverse=True
    )

    remaining_slots = feature_cap - len(required_present)

    return required_present + remaining[:remaining_slots]

def run_family_ablation(
    x_train: pl.DataFrame,
    y_train: pl.Series,
    x_val: pl.DataFrame,
    y_val: pl.Series,
    feature_columns: Sequence[str],
    scores: Mapping[str, float],
    feature_cap: int,
    required_columns: Sequence[str] = REQUIRED_FEATURES,
    min_relative_improvement: float = 0.005,
    random_state: int = 42,
    n_jobs: int | None = -1
) -> tuple[list[str], pl.DataFrame, float, float, list[str]]:
    feature_columns = list(feature_columns)

    if not feature_columns:
        empty_report = pl.DataFrame(
            schema=[
                "family",
                "feature_count_without_family",
                "baseline_rmse",
                "ablation_rmse",
                "relative_rmse_change",
                "protected",
                "evaluated",
                "dropped",
            ]
        )

        return [], empty_report, float("inf"), float("inf"), []

    families = sorted(
        {infer_feature_family(column) for column in feature_columns}
    )

    protected_families = {
        infer_feature_family(column)
        for column in required_columns
        if column in feature_columns
    }

    baseline_rmse = evaluate_validation_rmse(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        feature_columns=feature_columns,
        random_state=random_state,
        n_jobs=n_jobs
    )

    records: list[dict[str, Any]] = []
    dropped_families: list[str] = []

    for family in families:
        candidate_features = [
            column
            for column in feature_columns
            if infer_feature_family(column) != family
        ]

        protected = family in protected_families

        if not candidate_features:
            records.append(
                {
                    "family": family,
                    "feature_count_without_family": 0,
                    "baseline_rmse": baseline_rmse,
                    "ablation_rmse": np.nan,
                    "relative_rmse_change": np.nan,
                    "protected": protected,
                    "evaluated": False,
                    "dropped": False,
                }
            )
            continue
            
        if protected:
            ablation_rmse = baseline_rmse
            relative_change = 0.0
            evaluated = False
            dropped = False
        else:
            ablation_rmse = evaluate_validation_rmse(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                feature_columns=candidate_features,
                random_state=random_state,
                n_jobs=n_jobs,
            )

            if baseline_rmse > 0.0:
                relative_change = (baseline_rmse - ablation_rmse) / baseline_rmse
            else:
                relative_change = 0.0

            evaluated = True
            dropped = relative_change >= min_relative_improvement

            if dropped:
                dropped_families.append(family)

        records.append(
            {
                "family": family,
                "feature_count_without_family": len(candidate_features),
                "baseline_rmse": baseline_rmse,
                "ablation_rmse": ablation_rmse,
                "relative_rmse_change": relative_change,
                "protected": protected,
                "evaluated": evaluated,
                "dropped": dropped,
            }
        )

    retained_features = [
        column
        for column in feature_columns
        if infer_feature_family(column) not in dropped_families
    ]
           
    if not retained_features:
       retained_features = feature_columns
       dropped_families = []

    final_features = enforce_feature_cap(
        feature_columns=retained_features,
        scores=scores,
        feature_cap=feature_cap,
        required_columns=required_columns,
    )

    final_rmse = evaluate_validation_rmse(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        feature_columns=final_features,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    report = pl.DataFrame(records)

    return final_features, report, final_rmse, baseline_rmse, dropped_families
        