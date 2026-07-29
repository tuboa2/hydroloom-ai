from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

from ..preprocessing.feature_groups import (
    RAW_CLUSTER_FEATURES,
    REQUIRED_FEATURES,
    infer_feature_family,
)
from ..preprocessing.pipeline_factory import build_preprocessor

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
        column for column in RAW_CLUSTER_FEATURES
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
    