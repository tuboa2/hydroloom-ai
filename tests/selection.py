from __future__ import annotations

import numpy as np
import pandas as pd
from services.wqi_predictor.pipeline.phase3 import apply_south_upfront_exclusions
from services.wqi_predictor.preprocessing.feature_groups import (
    REQUIRED_FEATURES,
    assign_numeric_transformer,
    infer_feature_family,
)
from services.wqi_predictor.preprocessing.pipeline_factory import build_preprocessor

from services.wqi_predictor.selection.ablation import (
    apply_cluster_feature_gate,
    enforce_feature_cap,
    run_family_ablation,
)
from services.wqi_predictor.selection.cv import make_time_series_cv
from services.wqi_predictor.selection.screening import stage1_screening
from services.wqi_predictor.selection.stability import run_stability_selection


def _make_phase3_frame(n: int = 240, hemisphere: str = "north") -> pd.DataFrame:
    rng = np.random.default_rng(42)

    day_index = np.arange(n)
    day_of_year = day_index % 365

    wqi_lag1 = rng.normal(60.0, 10.0, size=n)
    wqi_lag7 = wqi_lag1 + rng.normal(0.0, 1.0, size=n)
    seasonal_wqi_mean_doy = 60.0 + 5.0 * np.sin(2.0 * np.pi * day_of_year / 365.0)

    daily_rainfall_mm = rng.exponential(2.0, size=n)
    daily_max_temp_celsius = rng.normal(20.0, 5.0, size=n)

    cluster_heavy = rng.uniform(100.0, 1000.0, size=n)
    cluster_conservation = rng.uniform(100.0, 1000.0, size=n)
    total_cluster_demand = cluster_heavy + cluster_conservation

    data = {
        "daily_max_temp_celsius": daily_max_temp_celsius,
        "temp_anomaly_celsius": rng.normal(0.0, 2.0, size=n),
        "cumulative_heat_index": rng.normal(10.0, 3.0, size=n),
        "daily_rainfall_mm": daily_rainfall_mm,
        "rolling_7d_rainfall_mm": rng.exponential(1.0, size=n),
        "cumulative_storm_rainfall_mm": rng.exponential(3.0, size=n),
        "antecedent_moisture_condition": rng.uniform(0.0, 1.0, size=n),
        "consecutive_dry_days": rng.integers(0, 10, size=n),
        "daily_runoff_volume_m3": rng.exponential(5.0, size=n),
        "total_suspended_solids_mg_L": rng.exponential(2.0, size=n),
        "nutrient_load_index": rng.exponential(1.0, size=n),
        "heat_x_nutrient_synergy": rng.exponential(1.0, size=n),
        "cluster_heavy_users_daily_mean_liters": cluster_heavy,
        "cluster_conservationists_daily_mean_liters": cluster_conservation,
        "total_cluster_demand": total_cluster_demand,
        "heavy_share": cluster_heavy / (total_cluster_demand + 1e-6),
        "conservationist_share": cluster_conservation / (total_cluster_demand + 1e-6),
        "cluster_demand_pc1": rng.normal(0.0, 1.0, size=n),
        "wqi_lag1": wqi_lag1,
        "wqi_lag7": wqi_lag7,
        "seasonal_wqi_mean_doy": seasonal_wqi_mean_doy,
        "day_of_year_sin": np.sin(2.0 * np.pi * day_of_year / 365.0),
        "day_of_year_cos": np.cos(2.0 * np.pi * day_of_year / 365.0),
        "rainfall_x_antecedent_moisture": daily_rainfall_mm * rng.uniform(0.0, 1.0, size=n),
    }

    if hemisphere == "north":
        data["season_label"] = rng.choice(
            ["Winter", "Spring", "Summer", "Autumn"],
            size=n,
        )
        data["tiered_pricing_regime"] = rng.integers(0, 3, size=n)
        data["watering_ban_active"] = rng.integers(0, 2, size=n)

    target = (
        0.65 * wqi_lag1
        + 0.10 * seasonal_wqi_mean_doy
        - 0.35 * daily_rainfall_mm
        + rng.normal(0.0, 1.0, size=n)
    )

    data["water_quality_index"] = target

    return pd.DataFrame(data)


def test_feature_family_inference() -> None:
    assert infer_feature_family("wqi_lag1") == "target_lags"
    assert infer_feature_family("wqi_roll_mean_7") == "target_lags"
    assert infer_feature_family("seasonal_wqi_mean_doy") == "seasonal_target"
    assert infer_feature_family("daily_rainfall_mm_lag1") == "exogenous_derived"
    assert infer_feature_family("total_cluster_demand") == "cluster_aggregates"
    assert infer_feature_family("cluster_heavy_users_daily_mean_liters") == "cluster_raw"
    assert infer_feature_family("season_label") == "policy"
    assert infer_feature_family("rainfall_x_antecedent_moisture") == "interactions"
    assert infer_feature_family("daily_max_temp_celsius") == "raw_environment"


def test_numeric_transformer_assignment() -> None:
    assert assign_numeric_transformer("daily_max_temp_celsius") == "standard"
    assert assign_numeric_transformer("cumulative_heat_index_lag7") == "standard"
    assert assign_numeric_transformer("daily_rainfall_mm") == "power"
    assert assign_numeric_transformer("daily_runoff_volume_m3_lag1") == "power"
    assert assign_numeric_transformer("total_cluster_demand") == "robust"
    assert assign_numeric_transformer("wqi_lag1") == "robust"


def test_preprocessor_transforms_without_nulls() -> None:
    frame = _make_phase3_frame()
    target = frame.pop("water_quality_index")

    features = list(frame.columns)

    preprocessor = build_preprocessor(features)
    transformed = preprocessor.fit_transform(frame[features], target)

    assert transformed.shape[0] == frame.shape[0]
    assert not transformed.isna().any().any()


def test_stage1_screening_removes_bad_columns() -> None:
    frame = _make_phase3_frame()
    frame.pop("water_quality_index")

    frame["zero_variance"] = 1.0
    frame["high_missing"] = np.nan
    frame["rain_duplicate"] = frame["daily_rainfall_mm"]
    frame["antecedent_moisture_condition_lead5"] = 0.0

    features = list(frame.columns)

    retained, report = stage1_screening(frame, features)

    assert "zero_variance" not in retained
    assert "high_missing" not in retained
    assert "antecedent_moisture_condition_lead5" not in retained

    assert not ("daily_rainfall_mm" in retained and "rain_duplicate" in retained)

    assert report["counts"]["input"] == len(features)
    assert report["counts"]["retained"] == len(retained)


def test_time_series_cv_preserves_temporal_order_and_gap() -> None:
    x = pd.DataFrame({"feature": np.arange(120)})

    cv = make_time_series_cv(n_splits=3, gap=7)

    for train_idx, val_idx in cv.split(x):
        assert train_idx.max() < val_idx.min()
        assert val_idx.min() - train_idx.max() > 7


def test_stability_selection_returns_subset_and_scores() -> None:
    frame = _make_phase3_frame(n=180)
    target = frame.pop("water_quality_index")

    features = [
        "wqi_lag1",
        "wqi_lag7",
        "seasonal_wqi_mean_doy",
        "daily_rainfall_mm",
        "daily_max_temp_celsius",
        "total_cluster_demand",
        "rainfall_x_antecedent_moisture",
        "season_label",
        "tiered_pricing_regime",
    ]

    selected, report, scores = run_stability_selection(
        x_train=frame,
        y_train=target,
        feature_columns=features,
        n_splits=2,
        gap=0,
        seeds=(42,),
        min_stability=0.0,
        run_score_threshold=0.0,
        candidate_cap=5,
        min_fallback_features=3,
        n_repeats=2,
        n_jobs=None,
    )

    assert set(selected).issubset(set(features))
    assert len(selected) <= 5

    assert {"feature", "stability", "ranking_score"}.issubset(report.columns)
    assert set(scores.keys()) == set(features)


def test_stability_selection_uses_multi_metric_scoring() -> None:
    frame = _make_phase3_frame(n=180)
    target = frame.pop("water_quality_index")

    features = [
        "wqi_lag1",
        "wqi_lag7",
        "seasonal_wqi_mean_doy",
        "daily_rainfall_mm",
        "daily_max_temp_celsius",
        "total_cluster_demand",
        "cluster_demand_pc1",
    ]

    _, report, _ = run_stability_selection(
        x_train=frame,
        y_train=target,
        feature_columns=features,
        n_splits=2,
        gap=0,
        seeds=(42,),
        min_stability=0.0,
        run_score_threshold=0.0,
        candidate_cap=None,
        min_fallback_features=3,
        n_repeats=2,
        n_jobs=None,
    )

    expected_columns = {
        "feature",
        "stability",
        "mean_score",
        "mean_perm_importance",
        "mean_split_importance",
        "ranking_score",
    }

    assert expected_columns.issubset(report.columns)

    assert report["mean_split_importance"].fillna(0.0).ge(0.0).all()
    assert report["mean_score"].fillna(0.0).ge(0.0).all()

    assert (report["mean_split_importance"].fillna(0.0) > 0.0).any()


def test_cluster_gate_replaces_raw_clusters_when_all_non_positive() -> None:
    selected_features = [
        "cluster_heavy_users_daily_mean_liters",
        "cluster_conservationists_daily_mean_liters",
        "daily_rainfall_mm",
        "heavy_share",
        "total_cluster_demand",
        "cluster_demand_pc1",
    ]

    available_features = selected_features + [
        "cluster_outdoor_landscape_daily_mean_liters",
        "cluster_standard_consumers_daily_mean_liters",
        "conservationist_share",
        "outdoor_share",
        "standard_share",
    ]

    permutation_importances = {
        "cluster_heavy_users_daily_mean_liters": -0.01,
        "cluster_conservationists_daily_mean_liters": 0.0,
        "cluster_outdoor_landscape_daily_mean_liters": -0.02,
        "cluster_standard_consumers_daily_mean_liters": 0.0,
        "heavy_share": 0.20,
        "conservationist_share": 0.10,
        "outdoor_share": 0.05,
        "standard_share": 0.01,
        "total_cluster_demand": 0.15,
        "cluster_demand_pc1": 0.12,
        "daily_rainfall_mm": 0.30,
    }

    scores = {
        "cluster_heavy_users_daily_mean_liters": 0.20,
        "cluster_conservationists_daily_mean_liters": 0.18,
        "cluster_outdoor_landscape_daily_mean_liters": 0.16,
        "cluster_standard_consumers_daily_mean_liters": 0.14,
        "heavy_share": 0.90,
        "conservationist_share": 0.70,
        "outdoor_share": 0.60,
        "standard_share": 0.50,
        "total_cluster_demand": 0.85,
        "cluster_demand_pc1": 0.80,
        "daily_rainfall_mm": 0.95,
    }

    result = apply_cluster_feature_gate(
        selected_features=selected_features,
        available_features=available_features,
        permutation_importances=permutation_importances,
        scores=scores,
    )

    assert result.report["triggered"] is True

    assert "cluster_heavy_users_daily_mean_liters" not in result.selected_features
    assert "cluster_conservationists_daily_mean_liters" not in result.selected_features

    assert "total_cluster_demand" in result.selected_features
    assert "cluster_demand_pc1" in result.selected_features
    assert "heavy_share" in result.selected_features
    assert "daily_rainfall_mm" in result.selected_features


def test_cluster_gate_does_not_trigger_when_raw_cluster_importance_positive() -> None:
    selected_features = [
        "cluster_heavy_users_daily_mean_liters",
        "cluster_conservationists_daily_mean_liters",
        "total_cluster_demand",
        "cluster_demand_pc1",
    ]

    available_features = selected_features + [
        "cluster_outdoor_landscape_daily_mean_liters",
        "cluster_standard_consumers_daily_mean_liters",
        "heavy_share",
    ]

    permutation_importances = {
        "cluster_heavy_users_daily_mean_liters": 0.05,
        "cluster_conservationists_daily_mean_liters": 0.0,
        "cluster_outdoor_landscape_daily_mean_liters": -0.01,
        "cluster_standard_consumers_daily_mean_liters": 0.0,
    }

    scores = {feature: 0.5 for feature in available_features}

    result = apply_cluster_feature_gate(
        selected_features=selected_features,
        available_features=available_features,
        permutation_importances=permutation_importances,
        scores=scores,
    )

    assert result.report["triggered"] is False
    assert result.selected_features == selected_features


def test_cluster_gate_selects_strongest_raw_cluster_when_no_shares_exist() -> None:
    selected_features = [
        "cluster_heavy_users_daily_mean_liters",
        "cluster_conservationists_daily_mean_liters",
        "cluster_outdoor_landscape_daily_mean_liters",
        "cluster_standard_consumers_daily_mean_liters",
    ]

    available_features = selected_features + [
        "total_cluster_demand",
        "cluster_demand_pc1",
    ]

    permutation_importances = {
        "cluster_heavy_users_daily_mean_liters": 0.0,
        "cluster_conservationists_daily_mean_liters": -0.01,
        "cluster_outdoor_landscape_daily_mean_liters": -0.02,
        "cluster_standard_consumers_daily_mean_liters": -0.03,
        "total_cluster_demand": 0.10,
        "cluster_demand_pc1": 0.08,
    }

    scores = {
        "cluster_heavy_users_daily_mean_liters": 0.90,
        "cluster_conservationists_daily_mean_liters": 0.70,
        "cluster_outdoor_landscape_daily_mean_liters": 0.60,
        "cluster_standard_consumers_daily_mean_liters": 0.50,
        "total_cluster_demand": 0.85,
        "cluster_demand_pc1": 0.80,
    }

    result = apply_cluster_feature_gate(
        selected_features=selected_features,
        available_features=available_features,
        permutation_importances=permutation_importances,
        scores=scores,
    )

    assert result.report["triggered"] is True

    assert "cluster_heavy_users_daily_mean_liters" in result.selected_features
    assert "cluster_conservationists_daily_mean_liters" not in result.selected_features
    assert "cluster_outdoor_landscape_daily_mean_liters" not in result.selected_features
    assert "cluster_standard_consumers_daily_mean_liters" not in result.selected_features

    assert "total_cluster_demand" in result.selected_features
    assert "cluster_demand_pc1" in result.selected_features


def test_enforce_feature_cap_keeps_required_features() -> None:
    features = [
        "wqi_lag1",
        "wqi_lag7",
        "seasonal_wqi_mean_doy",
        "daily_rainfall_mm",
        "daily_max_temp_celsius",
        "total_cluster_demand",
    ]

    scores = {
        "wqi_lag1": 0.95,
        "wqi_lag7": 0.90,
        "seasonal_wqi_mean_doy": 0.80,
        "daily_rainfall_mm": 0.70,
        "daily_max_temp_celsius": 0.60,
        "total_cluster_demand": 0.50,
    }

    selected = enforce_feature_cap(
        feature_columns=features,
        scores=scores,
        feature_cap=4,
        required_columns=REQUIRED_FEATURES,
    )

    assert len(selected) == 4
    assert "wqi_lag1" in selected
    assert "wqi_lag7" in selected
    assert "seasonal_wqi_mean_doy" in selected


def test_family_ablation_returns_capped_final_features() -> None:
    frame = _make_phase3_frame(n=200)
    target = frame.pop("water_quality_index")

    train = frame.iloc[:150].reset_index(drop=True)
    validation = frame.iloc[150:].reset_index(drop=True)

    y_train = target.iloc[:150].reset_index(drop=True)
    y_validation = target.iloc[150:].reset_index(drop=True)

    features = [
        "wqi_lag1",
        "wqi_lag7",
        "seasonal_wqi_mean_doy",
        "daily_rainfall_mm",
        "daily_max_temp_celsius",
        "total_cluster_demand",
        "rainfall_x_antecedent_moisture",
    ]

    scores = {
        "wqi_lag1": 0.95,
        "wqi_lag7": 0.90,
        "seasonal_wqi_mean_doy": 0.80,
        "daily_rainfall_mm": 0.70,
        "daily_max_temp_celsius": 0.60,
        "total_cluster_demand": 0.50,
        "rainfall_x_antecedent_moisture": 0.40,
    }

    final_features, report, final_rmse, baseline_rmse, dropped_families = run_family_ablation(
        x_train=train,
        y_train=y_train,
        x_validation=validation,
        y_validation=y_validation,
        feature_columns=features,
        scores=scores,
        feature_cap=4,
        required_columns=REQUIRED_FEATURES,
        min_relative_improvement=0.0,
        random_state=42,
        n_jobs=None,
    )

    assert len(final_features) <= 4
    assert isinstance(final_rmse, float)
    assert isinstance(baseline_rmse, float)
    assert isinstance(dropped_families, list)
    assert not report.empty


def test_south_upfront_exclusions_remove_default_columns() -> None:
    features = [
        "daily_rainfall_mm",
        "season_label",
        "tiered_pricing_regime",
        "watering_ban_active",
        "demand_x_runoff_pressure",
        "drought_x_heat_stress",
        "cluster_demand_pc1",
    ]

    retained, dropped = apply_south_upfront_exclusions(
        feature_columns=features,
        hemisphere="south",
        enabled_features=(),
    )

    assert "season_label" not in retained
    assert "tiered_pricing_regime" not in retained
    assert "watering_ban_active" not in retained
    assert "demand_x_runoff_pressure" not in retained
    assert "drought_x_heat_stress" not in retained

    assert "daily_rainfall_mm" in retained
    assert "cluster_demand_pc1" in retained

    assert set(dropped) == {
        "season_label",
        "tiered_pricing_regime",
        "watering_ban_active",
        "demand_x_runoff_pressure",
        "drought_x_heat_stress",
    }


def test_south_upfront_exclusions_allow_explicit_enablement() -> None:
    features = [
        "daily_rainfall_mm",
        "season_label",
        "tiered_pricing_regime",
        "watering_ban_active",
        "demand_x_runoff_pressure",
        "drought_x_heat_stress",
    ]

    retained, dropped = apply_south_upfront_exclusions(
        feature_columns=features,
        hemisphere="south",
        enabled_features=("season_label",),
    )

    assert "season_label" in retained
    assert "tiered_pricing_regime" not in retained
    assert "watering_ban_active" not in retained
    assert "demand_x_runoff_pressure" not in retained
    assert "drought_x_heat_stress" not in retained

    assert "season_label" not in dropped


def test_south_upfront_exclusions_do_not_affect_north() -> None:
    features = [
        "daily_rainfall_mm",
        "season_label",
        "tiered_pricing_regime",
        "watering_ban_active",
        "demand_x_runoff_pressure",
        "drought_x_heat_stress",
    ]

    retained, dropped = apply_south_upfront_exclusions(
        feature_columns=features,
        hemisphere="north",
        enabled_features=(),
    )

    assert retained == features
    assert dropped == []
