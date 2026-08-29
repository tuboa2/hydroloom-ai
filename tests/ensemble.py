from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import Ridge

from services.wqi_predictor.ensemble.blending import (
    arithmetic_mean_predictions,
    mean_median_blend_predictions,
    select_mean_median_lambda,
    trimmed_mean_predictions,
)
from services.wqi_predictor.ensemble.diagnostics import (
    compute_comprehensive_metrics,
    compute_decile_metrics,
    compute_monthly_metrics,
    compute_zone_metrics,
)
from services.wqi_predictor.ensemble.integrity import (
    FORBIDDEN_FEATURES,
    compute_canonical_json_hash,
    compute_sha256,
    verify_candidate_integrity,
)
from services.wqi_predictor.ensemble.leakage_audit import LeakageAuditor
from services.wqi_predictor.ensemble.level0_builder import (
    MANDATORY_SEEDS,
    build_level0_specs,
    build_model_specs,
    build_seed_params,
    instantiate_model,
)
from services.wqi_predictor.ensemble.oof_generator import generate_oof_matrix
from services.wqi_predictor.ensemble.postprocess import (
    apply_postprocessing,
    select_bounds_calibration,
)
from services.wqi_predictor.ensemble.refit import execute_production_refit
from services.wqi_predictor.ensemble.residual_corrector import (
    ResidualAutoregressiveCorrector,
    compute_ljung_box_stat,
    evaluate_residual_activation_gate,
)
from services.wqi_predictor.ensemble.stacking import (
    fit_elasticnet_stack,
    fit_ridge_stack,
    predict_stack,
)
from services.wqi_predictor.ensemble.state_machine import (
    ORDERED_STATES,
    Phase5StateMachine,
)
from services.wqi_predictor.ensemble.test_guard import TestSetAccessGuard
from services.wqi_predictor.ensemble.types import (
    ArtifactIntegrityError,
    EnsembleStrategy,
    FinalFrozenState,
    GateViolationError,
    Hemisphere,
    LeakageError,
    Level0ModelSpec,
    ModelSpec,
    OOFGenerationError,
    PostprocessingConfig,
    ResidualCorrectionError,
    TestSetAccessError,
    WeightOptimizationError,
    WQIZone,
    WQIState,
)
from services.wqi_predictor.ensemble.weighted_blend import (
    fit_constrained_weights,
    predict_constrained_weights,
)
from services.wqi_predictor.pipeline.ensemble import Phase5EnsemblePipeline


# ==============================================================================
# Helper fixtures & synthetic generators for Kaggle CPU environment testing
# ==============================================================================


def _generate_synthetic_splits(
    hemisphere: str = "north", n_features: int = 10
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, list[str]]:
    rng = np.random.default_rng(42)
    feature_names = [f"feat_{i}" for i in range(n_features)]

    # 1095 train days (Years 0-2), 365 val days (Year 3), 365 test days (Year 4)
    total_days = 1825

    base_signal = np.sin(np.linspace(0, 10 * np.pi, total_days)) * 25.0 + 50.0
    noise = rng.normal(loc=0.0, scale=3.0, size=total_days)
    wqi = np.clip(base_signal + noise, 5.0, 95.0)

    data: dict[str, Any] = {
        "day_index": np.arange(total_days, dtype=np.int64),
        "year_index": (np.arange(total_days) // 365).astype(np.int64),
        "water_quality_index": wqi.astype(np.float64),
    }

    for f in feature_names:
        data[f] = (wqi + rng.normal(loc=0.0, scale=4.0, size=total_days)).astype(np.float64)

    full_df = pl.DataFrame(data)

    train_df = full_df.filter(pl.col("day_index") < 1095)
    val_df = full_df.filter((pl.col("day_index") >= 1095) & (pl.col("day_index") < 1460))
    test_df = full_df.filter(pl.col("day_index") >= 1460)

    return train_df, val_df, test_df, feature_names


def _create_synthetic_phase4_registry(
    artifacts_root: Path, hemisphere: Hemisphere, feature_names: list[str]
) -> dict[str, Any]:
    hemi_model_dir = artifacts_root / "model" / hemisphere.value
    hemi_model_dir.mkdir(parents=True, exist_ok=True)

    candidates = [
        {
            "name": f"model-{hemisphere.value}-catboost-rmse",
            "study_name": f"model-{hemisphere.value}-catboost-rmse",
            "model_family": "catboost",
            "loss_name": "rmse",
            "best_iteration_count": 5,
            "passed_baseline_gate": True,
            "leakage_detected": False,
            "selected_features": feature_names,
            "hyperparameters": {
                "depth": 3,
                "learning_rate": 0.1,
                "iterations": 5,
                "verbose": False,
            },
        },
        {
            "name": f"model-{hemisphere.value}-lightgbm-rmse",
            "study_name": f"model-{hemisphere.value}-lightgbm-rmse",
            "model_family": "lightgbm",
            "loss_name": "rmse",
            "best_iteration_count": 5,
            "passed_baseline_gate": True,
            "leakage_detected": False,
            "selected_features": feature_names,
            "hyperparameters": {
                "num_leaves": 7,
                "learning_rate": 0.1,
                "n_estimators": 5,
                "verbosity": -1,
            },
        },
        {
            "name": f"model-{hemisphere.value}-ridge-rmse",
            "study_name": f"model-{hemisphere.value}-ridge-rmse",
            "model_family": "ridge",
            "loss_name": "rmse",
            "best_iteration_count": 1,
            "passed_baseline_gate": True,
            "leakage_detected": False,
            "selected_features": feature_names,
            "hyperparameters": {
                "alpha": 1.0,
            },
        },
    ]

    registry_payload = {
        "hemisphere": hemisphere.value,
        "feature_count": len(feature_names),
        "feature_set_hash": "synth_feat_hash_12345",
        "passed_candidates": candidates,
        "candidates": candidates,
    }

    with open(hemi_model_dir / "candidate_registry.json", "w", encoding="utf-8") as f:
        json.dump(registry_payload, f, indent=2)

    sel_dir = artifacts_root / "selection" / hemisphere.value
    sel_dir.mkdir(parents=True, exist_ok=True)
    with open(sel_dir / "selected_features.json", "w", encoding="utf-8") as f:
        json.dump({"final_features": feature_names}, f, indent=2)

    return registry_payload


# ==============================================================================
# 1. Constrained Weighted Average Optimizer Tests (Strategy A)
# ==============================================================================


class TestConstrainedWeightedBlend:
    def test_slsqp_weights_sum_to_one_and_non_negative(self):
        rng = np.random.default_rng(42)
        n_samples = 500
        n_models = 6

        # Generate correlated candidate predictions
        y_true = rng.normal(50.0, 10.0, size=n_samples)
        z_oof = np.column_stack(
            [y_true + rng.normal(0.0, scale=i + 1.0, size=n_samples) for i in range(n_models)]
        )
        columns = [f"m_{i}" for i in range(n_models)]

        res = fit_constrained_weights(z_oof=z_oof, y_oof=y_true, columns=columns)

        assert res["strategy"] == EnsembleStrategy.CONSTRAINED_WEIGHTED.value
        assert res["converged"] is True

        weights = np.array([res["weights"][c] for c in columns])
        assert np.all(weights >= -1e-8)
        assert np.isclose(np.sum(weights), 1.0, atol=1e-5)

        # Model 0 has smallest noise (scale=1.0) so it should have highest weight
        assert weights[0] == np.max(weights)

    def test_predict_constrained_weights(self):
        z_mat = np.array([[10.0, 20.0], [30.0, 40.0]])
        weights_map = {"m1": 0.75, "m2": 0.25}
        preds = predict_constrained_weights(z_mat, weights_map, ["m1", "m2"])
        expected = np.array([12.5, 32.5])
        np.testing.assert_allclose(preds, expected)

    def test_slsqp_zero_variance_fallback(self):
        # All models predict identical constants (zero variance)
        z_oof = np.full((100, 4), fill_value=42.0)
        y_true = np.full(100, fill_value=42.0)
        cols = ["m0", "m1", "m2", "m3"]

        res = fit_constrained_weights(z_oof=z_oof, y_oof=y_true, columns=cols)
        assert res["fallback"] is True
        for col in cols:
            assert res["weights"][col] == 0.25

    def test_dimension_mismatch_error(self):
        z_oof = np.zeros((100, 3))
        y_true = np.zeros(80)
        with pytest.raises(WeightOptimizationError):
            fit_constrained_weights(z_oof=z_oof, y_oof=y_true, columns=["m1", "m2", "m3"])


# ==============================================================================
# 2. Stacking Meta-Learner Tests (Strategies B1 & B2)
# ==============================================================================


class TestStackingMetaLearners:
    def test_ridge_stack_cv_fit_and_predict(self):
        rng = np.random.default_rng(42)
        n = 300
        y = rng.normal(50.0, 5.0, size=n)
        z = np.column_stack([y + rng.normal(0, 1, size=n), y + rng.normal(0, 2, size=n)])
        cols = ["m0", "m1"]

        ridge_model, meta = fit_ridge_stack(z_oof=z, y_oof=y, columns=cols, n_splits=3)
        assert meta["strategy"] == EnsembleStrategy.RIDGE_STACK.value
        assert "selected_alpha" in meta
        assert len(meta["coefficients"]) == 2

        preds = predict_stack(ridge_model, z[:10])
        assert preds.shape == (10,)
        assert np.all(np.isfinite(preds))

    def test_elasticnet_stack_cv_fit_and_predict(self):
        rng = np.random.default_rng(42)
        n = 200
        y = rng.normal(50.0, 5.0, size=n)
        z = np.column_stack([y + rng.normal(0, 1, size=n), y + rng.normal(0, 2, size=n)])
        cols = ["m0", "m1"]

        enet_model, meta = fit_elasticnet_stack(z_oof=z, y_oof=y, columns=cols, n_splits=3)
        assert meta["strategy"] == EnsembleStrategy.ELASTICNET_STACK.value
        assert "selected_l1_ratio" in meta

        preds = predict_stack(enet_model, z[:5])
        assert preds.shape == (5,)

    def test_extrapolation_warning_on_large_coefficients(self):
        meta = {
            "strategy": EnsembleStrategy.RIDGE_STACK.value,
            "max_coefficient_magnitude": 15.0,
            "extrapolation_warning": True,
        }
        assert meta["extrapolation_warning"] is True


# ==============================================================================
# 3. Simple & Trimmed Mean & Mean-Median Blend Tests (Strategies C & D)
# ==============================================================================


class TestSimpleAndTrimmedMeanBlending:
    def test_arithmetic_mean_predictions(self):
        z = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
        res = arithmetic_mean_predictions(z)
        np.testing.assert_allclose(res, np.array([20.0, 50.0]))

    def test_trimmed_mean_10_percent_cut(self):
        # 10 models: 10% trim removes k=1 from top and bottom
        z = np.tile(np.arange(1.0, 11.0), (3, 1))  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        res = trimmed_mean_predictions(z, trim_fraction=0.10)
        # Remaining: 2..9, average = 5.5
        np.testing.assert_allclose(res, np.array([5.5, 5.5, 5.5]))

    def test_trimmed_mean_fallback_few_models(self):
        # Fewer than 5 models -> falls back to arithmetic mean
        z = np.array([[10.0, 20.0, 30.0]])
        res = trimmed_mean_predictions(z, trim_fraction=0.10)
        np.testing.assert_allclose(res, np.array([20.0]))

    def test_mean_median_blend_and_lambda_selection(self):
        z = np.array([[10.0, 10.0, 100.0]])  # mean = 40.0, median = 10.0
        # lam = 0.5 -> 0.5*40 + 0.5*10 = 25.0
        preds = mean_median_blend_predictions(z, lam=0.5)
        np.testing.assert_allclose(preds, np.array([25.0]))

        y_true = np.array([10.0])
        best_lam, best_rmse = select_mean_median_lambda(z_val=z, y_val=y_true)
        # lam = 0.0 gives median (10.0), matching y_true perfectly (rmse = 0.0)
        assert best_lam == 0.0
        assert np.isclose(best_rmse, 0.0)


# ==============================================================================
# 4. Residual Autoregressive Corrector Tests (§5)
# ==============================================================================


class TestResidualAutoregressiveCorrector:
    def test_residual_dataset_lag_construction(self):
        corrector = ResidualAutoregressiveCorrector()
        y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
        base_preds = np.array([8.0, 18.0, 28.0, 38.0, 48.0, 58.0, 68.0, 78.0, 88.0, 98.0])

        x_res, residuals = corrector.build_residual_dataset(y_true=y_true, base_preds=base_preds)
        assert len(residuals) == 10
        assert np.all(residuals == 2.0)
        assert x_res.shape == (10, 3)  # lag1, lag7, base_yhat

        # Lag 1 at index 1 should be residuals[0] = 2.0
        assert x_res[1, 0] == 2.0
        # Lag 7 at index 7 should be residuals[0] = 2.0
        assert x_res[7, 1] == 2.0

    def test_sequential_inference_with_effective_clip(self):
        corrector = ResidualAutoregressiveCorrector(alpha=1.0, max_clip=5.0)
        corrector.model = Ridge(alpha=1.0)
        # Fit dummy model
        x_dummy = np.zeros((20, 3))
        y_dummy = np.ones(20) * 10.0  # large residual
        corrector.model.fit(x_dummy, y_dummy)
        corrector.effective_clip = 3.0

        base_test = np.array([50.0, 50.0, 50.0])
        y_train_end = [2.0] * 14

        corrected = corrector.predict_sequential(
            base_test_preds=base_test, y_train_end_residuals=y_train_end
        )
        assert len(corrected) == 3
        # Ensure correction did not exceed effective clip of 3.0
        for c in corrected:
            assert c <= 53.0
            assert c >= 47.0

    def test_ljung_box_autocorrelation_statistic(self):
        # Autocorrelated signal
        residuals = np.sin(np.linspace(0, 4 * np.pi, 100))
        lb = compute_ljung_box_stat(residuals)
        assert lb["q_stat_sum"] > 0.0
        assert len(lb["per_lag_q"]) == 4

    def test_residual_activation_gate_logic(self):
        rng = np.random.default_rng(42)
        n = 50
        y_val = np.sin(np.linspace(0, 4 * np.pi, n)) * 20.0 + 50.0
        # Base predictions have persistent lag autocorrelation and higher RMSE
        base_val = y_val - 5.0 + np.sin(np.linspace(0, 4 * np.pi, n)) * 3.0
        # Corrected predictions eliminate lag bias and error
        corr_val = y_val + rng.normal(0, 0.1, n)

        res_gate = evaluate_residual_activation_gate(
            y_val=y_val,
            base_val_preds=base_val,
            corrected_val_preds=corr_val,
            pre_max_ae=10.0,
        )
        assert res_gate.enabled is True
        assert "RMSE Gain" in res_gate.reason


# ==============================================================================
# 5. Postprocessing & Bounds Calibration Tests (§6)
# ==============================================================================


class TestPostprocessingAndCalibration:
    def test_bounds_calibration_selection(self):
        y_val = np.array([20.0, 30.0, 80.0, 85.0])
        val_preds = np.array([20.0, 30.0, 82.0, 87.0])
        train_wqi = np.array([10.0, 20.0, 80.0])  # max = 80.0 -> dyn_max = 85.0

        cfg = select_bounds_calibration(y_val=y_val, val_preds=val_preds, train_wqi=train_wqi)
        assert cfg.clip_min == 0.0
        assert cfg.clip_max in {85.0, 100.0}
        assert cfg.bounds_rule in {"dynamic", "full"}

    def test_apply_postprocessing_clipping(self):
        cfg = PostprocessingConfig(
            clip_min=0.0,
            clip_max=85.0,
            bounds_rule="dynamic",
            max_train_wqi=80.0,
            dynamic_candidate_max=85.0,
            selected_by="test",
        )
        preds = np.array([-5.0, 20.0, 95.0])
        clipped = apply_postprocessing(preds, cfg)
        np.testing.assert_allclose(clipped, np.array([0.0, 20.0, 85.0]))


# ==============================================================================
# 6. Level-0 Model Builder Tests (§3)
# ==============================================================================


class TestLevel0ModelBuilder:
    def test_multi_seed_expansion(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifacts_root = Path(tmp_dir)
            report = {
                "accepted_candidates": [
                    {
                        "name": "cand1",
                        "study_name": "cand1",
                        "model_family": "lightgbm",
                        "loss_name": "rmse",
                        "best_iteration_count": 50,
                        "hyperparameters": {"num_leaves": 31},
                        "feature_set_hash": "h123",
                    },
                    {
                        "name": "cand2",
                        "study_name": "cand2",
                        "model_family": "catboost",
                        "loss_name": "mae",
                        "best_iteration_count": 60,
                        "hyperparameters": {"depth": 6},
                        "feature_set_hash": "h123",
                    },
                ]
            }

            specs = build_level0_specs(
                hemisphere=Hemisphere.NORTH,
                integrity_report=report,
                artifacts_root=artifacts_root,
            )

            # 2 candidates * 5 mandatory seeds = 10 model specs
            assert len(specs) == 10
            seeds_found = {s.seed for s in specs}
            assert seeds_found == set(MANDATORY_SEEDS)

    def test_model_instantiation_and_fit(self):
        spec_lgb = ModelSpec(
            candidate_name="lgb",
            study_name="lgb",
            model_family="lightgbm",
            loss_name="rmse",
            seed=42,
            feature_set_hash="h",
            hyperparameters={"n_estimators": 5, "num_leaves": 7, "verbosity": -1},
            best_iteration_count=5,
            artifact_dir=Path("/tmp"),
            model_id="lgb_1",
        )
        model = instantiate_model(spec_lgb)
        x = np.random.normal(size=(50, 4))
        y = np.random.normal(size=50)
        model.fit(x, y)
        preds = model.predict(x)
        assert len(preds) == 50


# ==============================================================================
# 7. OOF Matrix Generator Tests (§3)
# ==============================================================================


class TestOOFMatrixGenerator:
    def test_timeseries_split_oof_generation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            train_df, val_df, _, feats = _generate_synthetic_splits("north", n_features=5)

            spec = ModelSpec(
                candidate_name="ridge",
                study_name="ridge",
                model_family="ridge",
                loss_name="rmse",
                seed=42,
                feature_set_hash="h",
                hyperparameters={"alpha": 1.0},
                best_iteration_count=1,
                artifact_dir=out_dir,
                model_id="ridge::seed=42",
            )

            oof_res, val_res = generate_oof_matrix(
                hemisphere=Hemisphere.NORTH,
                train_df=train_df,
                val_df=val_df,
                specs=[spec],
                selected_features=feats,
                n_splits=5,
                output_dir=out_dir,
            )

            assert oof_res.oof_matrix.shape == (1095, 1)
            assert len(oof_res.burn_in_indices) > 0
            assert np.sum(oof_res.valid_oof_mask) + len(oof_res.burn_in_indices) == 1095
            assert val_res.predictions.shape == (365, 1)
            assert np.all(np.isfinite(val_res.predictions))

    def test_non_finite_prediction_rejection(self):
        train_df, val_df, _, feats = _generate_synthetic_splits("north", n_features=5)
        # Empty specs
        with pytest.raises(OOFGenerationError):
            generate_oof_matrix(
                hemisphere=Hemisphere.NORTH,
                train_df=train_df,
                val_df=val_df,
                specs=[],
                selected_features=feats,
                output_dir=Path("/tmp"),
            )


# ==============================================================================
# 8. State Machine & Transition Tests (§1)
# ==============================================================================


class TestPhase5StateMachine:
    def test_forward_only_progression(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state_file = Path(tmp_dir) / "state.json"
            sm = Phase5StateMachine(Hemisphere.NORTH, state_file)
            assert sm.current_state == WQIState.INITIALIZED

            for state in ORDERED_STATES[1:]:
                sm.transition_to(state, {"step": state.value})
                assert sm.current_state == state

            assert sm.current_state == WQIState.REGISTERED
            assert len(sm.history) == 12

    def test_backward_transition_rejection(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state_file = Path(tmp_dir) / "state.json"
            sm = Phase5StateMachine(Hemisphere.NORTH, state_file)
            sm.transition_to(WQIState.INPUTS_VERIFIED)

            with pytest.raises(GateViolationError):
                sm.transition_to(WQIState.INITIALIZED)

    def test_state_persistence_and_reload(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state_file = Path(tmp_dir) / "state.json"
            sm1 = Phase5StateMachine(Hemisphere.NORTH, state_file)
            sm1.transition_to(WQIState.INPUTS_VERIFIED)
            sm1.transition_to(WQIState.CANDIDATES_FROZEN)

            # Reload into a second instance
            sm2 = Phase5StateMachine(Hemisphere.NORTH, state_file)
            assert sm2.current_state == WQIState.CANDIDATES_FROZEN
            assert len(sm2.history) == 2


# ==============================================================================
# 9. Test Set Access Guard & Leakage Auditor Tests (§6 & §25)
# ==============================================================================


class TestTestSetAccessGuardAndLeakageAuditor:
    def test_single_touch_access_guard_and_double_access_rejection(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            receipt_path = base_dir / "test_access_receipt.json"
            test_data_path = base_dir / "test_data.parquet"
            pl.DataFrame({"x": [1, 2, 3]}).write_parquet(test_data_path)

            pred_path = base_dir / "preds.parquet"
            metric_path = base_dir / "metrics.json"
            pl.DataFrame({"p": [1, 2, 3]}).write_parquet(pred_path)
            with open(metric_path, "w", encoding="utf-8") as f:
                json.dump({"rmse": 1.23}, f)

            # First touch - allowed
            with TestSetAccessGuard(
                hemisphere=Hemisphere.NORTH,
                receipt_path=receipt_path,
                final_state_hash="state_123",
                feature_set_hash="feat_123",
                test_data_path=test_data_path,
            ) as guard:
                receipt = guard.commit_receipt(
                    predictions_path=pred_path,
                    metrics_path=metric_path,
                )
                assert receipt.final_state_hash == "state_123"

            assert receipt_path.exists()

            # Second touch - strictly forbidden!
            with pytest.raises(TestSetAccessError):
                with TestSetAccessGuard(
                    hemisphere=Hemisphere.NORTH,
                    receipt_path=receipt_path,
                    final_state_hash="state_123",
                    feature_set_hash="feat_123",
                    test_data_path=test_data_path,
                ):
                    pass

    def test_leakage_auditor_full_audit(self):
        auditor = LeakageAuditor("north")
        audit_res = auditor.run_full_audit(
            train_days=list(range(1095)),
            val_days=list(range(1095, 1460)),
            test_days=list(range(1460, 1825)),
            selected_features=["wqi_lag1", "seasonal_wqi_mean_doy"],
            burn_in_indices=[0, 1, 2, 3],
            valid_oof_mask=[False, False, False, False] + [True] * (1095 - 4),
            level1_train_data=np.zeros((1091, 2)),
            test_receipt_present=True,
        )
        assert audit_res["audit_status"] == "PASSED_100%"
        assert audit_res["passed_checks"] >= 7

    def test_leakage_auditor_forbidden_feature_rejection(self):
        auditor = LeakageAuditor("north")
        with pytest.raises(LeakageError):
            auditor.run_full_audit(
                train_days=list(range(1095)),
                val_days=list(range(1095, 1460)),
                test_days=list(range(1460, 1825)),
                selected_features=["water_quality_index"],  # FORBIDDEN!
                burn_in_indices=[0],
                valid_oof_mask=[False] + [True] * 1094,
                level1_train_data=None,
                test_receipt_present=True,
            )


# ==============================================================================
# 10. Comprehensive Diagnostics & Zone Metrics Tests (§20)
# ==============================================================================


class TestDiagnosticsAndZoneMetrics:
    def test_compute_comprehensive_metrics_7_suite(self):
        y_t = np.array([20.0, 40.0, 60.0, 80.0, 100.0])
        y_p = np.array([21.0, 39.0, 62.0, 78.0, 99.0])
        metrics = compute_comprehensive_metrics(y_t, y_p)

        assert "rmse" in metrics
        assert "mae" in metrics
        assert "nse" in metrics
        assert "rmsle" in metrics
        assert "r2" in metrics
        assert "explained_variance" in metrics
        assert "max_absolute_error" in metrics
        assert metrics["rmse"] > 0.0
        assert metrics["r2"] > 0.95

    def test_compute_zone_metrics(self):
        y_t = np.array([10.0, 30.0, 55.0, 75.0, 90.0])
        y_p = np.array([11.0, 31.0, 56.0, 76.0, 91.0])
        zone_metrics = compute_zone_metrics(y_t, y_p)
        assert len(zone_metrics) == 5
        zones = {zm.zone_name for zm in zone_metrics}
        assert zones == {
            WQIZone.CRITICAL.value,
            WQIZone.POOR.value,
            WQIZone.MARGINAL.value,
            WQIZone.GOOD.value,
            WQIZone.EXCELLENT.value,
        }

    def test_compute_monthly_and_decile_metrics(self):
        y_t = np.linspace(10.0, 90.0, 365)
        y_p = y_t + np.random.normal(0, 1, 365)
        m_metrics = compute_monthly_metrics(y_t, y_p)
        d_metrics = compute_decile_metrics(y_t, y_p)
        assert len(m_metrics) == 12
        assert len(d_metrics) == 10


# ==============================================================================
# 11. Production Refit Tests (§6)
# ==============================================================================


class TestProductionRefit:
    def test_execute_production_refit_1460_rows(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            train_df, val_df, _, feats = _generate_synthetic_splits("north", n_features=4)
            full_hist_df = pl.concat([train_df, val_df])
            assert full_hist_df.height == 1460

            spec = ModelSpec(
                candidate_name="ridge_refit",
                study_name="ridge_refit",
                model_family="ridge",
                loss_name="rmse",
                seed=42,
                feature_set_hash="feat_hash",
                hyperparameters={"alpha": 1.0},
                best_iteration_count=1,
                artifact_dir=out_dir,
                model_id="ridge_refit::seed=42",
            )

            frozen_state = FinalFrozenState(
                hemisphere=Hemisphere.NORTH,
                feature_set_hash="feat_hash",
                selected_features=tuple(feats),
                level0_specs=(spec,),
                strategy=EnsembleStrategy.CONSTRAINED_WEIGHTED,
                strategy_parameters={"weights": {spec.model_id: 1.0}},
                residual_enabled=True,
                residual_model_hash=None,
                postprocessing=PostprocessingConfig(0.0, 100.0, "full", 95.0, None, "test"),
                frozen_state_hash="state_hash_123",
            )

            manifest = execute_production_refit(
                hemisphere=Hemisphere.NORTH,
                full_historical_df=full_hist_df,
                frozen_state=frozen_state,
                output_dir=out_dir,
            )

            assert manifest["refit_sample_count"] == 1460
            assert (out_dir / "manifest.json").exists()
            assert (out_dir / "level0_models" / f"{spec.model_id}.joblib").exists()
            assert (out_dir / "residual" / "residual_corrector.joblib").exists()


# ==============================================================================
# 12. End-to-End Synthetic Pipeline & Kaggle Execution Benchmark Tests
# ==============================================================================


class TestEndToEndEnsemblePipelineKaggleReady:
    def test_north_pipeline_end_to_end_synthetic(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root_dir = Path(tmp_dir)
            data_dir = root_dir / "data"
            art_dir = root_dir / "artifacts"
            data_dir.mkdir(parents=True, exist_ok=True)

            train_df, val_df, test_df, feats = _generate_synthetic_splits("north", n_features=6)
            train_df.write_parquet(data_dir / "wqi_north_train.parquet")
            val_df.write_parquet(data_dir / "wqi_north_val.parquet")
            test_df.write_parquet(data_dir / "wqi_north_test.parquet")

            _create_synthetic_phase4_registry(art_dir, Hemisphere.NORTH, feats)

            pipeline = Phase5EnsemblePipeline(
                hemisphere=Hemisphere.NORTH,
                data_dir=data_dir,
                artifacts_root=art_dir,
            )

            t0 = time.time()
            res = pipeline.run()
            elapsed = time.time() - t0

            assert res["state"] == WQIState.REGISTERED.value
            assert res["hemisphere"] == "north"
            assert "test_rmse" in res
            assert res["test_rmse"] > 0.0
            assert elapsed < 60.0  # Kaggle CPU execution time guard

    def test_south_pipeline_duplicate_collapse_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root_dir = Path(tmp_dir)
            data_dir = root_dir / "data"
            art_dir = root_dir / "artifacts"
            data_dir.mkdir(parents=True, exist_ok=True)

            train_df, val_df, test_df, feats = _generate_synthetic_splits("south", n_features=6)
            train_df.write_parquet(data_dir / "wqi_south_train.parquet")
            val_df.write_parquet(data_dir / "wqi_south_val.parquet")
            test_df.write_parquet(data_dir / "wqi_south_test.parquet")

            # Create candidate registry with duplicate CatBoost entries
            hemi_model_dir = art_dir / "model" / "south"
            hemi_model_dir.mkdir(parents=True, exist_ok=True)
            dup_candidates = [
                {
                    "name": "model-south-catboost-rmse",
                    "study_name": "model-south-catboost-rmse",
                    "model_family": "catboost",
                    "loss_name": "rmse",
                    "best_iteration_count": 5,
                    "passed_baseline_gate": True,
                    "leakage_detected": False,
                    "selected_features": feats,
                    "hyperparameters": {"depth": 3, "learning_rate": 0.1, "iterations": 5, "verbose": False},
                },
                {
                    "name": "model-south-catboost-logcosh",
                    "study_name": "model-south-catboost-logcosh",
                    "model_family": "catboost",
                    "loss_name": "rmse",  # duplicate params
                    "best_iteration_count": 5,
                    "passed_baseline_gate": True,
                    "leakage_detected": False,
                    "selected_features": feats,
                    "hyperparameters": {"depth": 3, "learning_rate": 0.1, "iterations": 5, "verbose": False},
                },
            ]
            with open(hemi_model_dir / "candidate_registry.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "hemisphere": "south",
                        "baseline": {"val_rmse": 50.0},
                        "feature_set_hash": "h_south",
                        "passed_candidates": dup_candidates,
                    },
                    f,
                )

            pipeline = Phase5EnsemblePipeline(
                hemisphere=Hemisphere.SOUTH,
                data_dir=data_dir,
                artifacts_root=art_dir,
            )

            res = pipeline.run()
            assert res["state"] == WQIState.REGISTERED.value
            assert res["hemisphere"] == "south"
            assert res["test_rmse"] > 0.0

    def test_custom_artifacts_path_aliases_and_string_paths(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root_dir = Path(tmp_dir)
            custom_art = str(root_dir / "my_custom_artifacts_folder")
            custom_data = str(root_dir / "my_custom_data_folder")

            pipeline = Phase5EnsemblePipeline(
                hemisphere="north",
                data_path=custom_data,
                artifacts_path=custom_art,
            )

            assert pipeline.artifacts_root == Path(custom_art).resolve()
            assert pipeline.data_dir == Path(custom_data).resolve()
            assert pipeline.hemisphere == Hemisphere.NORTH
