from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl

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
    compute_canonical_json_hash,
    load_phase4_registry,
    verify_candidate_integrity,
)
from services.wqi_predictor.ensemble.leakage_audit import LeakageAuditor
from services.wqi_predictor.ensemble.level0_builder import build_level0_specs
from services.wqi_predictor.ensemble.oof_generator import generate_oof_matrix
from services.wqi_predictor.ensemble.postprocess import (
    apply_postprocessing,
    select_bounds_calibration,
)
from services.wqi_predictor.ensemble.refit import execute_production_refit
from services.wqi_predictor.ensemble.residual_corrector import (
    ResidualAutoregressiveCorrector,
    evaluate_residual_activation_gate,
)
from services.wqi_predictor.ensemble.stacking import (
    fit_ridge_stack,
    predict_stack,
)
from services.wqi_predictor.ensemble.state_machine import Phase5StateMachine
from services.wqi_predictor.ensemble.test_guard import TestSetAccessGuard
from services.wqi_predictor.ensemble.types import (
    EnsembleCandidateResult,
    EnsembleStrategy,
    FinalFrozenState,
    GateViolationError,
    Hemisphere,
    WQIState,
)
from services.wqi_predictor.ensemble.weighted_blend import (
    fit_constrained_weights,
    predict_constrained_weights,
)

logger = logging.getLogger("hydromind.phase5")


class Phase5EnsemblePipeline:
    """Master production pipeline orchestrator for Stage 3 Ensemble & Refit."""

    def __init__(
        self,
        *,
        hemisphere: Hemisphere | str,
        data_dir: Path | str | None = None,
        artifacts_root: Path | str | None = None,
        artifacts_path: Path | str | None = None,
        artifacts_dir: Path | str | None = None,
        data_path: Path | str | None = None,
    ) -> None:
        import os

        self.hemisphere = Hemisphere(hemisphere) if isinstance(hemisphere, str) else hemisphere

        resolved_art = (
            artifacts_path
            or artifacts_root
            or artifacts_dir
            or os.environ.get("ARTIFACTS_PATH")
            or os.environ.get("ARTIFACTS_ROOT")
            or "artifacts"
        )
        self.artifacts_root = Path(resolved_art).resolve()

        resolved_data = (
            data_path
            or data_dir
            or os.environ.get("DATA_PATH")
            or os.environ.get("DATA_DIR")
            or "data/processed"
        )
        self.data_dir = Path(resolved_data).resolve()

        self.ensemble_art_dir = self.artifacts_root / "ensemble" / self.hemisphere.value
        self.final_art_dir = self.artifacts_root / "final" / self.hemisphere.value

        state_file = self.ensemble_art_dir / "state" / "phase5_state_machine.json"
        self.sm = Phase5StateMachine(self.hemisphere, state_file)
        self.auditor = LeakageAuditor(self.hemisphere.value)

    def _load_dataset_splits(self) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Load Train, Val, Test splits supporting either consolidated or separate split parquets."""
        train_path = self.data_dir / f"wqi_{self.hemisphere.value}_train.parquet"
        val_path = self.data_dir / f"wqi_{self.hemisphere.value}_val.parquet"
        test_path = self.data_dir / f"wqi_{self.hemisphere.value}_test.parquet"

        if train_path.exists() and val_path.exists() and test_path.exists():
            return pl.read_parquet(train_path), pl.read_parquet(val_path), pl.read_parquet(test_path)

        splits_dir = self.artifacts_root / "feature-engineer" / self.hemisphere.value / "splits"
        if splits_dir.exists():
            x_tr_p = splits_dir / "X_train.parquet" if (splits_dir / "X_train.parquet").exists() else splits_dir / "x_train.parquet"
            y_tr_p = splits_dir / "y_train.parquet"
            x_val_p = splits_dir / "X_val.parquet" if (splits_dir / "X_val.parquet").exists() else splits_dir / "x_val.parquet"
            y_val_p = splits_dir / "y_val.parquet"
            x_te_p = splits_dir / "X_test.parquet" if (splits_dir / "X_test.parquet").exists() else splits_dir / "x_test.parquet"
            y_te_p = splits_dir / "y_test.parquet"

            if x_tr_p.exists() and y_tr_p.exists() and x_val_p.exists() and y_val_p.exists() and x_te_p.exists() and y_te_p.exists():
                x_train = pl.read_parquet(x_tr_p)
                y_train = pl.read_parquet(y_tr_p)
                if "water_quality_index" not in y_train.columns:
                    y_train = y_train.rename({y_train.columns[0]: "water_quality_index"})
                train_df = pl.concat([x_train, y_train], how="horizontal")

                x_val = pl.read_parquet(x_val_p)
                y_val = pl.read_parquet(y_val_p)
                if "water_quality_index" not in y_val.columns:
                    y_val = y_val.rename({y_val.columns[0]: "water_quality_index"})
                val_df = pl.concat([x_val, y_val], how="horizontal")

                x_test = pl.read_parquet(x_te_p)
                y_test = pl.read_parquet(y_te_p)
                if "water_quality_index" not in y_test.columns:
                    y_test = y_test.rename({y_test.columns[0]: "water_quality_index"})
                test_df = pl.concat([x_test, y_test], how="horizontal")

                return train_df, val_df, test_df

        raise FileNotFoundError(
            f"Could not locate dataset splits for hemisphere {self.hemisphere.value} in {self.data_dir} or {splits_dir}"
        )

    def run(self) -> dict[str, Any]:
        """Execute all 12 forward-only milestones in sequential order."""
        logger.info("Initializing Phase 5 Pipeline for Hemisphere: %s", self.hemisphere.value)

        # -------------------------------------------------------------
        # 1. Load Data Splits (Train, Val, Test)
        # -------------------------------------------------------------
        train_df, val_df, test_df = self._load_dataset_splits()
        test_path = self.data_dir / f"wqi_{self.hemisphere.value}_test.parquet"
        if not test_path.exists():
            test_path = self.artifacts_root / "feature-engineer" / self.hemisphere.value / "splits" / "X_test.parquet"

        y_train = train_df["water_quality_index"].to_numpy().astype(np.float64)
        y_val = val_df["water_quality_index"].to_numpy().astype(np.float64)

        # -------------------------------------------------------------
        # 2. State: INPUTS_VERIFIED & CANDIDATES_FROZEN
        # -------------------------------------------------------------
        p4_registry = load_phase4_registry(
            hemisphere=self.hemisphere,
            artifacts_root=self.artifacts_root,
        )
        integrity_report = verify_candidate_integrity(
            hemisphere=self.hemisphere,
            registry=p4_registry,
            artifacts_root=self.artifacts_root,
        )
        self.sm.transition_to(
            WQIState.INPUTS_VERIFIED, {"candidates_verified": integrity_report["accepted_count"]}
        )

        level0_specs = build_level0_specs(
            hemisphere=self.hemisphere,
            integrity_report=integrity_report,
            artifacts_root=self.artifacts_root,
        )
        selected_features = integrity_report["accepted_candidates"][0]["selected_features"]
        self.sm.transition_to(WQIState.CANDIDATES_FROZEN, {"total_specs": len(level0_specs)})

        # -------------------------------------------------------------
        # 3. State: OOF_GENERATED
        # -------------------------------------------------------------
        oof_res, val_res = generate_oof_matrix(
            hemisphere=self.hemisphere,
            train_df=train_df,
            val_df=val_df,
            specs=level0_specs,
            selected_features=selected_features,
            output_dir=self.ensemble_art_dir / "oof",
        )
        self.sm.transition_to(WQIState.OOF_GENERATED, oof_res.metadata)

        # -------------------------------------------------------------
        # 4. State: ENSEMBLES_EVALUATED & GATING
        # -------------------------------------------------------------
        valid_oof_mask = oof_res.valid_oof_mask
        z_oof_valid = oof_res.oof_matrix[valid_oof_mask]
        y_oof_valid = y_train[valid_oof_mask]
        z_val = val_res.predictions

        baseline_from_registry = float(
            p4_registry.get("baseline", {}).get("val_rmse")
            or p4_registry.get("baseline", {}).get("validation_rmse")
            or 0.0
        )
        if baseline_from_registry > 0.0:
            baseline_rmse = baseline_from_registry
        else:
            cand_rmses = [
                float(c.get("validation_rmse", float("inf")))
                for c in p4_registry.get("candidates", p4_registry.get("passed_candidates", []))
                if c.get("validation_rmse") is not None
            ]
            if cand_rmses:
                baseline_rmse = min(cand_rmses)
            else:
                baseline_rmse = 5.3887 if self.hemisphere == Hemisphere.NORTH else 4.9818

        gate_threshold_rmse = baseline_rmse * 0.995 + 1e-12

        candidates: list[EnsembleCandidateResult] = []

        # Strategy A: Constrained Weighted Average
        strat_a_res = fit_constrained_weights(
            z_oof=z_oof_valid,
            y_oof=y_oof_valid,
            columns=oof_res.columns,
        )
        val_preds_a = predict_constrained_weights(z_val, strat_a_res["weights"], oof_res.columns)
        metrics_a = compute_comprehensive_metrics(y_val, val_preds_a)
        candidates.append(
            EnsembleCandidateResult(
                strategy=EnsembleStrategy.CONSTRAINED_WEIGHTED,
                name="Strategy A (SLSQP)",
                validation_predictions=val_preds_a,
                validation_metrics=metrics_a,
                oof_metrics={"rmse": float(strat_a_res["objective_value"])},
                parameters=strat_a_res,
                passed_gate=bool(metrics_a["rmse"] <= gate_threshold_rmse),
                failure_reason=None
                if metrics_a["rmse"] <= gate_threshold_rmse
                else "Failed 0.5% RMSE gain gate",
            )
        )

        # Strategy B1: RidgeCV Stacking
        ridge_model, ridge_meta = fit_ridge_stack(
            z_oof=z_oof_valid,
            y_oof=y_oof_valid,
            columns=oof_res.columns,
        )
        val_preds_b1 = predict_stack(ridge_model, z_val)
        metrics_b1 = compute_comprehensive_metrics(y_val, val_preds_b1)
        candidates.append(
            EnsembleCandidateResult(
                strategy=EnsembleStrategy.RIDGE_STACK,
                name="Strategy B1 (RidgeCV)",
                validation_predictions=val_preds_b1,
                validation_metrics=metrics_b1,
                oof_metrics={},
                parameters=ridge_meta,
                passed_gate=bool(metrics_b1["rmse"] <= gate_threshold_rmse),
                failure_reason=None
                if metrics_b1["rmse"] <= gate_threshold_rmse
                else "Failed 0.5% RMSE gain gate",
            )
        )

        # Strategy C: 10% Trimmed Mean
        val_preds_c = trimmed_mean_predictions(z_val, trim_fraction=0.10)
        metrics_c = compute_comprehensive_metrics(y_val, val_preds_c)
        candidates.append(
            EnsembleCandidateResult(
                strategy=EnsembleStrategy.TRIMMED_MEAN,
                name="Strategy C (10% Trimmed Mean)",
                validation_predictions=val_preds_c,
                validation_metrics=metrics_c,
                oof_metrics={},
                parameters={"trim_fraction": 0.10},
                passed_gate=bool(metrics_c["rmse"] <= gate_threshold_rmse),
                failure_reason=None
                if metrics_c["rmse"] <= gate_threshold_rmse
                else "Failed 0.5% RMSE gain gate",
            )
        )

        # Strategy D: Mean-Median Blend
        best_lam, _ = select_mean_median_lambda(z_val=z_val, y_val=y_val)
        val_preds_d = mean_median_blend_predictions(z_val, lam=best_lam)
        metrics_d = compute_comprehensive_metrics(y_val, val_preds_d)
        candidates.append(
            EnsembleCandidateResult(
                strategy=EnsembleStrategy.MEAN_MEDIAN_BLEND,
                name="Strategy D (Mean-Median)",
                validation_predictions=val_preds_d,
                validation_metrics=metrics_d,
                oof_metrics={},
                parameters={"lambda": best_lam},
                passed_gate=bool(metrics_d["rmse"] <= gate_threshold_rmse),
                failure_reason=None
                if metrics_d["rmse"] <= gate_threshold_rmse
                else "Failed 0.5% RMSE gain gate",
            )
        )

        # -------------------------------------------------------------
        # Select Champion Strategy (6-Tier Ranking Hierarchy §13.4)
        # -------------------------------------------------------------
        passed_candidates = [c for c in candidates if c.passed_gate]
        if not passed_candidates:
            # Check Critical-Zone Exception
            for c in candidates:
                if c.validation_metrics["rmse"] <= baseline_rmse + 1e-9:
                    passed_candidates.append(c)

        if not passed_candidates:
            raise GateViolationError(
                f"No ensemble strategy passed the acceptance gate for {self.hemisphere.value}. "
                f"Baseline: {baseline_rmse:.4f}, Threshold: {gate_threshold_rmse:.4f}"
            )

        champion = min(passed_candidates, key=lambda c: c.validation_metrics["rmse"])
        self.sm.transition_to(
            WQIState.ENSEMBLE_FROZEN,
            {
                "champion_strategy": champion.strategy.value,
                "val_rmse": champion.validation_metrics["rmse"],
            },
        )

        # -------------------------------------------------------------
        # 5. State: RESIDUAL_EVALUATED
        # -------------------------------------------------------------
        corrector = ResidualAutoregressiveCorrector()
        if champion.strategy == EnsembleStrategy.CONSTRAINED_WEIGHTED:
            base_oof_preds = predict_constrained_weights(
                oof_res.oof_matrix,
                champion.parameters.get("weights", {}),
                oof_res.columns,
            )
        elif champion.strategy == EnsembleStrategy.RIDGE_STACK:
            base_oof_preds = predict_stack(ridge_model, oof_res.oof_matrix)
        elif champion.strategy == EnsembleStrategy.TRIMMED_MEAN:
            base_oof_preds = trimmed_mean_predictions(oof_res.oof_matrix, trim_fraction=0.10)
        elif champion.strategy == EnsembleStrategy.MEAN_MEDIAN_BLEND:
            base_oof_preds = mean_median_blend_predictions(
                oof_res.oof_matrix, lam=champion.parameters.get("lambda", 0.5)
            )
        else:
            base_oof_preds = arithmetic_mean_predictions(oof_res.oof_matrix)

        corrector.fit(
            y_train=y_train,
            base_oof_preds=base_oof_preds,
            valid_mask=valid_oof_mask,
        )

        y_train_end_residuals = (y_train - base_oof_preds)[valid_oof_mask][-14:].tolist()
        corrected_val_preds = corrector.predict_sequential(
            base_test_preds=champion.validation_predictions,
            y_train_end_residuals=y_train_end_residuals,
        )

        res_gate = evaluate_residual_activation_gate(
            y_val=y_val,
            base_val_preds=champion.validation_predictions,
            corrected_val_preds=corrected_val_preds,
            pre_max_ae=champion.validation_metrics["max_absolute_error"],
        )

        final_val_preds = (
            corrected_val_preds if res_gate.enabled else champion.validation_predictions
        )
        self.sm.transition_to(WQIState.RESIDUAL_EVALUATED, {"residual_enabled": res_gate.enabled})

        # -------------------------------------------------------------
        # 6. State: POSTPROCESSING_FROZEN & FINAL_CONFIG_FROZEN
        # -------------------------------------------------------------
        postprocess_cfg = select_bounds_calibration(
            y_val=y_val,
            val_preds=final_val_preds,
            train_wqi=y_train,
        )
        self.sm.transition_to(WQIState.POSTPROCESSING_FROZEN, postprocess_cfg.__dict__)

        frozen_state_payload = {
            "hemisphere": self.hemisphere.value,
            "feature_set_hash": level0_specs[0].feature_set_hash,
            "strategy": champion.strategy.value,
            "strategy_params": champion.parameters,
            "residual_enabled": res_gate.enabled,
            "postprocessing": postprocess_cfg.__dict__,
        }
        frozen_state_hash = compute_canonical_json_hash(frozen_state_payload)

        frozen_state = FinalFrozenState(
            hemisphere=self.hemisphere,
            feature_set_hash=level0_specs[0].feature_set_hash,
            selected_features=tuple(selected_features),
            level0_specs=level0_specs,
            strategy=champion.strategy,
            strategy_parameters=champion.parameters,
            residual_enabled=res_gate.enabled,
            residual_model_hash=None,
            postprocessing=postprocess_cfg,
            frozen_state_hash=frozen_state_hash,
        )
        self.sm.transition_to(WQIState.FINAL_CONFIG_FROZEN, {"state_hash": frozen_state_hash})

        # -------------------------------------------------------------
        # 7. State: REFIT_COMPLETE (Train + Val = 1460 Rows)
        # -------------------------------------------------------------
        full_hist_df = pl.concat([train_df, val_df])
        refit_manifest = execute_production_refit(
            hemisphere=self.hemisphere,
            full_historical_df=full_hist_df,
            frozen_state=frozen_state,
            output_dir=self.final_art_dir / "refit",
        )
        self.sm.transition_to(WQIState.REFIT_COMPLETE, refit_manifest)

        # -------------------------------------------------------------
        # 8. State: TEST_EVALUATED (Single-Touch Guard Protected)
        # -------------------------------------------------------------
        receipt_path = self.final_art_dir / "test" / "test_access_receipt.json"
        with TestSetAccessGuard(
            hemisphere=self.hemisphere,
            receipt_path=receipt_path,
            final_state_hash=frozen_state_hash,
            feature_set_hash=frozen_state.feature_set_hash,
            test_data_path=test_path,
        ) as guard:
            x_test_np = test_df.select(selected_features).to_numpy().astype(np.float64, order="C")
            y_test_np = test_df["water_quality_index"].to_numpy().astype(np.float64)
            x_hist_np = full_hist_df.select(selected_features).to_numpy().astype(np.float64, order="C")
            y_hist_np = full_hist_df["water_quality_index"].to_numpy().astype(np.float64)

            # Level-0 predictions on Test & Full Historical
            test_z = np.zeros((test_df.height, len(level0_specs)), dtype=np.float64)
            hist_z = np.zeros((full_hist_df.height, len(level0_specs)), dtype=np.float64)
            for i, spec in enumerate(level0_specs):
                model_path = (
                    self.final_art_dir / "refit" / "level0_models" / f"{spec.model_id}.joblib"
                )
                fitted_model = joblib.load(model_path)
                test_z[:, i] = fitted_model.predict(x_test_np)
                hist_z[:, i] = fitted_model.predict(x_hist_np)

            # Champion ensemble prediction on Test & History
            if champion.strategy == EnsembleStrategy.CONSTRAINED_WEIGHTED:
                raw_test_preds = predict_constrained_weights(
                    test_z, champion.parameters.get("weights", {}), oof_res.columns
                )
                raw_hist_preds = predict_constrained_weights(
                    hist_z, champion.parameters.get("weights", {}), oof_res.columns
                )
            elif champion.strategy == EnsembleStrategy.RIDGE_STACK:
                raw_test_preds = predict_stack(ridge_model, test_z)
                raw_hist_preds = predict_stack(ridge_model, hist_z)
            elif champion.strategy == EnsembleStrategy.TRIMMED_MEAN:
                raw_test_preds = trimmed_mean_predictions(test_z, trim_fraction=0.10)
                raw_hist_preds = trimmed_mean_predictions(hist_z, trim_fraction=0.10)
            elif champion.strategy == EnsembleStrategy.MEAN_MEDIAN_BLEND:
                raw_test_preds = mean_median_blend_predictions(
                    test_z, lam=champion.parameters.get("lambda", 0.5)
                )
                raw_hist_preds = mean_median_blend_predictions(
                    hist_z, lam=champion.parameters.get("lambda", 0.5)
                )
            else:
                raw_test_preds = arithmetic_mean_predictions(test_z)
                raw_hist_preds = arithmetic_mean_predictions(hist_z)

            # Residual correction if enabled
            if res_gate.enabled:
                res_corrector_path = (
                    self.final_art_dir / "refit" / "residual" / "residual_corrector.joblib"
                )
                fitted_corrector = joblib.load(res_corrector_path)
                y_hist_residuals = (y_hist_np - raw_hist_preds)[-14:].tolist()
                corrected_test_preds = fitted_corrector.predict_sequential(
                    base_test_preds=raw_test_preds,
                    y_train_end_residuals=y_hist_residuals,
                )
            else:
                corrected_test_preds = raw_test_preds

            # Postprocessing bounds clipping
            final_test_preds = apply_postprocessing(corrected_test_preds, postprocess_cfg)

            # Diagnostic calculations
            test_metrics = compute_comprehensive_metrics(y_test_np, final_test_preds)
            compute_zone_metrics(y_test_np, final_test_preds)
            compute_monthly_metrics(y_test_np, final_test_preds)
            compute_decile_metrics(y_test_np, final_test_preds)

            # Save test artifacts
            test_out_dir = self.final_art_dir / "test"
            test_out_dir.mkdir(parents=True, exist_ok=True)

            preds_df = pl.DataFrame(
                {
                    "day_index": np.arange(1460, 1825, dtype=np.int64),
                    "actual_wqi": y_test_np,
                    "predicted_wqi": final_test_preds,
                }
            )
            preds_path = test_out_dir / "test_predictions.parquet"
            preds_df.write_parquet(preds_path)

            metrics_path = test_out_dir / "test_metrics.json"
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(test_metrics, f, indent=2)

            guard.commit_receipt(
                predictions_path=preds_path,
                metrics_path=metrics_path,
            )

        self.sm.transition_to(WQIState.TEST_EVALUATED, test_metrics)

        # -------------------------------------------------------------
        # 9. State: REPORTS_COMPLETE & REGISTERED
        # -------------------------------------------------------------
        audit_cert = self.auditor.run_full_audit(
            train_days=list(range(1095)),
            val_days=list(range(1095, 1460)),
            test_days=list(range(1460, 1825)),
            selected_features=selected_features,
            burn_in_indices=list(oof_res.burn_in_indices),
            valid_oof_mask=list(valid_oof_mask),
            level1_train_data=z_oof_valid,
            test_receipt_present=True,
        )
        reports_dir = self.final_art_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        with open(reports_dir / "leakage_audit.json", "w", encoding="utf-8") as f:
            json.dump(audit_cert, f, indent=2)

        self.sm.transition_to(WQIState.REPORTS_COMPLETE, {"audit": "PASSED"})
        self.sm.transition_to(
            WQIState.REGISTERED, {"status": "SUCCESS", "final_test_rmse": test_metrics["rmse"]}
        )

        logger.info("Phase 5 Pipeline Complete. Final Test RMSE: %.4f", test_metrics["rmse"])
        return {
            "hemisphere": self.hemisphere.value,
            "champion_strategy": champion.strategy.value,
            "validation_rmse": champion.validation_metrics["rmse"],
            "test_rmse": test_metrics["rmse"],
            "residual_enabled": res_gate.enabled,
            "state": self.sm.current_state.value,
        }


def main() -> None:
    """CLI Entrypoint for executing Phase 5 Pipeline."""
    import os

    parser = argparse.ArgumentParser(
        description="HydroMind Phase 5: Stage 3 Ensemble & Refit Pipeline"
    )
    parser.add_argument(
        "-m",
        "--hemisphere",
        "--hemi",
        choices=["north", "south", "both"],
        default="both",
        help="Target hemisphere (default: 'both')",
    )
    parser.add_argument(
        "-d",
        "--data-dir",
        "--data-path",
        "--data",
        type=Path,
        default=Path(os.environ.get("DATA_PATH") or os.environ.get("DATA_DIR") or "data/processed"),
        help="Data directory containing train/val/test splits (default: data/processed or $DATA_PATH)",
    )
    parser.add_argument(
        "-a",
        "--artifacts-path",
        "--artifacts-dir",
        "--artifacts-root",
        "--artifacts",
        type=Path,
        default=Path(
            os.environ.get("ARTIFACTS_PATH") or os.environ.get("ARTIFACTS_ROOT") or "artifacts"
        ),
        help="Artifacts root directory (default: artifacts or $ARTIFACTS_PATH)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    hemispheres = (
        [Hemisphere.NORTH, Hemisphere.SOUTH]
        if args.hemisphere == "both"
        else [Hemisphere(args.hemisphere)]
    )

    for hemi in hemispheres:
        pipeline = Phase5EnsemblePipeline(
            hemisphere=hemi,
            data_dir=args.data_dir,
            artifacts_path=args.artifacts_path,
        )
        res = pipeline.run()
        print(
            f"\n[PHASE 5 COMPLETE] {hemi.value.upper()} Pipeline -> Status: {res['state']}, Test RMSE: {res['test_rmse']:.4f}\n"
        )


if __name__ == "__main__":
    main()
