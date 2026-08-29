"""HydroMind Ensemble Subsystem (Phase 5 / Stage 3)."""
from .blending import (
    arithmetic_mean_predictions,
    mean_median_blend_predictions,
    select_mean_median_lambda,
    trimmed_mean_predictions,
)
from .diagnostics import (
    compute_comprehensive_metrics,
    compute_decile_metrics,
    compute_monthly_metrics,
    compute_zone_metrics,
)
from .integrity import (
    FORBIDDEN_FEATURES,
    compute_canonical_json_hash,
    compute_sha256,
    load_model_registry,
    load_phase4_registry,
    verify_candidate_integrity,
)
from .leakage_audit import LeakageAuditor
from .level0_builder import (
    MANDATORY_SEEDS,
    build_level0_specs,
    build_model_specs,
    build_seed_params,
    instantiate_model,
)
from .oof_generator import generate_oof_matrix
from .postprocess import apply_postprocessing, select_bounds_calibration
from .refit import execute_production_refit
from .residual_corrector import (
    ResidualAutoregressiveCorrector,
    compute_ljung_box_stat,
    evaluate_residual_activation_gate,
)
from .stacking import (
    fit_elasticnet_stack,
    fit_ridge_stack,
    predict_stack,
)
from .state_machine import ORDERED_STATES, Phase5StateMachine
from .test_guard import TestSetAccessGuard
from .types import (
    ArtifactIntegrityError,
    EnsembleCandidateResult,
    EnsembleError,
    EnsembleStrategy,
    FinalFrozenState,
    GateViolationError,
    Hemisphere,
    LeakageError,
    Level0ModelSpec,
    ModelSpec,
    OOFGenerationError,
    OOFResult,
    PostprocessingConfig,
    ResidualCorrectionError,
    ResidualGateResult,
    TestEvaluationReceipt,
    TestSetAccessError,
    ValidationPredictionResult,
    WeightOptimizationError,
    WQIZone,
    WQIState,
)
from .weighted_blend import fit_constrained_weights, predict_constrained_weights

__all__ = [
    "ArtifactIntegrityError",
    "EnsembleCandidateResult",
    "EnsembleError",
    "EnsembleStrategy",
    "FORBIDDEN_FEATURES",
    "FinalFrozenState",
    "GateViolationError",
    "Hemisphere",
    "LeakageAuditor",
    "LeakageError",
    "Level0ModelSpec",
    "MANDATORY_SEEDS",
    "ModelSpec",
    "OOFGenerationError",
    "OOFResult",
    "ORDERED_STATES",
    "Phase5StateMachine",
    "PostprocessingConfig",
    "ResidualAutoregressiveCorrector",
    "ResidualCorrectionError",
    "ResidualGateResult",
    "TestEvaluationReceipt",
    "TestSetAccessError",
    "TestSetAccessGuard",
    "ValidationPredictionResult",
    "WQIZone",
    "WQIState",
    "WeightOptimizationError",
    "apply_postprocessing",
    "arithmetic_mean_predictions",
    "build_level0_specs",
    "build_model_specs",
    "build_seed_params",
    "compute_canonical_json_hash",
    "compute_comprehensive_metrics",
    "compute_decile_metrics",
    "compute_ljung_box_stat",
    "compute_monthly_metrics",
    "compute_sha256",
    "compute_zone_metrics",
    "evaluate_residual_activation_gate",
    "execute_production_refit",
    "fit_constrained_weights",
    "fit_elasticnet_stack",
    "fit_ridge_stack",
    "generate_oof_matrix",
    "instantiate_model",
    "load_model_registry",
    "load_phase4_registry",
    "mean_median_blend_predictions",
    "predict_constrained_weights",
    "predict_stack",
    "select_bounds_calibration",
    "select_mean_median_lambda",
    "trimmed_mean_predictions",
    "verify_candidate_integrity",
]
