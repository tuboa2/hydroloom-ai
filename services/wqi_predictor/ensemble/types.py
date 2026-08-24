from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class Hemisphere(str, Enum):
    NORTH = "north"
    SOUTH = "south"


class EnsembleStrategy(str, Enum):
    CONSTRAINED_WEIGHTED = "constrained_weighted_average"
    RIDGE_STACK = "ridge_stack"
    ELASTICNET_STACK = "elasticnet_stack"
    ARITHMETIC_MEAN = "arithmetic_mean"
    TRIMMED_MEAN = "trimmed_mean"
    MEAN_MEDIAN_BLEND = "mean_median_blend"


class WQIState(str, Enum):
    INITIALIZED = "INITIALIZED"
    INPUTS_VERIFIED = "INPUTS_VERIFIED"
    CANDIDATES_FROZEN = "CANDIDATES_FROZEN"
    OOF_GENERATED = "OOF_GENERATED"
    ENSEMBLES_EVALUATED = "ENSEMBLES_EVALUATED"
    ENSEMBLE_FROZEN = "ENSEMBLE_FROZEN"
    RESIDUAL_EVALUATED = "RESIDUAL_EVALUATED"
    POSTPROCESSING_FROZEN = "POSTPROCESSING_FROZEN"
    FINAL_CONFIG_FROZEN = "FINAL_CONFIG_FROZEN"
    REFIT_COMPLETE = "REFIT_COMPLETE"
    TEST_EVALUATED = "TEST_EVALUATED"
    REPORTS_COMPLETE = "REPORTS_COMPLETE"
    REGISTERED = "REGISTERED"


class WQIZone(str, Enum):
    CRITICAL = "Critical"  # 0 <= WQI < 25
    POOR = "Poor"  # 25 <= WQI < 50
    MARGINAL = "Marginal"  # 50 <= WQI < 70
    GOOD = "Good"  # 70 <= WQI < 85
    EXCELLENT = "Excellent"  # 85 <= WQI <= 100


class EnsembleError(RuntimeError):
    """Raised for generic ensemble construction or fitting failures."""


class GateViolationError(RuntimeError):
    """Raised when a state machine transition or acceptance gate invariant is violated."""


class LeakageError(RuntimeError):
    """Raised when temporal, feature, target, or Test dataset leakage is detected."""


class OOFGenerationError(RuntimeError):
    """Raised when out-of-fold prediction generation produces invalid or leaked data."""


class WeightOptimizationError(RuntimeError):
    """Raised when SLSQP constrained weight optimization fails or violates invariants."""


class ResidualCorrectionError(RuntimeError):
    """Raised when residual autoregressive model construction or inference fails."""


class TestSetAccessError(RuntimeError):
    """Raised when Year 4 Test data is accessed outside the strict single-touch guard."""


class ArtifactIntegrityError(RuntimeError):
    """Raised when required candidate models, hashes, or metadata artifacts are corrupt or missing."""


@dataclass(frozen=True)
class ModelSpec:
    candidate_name: str
    study_name: str
    model_family: str
    loss_name: str
    seed: int
    feature_set_hash: str
    hyperparameters: Mapping[str, Any]
    best_iteration_count: int
    artifact_dir: Path
    model_id: str


@dataclass(frozen=True)
class OOFResult:
    oof_matrix: np.ndarray  # shape: (N_train, M)
    columns: tuple[str, ...]  # model_ids
    train_row_index: np.ndarray  # 0..1094
    fold_id: np.ndarray  # 0..4, or -1 for burn-in
    valid_oof_mask: np.ndarray  # True for coverable non-burn-in rows
    burn_in_indices: np.ndarray  # indices excluded from Level-1 fitting
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class ValidationPredictionResult:
    columns: tuple[str, ...]
    predictions: np.ndarray  # shape: (365, M)
    row_index: np.ndarray  # 1095..1459
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class EnsembleCandidateResult:
    strategy: EnsembleStrategy
    name: str
    validation_predictions: np.ndarray
    validation_metrics: Mapping[str, float]
    oof_metrics: Mapping[str, float]
    parameters: Mapping[str, Any]
    passed_gate: bool
    failure_reason: str | None


@dataclass(frozen=True)
class ResidualGateResult:
    enabled: bool
    reason: str
    base_metrics: Mapping[str, float]
    corrected_metrics: Mapping[str, float]
    ljung_box_report: Mapping[str, Any]


@dataclass(frozen=True)
class PostprocessingConfig:
    clip_min: float
    clip_max: float
    bounds_rule: str  # 'full' or 'dynamic'
    max_train_wqi: float
    dynamic_candidate_max: float | None
    selected_by: str


@dataclass(frozen=True)
class FinalFrozenState:
    hemisphere: Hemisphere
    feature_set_hash: str
    selected_features: tuple[str, ...]
    level0_specs: tuple[ModelSpec, ...]
    strategy: EnsembleStrategy
    strategy_parameters: Mapping[str, Any]
    residual_enabled: bool
    residual_model_hash: str | None
    postprocessing: PostprocessingConfig
    frozen_state_hash: str


@dataclass(frozen=True)
class TestEvaluationReceipt:
    hemisphere: Hemisphere
    evaluated_at_utc: str
    final_state_hash: str
    feature_set_hash: str
    data_hash: str
    prediction_sha256: str
    metrics_sha256: str


@dataclass(frozen=True)
class DiagnosticZoneMetric:
    zone_name: str
    observation_count: int
    prediction_count: int
    mae: float | None
    rmse: float | None
    bias: float | None
    max_absolute_error: float | None


@dataclass(frozen=True)
class MonthlyDiagnosticMetric:
    month: int
    observation_count: int
    rmse: float
    mae: float
    nse: float | None
    max_absolute_error: float


@dataclass(frozen=True)
class DecileDiagnosticMetric:
    decile: int
    observation_count: int
    rmse: float
    mae: float
    bias: float
