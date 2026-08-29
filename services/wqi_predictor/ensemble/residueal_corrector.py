"""Backward-compatibility alias for residual_corrector.py."""
from .residual_corrector import (
    ResidualAutoregressiveCorrector,
    compute_ljung_box_stat,
    evaluate_residual_activation_gate,
)

__all__ = [
    "ResidualAutoregressiveCorrector",
    "compute_ljung_box_stat",
    "evaluate_residual_activation_gate",
]
