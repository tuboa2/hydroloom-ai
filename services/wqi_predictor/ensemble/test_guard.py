from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .integrity import compute_sha256
from .types import (
    Hemisphere,
    TestEvaluationReceipt,
    TestSetAccessError,
)


class TestSetAccessGuard:
    """Context manager protecting the Year 4 Test dataset from multiple evaluations."""

    __test__ = False

    def __init__(
        self,
        *,
        hemisphere: Hemisphere,
        receipt_path: Path,
        final_state_hash: str,
        feature_set_hash: str,
        test_data_path: Path,
    ) -> None:
        self.hemisphere = hemisphere
        self.receipt_path = receipt_path
        self.final_state_hash = final_state_hash
        self.feature_set_hash = feature_set_hash
        self.test_data_path = test_data_path

    def __enter__(self) -> TestSetAccessGuard:
        if self.receipt_path.exists():
            raise TestSetAccessError(
                f"Single-touch test access violated! Receipt already exists at {self.receipt_path}. "
                "Re-evaluating Test data is strictly forbidden."
            )
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        # If an error occurred inside the guard block, do not write a valid receipt
        pass

    def commit_receipt(
        self,
        *,
        predictions_path: Path,
        metrics_path: Path,
    ) -> TestEvaluationReceipt:
        """Write atomic test access receipt upon successful single-pass evaluation."""
        if self.receipt_path.exists():
            raise TestSetAccessError("Receipt already exists during commit attempt.")

        data_hash = compute_sha256(self.test_data_path)
        pred_hash = compute_sha256(predictions_path)
        metric_hash = compute_sha256(metrics_path)
        timestamp = datetime.now(UTC).isoformat()

        receipt = TestEvaluationReceipt(
            hemisphere=self.hemisphere,
            evaluated_at_utc=timestamp,
            final_state_hash=self.final_state_hash,
            feature_set_hash=self.feature_set_hash,
            data_hash=data_hash,
            prediction_sha256=pred_hash,
            metrics_sha256=metric_hash,
        )

        self.receipt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.receipt_path, "w", encoding="utf-8") as f:
            json.dump(receipt.__dict__, f, indent=2)

        return receipt
