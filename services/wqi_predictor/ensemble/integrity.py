from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from services.wqi_predictor.ensemble.types import (
    ArtifactIntegrityError,
    Hemisphere,
    LeakageError,
)

FORBIDDEN_FEATURES: frozenset[str] = frozenset(
    {
        "hemisphere",
        "day_index",
        "year_index",
        "water_quality_index",
    }
)


def compute_sha256(filepath: Path) -> str:
    if not filepath.exists():
        raise ArtifactIntegrityError(f"Target file for hashing does not exist: {filepath}")

    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_canonical_json_hash(payload: Mapping[str, Any]) -> str:
    canonical_str = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()


def load_model_registry(
    *,
    hemisphere: Hemisphere,
    artifacts_root: Path,
) -> dict[str, Any]:
    registry_path = artifacts_root / "model" / hemisphere.value / "registry" / "manifest.json"
    if not registry_path.exists():
        raise ArtifactIntegrityError(
            f"Phase 4 registry manifest missing for hemisphere {hemisphere.value} at {registry_path}"
        )

    try:
        with open(registry_path, encoding="utf-8") as f:
            registry = json.load(f)
    except Exception as exc:
        raise ArtifactIntegrityError(
            f"Failed to parse Phase 4 registry JSON at {registry_path}: {exc}"
        ) from exc

    return registry


def verify_candidate_integrity(
    *,
    hemisphere: Hemisphere,
    registry: Mapping[str, Any],
    artifacts_root: Path,
) -> dict[str, Any]:
    candidates = registry.get("candidates", [])
    if not candidates:
        raise ArtifactIntegrityError(
            f"No candidates found in Phase 4 registry for {hemisphere.value}"
        )

    accepted_candidates: list[dict[str, Any]] = []
    collapsed_candidates: list[dict[str, Any]] = []
    seen_signatures: set[str] = set()

    for cand in candidates:
        name = cand.get("name", "unknown")

        # 1. Baseline gate verification
        if not cand.get("passed_baseline_gate", False):
            continue

        # 2. Leakage flag verification
        if cand.get("leakage_detected", True):
            raise LeakageError(f"Candidate {name} flagged with leakage_detected=True in Phase 4.")

        # 3. Forbidden feature audit
        features = cand.get("selected_features", [])
        for feat in features:
            if feat in FORBIDDEN_FEATURES:
                raise LeakageError(f"Candidate {name} contains forbidden leakage feature: {feat}")

        # 4. Best iteration count check for GBDTs
        family = cand.get("model_family", "").lower()
        if family in {"lightgbm", "xgboost", "catboost"}:
            best_iter = cand.get("best_iteration_count")
            if best_iter is None or int(best_iter) <= 0:
                raise ArtifactIntegrityError(
                    f"Boosted tree candidate {name} missing valid best_iteration_count."
                )

        # 5. Duplicate model stream collapse (Section 3.2 South CatBoost anomaly)
        # Unique signature: model_family + hyperparameters summary
        sig_payload = {
            "family": family,
            "loss": cand.get("loss_name", ""),
            "params": cand.get("hyperparameters", {}),
        }
        sig_hash = compute_canonical_json_hash(sig_payload)

        if sig_hash in seen_signatures:
            collapsed_candidates.append(
                {
                    "name": name,
                    "reason": "Exact architectural and hyperparameter duplicate of previously accepted candidate.",
                    "action": "Collapsed into primary representative.",
                }
            )
            continue

        seen_signatures.add(sig_hash)
        accepted_candidates.append(cand)

    if not accepted_candidates:
        raise ArtifactIntegrityError(
            f"Zero valid candidates survived integrity audit for hemisphere {hemisphere.value}."
        )

    return {
        "hemisphere": hemisphere.value,
        "total_evaluated": len(candidates),
        "accepted_count": len(accepted_candidates),
        "accepted_candidates": accepted_candidates,
        "collapsed_count": len(collapsed_candidates),
        "collapsed_candidates": collapsed_candidates,
        "audit_passed": True,
    }
