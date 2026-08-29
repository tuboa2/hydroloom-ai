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
    candidate_paths = [
        artifacts_root / "model" / hemisphere.value / "candidate_registry.json",
        artifacts_root / "model" / hemisphere.value / "registry" / "manifest.json",
        artifacts_root / "model" / hemisphere.value / "manifest.json",
    ]

    registry_path = None
    for p in candidate_paths:
        if p.exists():
            registry_path = p
            break

    if registry_path is None:
        raise ArtifactIntegrityError(
            f"Phase 4 registry manifest missing for hemisphere {hemisphere.value}. Looked in: {[str(p) for p in candidate_paths]}"
        )

    try:
        with open(registry_path, encoding="utf-8") as f:
            registry = json.load(f)
    except Exception as exc:
        raise ArtifactIntegrityError(
            f"Failed to parse Phase 4 registry JSON at {registry_path}: {exc}"
        ) from exc

    return registry


load_phase4_registry = load_model_registry


def verify_candidate_integrity(
    *,
    hemisphere: Hemisphere,
    registry: Mapping[str, Any],
    artifacts_root: Path,
) -> dict[str, Any]:
    candidates = registry.get("candidates", registry.get("passed_candidates", []))
    if not candidates:
        raise ArtifactIntegrityError(
            f"No candidates found in Phase 4 registry for {hemisphere.value}"
        )

    # Load fallback selected features if not present on each candidate
    selection_features: list[str] = []
    sel_path = artifacts_root / "selection" / hemisphere.value / "selected_features.json"
    if sel_path.exists():
        try:
            with open(sel_path, encoding="utf-8") as f:
                sel_data = json.load(f)
                selection_features = sel_data.get("final_features", [])
        except Exception:
            pass

    feature_set_hash = registry.get("feature_set_hash", "frozen")

    accepted_candidates: list[dict[str, Any]] = []
    collapsed_candidates: list[dict[str, Any]] = []
    seen_signatures: set[str] = set()

    for cand in candidates:
        name = cand.get("name", cand.get("study_name", "unknown"))
        study_name = cand.get("study_name", name)
        family = cand.get("model_family", "").lower()
        loss_name = cand.get("loss_name", cand.get("loss", "rmse"))

        # 1. Baseline gate verification
        if not cand.get("passed_baseline_gate", False):
            continue

        # 2. Leakage flag verification
        if cand.get("leakage_detected", True):
            raise LeakageError(f"Candidate {name} flagged with leakage_detected=True in Phase 4.")

        # Resolve selected features
        features = cand.get("selected_features", selection_features)
        if not features and selection_features:
            features = selection_features

        # 3. Forbidden feature audit
        for feat in features:
            if feat in FORBIDDEN_FEATURES:
                raise LeakageError(f"Candidate {name} contains forbidden leakage feature: {feat}")

        # 4. Best iteration count check for GBDTs
        best_iter = cand.get("best_iteration_count")
        if family in {"lightgbm", "xgboost", "catboost"}:
            if best_iter is None or int(best_iter) <= 0:
                raise ArtifactIntegrityError(
                    f"Boosted tree candidate {name} missing valid best_iteration_count."
                )
        else:
            best_iter = best_iter or 100

        # Resolve hyperparameters
        hyperparameters = dict(cand.get("hyperparameters", {}))
        if not hyperparameters and cand.get("artifact_dir"):
            art_p = Path(cand["artifact_dir"])
            if not art_p.is_absolute():
                art_p = artifacts_root.parent / art_p if not (artifacts_root / art_p).exists() else artifacts_root / art_p
            best_params_p = art_p / "best_params.json"
            if best_params_p.exists():
                try:
                    with open(best_params_p, encoding="utf-8") as f:
                        hyperparameters = json.load(f)
                except Exception:
                    pass

        # 5. Duplicate model stream collapse (Section 3.2 South CatBoost anomaly)
        sig_payload = {
            "family": family,
            "loss": loss_name,
            "params": hyperparameters,
            "best_iter": int(best_iter) if best_iter is not None else 100,
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
        accepted_candidates.append(
            {
                "name": name,
                "study_name": study_name,
                "model_family": family,
                "loss_name": loss_name,
                "best_iteration_count": int(best_iter),
                "hyperparameters": hyperparameters,
                "selected_features": features,
                "feature_set_hash": cand.get("feature_set_hash", feature_set_hash),
            }
        )

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
