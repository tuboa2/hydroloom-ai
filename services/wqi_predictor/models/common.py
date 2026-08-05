from __future__ import annotations

import re
import functools
import hashlib
import logging

from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import joblib
import orjson
import numpy as np
import polars as pl

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, PowerTransformer, RobustScaler, StandardScaler

from ..config import (
    MODEL_CONFIG,
    FORBIDDEN_FEATURES,
    CATEGORICAL_FEATURES,
    PASSTHROUGH_FEATURES,
    STANDARD_FEATURES,
    POWER_FEATURES,
    _FAMILY_ABBREVIATIONS,
)

_IO_POOL = ThreadPoolExecutor(max_workers=4)

logger = logging.getLogger(__name__)

ENGINEERED_PATTERN = re.compile(
    r"_(?:diff|zscore|roll_mean|roll_std|roll_min|roll_max|lag|ewm_)"
)

class ModelError(RuntimeError):
     """Base exception for Model failures."""

class FeatureContractError(ModelError):
    """Raised when frozen feature contract validation fails."""

class ArtifactError(ModelError):
    """Raised when required artifacts are missing or invalid."""

def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved

def _json_default(obj: Any) -> Any:
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)

    dict_repr = getattr(obj, "__dict__", None)
    if dict_repr is not None:
        return dict_repr

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()

    return str(obj)

def write_json(path: str | Path, payload: Any) -> Path:
    resolved = Path(path)
    ensure_dir(resolved.parent)

    resolved.write_bytes(
        orjson.dumps(
            payload,
            default=_json_default,
            option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY,
        )
    )
    
    return resolved

def read_json(path: str | Path) -> Any:
    resolved = Path(path)

    if not resolved.exists():
        raise ArtifactError(f"Required JSON artifact not found: {resolved}")

    return orjson.loads(resolved.read_bytes())

def clip_predictions(y_pred: np.ndarray | Sequence[float]) -> np.ndarray:
    c_min = float(MODEL_CONFIG["prediction_clip_min"])
    c_max = float(MODEL_CONFIG["prediction_clip_max"])

    arr = np.array(y_pred, dtype=np.float32, copy=True).ravel()

    if not np.all(np.isfinite(arr)):
        logger.warning("Non-finite predictions detected. Replacing with finite bounds.")
        np.nan_to_num(arr, copy=False, nan=c_min, posinf=c_max, neginf=c_min)

    np.clip(arr, c_min, c_max, out=arr)

    return arr

def validate_feature_contract(
    x_train: pl.DataFrame,
    x_val: pl.DataFrame,
    selected_features: Sequence[str],
) -> list[str]:
    if x_train is None or x_val is None:
        raise FeatureContractError("x_train and x_val must be provided.")

    features = list(selected_features) if not isinstance(selected_features, list) else selected_features
    if not features:
        raise FeatureContractError("Selected feature list is empty.")

    feature_set = set(features)

    if len(features) != len(feature_set):
        counts = Counter(features)
        duplicated = [f for f, count in counts.items() if count > 1]
        raise FeatureContractError(f"Duplicated features in contract: {sorted(duplicated)}")

    forbidden = feature_set & FORBIDDEN_FEATURES
    if forbidden:
        raise FeatureContractError(f"Forbidden features present in contract: {sorted(forbidden)}")

    train_cols = set(x_train.columns)
    missing_train = feature_set - train_cols
    if missing_train:
        raise FeatureContractError(f"Features missing from X_train: {sorted(missing_train)}")

    val_cols = set(x_val.columns)
    missing_val = feature_set - val_cols
    if missing_val:
        raise FeatureContractError(f"Features missing from X_validation: {sorted(missing_val)}")

    logger.info(
        "Feature contract validated: %d features, train_rows=%d, validation_rows=%d",
        len(features),
        x_train.height,
        x_val.height,
    )

    return features

def hash_features(feature_columns: Sequence[str], sort: bool = True) -> str:
    hasher = hashlib.sha256()
    
    cols = sorted(feature_columns) if sort else feature_columns

    for col in cols:
        hasher.update(col.encode("utf-8"))
        hasher.update(b"\x00")  
        
    return hasher.hexdigest()

@functools.lru_cache(maxsize=8192)
def _feature_kind(feature_name: str) -> str:
    name = feature_name.strip() if feature_name.startswith(" ") or feature_name.endswith(" ") else feature_name

    if name in CATEGORICAL_FEATURES:
        return "categorical"
    if name in PASSTHROUGH_FEATURES:
        return "passthrough"
    if name in STANDARD_FEATURES:
        return "standard"
    if name in POWER_FEATURES and not ENGINEERED_PATTERN.search(name):
        return "power"

    return "robust"

def build_fold_preprocessor(feature_columns: Sequence[str]) -> ColumnTransformer:
    if not feature_columns:
        raise ModelError("Cannot build preprocessor from empty feature list.")

    groups: dict[str, list[str]] = defaultdict(list)
    for feature in feature_columns:
        groups[_feature_kind(feature)].append(feature)

    transformers: list[tuple[str, Any, list[str]]] = []

    if std_cols := groups.get("standard"):
        transformers.append(("num_standard", StandardScaler(), std_cols))

    if rob_cols := groups.get("robust"):
        transformers.append(("num_robust", RobustScaler(), rob_cols))

    if pwr_cols := groups.get("power"):
        transformers.append((
            "num_power",
            PowerTransformer(method="yeo-johnson", standardize=True),
            pwr_cols,
        ))

    if pass_cols := groups.get("passthrough"):
        transformers.append(("num_passthrough", "passthrough", pass_cols))

    if cat_cols := groups.get("categorical"):
        transformers.append((
            "cat_ordinal",
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                dtype=np.float32,
            ),
            cat_cols,
        ))

    if not transformers:
        raise ModelError("No preprocessing transformers were constructed.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )

@functools.lru_cache(maxsize=256)
def make_study_name(hemisphere: str, model_family: str, loss_name: str) -> str:
    fam_lower = model_family.lower()
    family = _FAMILY_ABBREVIATIONS.get(fam_lower, fam_lower)
    loss = "diversity" if fam_lower == "linear" else loss_name.lower()

    return f"model-{hemisphere.lower()}-{family}-{loss}"

@functools.lru_cache(maxsize=256)
def make_candidate_dirname(model_family: str, loss_name: str) -> str:
    fam_lower = model_family.lower()
    family = _FAMILY_ABBREVIATIONS.get(fam_lower, fam_lower)
    loss = "diversity" if fam_lower == "linear" else loss_name.lower()

    return f"{family}_{loss}"

def save_candidate_artifacts(
    candidate_dir: str | Path,
    model: Any,
    preprocessor: Any,
    best_params: Mapping[str, Any],
    cv_metrics: pl.DataFrame | None,
    val_metrics: Mapping[str, Any],
    report_card: Mapping[str, Any] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> Path:
    resolved_dir = ensure_dir(candidate_dir)

    futures = []

    futures.append(
        _IO_POOL.submit(
            joblib.dump, model, resolved_dir / "model.joblib", compress=("zstd", 3)
        )
    )
    futures.append(
        _IO_POOL.submit(
            joblib.dump, preprocessor, resolved_dir / "preprocessor.joblib", compress=("zstd", 3)
        )
    )

    futures.append(
        _IO_POOL.submit(write_json, resolved_dir / "best_params.json", best_params)
    )
    futures.append(
        _IO_POOL.submit(write_json, resolved_dir / "val_metrics.json", val_metrics)
    )

    cv_path = resolved_dir / "cv_metrics.csv"
    if cv_metrics is not None:
        futures.append(_IO_POOL.submit(cv_metrics.write_csv, cv_path))
    else:
        futures.append(_IO_POOL.submit(cv_path.write_bytes, b"fold,rmse\n"))

    if report_card is not None:
        futures.append(
            _IO_POOL.submit(write_json, resolved_dir / "report_card.json", report_card)
        )

    if extra_metadata is not None:
        futures.append(
            _IO_POOL.submit(
                write_json, resolved_dir / "candidate_metadata.json", extra_metadata
            )
        )

    for future in futures:
        future.result()

    logger.info("Saved candidate artifacts to %s", resolved_dir)
    
    return resolved_dir

def load_candidate_artifacts(
    candidate_dir: str | Path,
    load_models: bool = True,
) -> dict[str, Any]:
    """Fast, concurrent artifact loader for candidate models and metrics."""
    resolved_dir = Path(candidate_dir)

    if not resolved_dir.is_dir():
        raise ArtifactError(f"Candidate directory not found: {resolved_dir}")

    payload: dict[str, Any] = {"candidate_dir": str(resolved_dir)}
    futures: dict[str, Any] = {}

    json_targets = {
        "best_params": resolved_dir / "best_params.json",
        "val_metrics": resolved_dir / "val_metrics.json",
        "report_card": resolved_dir / "report_card.json",
        "candidate_metadata": resolved_dir / "candidate_metadata.json",
    }

    for key, path in json_targets.items():
        if path.is_file():
            futures[key] = _IO_POOL.submit(read_json, path)

    cv_path = resolved_dir / "cv_metrics.csv"
    if cv_path.is_file():
        futures["cv_metrics"] = _IO_POOL.submit(pl.read_csv, cv_path)

    if load_models:
        model_path = resolved_dir / "model.joblib"
        if model_path.is_file():
            futures["model"] = _IO_POOL.submit(joblib.load, model_path)

        preprocessor_path = resolved_dir / "preprocessor.joblib"
        if preprocessor_path.is_file():
            futures["preprocessor"] = _IO_POOL.submit(joblib.load, preprocessor_path)

    for key, future in futures.items():
        payload[key] = future.result()

    return payload
    