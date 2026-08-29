from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import polars as pl

from .. import config
from ..utils.logging_config import get_logger
from . import validator

logger = get_logger(__name__)


@dataclass(frozen=True)
class IngestedHemisphere:
    hemisphere: Literal["north", "south"]
    raw_frame: pl.DataFrame
    feature_frame: pl.DataFrame
    target: pl.Series
    data_hash: str
    metadata: dict[str, Any]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_no_future_leakage_columns(dataframe: pl.DataFrame) -> None:
    future_columns = [
        column
        for column in dataframe.columns
        if validator.matches_any_pattern(column, config.FORBIDDEN_RAISE_PATTERNS)
    ]

    if future_columns:
        logger.error("Future-leakage columns detected: %s", sorted(future_columns))
        raise validator.DataValidationError(
            f"Future-leakage columns are forbidden: {sorted(future_columns)}."
        )


def drop_forbidden_columns(dataframe: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    forbidden = [
        column
        for column in dataframe.columns
        if validator.matches_any_pattern(column, config.FORBIDDEN_DROP_PATTERNS)
    ]

    if not forbidden:
        return dataframe.clone(), []

    logger.info("Dropping %d forbidden columns: %s", len(forbidden), forbidden)
    return dataframe.drop(forbidden), forbidden


def build_feature_frame(dataframe: pl.DataFrame) -> pl.DataFrame:
    drop_columns = [column for column in config.FEATURE_EXCLUSIONS if column in dataframe.columns]
    logger.debug("Building feature frame — dropping %d exclusion columns.", len(drop_columns))
    return dataframe.drop(drop_columns)


def to_float(val):
    return float(val) if val is not None else None


def _target_metadata(target: pl.Series) -> dict[str, Any]:
    return {
        "mean": to_float(target.mean()),
        "std": to_float(target.std()),
        "min": to_float(target.min()),
        "max": to_float(target.max()),
        "zero_count": int((target == 0).sum() or 0),
    }


def load_hemisphere(hemisphere: str, data_dir: Path | None = None) -> IngestedHemisphere:
    """Load, validate, and prepare a hemisphere dataset for downstream processing.

    Args:
        hemisphere: Either ``"north"`` or ``"south"``.
        data_dir: Override for the default data directory.

    Returns:
        An :class:`IngestedHemisphere` with validated raw and feature frames.

    Raises:
        FileNotFoundError: If the parquet file is missing.
        DataValidationError: If governance checks fail.
    """
    resolved_data_dir = Path(data_dir or config.DATA_DIR)
    path = resolved_data_dir / f"{hemisphere}_raw.parquet"

    logger.info("Loading hemisphere=%s from %s", hemisphere, path)

    if not path.is_file():
        logger.error("Missing parquet file: %s", path)
        raise FileNotFoundError(f"Missing required parquet file: {path}.")

    data_hash = _sha256(path)
    logger.debug("Data SHA-256: %s", data_hash)

    dataframe = pl.read_parquet(path)
    dataframe = dataframe.sort(config.DAY_INDEX_COLUMN)

    logger.info(
        "Loaded %s: rows=%d, columns=%d",
        hemisphere,
        dataframe.shape[0],
        dataframe.shape[1],
    )

    validator.validate_schema(dataframe, hemisphere)
    assert_no_future_leakage_columns(dataframe)

    governed_frame, dropped_forbidden = drop_forbidden_columns(dataframe)
    validator.validate_no_nulls(governed_frame)
    validator.validate_hemisphere_constant(governed_frame, hemisphere)
    validator.validate_temporal_index(governed_frame)
    validator.validate_target(governed_frame)
    logger.info("All governance checks passed for %s.", hemisphere)

    feature_frame = build_feature_frame(governed_frame)
    target = governed_frame[config.TARGET_COLUMN].clone()

    null_count = int(sum(governed_frame.null_count().row(0)))

    metadata: dict[str, Any] = {
        "hemisphere": hemisphere,
        "source_path": str(path),
        "data_sha256": data_hash,
        "shape": {
            "rows": int(governed_frame.shape[0]),
            "columns": int(governed_frame.shape[1]),
        },
        "null_count": null_count,
        "dropped_forbidden_columns": dropped_forbidden,
        "feature_columns": list(feature_frame.columns),
        "feature_count": int(feature_frame.shape[1]),
        "target": _target_metadata(target),
    }

    logger.info(
        "Ingestion complete for %s: features=%d, null_count=%d, target_mean=%.4f, target_std=%.4f",
        hemisphere,
        metadata["feature_count"],
        null_count,
        metadata["target"]["mean"] or 0.0,
        metadata["target"]["std"] or 0.0,
    )

    return IngestedHemisphere(
        hemisphere=hemisphere,
        raw_frame=governed_frame,
        feature_frame=feature_frame,
        target=target,
        data_hash=data_hash,
        metadata=metadata,
    )
