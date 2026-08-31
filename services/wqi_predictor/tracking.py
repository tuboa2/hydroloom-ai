from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow

import wandb

from .config import RANDOM_STATE
from .utils.logging_config import get_logger

logger = get_logger(__name__)

MAX_PARAM_LENGTH = 500

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"


@dataclass(frozen=True)
class TrackingConfig:
    project: str = "hydroloom-service-a"
    experiment: str = "setup-data-governance"
    run_name: str | None = None
    mlflow_tracking_uri: str | None = None
    wandb_mode: str = "online"
    enabled: bool = True


def _flatten_dict(data: Mapping[str, Any], parent: str = "", sep: str = "/") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{parent}{sep}{key}" if parent else str(key)
        if isinstance(value, Mapping):
            flattened.update(_flatten_dict(value, full_key, sep))
        else:
            flattened[full_key] = value
    return flattened


def _truncate(value: str) -> str:
    if len(value) <= MAX_PARAM_LENGTH:
        return value
    return value[: MAX_PARAM_LENGTH - 3] + "..."


def _sanitize_param_value(value: Any) -> Any:
    # cleaning and serializing parameters for safe logging, metrics, or DB storage
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, str):
            return _truncate(value)
        return value
    if isinstance(value, Path):
        return _truncate(str(value))
    if isinstance(value, (list, tuple, set, frozenset)):
        serialized = json.dumps(sorted(map(str, value)))
        return _truncate(serialized)
    serialized = json.dumps(value, default=str)
    return _truncate(serialized)


def _sanitize_params(params: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _sanitize_param_value(value) for key, value in params.items()}


def _sanitize_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    sanitized: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            sanitized[key] = float(int(value))
            continue
        if isinstance(value, (int, float)) and math.isfinite(value):
            sanitized[key] = float(value)
    return sanitized


class ExperimentTracker:
    # thin dual-tracking wrapper around mlflow and wandb
    def __init__(self, config: TrackingConfig | None = None) -> None:
        self._config = config or TrackingConfig()
        self._wandb_run = None
        self._mlflow_active = False

    def __enter__(self) -> ExperimentTracker:
        if not self._config.enabled:
            logger.info("Experiment tracking is disabled — skipping init.")
            return self

        logger.info(
            "Initializing experiment tracking: project=%s, experiment=%s, run=%s",
            self._config.project,
            self._config.experiment,
            self._config.run_name,
        )

        if self._config.mlflow_tracking_uri:
            mlflow.set_tracking_uri(self._config.mlflow_tracking_uri)
        mlflow.set_experiment(self._config.experiment)
        mlflow.start_run(run_name=self._config.run_name)
        self._mlflow_active = True
        logger.debug("MLflow run started: %s", mlflow.active_run().info.run_id)

        wandb_mode = self._config.wandb_mode
        if wandb_mode == "online" and not os.getenv("WANDB_API_KEY"):
            wandb_mode = "offline"
            logger.warning("WANDB_API_KEY not set — falling back to offline mode.")

        run_name = self._config.run_name or mlflow.active_run().info.run_name
        self._wandb_run = wandb.init(
            project=self._config.project,
            name=run_name,
            mode=wandb_mode,
            reinit="finish_previous",
            config={
                "experiment": self._config.experiment,
                "seed": RANDOM_STATE,
            },
        )
        logger.info(
            "W&B run initialized: name=%s, mode=%s, id=%s",
            run_name,
            wandb_mode,
            getattr(self._wandb_run, "id", "unknown"),
        )

        return self

    def log_params(self, params: Mapping[str, Any]) -> None:
        if not self._config.enabled:
            return
        flat_params = _sanitize_params(_flatten_dict(params))
        logger.debug("Logging %d params to trackers.", len(flat_params))
        mlflow.log_params(flat_params)
        if self._wandb_run is not None:
            self._wandb_run.config.update(flat_params, allow_val_change=True)

    def log_metrics(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        if not self._config.enabled:
            return
        flat_metrics = _sanitize_metrics(_flatten_dict(metrics))
        if not flat_metrics:
            return
        logger.debug(
            "Logging %d metrics (step=%s) to trackers.",
            len(flat_metrics),
            step,
        )
        mlflow.log_metrics(flat_metrics, step=step)
        if self._wandb_run is not None:
            self._wandb_run.log(flat_metrics, step=step)

    def log_artifact(self, path: str | Path) -> None:
        if not self._config.enabled:
            return
        artifact_path = Path(path) if not isinstance(path, Path) else path
        if not artifact_path.exists():
            artifact_path.mkdir(parents=True, exist_ok=True)
        logger.debug("Logging artifact: %s", artifact_path)
        mlflow.log_artifact(str(artifact_path))
        if self._wandb_run is None:
            return
        if artifact_path.is_dir():
            for file_path in artifact_path.rglob("*"):
                if file_path.is_file():
                    self._wandb_run.save(str(file_path))
        else:
            self._wandb_run.save(str(artifact_path))

    def set_summary(self, summary: Mapping[str, Any]) -> None:
        """Write final summary values to W&B (displayed on the run overview)."""
        if not self._config.enabled:
            return
        flat = _sanitize_metrics(_flatten_dict(summary))
        if self._wandb_run is not None:
            for key, value in flat.items():
                self._wandb_run.summary[key] = value
            logger.debug("W&B summary updated with %d values.", len(flat))

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if not self._config.enabled:
            return False

        if exc_type is not None:
            logger.error(
                "Experiment tracker exiting due to exception: %s: %s",
                exc_type.__name__,
                exc_value,
                exc_info=True,
            )

        # Exception-safe cleanup: always attempt both W&B and MLflow teardown.
        try:
            if self._wandb_run is not None:
                self._wandb_run.finish()
                logger.info("W&B run finished successfully.")
        except Exception:
            logger.exception("Failed to finish W&B run.")
        finally:
            try:
                if self._mlflow_active:
                    mlflow.end_run()
                    logger.info("MLflow run ended successfully.")
            except Exception:
                logger.exception("Failed to end MLflow run.")

        return False
