from __future__ import annotations
import json
import math
import os
import mlflow
import wandb
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from .config import RANDOM_STATE

MAX_PARAM_LENGTH = 500

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

@dataclass(frozen=True)
class TrackingConfig:
    project: str = "hydroloom-service-a"
    experiment: str = "preprocess-data-governance"
    run_name: str | None = None
    mlflow_tracking_uri: str | None = None
    wandb_mode: str = "online"
    enabled: bool = True

def _flatten_dict(
    data: Mapping[str, Any],
    parent: str = "",
    sep: str = "/"
) -> dict[str, Any]:
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
    return { key: _sanitize_param_value(value) for key, value in params.items() }

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

    def __enter__(self) -> "ExperimentTracker":
        if not self._config.enabled:
            return self
        if self._config.mlflow_tracking_uri:
            mlflow.set_tracking_uri(self._config.mlflow_tracking_uri)
        mlflow.set_experiment(self._config.experiment)
        mlflow.start_run(run_name=self._config.run_name)
        self._mlflow_active = True

        wandb_mode = self._config.wandb_mode
        if wandb_mode == "online" and not os.getenv("WANDB_API_KEY"):
            wandb_mode = "offline"
        run_name = self._config.run_name or mlflow.active_run().info.run_name
        self._wanb_run = wandb.init(
            project=self._config.project,
            name=run_name,
            mode=wandb_mode,
            reinit="finish_previous",
            config={
                "experiment": self._config.experiment,
                "seed": RANDOM_STATE,
            }
        )

        return self

    def log_params(self, params: Mapping[str, Any]) -> None:
        if not self._config.enabled:
            return
        flat_params = _sanitize_params(_flatten_dict(params))
        mlflow.log_params(flat_params)
        if self._wandb_run is not None:
            self._wanb_run.config.update(flat_params, allow_val_change=True)

    def log_metrics(
        self,
        metrics: Mapping[str, Any],
        step: int | None = None
    ) -> None:
        if not self._config.enabled:
            return
        flat_metrics = _sanitize_metrics(_flatten_dict(metrics))
        if not flat_metrics:
            return
        mlflow.log_metrics(flat_metrics, step=step)
        if self._wandb_run is not None:
            self._wandb_run.log(flat_metrics, step=step)

    def log_artifact(self, path: str | Path) -> None:
        if not self._config.enabled:
            return
        artifact_path = Path(path) if not isinstance(path, Path) else path
        if not artifact_path.exists():
            artifact_path.mkdir(parents=True, exist_ok=True)
        mlflow.log_artifact(str(artifact_path))
        if self._wandb_run is None:
            return
        if artifact_path.is_dir():
            for file_path in artifact_path.rglob("*"):
                if file_path.is_file():
                    self._wandb_run.save(str(file_path))
        else:
            self._wandb_run.save(str(artifact_path))

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if not self._config.enabled:
            return False
        if self._wandb_run is not None:
            self._wandb_run.finish()
        if self._mlflow_active:
            mlflow.end_run()
        return False
            