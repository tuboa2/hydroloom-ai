from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_REPO_ID = os.getenv("HF_REPO_ID", "tuboa2/hydroloom-ai")
DEFAULT_CACHE_DIR = Path(os.getenv("HYDROLOOM_CACHE_DIR", "/tmp/hydroloom_cache"))


class ArtifactLoader:
    """Thread-safe artifact loader with Hugging Face Hub integration and local fallback."""

    def __init__(
        self,
        repo_id: str = DEFAULT_REPO_ID,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
    ) -> None:
        self.repo_id = repo_id
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: dict[str, Any] = {}

    def get_file(self, filename: str, repo_type: str = "space") -> Path:
        """Resolve a file path from local filesystem, cache, or Hugging Face Hub."""
        # 1. Check direct local repo paths
        local_candidates = [
            ROOT_DIR / filename,
            ROOT_DIR / "artifacts" / filename,
            ROOT_DIR / "services" / "behavior_clustering" / "models" / filename,
            ROOT_DIR / "services" / "behavior_clustering" / "model" / filename,
            self.cache_dir / filename,
        ]
        for candidate in local_candidates:
            if candidate.exists() and candidate.is_file():
                return candidate

        # 2. Attempt download from Hugging Face Hub (only if HF_TOKEN is configured or enabled)
        token = os.getenv("HF_TOKEN")
        if not token and os.getenv("ENVIRONMENT") == "production":
            # Avoid blocking network timeouts during free-tier startup
            raise FileNotFoundError(f"Artifact '{filename}' not found locally (offline mode).")

        try:
            from huggingface_hub import hf_hub_download

            downloaded = hf_hub_download(
                repo_id=self.repo_id,
                filename=filename,
                repo_type=repo_type,
                cache_dir=str(self.cache_dir),
                token=token,
                local_files_only=False,
            )
            return Path(downloaded)
        except Exception as err:
            logger.warning(
                "Could not download %s from Hugging Face Hub (%s): %s",
                filename,
                self.repo_id,
                err,
            )
            raise FileNotFoundError(
                f"Artifact '{filename}' not found locally or on Hugging Face Hub."
            ) from err

    def load_joblib(self, filename: str, repo_type: str = "space") -> Any:
        """Load a serialized Joblib object with in-memory caching."""
        if filename in self._memory_cache:
            return self._memory_cache[filename]

        path = self.get_file(filename, repo_type=repo_type)
        obj = joblib.load(path)
        self._memory_cache[filename] = obj
        return obj

    def load_json(self, filename: str, repo_type: str = "space") -> dict[str, Any]:
        """Load a JSON metadata file with in-memory caching."""
        if filename in self._memory_cache:
            return self._memory_cache[filename]

        path = self.get_file(filename, repo_type=repo_type)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self._memory_cache[filename] = data
        return data

    def load_clustering_model(self, hemisphere: str = "north") -> Any:
        """Load K-Means clustering model for specified hemisphere."""
        filename = f"kmeans_{hemisphere.lower()}_k4.joblib"
        try:
            model = self.load_joblib(filename)
            if hasattr(model, "cluster_centers_"):
                model.cluster_centers_ = model.cluster_centers_.astype(np.float64)
            return model
        except Exception as e:
            logger.warning("Failed to load clustering model %s: %s. Using fallback.", filename, e)
            return self._fallback_kmeans(hemisphere)

    def _fallback_kmeans(self, hemisphere: str) -> Any:
        """Lightweight fallback KMeans clusterer with pre-calculated archetype centers."""
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=4, random_state=2033, n_init=1)
        # Synthetic baseline centers for 4 archetypes:
        # [log_per_capita, dry_spike, efficiency_penalty, landscape_demand]
        if hemisphere.lower() == "north":
            centers = np.array([
                [4.2, 1.05, 0.05, 0.20],  # Cluster 0: Conservationist
                [4.8, 1.25, 0.15, 0.45],  # Cluster 1: Average Consumer
                [5.3, 1.60, 0.35, 0.85],  # Cluster 2: Landscape Heavy
                [5.8, 1.80, 0.65, 0.55],  # Cluster 3: High Volume
            ], dtype=np.float64)
        else:
            centers = np.array([
                [4.1, 1.08, 0.04, 0.25],  # Cluster 0: Conservationist
                [4.7, 1.28, 0.18, 0.50],  # Cluster 1: Average Consumer
                [5.4, 1.65, 0.38, 0.90],  # Cluster 2: Landscape Heavy
                [5.9, 1.85, 0.70, 0.60],  # Cluster 3: High Volume
            ], dtype=np.float64)

        kmeans.cluster_centers_ = centers
        kmeans._n_threads = 1
        return kmeans


# Global singleton instance
hub_loader = ArtifactLoader()
