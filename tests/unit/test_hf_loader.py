from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from services.wqi_predictor.deployment.hf_loader import ArtifactLoader


def test_artifact_loader_initialization():
    """Verify ArtifactLoader initializes with proper directories."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        loader = ArtifactLoader(repo_id="tuboa2/hydroloom-ai", cache_dir=tmp_dir)
        assert loader.repo_id == "tuboa2/hydroloom-ai"
        assert loader.cache_dir == Path(tmp_dir)
        assert loader.cache_dir.exists()


def test_load_clustering_model_north_and_south():
    """Verify loading K-Means models for North and South hemispheres."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        loader = ArtifactLoader(repo_id="tuboa2/hydroloom-ai", cache_dir=tmp_dir)

        model_north = loader.load_clustering_model("north")
        assert model_north is not None
        assert hasattr(model_north, "cluster_centers_")
        assert model_north.cluster_centers_.shape == (4, 4)

        model_south = loader.load_clustering_model("south")
        assert model_south is not None
        assert hasattr(model_south, "cluster_centers_")
        assert model_south.cluster_centers_.shape == (4, 4)


def test_fallback_kmeans_centroids():
    """Verify fallback KMeans clusterer produces valid cluster predictions."""
    loader = ArtifactLoader()
    fallback_km = loader._fallback_kmeans("north")

    # Predict archetype for typical average household input
    # [log_per_capita=4.8, dry_spike=1.25, penalty=0.15, landscape=0.45]
    sample = np.array([[4.8, 1.25, 0.15, 0.45]], dtype=np.float64)
    pred = fallback_km.predict(sample)

    assert len(pred) == 1
    assert pred[0] in [0, 1, 2, 3]


def test_memory_caching():
    """Verify in-memory cache stores objects across subsequent calls."""
    loader = ArtifactLoader()
    model1 = loader.load_clustering_model("north")
    model2 = loader.load_clustering_model("north")

    # Both references should be identical from cache
    assert model1 is model2
