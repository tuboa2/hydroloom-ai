from __future__ import annotations
import logging
import joblib
import numpy as np
import polars as pl
from pathlib import Path
from numpy.random import Generator
from sklearn.cluster import KMeans
from typing import Literal

logger = logging.getLogger(__name__)

# ─── Archetype Name Constants ──────────────────────────────────────────
# These map to the §6 feature column names. The order here is irrelevant;
# the mapping is determined dynamically from centroid geometry at runtime.
ARCHETYPE_HEAVY_USERS = "heavy_users"
ARCHETYPE_CONSERVATIONISTS = "conservationists"
ARCHETYPE_OUTDOOR_LANDSCAPE = "outdoor_landscape"
ARCHETYPE_STANDARD_CONSUMERS = "standard_consumers"

# Feature column name template (§10.1 schema)
CLUSTER_FEATURE_TEMPLATE = "cluster_{archetype}_daily_mean_liters"

# Subsample size per §0.4 — 5,000 households from the full 100K population
DEFAULT_CATCHMENT_SUBSAMPLE = 5_000

# Minimum households per cluster after subsampling to prevent
# degenerate mean estimates. If any cluster has fewer than this,
# a warning is emitted (but computation proceeds).
MIN_CLUSTER_SIZE = 10

# Expected number of clusters from Service B
EXPECTED_N_CLUSTERS = 4

def resolve_archetype_mapping(kmeans_model: KMeans) -> dict[int, str]: 
    centers = kmeans_model.cluster_centers_

    # ── Pre-conditions ──
    if centers.shape != (EXPECTED_N_CLUSTERS, 4):
        raise ValueError(
            f"Expected KMeans centroids shape (4, 4), got {centers.shape}. "
            f"Ensure the model was trained on the 4-feature Service B schema."
        )

    labels: dict[int, str] = {}

    # Column 0: log_per_capita_usage → identifies volume extremes
    usages = centers[:, 0]
    sorted_usage_indices = np.argsort(usages)

    # Lowest usage → Conservationists
    labels[int(sorted_usage_indices[0])] = ARCHETYPE_CONSERVATIONISTS
    # Highest usage → Heavy Users
    labels[int(sorted_usage_indices[-1])] = ARCHETYPE_HEAVY_USERS

    # Remaining two clusters: differentiated by landscape_demand_index (col 3)
    remaining = [i for i in range(EXPECTED_N_CLUSTERS) if i not in labels]
    assert len(remaining) == 2, (
        f"Expected 2 remaining clusters, got {len(remaining)}"
    )

    landscape_indices = centers[:, 3]
    if landscape_indices[remaining[0]] > landscape_indices[remaining[1]]:
        labels[remaining[0]] = ARCHETYPE_OUTDOOR_LANDSCAPE
        labels[remaining[1]] = ARCHETYPE_STANDARD_CONSUMERS
    else:
        labels[remaining[1]] = ARCHETYPE_OUTDOOR_LANDSCAPE
        labels[remaining[0]] = ARCHETYPE_STANDARD_CONSUMERS

    # ── Post-conditions ──
    assert len(labels) == EXPECTED_N_CLUSTERS, (
        f"Mapping produced {len(labels)} entries, expected {EXPECTED_N_CLUSTERS}"
    )
    assert set(labels.keys()) == set(range(EXPECTED_N_CLUSTERS)), (
        f"Cluster IDs must be {{0,1,2,3}}, got {set(labels.keys())}"
    )
    expected_archetypes = {
        ARCHETYPE_HEAVY_USERS,
        ARCHETYPE_CONSERVATIONISTS,
        ARCHETYPE_OUTDOOR_LANDSCAPE,
        ARCHETYPE_STANDARD_CONSUMERS,
    }
    assert set(labels.values()) == expected_archetypes, (
        f"Archetype set mismatch: {set(labels.values())} != {expected_archetypes}"
    )

    return labels


def predict_cluster_labels(
    kmeans_model: KMeans,
    scaled_features: np.ndarray,
) -> np.ndarray:
    # ── Pre-conditions ──
    assert scaled_features.ndim == 2, (
        f"Expected 2D array, got {scaled_features.ndim}D"
    )
    n_features = kmeans_model.cluster_centers_.shape[1]
    assert scaled_features.shape[1] == n_features, (
        f"Feature count mismatch: input has {scaled_features.shape[1]}, "
        f"model expects {n_features}"
    )

    # 1. Enforce 64-bit precision on the input data
    features_64bit = scaled_features.astype(np.float64)
    
    # 2. Patch the scikit-learn version mismatch by enforcing 64-bit precision 
    # on the unpickled model's internal cluster centers.
    kmeans_model.cluster_centers_ = kmeans_model.cluster_centers_.astype(np.float64)

    # Wrap in np.asarray to enforce a strictly typed ndarray for basedpyright
    labels = np.asarray(kmeans_model.predict(features_64bit))

    # ── Post-conditions ──
    assert labels.shape == (scaled_features.shape[0],), (
        f"Label shape mismatch: expected {(scaled_features.shape[0],)}, got {labels.shape}"
    )
    
    unique_labels = set(np.unique(labels))
    assert unique_labels.issubset({0, 1, 2, 3}), (
        f"Unexpected cluster labels: {unique_labels}"
    )

    return labels
    
def subsample_household_indices(
    n_households: int,
    subsample_size: int,
    rng: Generator,
) -> np.ndarray:  
    # ── Pre-conditions ──
    if subsample_size <= 0:
        raise ValueError(
            f"subsample_size must be > 0, got {subsample_size}"
        )
    if subsample_size > n_households:
        raise ValueError(
            f"subsample_size ({subsample_size}) cannot exceed "
            f"n_households ({n_households})"
        )

    indices = rng.choice(n_households, size=subsample_size, replace=False)
    indices.sort()  # Sort for cache-friendly memory access during slicing

    # ── Post-conditions ──
    assert indices.shape == (subsample_size,)
    assert len(np.unique(indices)) == subsample_size, "Duplicate indices detected"
    assert np.all(indices >= 0) and np.all(indices < n_households)

    return indices


def compute_cluster_daily_means(
    water_usage_matrix: np.ndarray,
    cluster_labels: np.ndarray,
    archetype_mapping: dict[int, str],
    n_days: int,
) -> dict[str, np.ndarray]:
    n_subsample = water_usage_matrix.shape[0]

    # ── Pre-conditions ──
    assert water_usage_matrix.shape == (n_subsample, n_days), (
        f"Matrix shape {water_usage_matrix.shape} != ({n_subsample}, {n_days})"
    )
    assert water_usage_matrix.dtype == np.float32, (
        f"Expected float32, got {water_usage_matrix.dtype}"
    )
    assert cluster_labels.shape == (n_subsample,), (
        f"Labels shape {cluster_labels.shape} != ({n_subsample},)"
    )

    result: dict[str, np.ndarray] = {}

    for cluster_id, archetype_name in archetype_mapping.items():
        mask = cluster_labels == cluster_id
        cluster_count = int(mask.sum())

        feature_name = CLUSTER_FEATURE_TEMPLATE.format(
            archetype=archetype_name
        )

        if cluster_count == 0:
            # Edge case: empty cluster after subsampling.
            # This should be extremely rare with 5K subsample from 100K,
            # but we handle it defensively.
            logger.warning(
                "Cluster %d (%s) has 0 households in subsample. "
                "Filling with zeros. Consider increasing subsample size.",
                cluster_id,
                archetype_name,
            )
            result[feature_name] = np.zeros(n_days, dtype=np.float32)
            continue

        if cluster_count < MIN_CLUSTER_SIZE:
            logger.warning(
                "Cluster %d (%s) has only %d households in subsample "
                "(minimum recommended: %d). Mean estimates may be noisy.",
                cluster_id,
                archetype_name,
                cluster_count,
                MIN_CLUSTER_SIZE,
            )

        # Vectorized mean: boolean mask selects rows, .mean(axis=0) reduces
        # across households for each day column.
        # Time: O(cluster_count × n_days)
        # Memory: O(cluster_count × n_days) for the masked view (no copy
        # with basic indexing in NumPy when mask is boolean)
        cluster_mean = water_usage_matrix[mask, :].mean(axis=0).astype(
            np.float32
        )

        # ── Loop invariant: no NaN values after mean ──
        # NaN can only occur if cluster_count == 0, which is handled above.
        assert not np.any(np.isnan(cluster_mean)), (
            f"NaN detected in {feature_name} mean. "
            f"Cluster count: {cluster_count}"
        )

        # ── Loop invariant: all values non-negative ──
        # Water usage matrix is guaranteed non-negative by
        # HouseholdDemographicSimulator's physiological floor.
        assert np.all(cluster_mean >= 0), (
            f"Negative values in {feature_name}. "
            f"Min: {cluster_mean.min():.4f}"
        )

        # Round to 2 decimal places, matching existing pipeline convention
        np.round(cluster_mean, 2, out=cluster_mean)

        result[feature_name] = cluster_mean

        logger.info(
            "  %s | n=%d | mean=%.1f | std=%.1f | min=%.1f | max=%.1f",
            feature_name,
            cluster_count,
            cluster_mean.mean(),
            cluster_mean.std(),
            cluster_mean.min(),
            cluster_mean.max(),
        )

    # ── Post-conditions ──
    assert len(result) == EXPECTED_N_CLUSTERS, (
        f"Expected {EXPECTED_N_CLUSTERS} feature arrays, got {len(result)}"
    )
    for name, arr in result.items():
        assert arr.shape == (n_days,), f"{name}: shape {arr.shape} != ({n_days},)"
        assert arr.dtype == np.float32, f"{name}: dtype {arr.dtype} != float32"
        assert not np.any(np.isnan(arr)), f"{name}: contains NaN"

    return result


class ClusterSimulator:
    def __init__(
        self,
        rng: Generator,
        n_days: int = 1825,
        subsample_size: int = DEFAULT_CATCHMENT_SUBSAMPLE,
        service_b_data_dir: str = "",
        service_b_models_dir: str = "",
    ) -> None: 
        if n_days <= 0:
            raise ValueError(f"n_days must be > 0, got {n_days}")
        if subsample_size <= 0:
            raise ValueError(
                f"subsample_size must be > 0, got {subsample_size}"
            )

        self._rng = rng
        self._n_days = n_days
        self._subsample_size = subsample_size
        self._service_b_data_dir = service_b_data_dir
        self._service_b_models_dir = service_b_models_dir
        self._logger = logging.getLogger(__name__)

    def _load_kmeans_model(
        self,
        hemisphere: Literal["north", "south"],
    ) -> KMeans:
        model_path = (
            Path(self._service_b_models_dir)
            / f"kmeans_{hemisphere}_k4.joblib"
        )
        if not model_path.exists():
            raise FileNotFoundError(
                f"KMeans model not found at {model_path}. "
                f"Ensure Service B training has been completed."
            )

        model = joblib.load(model_path)
        self._logger.info(
            "Loaded KMeans model for %s hemisphere from %s",
            hemisphere,
            model_path,
        )

        # Validate model structure
        assert isinstance(model, KMeans), (
            f"Expected KMeans, got {type(model).__name__}"
        )
        assert model.cluster_centers_.shape == (EXPECTED_N_CLUSTERS, 4), (
            f"Centroid shape mismatch: {model.cluster_centers_.shape}"
        )

        return model

    def _load_scaled_features(
        self,
        hemisphere: Literal["north", "south"],
    ) -> np.ndarray:
        parquet_path = (
            Path(self._service_b_data_dir)
            / f"{hemisphere}_final.parquet"
        )
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Scaled features not found at {parquet_path}. "
                f"Ensure Service B feature engineering has been completed."
            )

        df = pl.read_parquet(parquet_path)
        features = df.to_numpy().astype(np.float64)

        self._logger.info(
            "Loaded scaled features for %s: shape=%s, dtype=%s",
            hemisphere,
            features.shape,
            features.dtype,
        )

        assert features.ndim == 2, f"Expected 2D array, got {features.ndim}D"
        assert features.shape[1] == 4, (
            f"Expected 4 features, got {features.shape[1]}"
        )

        return features

    def generate_features(
        self,
        water_usage_matrix: np.ndarray,
        *,
        hemisphere: Literal["north", "south"],
        daily_rainfall_mm: np.ndarray,
        consecutive_dry_days: np.ndarray,
    ) -> dict[str, np.ndarray]:       
        n_households = water_usage_matrix.shape[0]

        # ── Pre-conditions ──
        assert water_usage_matrix.shape[1] == self._n_days, (
            f"Matrix has {water_usage_matrix.shape[1]} days, "
            f"expected {self._n_days}"
        )
        assert water_usage_matrix.dtype == np.float32, (
            f"Expected float32, got {water_usage_matrix.dtype}"
        )

        self._logger.info(
            "=== Domain 5: Cluster Metrics for %s hemisphere ===",
            hemisphere.upper(),
        )
        self._logger.info(
            "Input water usage matrix: %s (%.1f MB)",
            water_usage_matrix.shape,
            water_usage_matrix.nbytes / 1e6,
        )

        # Step 1: Load KMeans model
        kmeans_model = self._load_kmeans_model(hemisphere)

        # Step 2: Load scaled features for cluster prediction
        scaled_features = self._load_scaled_features(hemisphere)
        assert scaled_features.shape[0] == n_households, (
            f"Household count mismatch: matrix has {n_households} rows, "
            f"scaled features has {scaled_features.shape[0]} rows"
        )

        # Step 3: Predict cluster labels for ALL households
        all_labels = predict_cluster_labels(kmeans_model, scaled_features)
        self._logger.info(
            "Cluster distribution (full pop): %s",
            {int(k): int(v) for k, v in zip(
                *np.unique(all_labels, return_counts=True)
            )},
        )

        # Step 4: Resolve archetype mapping from centroids
        archetype_mapping = resolve_archetype_mapping(kmeans_model)
        self._logger.info("Archetype mapping: %s", archetype_mapping)

        # Step 5: Subsample 5K household indices
        subsample_indices = subsample_household_indices(
            n_households=n_households,
            subsample_size=self._subsample_size,
            rng=self._rng,
        )
        self._logger.info(
            "Subsampled %d households from %d total",
            self._subsample_size,
            n_households,
        )

        subsample_usage = water_usage_matrix[subsample_indices, :].copy()
        assert subsample_usage.shape == (
            self._subsample_size,
            self._n_days,
        )
        # Step 7: Slice cluster labels to subsample
        subsample_labels = all_labels[subsample_indices]
        self._logger.info(
            "Cluster distribution (subsample): %s",
            {int(k): int(v) for k, v in zip(
                *np.unique(subsample_labels, return_counts=True)
            )},
        )
        
        # --- V2 FIX: Archetype Behavioral Scaling & Weather Reactivity ---
        # The raw demographic matrix lacks the variance required by §6.
        # We inject archetype-specific multipliers and weather-reactive logic.
        if hemisphere == "north":
            mults = {
                ARCHETYPE_HEAVY_USERS: np.float32(1.85),
                ARCHETYPE_STANDARD_CONSUMERS: np.float32(0.85),
                ARCHETYPE_OUTDOOR_LANDSCAPE: np.float32(1.10),
                ARCHETYPE_CONSERVATIONISTS: np.float32(0.45),
            }
        else:
            mults = {
                ARCHETYPE_HEAVY_USERS: np.float32(0.62),
                ARCHETYPE_STANDARD_CONSUMERS: np.float32(0.27),
                ARCHETYPE_OUTDOOR_LANDSCAPE: np.float32(0.42),
                ARCHETYPE_CONSERVATIONISTS: np.float32(0.17),
            }
            
        for cluster_id, archetype_name in archetype_mapping.items():
            mask = subsample_labels == cluster_id
            subsample_usage[mask] *= mults[archetype_name]
            
        # Weather Reactivity for Outdoor/Landscape Cluster
        outdoor_id = next(k for k, v in archetype_mapping.items() if v == ARCHETYPE_OUTDOOR_LANDSCAPE)
        outdoor_mask = subsample_labels == outdoor_id
        
        max_outdoor_demand = np.float32(400.0 if hemisphere == "north" else 250.0)
        cdd_factor = (np.minimum(consecutive_dry_days, 14) / 14.0).astype(np.float32)
        is_dry_today = (daily_rainfall_mm < 0.1).astype(np.float32)
        outdoor_demand_profile = (max_outdoor_demand * cdd_factor * is_dry_today).astype(np.float32)
        
        subsample_usage[outdoor_mask] += outdoor_demand_profile[np.newaxis, :]
        
        # Watering Ban Compliance (§7.1 Hysteresis)
        ban_condition = (consecutive_dry_days >= 10).astype(np.float32)
        outdoor_compliance = (1.0 - 0.40 * ban_condition).astype(np.float32)
        standard_compliance = (1.0 - 0.10 * ban_condition).astype(np.float32)
        
        subsample_usage[outdoor_mask] *= outdoor_compliance[np.newaxis, :]
        
        standard_id = next(k for k, v in archetype_mapping.items() if v == ARCHETYPE_STANDARD_CONSUMERS)
        standard_mask = subsample_labels == standard_id
        subsample_usage[standard_mask] *= standard_compliance[np.newaxis, :]
        
        np.maximum(subsample_usage, 0.0, out=subsample_usage)

        # Step 8: Compute daily cluster means
        result = compute_cluster_daily_means(
            water_usage_matrix=subsample_usage,
            cluster_labels=subsample_labels,
            archetype_mapping=archetype_mapping,
            n_days=self._n_days,
        )

        # ── Post-conditions: temporal variance check ──
        # §0.4 CAUTION: "Static population means are removed as they carry
        # zero intra-hemisphere variance."
        # Our dynamic daily means MUST have temporal variance.
        for name, arr in result.items():
            std_val = float(arr.std())
            if std_val < 1e-6:
                raise RuntimeError(
                    f"FATAL: {name} has near-zero temporal variance "
                    f"(std={std_val:.8f}). This violates §0.4 — the feature "
                    f"would be functionally static and must not be included."
                )
            self._logger.info(
                "Variance check PASSED: %s std=%.2f", name, std_val
            )

        return result
