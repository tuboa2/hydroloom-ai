from __future__ import annotations

import numpy as np
import pytest
from numpy.random import Generator
from sklearn.cluster import KMeans

from sims.cluster import (
    ARCHETYPE_CONSERVATIONISTS,
    ARCHETYPE_HEAVY_USERS,
    ARCHETYPE_OUTDOOR_LANDSCAPE,
    ARCHETYPE_STANDARD_CONSUMERS,
    CLUSTER_FEATURE_TEMPLATE,
    ClusterSimulator,
    compute_cluster_daily_means,
    predict_cluster_labels,
    resolve_archetype_mapping,
    subsample_household_indices,
)


# ─── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def rng() -> Generator:
    """Deterministic RNG for reproducible tests."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def synthetic_kmeans_model() -> KMeans:
    """Create a fitted KMeans model with known centroid geometry.

    Centroids are designed so the archetype mapping is deterministic:
        Cluster 0: Conservationists (lowest log_per_capita_usage = 1.0)
        Cluster 1: Heavy Users (highest log_per_capita_usage = 8.0)
        Cluster 2: Outdoor/Landscape (higher landscape_demand_index = 0.9)
        Cluster 3: Standard Consumers (lower landscape_demand_index = 0.4)
    """
    model = KMeans(n_clusters=4, random_state=99, n_init="auto")

    # Create synthetic training data that produces known centroids
    # 4 features: log_per_capita_usage, dry_day_spike_factor,
    #             efficiency_penalty_ratio, landscape_demand_index
    np.random.seed(99)
    data = np.array([
        # Cluster 0: Conservationists — low usage
        *[[1.0, 1.0, 1.2, 0.3] for _ in range(25)],
        # Cluster 1: Heavy Users — high usage
        *[[8.0, 2.0, 1.8, 0.7] for _ in range(25)],
        # Cluster 2: Outdoor/Landscape — moderate usage, high landscape
        *[[4.0, 1.5, 1.4, 0.9] for _ in range(25)],
        # Cluster 3: Standard Consumers — moderate usage, low landscape
        *[[3.5, 1.2, 1.3, 0.4] for _ in range(25)],
    ], dtype=np.float64)

    model.fit(data)
    return model


@pytest.fixture
def synthetic_water_usage_matrix(rng: Generator) -> np.ndarray:
    """Create a synthetic 200 × 100 water usage matrix.

    200 households, 100 days (smaller than production for speed).
    Values range from 50 to 2000 L/day to cover all archetype ranges.
    """
    n_households = 200
    n_days = 100
    # Base usage varies by "cluster" — first 50 are low, next 50 high, etc.
    base = np.array(
        [150.0] * 50  # conservationists
        + [1200.0] * 50  # heavy users
        + [600.0] * 50  # outdoor
        + [400.0] * 50  # standard
    ).reshape(-1, 1)

    # Add daily variation (temporal variance)
    daily_noise = rng.normal(0, 30, size=(n_households, n_days))
    seasonal = 100 * np.sin(2 * np.pi * np.arange(n_days) / 100)

    matrix = (base + daily_noise + seasonal[np.newaxis, :]).astype(np.float32)
    np.maximum(matrix, 10.0, out=matrix)  # Floor at 10 L/day
    return matrix


@pytest.fixture
def synthetic_cluster_labels() -> np.ndarray:
    """Cluster labels matching the synthetic water usage matrix structure.

    200 households: 50 per cluster (0, 1, 2, 3).
    """
    return np.array([0] * 50 + [1] * 50 + [2] * 50 + [3] * 50)


@pytest.fixture
def synthetic_archetype_mapping() -> dict[int, str]:
    """Direct archetype mapping matching the synthetic fixtures."""
    return {
        0: ARCHETYPE_CONSERVATIONISTS,
        1: ARCHETYPE_HEAVY_USERS,
        2: ARCHETYPE_OUTDOOR_LANDSCAPE,
        3: ARCHETYPE_STANDARD_CONSUMERS,
    }


# ─── Unit Tests: resolve_archetype_mapping ──────────────────────────────


class TestResolveArchetypeMapping:
    """Tests for the centroid-to-archetype mapping logic."""

    def test_correct_mapping_from_known_centroids(
        self, synthetic_kmeans_model: KMeans
    ) -> None:
        """Verify archetype assignment from known centroid geometry."""
        mapping = resolve_archetype_mapping(synthetic_kmeans_model)

        assert len(mapping) == 4
        assert set(mapping.keys()) == {0, 1, 2, 3}
        assert set(mapping.values()) == {
            ARCHETYPE_HEAVY_USERS,
            ARCHETYPE_CONSERVATIONISTS,
            ARCHETYPE_OUTDOOR_LANDSCAPE,
            ARCHETYPE_STANDARD_CONSUMERS,
        }

    def test_heavy_users_has_max_log_per_capita(
        self, synthetic_kmeans_model: KMeans
    ) -> None:
        """Heavy Users must be the cluster with highest centroid[:, 0]."""
        mapping = resolve_archetype_mapping(synthetic_kmeans_model)
        centers = synthetic_kmeans_model.cluster_centers_
        heavy_id = [
            k for k, v in mapping.items() if v == ARCHETYPE_HEAVY_USERS
        ][0]
        assert centers[heavy_id, 0] == centers[:, 0].max()

    def test_conservationists_has_min_log_per_capita(
        self, synthetic_kmeans_model: KMeans
    ) -> None:
        """Conservationists must be the cluster with lowest centroid[:, 0]."""
        mapping = resolve_archetype_mapping(synthetic_kmeans_model)
        centers = synthetic_kmeans_model.cluster_centers_
        cons_id = [
            k for k, v in mapping.items()
            if v == ARCHETYPE_CONSERVATIONISTS
        ][0]
        assert centers[cons_id, 0] == centers[:, 0].min()

    def test_wrong_shape_raises_value_error(self) -> None:
        """Model with wrong centroid shape must raise ValueError."""
        model = KMeans(n_clusters=3, n_init="auto")
        model.cluster_centers_ = np.zeros((3, 4))
        with pytest.raises(ValueError, match="Expected KMeans centroids shape"):
            resolve_archetype_mapping(model)

    def test_all_archetypes_unique(
        self, synthetic_kmeans_model: KMeans
    ) -> None:
        """No two clusters should map to the same archetype."""
        mapping = resolve_archetype_mapping(synthetic_kmeans_model)
        values = list(mapping.values())
        assert len(values) == len(set(values))


# ─── Unit Tests: predict_cluster_labels ─────────────────────────────────


class TestPredictClusterLabels:
    """Tests for cluster label prediction."""

    def test_output_shape_matches_input(
        self, synthetic_kmeans_model: KMeans
    ) -> None:
        """Output must have same length as input rows."""
        features = np.random.default_rng(0).random((50, 4))
        labels = predict_cluster_labels(synthetic_kmeans_model, features)
        assert labels.shape == (50,)

    def test_labels_in_valid_range(
        self, synthetic_kmeans_model: KMeans
    ) -> None:
        """All labels must be in {0, 1, 2, 3}."""
        features = np.random.default_rng(0).random((100, 4))
        labels = predict_cluster_labels(synthetic_kmeans_model, features)
        assert set(np.unique(labels)).issubset({0, 1, 2, 3})

    def test_wrong_feature_count_raises_assertion(
        self, synthetic_kmeans_model: KMeans
    ) -> None:
        """Mismatched feature count must fail assertion."""
        features = np.random.default_rng(0).random((50, 3))
        with pytest.raises(AssertionError, match="Feature count mismatch"):
            predict_cluster_labels(synthetic_kmeans_model, features)

    def test_1d_input_raises_assertion(
        self, synthetic_kmeans_model: KMeans
    ) -> None:
        """1-D input must fail assertion."""
        features = np.random.default_rng(0).random(4)
        with pytest.raises(AssertionError, match="Expected 2D"):
            predict_cluster_labels(synthetic_kmeans_model, features)


# ─── Unit Tests: subsample_household_indices ────────────────────────────


class TestSubsampleHouseholdIndices:
    """Tests for the catchment subsampling function."""

    def test_correct_size(self, rng: Generator) -> None:
        """Output must have exactly subsample_size elements."""
        indices = subsample_household_indices(100_000, 5_000, rng)
        assert indices.shape == (5_000,)

    def test_no_duplicates(self, rng: Generator) -> None:
        """All sampled indices must be unique (replace=False)."""
        indices = subsample_household_indices(100_000, 5_000, rng)
        assert len(np.unique(indices)) == 5_000

    def test_sorted_output(self, rng: Generator) -> None:
        """Output must be sorted for cache-friendly access."""
        indices = subsample_household_indices(100_000, 5_000, rng)
        assert np.all(indices[:-1] <= indices[1:])

    def test_within_bounds(self, rng: Generator) -> None:
        """All indices must be in [0, n_households)."""
        n = 100_000
        indices = subsample_household_indices(n, 5_000, rng)
        assert np.all(indices >= 0)
        assert np.all(indices < n)

    def test_reproducibility(self) -> None:
        """Same seed must produce identical indices."""
        rng1 = np.random.default_rng(seed=2032)
        rng2 = np.random.default_rng(seed=2032)
        idx1 = subsample_household_indices(100_000, 5_000, rng1)
        idx2 = subsample_household_indices(100_000, 5_000, rng2)
        np.testing.assert_array_equal(idx1, idx2)

    def test_subsample_exceeds_population_raises(
        self, rng: Generator
    ) -> None:
        """subsample > population must raise ValueError."""
        with pytest.raises(ValueError, match="cannot exceed"):
            subsample_household_indices(100, 200, rng)

    def test_zero_subsample_raises(self, rng: Generator) -> None:
        """subsample_size = 0 must raise ValueError."""
        with pytest.raises(ValueError, match="must be > 0"):
            subsample_household_indices(100, 0, rng)

    def test_full_population_subsample(self, rng: Generator) -> None:
        """subsample == population must return all indices."""
        indices = subsample_household_indices(100, 100, rng)
        assert indices.shape == (100,)
        assert len(np.unique(indices)) == 100


# ─── Unit Tests: compute_cluster_daily_means ────────────────────────────


class TestComputeClusterDailyMeans:
    """Tests for the core aggregation function."""

    def test_output_has_four_features(
        self,
        synthetic_water_usage_matrix: np.ndarray,
        synthetic_cluster_labels: np.ndarray,
        synthetic_archetype_mapping: dict[int, str],
    ) -> None:
        """Must produce exactly 4 feature arrays."""
        n_days = synthetic_water_usage_matrix.shape[1]
        result = compute_cluster_daily_means(
            synthetic_water_usage_matrix,
            synthetic_cluster_labels,
            synthetic_archetype_mapping,
            n_days,
        )
        assert len(result) == 4

    def test_output_shapes(
        self,
        synthetic_water_usage_matrix: np.ndarray,
        synthetic_cluster_labels: np.ndarray,
        synthetic_archetype_mapping: dict[int, str],
    ) -> None:
        """All output arrays must have shape (n_days,)."""
        n_days = synthetic_water_usage_matrix.shape[1]
        result = compute_cluster_daily_means(
            synthetic_water_usage_matrix,
            synthetic_cluster_labels,
            synthetic_archetype_mapping,
            n_days,
        )
        for name, arr in result.items():
            assert arr.shape == (n_days,), f"{name}: shape {arr.shape}"

    def test_output_dtype_float32(
        self,
        synthetic_water_usage_matrix: np.ndarray,
        synthetic_cluster_labels: np.ndarray,
        synthetic_archetype_mapping: dict[int, str],
    ) -> None:
        """All output arrays must be float32 (§10.1)."""
        n_days = synthetic_water_usage_matrix.shape[1]
        result = compute_cluster_daily_means(
            synthetic_water_usage_matrix,
            synthetic_cluster_labels,
            synthetic_archetype_mapping,
            n_days,
        )
        for name, arr in result.items():
            assert arr.dtype == np.float32, f"{name}: dtype {arr.dtype}"

    def test_no_nan_values(
        self,
        synthetic_water_usage_matrix: np.ndarray,
        synthetic_cluster_labels: np.ndarray,
        synthetic_archetype_mapping: dict[int, str],
    ) -> None:
        """No NaN values should be present in any output."""
        n_days = synthetic_water_usage_matrix.shape[1]
        result = compute_cluster_daily_means(
            synthetic_water_usage_matrix,
            synthetic_cluster_labels,
            synthetic_archetype_mapping,
            n_days,
        )
        for name, arr in result.items():
            assert not np.any(np.isnan(arr)), f"NaN in {name}"

    def test_non_negative_values(
        self,
        synthetic_water_usage_matrix: np.ndarray,
        synthetic_cluster_labels: np.ndarray,
        synthetic_archetype_mapping: dict[int, str],
    ) -> None:
        """Water usage means must be non-negative."""
        n_days = synthetic_water_usage_matrix.shape[1]
        result = compute_cluster_daily_means(
            synthetic_water_usage_matrix,
            synthetic_cluster_labels,
            synthetic_archetype_mapping,
            n_days,
        )
        for name, arr in result.items():
            assert np.all(arr >= 0), f"Negative in {name}: min={arr.min()}"

    def test_heavy_users_higher_than_conservationists(
        self,
        synthetic_water_usage_matrix: np.ndarray,
        synthetic_cluster_labels: np.ndarray,
        synthetic_archetype_mapping: dict[int, str],
    ) -> None:
        """Heavy Users mean must exceed Conservationists mean on average.

        This validates the semantic ordering from §6.1 vs §6.2 ideal ranges.
        """
        n_days = synthetic_water_usage_matrix.shape[1]
        result = compute_cluster_daily_means(
            synthetic_water_usage_matrix,
            synthetic_cluster_labels,
            synthetic_archetype_mapping,
            n_days,
        )
        heavy_key = CLUSTER_FEATURE_TEMPLATE.format(
            archetype=ARCHETYPE_HEAVY_USERS
        )
        cons_key = CLUSTER_FEATURE_TEMPLATE.format(
            archetype=ARCHETYPE_CONSERVATIONISTS
        )
        assert result[heavy_key].mean() > result[cons_key].mean(), (
            f"Heavy Users mean ({result[heavy_key].mean():.1f}) should "
            f"exceed Conservationists ({result[cons_key].mean():.1f})"
        )

    def test_temporal_variance_present(
        self,
        synthetic_water_usage_matrix: np.ndarray,
        synthetic_cluster_labels: np.ndarray,
        synthetic_archetype_mapping: dict[int, str],
    ) -> None:
        """All features must have temporal variance (std > 0).

        This is a critical invariant from §0.4 CAUTION — static features
        must not leak into the output.
        """
        n_days = synthetic_water_usage_matrix.shape[1]
        result = compute_cluster_daily_means(
            synthetic_water_usage_matrix,
            synthetic_cluster_labels,
            synthetic_archetype_mapping,
            n_days,
        )
        for name, arr in result.items():
            assert arr.std() > 0, (
                f"Zero temporal variance in {name} violates §0.4"
            )

    def test_feature_name_format(
        self,
        synthetic_water_usage_matrix: np.ndarray,
        synthetic_cluster_labels: np.ndarray,
        synthetic_archetype_mapping: dict[int, str],
    ) -> None:
        """Output keys must match §10.1 schema column names."""
        n_days = synthetic_water_usage_matrix.shape[1]
        result = compute_cluster_daily_means(
            synthetic_water_usage_matrix,
            synthetic_cluster_labels,
            synthetic_archetype_mapping,
            n_days,
        )
        expected_keys = {
            "cluster_heavy_users_daily_mean_liters",
            "cluster_conservationists_daily_mean_liters",
            "cluster_outdoor_landscape_daily_mean_liters",
            "cluster_standard_consumers_daily_mean_liters",
        }
        assert set(result.keys()) == expected_keys

    def test_empty_cluster_produces_zeros(self) -> None:
        """If a cluster has 0 members, output must be zeros (not NaN)."""
        n_days = 10
        matrix = np.ones((20, n_days), dtype=np.float32) * 100
        # All 20 households in cluster 0; clusters 1,2,3 are empty
        labels = np.zeros(20, dtype=int)
        mapping = {
            0: ARCHETYPE_CONSERVATIONISTS,
            1: ARCHETYPE_HEAVY_USERS,
            2: ARCHETYPE_OUTDOOR_LANDSCAPE,
            3: ARCHETYPE_STANDARD_CONSUMERS,
        }
        result = compute_cluster_daily_means(matrix, labels, mapping, n_days)
        # The empty clusters should have zeros
        heavy_key = CLUSTER_FEATURE_TEMPLATE.format(
            archetype=ARCHETYPE_HEAVY_USERS
        )
        np.testing.assert_array_equal(
            result[heavy_key], np.zeros(n_days, dtype=np.float32)
        )

    def test_manual_mean_calculation(self) -> None:
        """Verify mean calculation against hand-computed values."""
        n_days = 3
        # 4 households, 3 days
        matrix = np.array(
            [
                [100.0, 200.0, 300.0],  # Cluster 0
                [400.0, 500.0, 600.0],  # Cluster 0
                [1000.0, 1100.0, 1200.0],  # Cluster 1
                [700.0, 800.0, 900.0],  # Cluster 1
            ],
            dtype=np.float32,
        )
        labels = np.array([0, 0, 1, 1])
        mapping = {
            0: ARCHETYPE_CONSERVATIONISTS,
            1: ARCHETYPE_HEAVY_USERS,
        }
        # Only 2 clusters in mapping — this will fail the 4-cluster postcondition.
        # So we add dummy clusters:
        mapping[2] = ARCHETYPE_OUTDOOR_LANDSCAPE
        mapping[3] = ARCHETYPE_STANDARD_CONSUMERS

        result = compute_cluster_daily_means(matrix, labels, mapping, n_days)

        cons_key = CLUSTER_FEATURE_TEMPLATE.format(
            archetype=ARCHETYPE_CONSERVATIONISTS
        )
        heavy_key = CLUSTER_FEATURE_TEMPLATE.format(
            archetype=ARCHETYPE_HEAVY_USERS
        )

        # Cluster 0 mean: [(100+400)/2, (200+500)/2, (300+600)/2]
        # = [250, 350, 450]
        np.testing.assert_array_almost_equal(
            result[cons_key], [250.0, 350.0, 450.0], decimal=1
        )
        # Cluster 1 mean: [(1000+700)/2, (1100+800)/2, (1200+900)/2]
        # = [850, 950, 1050]
        np.testing.assert_array_almost_equal(
            result[heavy_key], [850.0, 950.0, 1050.0], decimal=1
        )


# ─── Integration Tests ──────────────────────────────────────────────────


class TestClusterSimulatorInit:
    """Tests for ClusterSimulator initialization."""

    def test_valid_initialization(self, rng: Generator) -> None:
        """Valid parameters should not raise."""
        sim = ClusterSimulator(
            rng=rng,
            n_days=1825,
            subsample_size=5000,
            service_b_data_dir="/tmp/test",
            service_b_models_dir="/tmp/test",
        )
        assert sim._n_days == 1825
        assert sim._subsample_size == 5000

    def test_zero_days_raises(self, rng: Generator) -> None:
        """n_days = 0 must raise ValueError."""
        with pytest.raises(ValueError, match="n_days must be > 0"):
            ClusterSimulator(rng=rng, n_days=0)

    def test_negative_subsample_raises(self, rng: Generator) -> None:
        """Negative subsample_size must raise ValueError."""
        with pytest.raises(ValueError, match="subsample_size must be > 0"):
            ClusterSimulator(rng=rng, subsample_size=-1)


# ─── Property-Based Tests ───────────────────────────────────────────────


class TestClusterMeansProperties:
    """Property-based invariant tests for cluster mean computation."""

    @pytest.mark.parametrize("n_households", [100, 500, 2000])
    def test_output_shape_invariant_across_sizes(
        self, n_households: int, rng: Generator
    ) -> None:
        """Output shape must always be (n_days,) regardless of n_households."""
        n_days = 50
        matrix = rng.uniform(
            50, 2000, size=(n_households, n_days)
        ).astype(np.float32)
        labels = rng.integers(0, 4, size=n_households)
        mapping = {
            0: ARCHETYPE_CONSERVATIONISTS,
            1: ARCHETYPE_HEAVY_USERS,
            2: ARCHETYPE_OUTDOOR_LANDSCAPE,
            3: ARCHETYPE_STANDARD_CONSUMERS,
        }
        result = compute_cluster_daily_means(matrix, labels, mapping, n_days)
        for arr in result.values():
            assert arr.shape == (n_days,)

    @pytest.mark.parametrize("seed", [0, 42, 2032, 9999])
    def test_subsample_determinism(self, seed: int) -> None:
        """Same seed must always produce the same subsample."""
        rng1 = np.random.default_rng(seed)
        rng2 = np.random.default_rng(seed)
        idx1 = subsample_household_indices(10_000, 1_000, rng1)
        idx2 = subsample_household_indices(10_000, 1_000, rng2)
        np.testing.assert_array_equal(idx1, idx2)

    def test_mean_bounded_by_matrix_extremes(self, rng: Generator) -> None:
        """Cluster mean for any day must lie within [min, max] of the
        contributing households on that day."""
        n_households = 100
        n_days = 30
        matrix = rng.uniform(
            100, 500, size=(n_households, n_days)
        ).astype(np.float32)
        labels = rng.integers(0, 4, size=n_households)
        mapping = {
            0: ARCHETYPE_CONSERVATIONISTS,
            1: ARCHETYPE_HEAVY_USERS,
            2: ARCHETYPE_OUTDOOR_LANDSCAPE,
            3: ARCHETYPE_STANDARD_CONSUMERS,
        }
        result = compute_cluster_daily_means(matrix, labels, mapping, n_days)
        for cluster_id, archetype_name in mapping.items():
            mask = labels == cluster_id
            if mask.sum() == 0:
                continue
            cluster_data = matrix[mask, :]
            feature_key = CLUSTER_FEATURE_TEMPLATE.format(
                archetype=archetype_name
            )
            mean_arr = result[feature_key]
            for d in range(n_days):
                assert mean_arr[d] >= cluster_data[:, d].min() - 0.01
                assert mean_arr[d] <= cluster_data[:, d].max() + 0.01


# ─── Schema Compliance Tests ────────────────────────────────────────────


class TestSchemaCompliance:
    """Verify output matches §10.1 Final Schema specification."""

    def test_column_names_match_schema(
        self,
        synthetic_water_usage_matrix: np.ndarray,
        synthetic_cluster_labels: np.ndarray,
        synthetic_archetype_mapping: dict[int, str],
    ) -> None:
        """Column names must exactly match §10.1 table."""
        n_days = synthetic_water_usage_matrix.shape[1]
        result = compute_cluster_daily_means(
            synthetic_water_usage_matrix,
            synthetic_cluster_labels,
            synthetic_archetype_mapping,
            n_days,
        )
        schema_columns = {
            "cluster_heavy_users_daily_mean_liters",
            "cluster_conservationists_daily_mean_liters",
            "cluster_outdoor_landscape_daily_mean_liters",
            "cluster_standard_consumers_daily_mean_liters",
        }
        assert set(result.keys()) == schema_columns

    def test_dtype_is_float32(
        self,
        synthetic_water_usage_matrix: np.ndarray,
        synthetic_cluster_labels: np.ndarray,
        synthetic_archetype_mapping: dict[int, str],
    ) -> None:
        """§10.1 specifies Float32 for all cluster features."""
        n_days = synthetic_water_usage_matrix.shape[1]
        result = compute_cluster_daily_means(
            synthetic_water_usage_matrix,
            synthetic_cluster_labels,
            synthetic_archetype_mapping,
            n_days,
        )
        for name, arr in result.items():
            assert arr.dtype == np.float32, (
                f"§10.1 requires Float32 for {name}, got {arr.dtype}"
            )
