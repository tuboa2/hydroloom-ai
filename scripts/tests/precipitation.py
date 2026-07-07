from __future__ import annotations
import numpy as np
import pytest
from sims.precipitation import PrecipitationSimulator
from params import (
    AMC_PARAMS,
)

# ─── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def rng() -> np.random.Generator:
    """Provide a seeded RNG for deterministic tests."""
    return np.random.default_rng(seed=2032)


@pytest.fixture
def temporal_index() -> np.ndarray:
    """Standard 5-year temporal index."""
    return np.arange(1825, dtype=np.int32)


@pytest.fixture
def simulator(
    temporal_index: np.ndarray,
    rng: np.random.Generator,
) -> PrecipitationSimulator:
    """Instantiate the simulator with standard config."""
    return PrecipitationSimulator(
        temporal_index=temporal_index,
        rng=rng,
    )


@pytest.fixture
def north_temp(rng: np.random.Generator) -> np.ndarray:
    """Synthetic temperature array mimicking North hemisphere."""
    day_of_year = np.arange(1825, dtype=np.float32) % 365
    base = 16.0 - 11.5 * np.cos(2.0 * np.pi * (day_of_year - 30) / 365.0)
    noise = rng.normal(0, 2.0, size=1825).astype(np.float32)
    return np.clip(base + noise, 7.0, 35.0).astype(np.float32)


@pytest.fixture
def south_temp(rng: np.random.Generator) -> np.ndarray:
    """Synthetic temperature array mimicking South hemisphere."""
    day_of_year = np.arange(1825, dtype=np.float32) % 365
    base = 15.0 + 5.5 * np.cos(2.0 * np.pi * (day_of_year - 40) / 365.0)
    noise = rng.normal(0, 1.5, size=1825).astype(np.float32)
    return np.clip(base + noise, 9.0, 25.0).astype(np.float32)


@pytest.fixture
def north_seasons() -> np.ndarray:
    """Season labels for north hemisphere (1825 days)."""
    labels = np.empty(1825, dtype="<U6")
    for t in range(1825):
        day = t % 365
        if day <= 78 or day >= 355:
            labels[t] = "winter"
        elif 79 <= day <= 171:
            labels[t] = "spring"
        elif 172 <= day <= 265:
            labels[t] = "summer"
        else:
            labels[t] = "autumn"
    return labels


@pytest.fixture
def south_seasons() -> np.ndarray:
    """Season labels for south hemisphere (1825 days)."""
    labels = np.empty(1825, dtype="<U6")
    for t in range(1825):
        shifted_day = (t % 365 + 182) % 365
        if shifted_day <= 78 or shifted_day >= 355:
            labels[t] = "winter"
        elif 79 <= shifted_day <= 171:
            labels[t] = "spring"
        elif 172 <= shifted_day <= 265:
            labels[t] = "summer"
        else:
            labels[t] = "autumn"
    return labels


# ─── §4.1 Daily Rainfall Tests ─────────────────────────────────────────


class TestDailyRainfallMm:
    """Tests for bimodal daily rainfall generation."""

    def test_output_shape_north(
        self,
        simulator: PrecipitationSimulator,
        north_temp: np.ndarray,
    ) -> None:
        """D3-INV-010: Output arrays must have shape (1825,)."""
        result = simulator.generate_daily_rainfall_mm(
            daily_temp=north_temp, hemisphere="north"
        )
        assert result["daily_rainfall_mm"].shape == (1825,)
        assert result["stratiform_rainfall_mm"].shape == (1825,)
        assert result["convective_rainfall_mm"].shape == (1825,)
        assert result["wet_mask"].shape == (1825,)

    def test_output_shape_south(
        self,
        simulator: PrecipitationSimulator,
        south_temp: np.ndarray,
    ) -> None:
        """D3-INV-010: Output arrays must have shape (1825,)."""
        result = simulator.generate_daily_rainfall_mm(
            daily_temp=south_temp, hemisphere="south"
        )
        assert result["daily_rainfall_mm"].shape == (1825,)

    def test_output_dtype(
        self,
        simulator: PrecipitationSimulator,
        north_temp: np.ndarray,
    ) -> None:
        """Output must be float32."""
        result = simulator.generate_daily_rainfall_mm(
            daily_temp=north_temp, hemisphere="north"
        )
        assert result["daily_rainfall_mm"].dtype == np.float32
        assert result["stratiform_rainfall_mm"].dtype == np.float32
        assert result["convective_rainfall_mm"].dtype == np.float32

    def test_physical_bounds_inv001(
        self,
        simulator: PrecipitationSimulator,
        north_temp: np.ndarray,
    ) -> None:
        """D3-INV-001: daily_rainfall_mm ∈ [0.0, 150.0]."""
        result = simulator.generate_daily_rainfall_mm(
            daily_temp=north_temp, hemisphere="north"
        )
        rainfall = result["daily_rainfall_mm"]
        assert rainfall.min() >= 0.0
        assert rainfall.max() <= 150.0

    def test_convective_temperature_gate_inv008(
        self,
        simulator: PrecipitationSimulator,
    ) -> None:
        """D3-INV-008: Convective activates only when T > threshold."""
        # Create temperature array entirely below north threshold (25°C)
        cold_temp = np.full(1825, 20.0, dtype=np.float32)
        result = simulator.generate_daily_rainfall_mm(
            daily_temp=cold_temp, hemisphere="north"
        )
        assert np.all(result["convective_rainfall_mm"] == 0.0), (
            "Convective rainfall should be zero when all temps < 25°C (north)"
        )

    def test_convective_temperature_gate_south(
        self,
        simulator: PrecipitationSimulator,
    ) -> None:
        """D3-INV-008: South convective gate at 20°C."""
        cold_temp = np.full(1825, 18.0, dtype=np.float32)
        result = simulator.generate_daily_rainfall_mm(
            daily_temp=cold_temp, hemisphere="south"
        )
        assert np.all(result["convective_rainfall_mm"] == 0.0)

    def test_bimodal_independence_inv009(
        self,
        simulator: PrecipitationSimulator,
    ) -> None:
        """D3-INV-009: Stratiform and convective are independently sampled.

        With hot temperatures, both components should fire independently.
        Verify that combined rainfall can exceed either component alone.
        """
        hot_temp = np.full(1825, 32.0, dtype=np.float32)
        result = simulator.generate_daily_rainfall_mm(
            daily_temp=hot_temp, hemisphere="north"
        )
        strat = result["stratiform_rainfall_mm"]
        conv = result["convective_rainfall_mm"]
        combined = result["daily_rainfall_mm"]

        # On days where both fire, combined should roughly equal
        # strat + conv (before capping)
        both_wet = (strat > 0) & (conv > 0)
        if both_wet.sum() > 0:
            uncapped = strat[both_wet] + conv[both_wet]
            capped = np.minimum(uncapped, 150.0)
            np.testing.assert_allclose(
                combined[both_wet],
                np.round(capped, 2),
                atol=0.02,
            )

    def test_wet_mask_consistency(
        self,
        simulator: PrecipitationSimulator,
        north_temp: np.ndarray,
    ) -> None:
        """Wet mask must be 1 exactly when rainfall > 0."""
        result = simulator.generate_daily_rainfall_mm(
            daily_temp=north_temp, hemisphere="north"
        )
        rainfall = result["daily_rainfall_mm"]
        wet_mask = result["wet_mask"]
        expected_mask = (rainfall > 0.0).astype(np.uint8)
        np.testing.assert_array_equal(wet_mask, expected_mask)

    def test_both_hemispheres_produce_rainfall(
        self,
        simulator: PrecipitationSimulator,
        north_temp: np.ndarray,
        south_temp: np.ndarray,
    ) -> None:
        """Sanity check: both hemispheres produce nonzero rainfall."""
        n_result = simulator.generate_daily_rainfall_mm(
            daily_temp=north_temp, hemisphere="north"
        )
        s_result = simulator.generate_daily_rainfall_mm(
            daily_temp=south_temp, hemisphere="south"
        )
        assert n_result["daily_rainfall_mm"].sum() > 0.0
        assert s_result["daily_rainfall_mm"].sum() > 0.0


# ─── §4.2 Consecutive Dry Days Tests ───────────────────────────────────


class TestConsecutiveDryDays:
    """Tests for the consecutive dry day counter."""

    def test_output_shape_and_dtype(
        self,
        simulator: PrecipitationSimulator,
    ) -> None:
        """D3-INV-010: Shape (1825,), dtype int16."""
        rainfall = np.zeros(1825, dtype=np.float32)
        cdd = simulator.generate_consecutive_dry_days(rainfall)
        assert cdd.shape == (1825,)
        assert cdd.dtype == np.int16

    def test_all_dry_monotonic_increase(
        self,
        simulator: PrecipitationSimulator,
    ) -> None:
        """All-dry scenario: CDD should be 1, 2, 3, ..., 1825."""
        rainfall = np.zeros(1825, dtype=np.float32)
        cdd = simulator.generate_consecutive_dry_days(rainfall)
        expected = np.arange(1, 1826, dtype=np.int16)
        np.testing.assert_array_equal(cdd, expected)

    def test_all_wet_zero(
        self,
        simulator: PrecipitationSimulator,
    ) -> None:
        """D3-INV-003: All-wet scenario: CDD should be 0 everywhere."""
        rainfall = np.full(1825, 5.0, dtype=np.float32)
        cdd = simulator.generate_consecutive_dry_days(rainfall)
        assert np.all(cdd == 0)

    def test_wet_day_resets_inv003(
        self,
        simulator: PrecipitationSimulator,
    ) -> None:
        """D3-INV-003: CDD resets to 0 on any wet day."""
        rainfall = np.zeros(1825, dtype=np.float32)
        rainfall[10] = 5.0  # Single wet day
        cdd = simulator.generate_consecutive_dry_days(rainfall)
        assert cdd[10] == 0
        assert cdd[9] == 9  # 9 dry days before (days 1-9, CDD[0]=1)
        assert cdd[11] == 1  # First dry day after wet day

    def test_non_negative_inv002(
        self,
        simulator: PrecipitationSimulator,
    ) -> None:
        """D3-INV-002: CDD is always >= 0."""
        rainfall = np.zeros(1825, dtype=np.float32)
        rainfall[::3] = 1.0
        cdd = simulator.generate_consecutive_dry_days(rainfall)
        assert cdd.min() >= 0

    def test_boundary_t0_dry_inv012(
        self,
        simulator: PrecipitationSimulator,
    ) -> None:
        """D3-INV-012: CDD(0) = 1 when rainfall(0) = 0."""
        rainfall = np.zeros(1825, dtype=np.float32)
        cdd = simulator.generate_consecutive_dry_days(rainfall)
        assert cdd[0] == 1

    def test_boundary_t0_wet_inv012(
        self,
        simulator: PrecipitationSimulator,
    ) -> None:
        """D3-INV-012: CDD(0) = 0 when rainfall(0) > 0."""
        rainfall = np.zeros(1825, dtype=np.float32)
        rainfall[0] = 10.0
        cdd = simulator.generate_consecutive_dry_days(rainfall)
        assert cdd[0] == 0

    def test_alternating_pattern(
        self,
        simulator: PrecipitationSimulator,
    ) -> None:
        """Alternating wet/dry pattern: CDD should alternate 0, 1."""
        rainfall = np.zeros(1825, dtype=np.float32)
        rainfall[::2] = 5.0  # Even days are wet
        cdd = simulator.generate_consecutive_dry_days(rainfall)
        for t in range(1825):
            if t % 2 == 0:
                assert cdd[t] == 0, f"Day {t} is wet, CDD should be 0"
            else:
                assert cdd[t] == 1, f"Day {t} is dry after wet, CDD should be 1"


# ─── §4.3 Rolling 7-Day Rainfall Tests ─────────────────────────────────


class TestRolling7dRainfall:
    """Tests for the 7-day trailing moving average."""

    def test_output_shape_and_dtype(
        self,
        simulator: PrecipitationSimulator,
    ) -> None:
        """D3-INV-010: Shape (1825,), dtype float32."""
        rainfall = np.ones(1825, dtype=np.float32) * 7.0
        rolling = simulator.generate_rolling_7d_rainfall_mm(rainfall)
        assert rolling.shape == (1825,)
        assert rolling.dtype == np.float32

    def test_constant_rainfall_equals_value(
        self,
        simulator: PrecipitationSimulator,
    ) -> None:
        """Constant rainfall: 7-day average should equal the constant."""
        rainfall = np.full(1825, 10.0, dtype=np.float32)
        rolling = simulator.generate_rolling_7d_rainfall_mm(rainfall)
        # After day 6, all windows are full
        np.testing.assert_allclose(
            rolling[6:], 10.0, atol=0.02
        )

    def test_bounds_inv004(
        self,
        simulator: PrecipitationSimulator,
    ) -> None:
        """D3-INV-004: rolling_7d ∈ [0.0, 150.0]."""
        rainfall = np.full(1825, 150.0, dtype=np.float32)
        rolling = simulator.generate_rolling_7d_rainfall_mm(rainfall)
        assert rolling.min() >= 0.0
        assert rolling.max() <= 150.0

    def test_partial_window_day0(
        self,
        simulator: PrecipitationSimulator,
    ) -> None:
        """Day 0: window size = 1, so R7(0) = rainfall(0)."""
        rainfall = np.zeros(1825, dtype=np.float32)
        rainfall[0] = 21.0
        rolling = simulator.generate_rolling_7d_rainfall_mm(rainfall)
        np.testing.assert_allclose(rolling[0], 21.0, atol=0.02)

    def test_partial_window_day3(
        self,
        simulator: PrecipitationSimulator,
    ) -> None:
        """Day 3: window size = 4, average of days 0-3."""
        rainfall = np.zeros(1825, dtype=np.float32)
        rainfall[0:4] = [10.0, 20.0, 30.0, 40.0]
        rolling = simulator.generate_rolling_7d_rainfall_mm(rainfall)
        expected = (10.0 + 20.0 + 30.0 + 40.0) / 4.0
        np.testing.assert_allclose(rolling[3], expected, atol=0.02)

    def test_full_window_day6(
        self,
        simulator: PrecipitationSimulator,
    ) -> None:
        """Day 6: first full 7-day window."""
        rainfall = np.zeros(1825, dtype=np.float32)
        rainfall[0:7] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        rolling = simulator.generate_rolling_7d_rainfall_mm(rainfall)
        expected = sum(range(1, 8)) / 7.0
        np.testing.assert_allclose(rolling[6], expected, atol=0.02)

    def test_zero_rainfall(
        self,
        simulator: PrecipitationSimulator,
    ) -> None:
        """All dry: rolling average should be 0."""
        rainfall = np.zeros(1825, dtype=np.float32)
        rolling = simulator.generate_rolling_7d_rainfall_mm(rainfall)
        assert np.all(rolling == 0.0)


# ─── §4.4 Cumulative Storm Rainfall Tests ──────────────────────────────


class TestCumulativeStormRainfall:
    """Tests for the wet-spell accumulator."""

    def test_output_shape_and_dtype(
        self,
        simulator: PrecipitationSimulator,
    ) -> None:
        """D3-INV-010: Shape (1825,), dtype float32."""
        rainfall = np.zeros(1825, dtype=np.float32)
        csr = simulator.generate_cumulative_storm_rainfall_mm(rainfall)
        assert csr.shape == (1825,)
        assert csr.dtype == np.float32

    def test_dry_day_reset_inv005(
        self,
        simulator: PrecipitationSimulator,
    ) -> None:
        """D3-INV-005: CSR resets to 0 on any dry day."""
        rainfall = np.zeros(1825, dtype=np.float32)
        rainfall[0:5] = [10.0, 20.0, 0.0, 30.0, 40.0]
        csr = simulator.generate_cumulative_storm_rainfall_mm(rainfall)
        assert csr[0] == 10.0
        assert csr[1] == 30.0  # 10 + 20
        assert csr[2] == 0.0  # dry day reset
        assert csr[3] == 30.0
        assert csr[4] == 70.0  # 30 + 40

    def test_all_dry_zero(
        self,
        simulator: PrecipitationSimulator,
    ) -> None:
        """All dry: CSR should be 0 everywhere."""
        rainfall = np.zeros(1825, dtype=np.float32)
        csr = simulator.generate_cumulative_storm_rainfall_mm(rainfall)
        assert np.all(csr == 0.0)

    def test_continuous_wet_spell(
        self,
        simulator: PrecipitationSimulator,
    ) -> None:
        """Continuous wet spell: CSR should monotonically accumulate."""
        rainfall = np.full(1825, 5.0, dtype=np.float32)
        csr = simulator.generate_cumulative_storm_rainfall_mm(rainfall)
        # CSR at day t should be 5.0 * (t + 1)
        expected = np.arange(1, 1826, dtype=np.float32) * 5.0
        np.testing.assert_allclose(csr, expected, atol=0.02)

    def test_non_negative(
        self,
        simulator: PrecipitationSimulator,
    ) -> None:
        """CSR must never be negative."""
        rainfall = np.zeros(1825, dtype=np.float32)
        rainfall[::2] = 3.0
        csr = simulator.generate_cumulative_storm_rainfall_mm(rainfall)
        assert csr.min() >= 0.0


# ─── §4.5 AMC Tests ────────────────────────────────────────────────────


class TestAntecedentMoistureCondition:
    """Tests for the Antecedent Moisture Condition (AMC)."""

    def test_output_shape_and_dtype(
        self,
        simulator: PrecipitationSimulator,
        north_seasons: np.ndarray,
    ) -> None:
        """D3-INV-010: Shape (1825,), dtype float32."""
        rainfall = np.zeros(1825, dtype=np.float32)
        amc = simulator.generate_antecedent_moisture_condition(
            daily_rainfall_mm=rainfall,
            season_label=north_seasons,
            hemisphere="north",
        )
        assert amc.shape == (1825,)
        assert amc.dtype == np.float32

    def test_bounds_inv006(
        self,
        simulator: PrecipitationSimulator,
        north_seasons: np.ndarray,
    ) -> None:
        """D3-INV-006: AMC ∈ [0.0, 1.0]."""
        rainfall = np.full(1825, 50.0, dtype=np.float32)
        amc = simulator.generate_antecedent_moisture_condition(
            daily_rainfall_mm=rainfall,
            season_label=north_seasons,
            hemisphere="north",
        )
        assert amc.min() >= 0.0
        assert amc.max() <= 1.0

    def test_cold_start_inv011(
        self,
        simulator: PrecipitationSimulator,
        north_seasons: np.ndarray,
    ) -> None:
        """D3-INV-011: AMC(0) = sigmoid(-k * R5_mid) since R5(0) = 0.

        At t=0, there are no prior days so R5(0) = 0.
        AMC(0) = 1 / (1 + exp(-k * (0 - R5_mid)))
               = 1 / (1 + exp(k * R5_mid))
        This should be a small value (near 0) for typical R5_mid.
        """
        rainfall = np.zeros(1825, dtype=np.float32)
        amc = simulator.generate_antecedent_moisture_condition(
            daily_rainfall_mm=rainfall,
            season_label=north_seasons,
            hemisphere="north",
        )
        params = AMC_PARAMS["north"]
        # At t=0, season determines R5_mid
        season_0 = north_seasons[0]
        if season_0 in ("spring", "summer"):
            r5_mid = params.r5_mid_growing
        else:
            r5_mid = params.r5_mid_dormant
        expected = 1.0 / (1.0 + np.exp(params.k_amc * r5_mid))
        np.testing.assert_allclose(
            float(amc[0]), round(expected, 4), atol=1e-3
        )

    def test_time_causality_inv007_today_excluded(
        self,
        simulator: PrecipitationSimulator,
        north_seasons: np.ndarray,
    ) -> None:
        """D3-INV-007: Today's rainfall MUST NOT affect today's AMC.

        Create a scenario where only day 5 has rain. AMC(5) should
        NOT reflect day 5's rain, but AMC(6) through AMC(10) should.
        """
        rainfall = np.zeros(1825, dtype=np.float32)
        rainfall[5] = 100.0  # Heavy rain ONLY on day 5

        amc = simulator.generate_antecedent_moisture_condition(
            daily_rainfall_mm=rainfall,
            season_label=north_seasons,
            hemisphere="north",
        )

        # AMC(5) should NOT include day 5's rainfall.
        # R5(5) = sum(rainfall[0:5]) = 0.0
        # So AMC(5) should be the same as AMC(0) (both have R5 = 0)
        np.testing.assert_allclose(
            float(amc[5]), float(amc[0]), atol=1e-3,
            err_msg="AMC(5) was affected by day 5's rainfall — CAUSALITY LEAK!"
        )

        # AMC(6) should reflect day 5's rain: R5(6) = rainfall[5] = 100
        # AMC(6) > AMC(5) because R5(6) > R5(5)
        assert float(amc[6]) > float(amc[5]), (
            "AMC(6) should be higher than AMC(5) due to day 5's rain"
        )

    def test_time_causality_inv007_r5_at_0_is_zero(
        self,
        simulator: PrecipitationSimulator,
        north_seasons: np.ndarray,
    ) -> None:
        """D3-INV-007: R5(0) must be exactly 0 (no prior days)."""
        rainfall = np.full(1825, 50.0, dtype=np.float32)
        amc = simulator.generate_antecedent_moisture_condition(
            daily_rainfall_mm=rainfall,
            season_label=north_seasons,
            hemisphere="north",
        )
        # AMC(0) should reflect R5=0, NOT R5=50
        params = AMC_PARAMS["north"]
        season_0 = north_seasons[0]
        if season_0 in ("spring", "summer"):
            r5_mid = params.r5_mid_growing
        else:
            r5_mid = params.r5_mid_dormant
        expected_amc_0 = 1.0 / (1.0 + np.exp(params.k_amc * r5_mid))
        np.testing.assert_allclose(
            float(amc[0]), round(expected_amc_0, 4), atol=1e-3,
        )

    def test_high_rainfall_saturates_amc(
        self,
        simulator: PrecipitationSimulator,
        north_seasons: np.ndarray,
    ) -> None:
        """Sustained heavy rainfall should push AMC toward 1.0."""
        rainfall = np.full(1825, 100.0, dtype=np.float32)
        amc = simulator.generate_antecedent_moisture_condition(
            daily_rainfall_mm=rainfall,
            season_label=north_seasons,
            hemisphere="north",
        )
        # After ~5 days, AMC should be very close to 1.0
        assert float(amc[10]) > 0.95

    def test_zero_rainfall_low_amc(
        self,
        simulator: PrecipitationSimulator,
        north_seasons: np.ndarray,
    ) -> None:
        """Zero rainfall should keep AMC near 0."""
        rainfall = np.zeros(1825, dtype=np.float32)
        amc = simulator.generate_antecedent_moisture_condition(
            daily_rainfall_mm=rainfall,
            season_label=north_seasons,
            hemisphere="north",
        )
        assert float(amc.max()) < 0.5

    def test_season_dependent_midpoint(
        self,
        simulator: PrecipitationSimulator,
    ) -> None:
        """Growing season should have lower midpoint (easier saturation).

        Given the same R5, AMC should be higher in spring/summer
        (lower midpoint) than in autumn/winter (higher midpoint).
        """
        rainfall = np.full(1825, 20.0, dtype=np.float32)
        # Create artificial seasons: first 900 days spring, rest winter
        seasons = np.full(1825, "spring", dtype="<U6")
        seasons[900:] = "winter"

        amc = simulator.generate_antecedent_moisture_condition(
            daily_rainfall_mm=rainfall,
            season_label=seasons,
            hemisphere="north",
        )

        # After warmup, spring AMC should be higher than winter AMC
        # (same R5 but lower midpoint in spring)
        spring_mean = float(amc[10:900].mean())
        winter_mean = float(amc[910:].mean())
        assert spring_mean > winter_mean, (
            f"Spring AMC ({spring_mean:.4f}) should exceed "
            f"winter AMC ({winter_mean:.4f})"
        )

    def test_south_hemisphere(
        self,
        simulator: PrecipitationSimulator,
        south_seasons: np.ndarray,
    ) -> None:
        """South hemisphere should use south-specific parameters."""
        rainfall = np.full(1825, 15.0, dtype=np.float32)
        amc = simulator.generate_antecedent_moisture_condition(
            daily_rainfall_mm=rainfall,
            season_label=south_seasons,
            hemisphere="south",
        )
        assert amc.shape == (1825,)
        assert amc.min() >= 0.0
        assert amc.max() <= 1.0


# ─── Master Orchestrator Tests ──────────────────────────────────────────


class TestOrchestrator:
    """Integration tests for the full Domain 3 pipeline."""

    def test_all_features_returned(
        self,
        simulator: PrecipitationSimulator,
        north_temp: np.ndarray,
        north_seasons: np.ndarray,
    ) -> None:
        """Orchestrator returns all 5 features."""
        result = simulator.generate_features(
            daily_temp=north_temp,
            season_label=north_seasons,
            hemisphere="north",
        )
        expected_keys = {
            "daily_rainfall_mm",
            "consecutive_dry_days",
            "rolling_7d_rainfall_mm",
            "cumulative_storm_rainfall_mm",
            "antecedent_moisture_condition",
        }
        assert set(result.keys()) == expected_keys

    def test_all_shapes_consistent(
        self,
        simulator: PrecipitationSimulator,
        north_temp: np.ndarray,
        north_seasons: np.ndarray,
    ) -> None:
        """All features must have shape (1825,)."""
        result = simulator.generate_features(
            daily_temp=north_temp,
            season_label=north_seasons,
            hemisphere="north",
        )
        for key, arr in result.items():
            assert arr.shape == (1825,), f"{key}: shape {arr.shape} != (1825,)"

    def test_cross_feature_consistency(
        self,
        simulator: PrecipitationSimulator,
        north_temp: np.ndarray,
        north_seasons: np.ndarray,
    ) -> None:
        """Cross-validate: dry days imply zero CSR."""
        result = simulator.generate_features(
            daily_temp=north_temp,
            season_label=north_seasons,
            hemisphere="north",
        )
        rainfall = result["daily_rainfall_mm"]
        cdd = result["consecutive_dry_days"]
        csr = result["cumulative_storm_rainfall_mm"]

        dry_mask = rainfall == 0.0
        # Where rainfall is 0, CDD should be > 0 (unless it's the
        # very first day and it's wet before — but CDD(0) = 1 if dry)
        assert np.all(csr[dry_mask] == 0.0), (
            "CSR should be 0 on all dry days"
        )
        assert np.all(cdd[~dry_mask] == 0), (
            "CDD should be 0 on all wet days"
        )

    def test_south_hemisphere_full(
        self,
        simulator: PrecipitationSimulator,
        south_temp: np.ndarray,
        south_seasons: np.ndarray,
    ) -> None:
        """Full pipeline runs cleanly for south hemisphere."""
        result = simulator.generate_features(
            daily_temp=south_temp,
            season_label=south_seasons,
            hemisphere="south",
        )
        assert len(result) == 5
        for key, arr in result.items():
            assert arr.shape == (1825,), f"{key}: bad shape"

    def test_reproducibility(
        self,
        temporal_index: np.ndarray,
        north_temp: np.ndarray,
        north_seasons: np.ndarray,
    ) -> None:
        """Same seed should produce identical results."""
        rng1 = np.random.default_rng(seed=42)
        sim1 = PrecipitationSimulator(
            temporal_index=temporal_index, rng=rng1
        )
        result1 = sim1.generate_features(
            daily_temp=north_temp,
            season_label=north_seasons,
            hemisphere="north",
        )

        rng2 = np.random.default_rng(seed=42)
        sim2 = PrecipitationSimulator(
            temporal_index=temporal_index, rng=rng2
        )
        result2 = sim2.generate_features(
            daily_temp=north_temp,
            season_label=north_seasons,
            hemisphere="north",
        )

        for key in result1:
            np.testing.assert_array_equal(
                result1[key], result2[key],
                err_msg=f"Reproducibility failed for {key}",
            )
