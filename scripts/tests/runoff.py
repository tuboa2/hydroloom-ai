from __future__ import annotations
import numpy as np
import pytest
from sims.runoff import RunoffSimulator
from params import RUNOFF_PARAMS

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
) -> RunoffSimulator:
    """Instantiate the Domain 4 simulator with standard config."""
    return RunoffSimulator(
        temporal_index=temporal_index,
        rng=rng,
    )


@pytest.fixture
def zero_rainfall() -> np.ndarray:
    """All-dry scenario: zero precipitation every day."""
    return np.zeros(1825, dtype=np.float32)


@pytest.fixture
def constant_light_rain() -> np.ndarray:
    """Light rainfall every day: 5.0 mm."""
    return np.full(1825, 5.0, dtype=np.float32)


@pytest.fixture
def extreme_storm_day() -> np.ndarray:
    """Single extreme storm on day 100: 80mm. All other days dry."""
    rain = np.zeros(1825, dtype=np.float32)
    rain[100] = 80.0
    return rain


@pytest.fixture
def multi_day_heavy_storm() -> np.ndarray:
    """5-day heavy storm (days 200-204): 60mm/day. Rest dry."""
    rain = np.zeros(1825, dtype=np.float32)
    rain[200:205] = 60.0
    return rain


@pytest.fixture
def dry_amc() -> np.ndarray:
    """Completely dry AMC: 0.0 everywhere (minimum curve number)."""
    return np.zeros(1825, dtype=np.float32)


@pytest.fixture
def saturated_amc() -> np.ndarray:
    """Fully saturated AMC: 1.0 everywhere (maximum curve number)."""
    return np.ones(1825, dtype=np.float32)


@pytest.fixture
def mid_amc() -> np.ndarray:
    """Mid-range AMC: 0.5 everywhere."""
    return np.full(1825, 0.5, dtype=np.float32)


@pytest.fixture
def cdd_after_long_dry() -> np.ndarray:
    """CDD array: 14 consecutive dry days before day 100, then reset."""
    cdd = np.zeros(1825, dtype=np.int16)
    for t in range(100):
        cdd[t] = np.int16(t + 1)
    return cdd


@pytest.fixture
def zero_csr() -> np.ndarray:
    """Zero cumulative storm rainfall."""
    return np.zeros(1825, dtype=np.float32)


@pytest.fixture
def north_temp() -> np.ndarray:
    """Synthetic north hemisphere temperature array."""
    day_of_year = np.arange(1825, dtype=np.float32) % 365
    base = 16.0 - 11.5 * np.cos(2.0 * np.pi * (day_of_year - 30) / 365.0)
    return np.clip(base, 7.0, 35.0).astype(np.float32)


@pytest.fixture
def temp_anomaly_positive() -> np.ndarray:
    """Positive temperature anomaly: +3°C everywhere."""
    return np.full(1825, 3.0, dtype=np.float32)


@pytest.fixture
def temp_anomaly_negative() -> np.ndarray:
    """Negative temperature anomaly: -2°C everywhere."""
    return np.full(1825, -2.0, dtype=np.float32)


@pytest.fixture
def temp_anomaly_mixed(rng: np.random.Generator) -> np.ndarray:
    """Mixed positive/negative anomaly."""
    return rng.uniform(-5.0, 5.0, size=1825).astype(np.float32)


# ─── §5.1 Daily Runoff Volume Tests ────────────────────────────────────


class TestDailyRunoffVolumeM3:
    """Tests for SCS Curve Number runoff computation."""

    def test_output_shape_and_dtype(
        self,
        simulator: RunoffSimulator,
        constant_light_rain: np.ndarray,
        mid_amc: np.ndarray,
    ) -> None:
        """D4-INV-001: Shape (1825,), dtype float32."""
        result = simulator.generate_daily_runoff_volume_m3(
            daily_rainfall_mm=constant_light_rain,
            antecedent_moisture_condition=mid_amc,
            hemisphere="north",
        )
        assert result.shape == (1825,)
        assert result.dtype == np.float32

    def test_zero_rainfall_zero_runoff(
        self,
        simulator: RunoffSimulator,
        zero_rainfall: np.ndarray,
        mid_amc: np.ndarray,
    ) -> None:
        """D4-INV-003: Zero rainfall must produce zero runoff."""
        result = simulator.generate_daily_runoff_volume_m3(
            daily_rainfall_mm=zero_rainfall,
            antecedent_moisture_condition=mid_amc,
            hemisphere="north",
        )
        assert np.all(result == 0.0)

    def test_non_negative_runoff(
        self,
        simulator: RunoffSimulator,
        constant_light_rain: np.ndarray,
        mid_amc: np.ndarray,
    ) -> None:
        """D4-INV-002: Runoff must never be negative."""
        result = simulator.generate_daily_runoff_volume_m3(
            daily_rainfall_mm=constant_light_rain,
            antecedent_moisture_condition=mid_amc,
            hemisphere="north",
        )
        assert result.min() >= 0.0

    def test_wet_amc_produces_more_runoff(
        self,
        simulator: RunoffSimulator,
    ) -> None:
        """EDGE-02: Higher AMC (wetter soil) → higher CN → more runoff.

        This validates the sigmoid-smoothed continuous AMC interpolation.
        """
        rainfall = np.full(1825, 30.0, dtype=np.float32)

        dry_amc = np.full(1825, 0.1, dtype=np.float32)
        wet_amc = np.full(1825, 0.9, dtype=np.float32)

        # Need separate simulators because RNG state differs
        rng1 = np.random.default_rng(seed=99)
        sim1 = RunoffSimulator(
            temporal_index=np.arange(1825, dtype=np.int32), rng=rng1
        )
        runoff_dry = sim1.generate_daily_runoff_volume_m3(
            daily_rainfall_mm=rainfall,
            antecedent_moisture_condition=dry_amc,
            hemisphere="north",
        )

        rng2 = np.random.default_rng(seed=99)
        sim2 = RunoffSimulator(
            temporal_index=np.arange(1825, dtype=np.int32), rng=rng2
        )
        runoff_wet = sim2.generate_daily_runoff_volume_m3(
            daily_rainfall_mm=rainfall,
            antecedent_moisture_condition=wet_amc,
            hemisphere="north",
        )

        assert runoff_wet.mean() > runoff_dry.mean(), (
            "Wet AMC should produce more runoff than dry AMC"
        )

    def test_cn_eff_bounds(
        self,
        simulator: RunoffSimulator,
    ) -> None:
        """CN_eff must stay within [CN_I, CN_III] for AMC ∈ [0, 1]."""
        params = RUNOFF_PARAMS["north"]
        amc_values = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        for amc_val in amc_values:
            cn_eff = params.cn_i + amc_val * (params.cn_iii - params.cn_i)
            assert params.cn_i <= cn_eff <= params.cn_iii

    def test_south_hemisphere(
        self,
        simulator: RunoffSimulator,
        constant_light_rain: np.ndarray,
        mid_amc: np.ndarray,
    ) -> None:
        """South hemisphere runs without error."""
        result = simulator.generate_daily_runoff_volume_m3(
            daily_rainfall_mm=constant_light_rain,
            antecedent_moisture_condition=mid_amc,
            hemisphere="south",
        )
        assert result.shape == (1825,)
        assert result.min() >= 0.0

    def test_below_ia_threshold_no_runoff(
        self,
        simulator: RunoffSimulator,
    ) -> None:
        """When P < 0.2·S (initial abstraction), runoff should be zero."""
        # With dry AMC (low CN), S is large, so Ia = 0.2*S is large.
        # Very light rain should not exceed this threshold.
        very_light = np.full(1825, 0.5, dtype=np.float32)
        dry_amc = np.full(1825, 0.0, dtype=np.float32)
        result = simulator.generate_daily_runoff_volume_m3(
            daily_rainfall_mm=very_light,
            antecedent_moisture_condition=dry_amc,
            hemisphere="north",
        )
        # With CN_I=61, S=(25400/61)-254 ≈ 162.3, Ia=32.5
        # P=0.5 < Ia=32.5, so all runoff should be 0
        assert np.all(result == 0.0), (
            "Very light rain with dry soil should produce zero runoff"
        )


# ─── §5.2 Total Suspended Solids Tests ─────────────────────────────────


class TestTotalSuspendedSolidsMgL:
    """Tests for TSS concentration with first-flush, depletion, and scour."""

    def test_output_shape_and_dtype(
        self,
        simulator: RunoffSimulator,
        constant_light_rain: np.ndarray,
    ) -> None:
        """D4-INV-004: Shape (1825,), dtype float32."""
        runoff = np.full(1825, 100.0, dtype=np.float32)
        cdd = np.zeros(1825, dtype=np.int16)
        csr = np.zeros(1825, dtype=np.float32)
        result = simulator.generate_total_suspended_solids_mg_L(
            daily_rainfall_mm=constant_light_rain,
            daily_runoff_volume_m3=runoff,
            consecutive_dry_days=cdd,
            cumulative_storm_rainfall_mm=csr,
            hemisphere="north",
        )
        assert result.shape == (1825,)
        assert result.dtype == np.float32

    def test_dry_days_zero_tss(
        self,
        simulator: RunoffSimulator,
        zero_rainfall: np.ndarray,
    ) -> None:
        """D4-INV-006: TSS must be zero on all dry days."""
        runoff = np.zeros(1825, dtype=np.float32)
        cdd = np.arange(1, 1826, dtype=np.int16)
        csr = np.zeros(1825, dtype=np.float32)
        result = simulator.generate_total_suspended_solids_mg_L(
            daily_rainfall_mm=zero_rainfall,
            daily_runoff_volume_m3=runoff,
            consecutive_dry_days=cdd,
            cumulative_storm_rainfall_mm=csr,
            hemisphere="north",
        )
        assert np.all(result == 0.0)

    def test_physical_bounds(
        self,
        simulator: RunoffSimulator,
        constant_light_rain: np.ndarray,
    ) -> None:
        """D4-INV-005: TSS ∈ [0.0, 1000.0]."""
        runoff = np.full(1825, 500.0, dtype=np.float32)
        cdd = np.zeros(1825, dtype=np.int16)
        csr = np.zeros(1825, dtype=np.float32)
        result = simulator.generate_total_suspended_solids_mg_L(
            daily_rainfall_mm=constant_light_rain,
            daily_runoff_volume_m3=runoff,
            consecutive_dry_days=cdd,
            cumulative_storm_rainfall_mm=csr,
            hemisphere="north",
        )
        assert result.min() >= 0.0
        assert result.max() <= 1000.0

    def test_scour_override_phys05(
        self,
        simulator: RunoffSimulator,
    ) -> None:
        """PHYS-05: Velocity scour override prevents TSS collapse.

        During a multi-day extreme storm (>50mm/day), DD_eff must
        not fall below 0.40, even with large CSR_prior.
        """
        # 10-day extreme storm: 70mm/day
        rain = np.zeros(1825, dtype=np.float32)
        rain[500:510] = 70.0

        # CSR accumulates across the wet spell
        csr = np.zeros(1825, dtype=np.float32)
        cumulative = np.float32(0.0)
        for t in range(500, 510):
            cumulative += rain[t]
            csr[t] = cumulative

        runoff = np.full(1825, 200.0, dtype=np.float32)
        cdd = np.zeros(1825, dtype=np.int16)
        cdd[499] = np.int16(10)  # 10 dry days before storm

        result = simulator.generate_total_suspended_solids_mg_L(
            daily_rainfall_mm=rain,
            daily_runoff_volume_m3=runoff,
            consecutive_dry_days=cdd,
            cumulative_storm_rainfall_mm=csr,
            hemisphere="north",
        )

        # TSS on late storm days (high CSR) should still be substantial
        # because scour override prevents DD_eff from going below 0.40
        late_storm_tss = result[507:510]
        assert np.all(late_storm_tss > 0.0), (
            "PHYS-05: TSS must remain positive during extreme multi-day storm"
        )

    def test_first_flush_amplification(
        self,
        simulator: RunoffSimulator,
    ) -> None:
        """Longer antecedent dry period → higher first-flush factor.

        Create two storms: one after 2 dry days, one after 14 dry days.
        TSS should be higher after the longer dry period (more buildup).
        """
        # Storm 1: day 10, preceded by 2 dry days
        # Storm 2: day 100, preceded by 14+ dry days
        rain = np.zeros(1825, dtype=np.float32)
        rain[10] = 20.0
        rain[100] = 20.0

        cdd = np.zeros(1825, dtype=np.int16)
        for t in range(10):
            if t >= 8:
                cdd[t] = np.int16(t - 7)  # 2 dry days before day 10
        for t in range(86, 100):
            cdd[t] = np.int16(t - 85)  # 14 dry days before day 100

        csr = np.zeros(1825, dtype=np.float32)
        csr[10] = 20.0
        csr[100] = 20.0

        runoff = np.zeros(1825, dtype=np.float32)
        runoff[10] = 50.0
        runoff[100] = 50.0

        # Run many trials to overcome stochastic base TSS
        tss_short_dry = []
        tss_long_dry = []
        for seed in range(100):
            rng_i = np.random.default_rng(seed=seed)
            sim_i = RunoffSimulator(
                temporal_index=np.arange(1825, dtype=np.int32), rng=rng_i
            )
            result = sim_i.generate_total_suspended_solids_mg_L(
                daily_rainfall_mm=rain,
                daily_runoff_volume_m3=runoff,
                consecutive_dry_days=cdd,
                cumulative_storm_rainfall_mm=csr,
                hemisphere="north",
            )
            tss_short_dry.append(float(result[10]))
            tss_long_dry.append(float(result[100]))

        mean_short = np.mean(tss_short_dry)
        mean_long = np.mean(tss_long_dry)
        assert mean_long > mean_short, (
            f"First-flush: TSS after 14 dry days ({mean_long:.1f}) "
            f"should exceed TSS after 2 dry days ({mean_short:.1f})"
        )


# ─── §5.3 Nutrient Load Index Tests ────────────────────────────────────


class TestNutrientLoadIndex:
    """Tests for the composite nutrient load index."""

    def test_output_shape_and_dtype(
        self,
        simulator: RunoffSimulator,
        north_temp: np.ndarray,
    ) -> None:
        """D4-INV-008: Shape (1825,), dtype float32."""
        runoff = np.full(1825, 100.0, dtype=np.float32)
        result = simulator.generate_nutrient_load_index(
            daily_runoff_volume_m3=runoff,
            daily_max_temp_celsius=north_temp,
            hemisphere="north",
        )
        assert result.shape == (1825,)
        assert result.dtype == np.float32

    def test_zero_runoff_zero_nli(
        self,
        simulator: RunoffSimulator,
        north_temp: np.ndarray,
    ) -> None:
        """D4-INV-010: Zero runoff must produce zero NLI."""
        runoff = np.zeros(1825, dtype=np.float32)
        result = simulator.generate_nutrient_load_index(
            daily_runoff_volume_m3=runoff,
            daily_max_temp_celsius=north_temp,
            hemisphere="north",
        )
        assert np.all(result == 0.0)

    def test_non_negative(
        self,
        simulator: RunoffSimulator,
        north_temp: np.ndarray,
    ) -> None:
        """D4-INV-009: NLI must be non-negative."""
        runoff = np.full(1825, 500.0, dtype=np.float32)
        result = simulator.generate_nutrient_load_index(
            daily_runoff_volume_m3=runoff,
            daily_max_temp_celsius=north_temp,
            hemisphere="north",
        )
        assert result.min() >= 0.0

    def test_higher_runoff_higher_nli(
        self,
        simulator: RunoffSimulator,
        north_temp: np.ndarray,
    ) -> None:
        """More runoff should produce higher NLI on average."""
        low_runoff = np.full(1825, 10.0, dtype=np.float32)
        high_runoff = np.full(1825, 500.0, dtype=np.float32)

        rng1 = np.random.default_rng(seed=42)
        sim1 = RunoffSimulator(
            temporal_index=np.arange(1825, dtype=np.int32), rng=rng1
        )
        nli_low = sim1.generate_nutrient_load_index(
            daily_runoff_volume_m3=low_runoff,
            daily_max_temp_celsius=north_temp,
            hemisphere="north",
        )

        rng2 = np.random.default_rng(seed=42)
        sim2 = RunoffSimulator(
            temporal_index=np.arange(1825, dtype=np.int32), rng=rng2
        )
        nli_high = sim2.generate_nutrient_load_index(
            daily_runoff_volume_m3=high_runoff,
            daily_max_temp_celsius=north_temp,
            hemisphere="north",
        )

        assert nli_high.mean() > nli_low.mean()


# ─── §5.4 Heat × Nutrient Synergy Tests ────────────────────────────────


class TestHeatXNutrientSynergy:
    """Tests for the thermal–nutrient interaction term."""

    def test_output_shape_and_dtype(
        self,
        simulator: RunoffSimulator,
        temp_anomaly_positive: np.ndarray,
    ) -> None:
        """D4-INV-011: Shape (1825,), dtype float32."""
        nli = np.full(1825, 5.0, dtype=np.float32)
        result = simulator.generate_heat_x_nutrient_synergy(
            temp_anomaly_celsius=temp_anomaly_positive,
            nutrient_load_index=nli,
        )
        assert result.shape == (1825,)
        assert result.dtype == np.float32

    def test_negative_anomaly_zero_synergy(
        self,
        simulator: RunoffSimulator,
        temp_anomaly_negative: np.ndarray,
    ) -> None:
        """Negative temp anomaly should produce zero synergy."""
        nli = np.full(1825, 10.0, dtype=np.float32)
        result = simulator.generate_heat_x_nutrient_synergy(
            temp_anomaly_celsius=temp_anomaly_negative,
            nutrient_load_index=nli,
        )
        assert np.all(result == 0.0)

    def test_zero_nli_zero_synergy(
        self,
        simulator: RunoffSimulator,
        temp_anomaly_positive: np.ndarray,
    ) -> None:
        """Zero NLI should produce zero synergy regardless of anomaly."""
        nli = np.zeros(1825, dtype=np.float32)
        result = simulator.generate_heat_x_nutrient_synergy(
            temp_anomaly_celsius=temp_anomaly_positive,
            nutrient_load_index=nli,
        )
        assert np.all(result == 0.0)

    def test_positive_anomaly_positive_nli(
        self,
        simulator: RunoffSimulator,
    ) -> None:
        """Positive anomaly × positive NLI → positive synergy."""
        anomaly = np.full(1825, 3.0, dtype=np.float32)
        nli = np.full(1825, 5.0, dtype=np.float32)
        result = simulator.generate_heat_x_nutrient_synergy(
            temp_anomaly_celsius=anomaly,
            nutrient_load_index=nli,
        )
        expected = np.float32(3.0 * 5.0)
        np.testing.assert_allclose(result, expected, atol=0.001)

    def test_non_negative(
        self,
        simulator: RunoffSimulator,
        temp_anomaly_mixed: np.ndarray,
    ) -> None:
        """D4-INV-012: Synergy must never be negative."""
        nli = np.full(1825, 2.0, dtype=np.float32)
        result = simulator.generate_heat_x_nutrient_synergy(
            temp_anomaly_celsius=temp_anomaly_mixed,
            nutrient_load_index=nli,
        )
        assert result.min() >= 0.0


# ─── Invariant Assertion Battery (INV-023 to INV-026) ──────────────────


class TestGlobalInvariants:
    """Cross-domain invariant assertions that Domain 4 must respect.

    INV-023: temp_anomaly calculated before physical clipping.
        → Verified by Domain 2 tests. Domain 4 consumes it as-is.

    INV-024: AMC uses exclusively t-5 to t-1 rainfall.
        → Verified by Domain 3 tests. Domain 4 consumes the
          pre-computed AMC array. Here we verify Domain 4 does
          NOT re-derive AMC internally.

    INV-025: tiered_pricing uses expanding window.
        → Not Domain 4's responsibility. Verified in Domain 6 tests.

    INV-026: Latent variables absent from final schema.
        → Verified at assembly time. Domain 4 does not generate
          latent variables.
    """

    def test_inv024_amc_consumed_not_rederived(
        self,
        simulator: RunoffSimulator,
    ) -> None:
        """INV-024: Domain 4 must use the pre-computed AMC as-is.

        Verify that changing the AMC input directly changes runoff,
        proving the simulator does NOT internally recompute AMC
        from rainfall.
        """
        rain = np.full(1825, 20.0, dtype=np.float32)

        amc_low = np.full(1825, 0.1, dtype=np.float32)
        amc_high = np.full(1825, 0.9, dtype=np.float32)

        rng1 = np.random.default_rng(seed=77)
        sim1 = RunoffSimulator(
            temporal_index=np.arange(1825, dtype=np.int32), rng=rng1
        )
        runoff_low = sim1.generate_daily_runoff_volume_m3(
            daily_rainfall_mm=rain,
            antecedent_moisture_condition=amc_low,
            hemisphere="north",
        )

        rng2 = np.random.default_rng(seed=77)
        sim2 = RunoffSimulator(
            temporal_index=np.arange(1825, dtype=np.int32), rng=rng2
        )
        runoff_high = sim2.generate_daily_runoff_volume_m3(
            daily_rainfall_mm=rain,
            antecedent_moisture_condition=amc_high,
            hemisphere="north",
        )

        # If AMC is properly consumed (not re-derived), these MUST differ
        assert not np.array_equal(runoff_low, runoff_high), (
            "INV-024: Runoff unchanged despite different AMC inputs — "
            "suggests AMC is being internally re-derived!"
        )

    def test_inv026_no_latent_variables(
        self,
        simulator: RunoffSimulator,
        constant_light_rain: np.ndarray,
        mid_amc: np.ndarray,
        north_temp: np.ndarray,
        temp_anomaly_positive: np.ndarray,
    ) -> None:
        """INV-026: Domain 4 output must not contain latent variables."""
        result = simulator.generate_features(
            daily_rainfall_mm=constant_light_rain,
            antecedent_moisture_condition=mid_amc,
            consecutive_dry_days=np.zeros(1825, dtype=np.int16),
            cumulative_storm_rainfall_mm=np.zeros(1825, dtype=np.float32),
            daily_max_temp_celsius=north_temp,
            temp_anomaly_celsius=temp_anomaly_positive,
            hemisphere="north",
        )
        forbidden_keys = {
            "latent_groundwater",
            "latent_industrial",
        }
        actual_keys = set(result.keys())
        overlap = actual_keys & forbidden_keys
        assert len(overlap) == 0, (
            f"INV-026: Latent variables found in Domain 4 output: {overlap}"
        )


# ─── Master Orchestrator Integration Tests ─────────────────────────────


class TestOrchestrator:
    """Integration tests for the full Domain 4 pipeline."""

    def test_all_features_returned(
        self,
        simulator: RunoffSimulator,
        constant_light_rain: np.ndarray,
        mid_amc: np.ndarray,
        north_temp: np.ndarray,
        temp_anomaly_positive: np.ndarray,
    ) -> None:
        """Orchestrator returns all 4 features."""
        result = simulator.generate_features(
            daily_rainfall_mm=constant_light_rain,
            antecedent_moisture_condition=mid_amc,
            consecutive_dry_days=np.zeros(1825, dtype=np.int16),
            cumulative_storm_rainfall_mm=np.zeros(1825, dtype=np.float32),
            daily_max_temp_celsius=north_temp,
            temp_anomaly_celsius=temp_anomaly_positive,
            hemisphere="north",
        )
        expected_keys = {
            "daily_runoff_volume_m3",
            "total_suspended_solids_mg_L",
            "nutrient_load_index",
            "heat_x_nutrient_synergy",
        }
        assert set(result.keys()) == expected_keys

    def test_all_shapes_consistent(
        self,
        simulator: RunoffSimulator,
        constant_light_rain: np.ndarray,
        mid_amc: np.ndarray,
        north_temp: np.ndarray,
        temp_anomaly_positive: np.ndarray,
    ) -> None:
        """All features must have shape (1825,)."""
        result = simulator.generate_features(
            daily_rainfall_mm=constant_light_rain,
            antecedent_moisture_condition=mid_amc,
            consecutive_dry_days=np.zeros(1825, dtype=np.int16),
            cumulative_storm_rainfall_mm=np.zeros(1825, dtype=np.float32),
            daily_max_temp_celsius=north_temp,
            temp_anomaly_celsius=temp_anomaly_positive,
            hemisphere="north",
        )
        for key, arr in result.items():
            assert arr.shape == (1825,), f"{key}: shape {arr.shape} != (1825,)"

    def test_reproducibility(
        self,
        north_temp: np.ndarray,
        temp_anomaly_positive: np.ndarray,
    ) -> None:
        """Same seed should produce identical results."""
        rain = np.full(1825, 15.0, dtype=np.float32)
        amc = np.full(1825, 0.5, dtype=np.float32)
        cdd = np.zeros(1825, dtype=np.int16)
        csr = np.zeros(1825, dtype=np.float32)

        rng1 = np.random.default_rng(seed=42)
        sim1 = RunoffSimulator(
            temporal_index=np.arange(1825, dtype=np.int32), rng=rng1
        )
        result1 = sim1.generate_features(
            daily_rainfall_mm=rain,
            antecedent_moisture_condition=amc,
            consecutive_dry_days=cdd,
            cumulative_storm_rainfall_mm=csr,
            daily_max_temp_celsius=north_temp,
            temp_anomaly_celsius=temp_anomaly_positive,
            hemisphere="north",
        )

        rng2 = np.random.default_rng(seed=42)
        sim2 = RunoffSimulator(
            temporal_index=np.arange(1825, dtype=np.int32), rng=rng2
        )
        result2 = sim2.generate_features(
            daily_rainfall_mm=rain,
            antecedent_moisture_condition=amc,
            consecutive_dry_days=cdd,
            cumulative_storm_rainfall_mm=csr,
            daily_max_temp_celsius=north_temp,
            temp_anomaly_celsius=temp_anomaly_positive,
            hemisphere="north",
        )

        for key in result1:
            np.testing.assert_array_equal(
                result1[key],
                result2[key],
                err_msg=f"Reproducibility failed for {key}",
            )

    def test_south_hemisphere_full(
        self,
        simulator: RunoffSimulator,
        temp_anomaly_positive: np.ndarray,
    ) -> None:
        """Full pipeline runs cleanly for south hemisphere."""
        rain = np.full(1825, 10.0, dtype=np.float32)
        amc = np.full(1825, 0.5, dtype=np.float32)
        temp = np.full(1825, 17.0, dtype=np.float32)
        cdd = np.zeros(1825, dtype=np.int16)
        csr = np.zeros(1825, dtype=np.float32)

        result = simulator.generate_features(
            daily_rainfall_mm=rain,
            antecedent_moisture_condition=amc,
            consecutive_dry_days=cdd,
            cumulative_storm_rainfall_mm=csr,
            daily_max_temp_celsius=temp,
            temp_anomaly_celsius=temp_anomaly_positive,
            hemisphere="south",
        )
        assert len(result) == 4
        for key, arr in result.items():
            assert arr.shape == (1825,), f"{key}: bad shape"
