from __future__ import annotations
import logging
import numpy as np
from typing import Literal
from params import (
    PRECIPITATION_PARAMS,
    AMC_PARAMS,
    PrecipitationParams,
    AMCParams,
)

logger = logging.getLogger(__name__)

class PrecipitationSimulator:
    def __init__(
        self,
        temporal_index: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        self._temporal_index = temporal_index
        self._simulation_days: int = len(temporal_index)
        self._rng = rng
        self._logger = logging.getLogger(__name__)
        self._logger.info(
            "PrecipitationRegimeSimulator Initialized. Days: %d",
            self._simulation_days,
        )

    def generate_daily_rainfall_mm(
        self,
        daily_temp: np.ndarray,
        *,
        hemisphere: Literal["north", "south"],
    ) -> dict[str, np.ndarray]:        
        params: PrecipitationParams = PRECIPITATION_PARAMS[hemisphere]
        n: int = self._simulation_days
        strat = params.stratiform
        conv = params.convective

        temp_f32: np.ndarray = daily_temp.astype(np.float32)

        # ── Component A: Stratiform ────────────────────────────────
        # Sigmoid probability: higher k with positive sign = inverse
        # relationship (cooler temps → more frontal rain)
        strat_exponent: np.ndarray = strat.sigmoid_k * (
            temp_f32 - strat.sigmoid_midpoint
        )
        strat_prob: np.ndarray = np.float32(1.0) / (
            np.float32(1.0) + np.exp(strat_exponent)
        )

        # Bernoulli mask: independent uniform draw
        strat_uniform: np.ndarray = self._rng.uniform(
            0.0, 1.0, size=n
        ).astype(np.float32)
        strat_wet: np.ndarray = (strat_uniform < strat_prob).astype(
            np.uint8
        )

        # Gamma volume: shape=α, scale=β
        strat_volume: np.ndarray = self._rng.gamma(
            shape=strat.gamma_shape,
            scale=strat.gamma_scale,
            size=n,
        ).astype(np.float32)

        # Apply wet floor
        np.maximum(strat_volume, np.float32(strat.wet_floor_mm), out=strat_volume)

        # Masked stratiform rainfall
        stratiform_rainfall: np.ndarray = (
            strat_wet.astype(np.float32) * strat_volume
        )

        # ── Component B: Convective ────────────────────────────────
        # Temperature gate: only activate above threshold (D3-INV-008)
        conv_gate: np.ndarray = (
            temp_f32 > np.float32(conv.activation_temp)
        ).astype(np.uint8)

        # Sigmoid probability within activated region (negative k =
        # direct relationship: hotter → more likely)
        conv_exponent: np.ndarray = conv.sigmoid_k * (
            temp_f32 - conv.sigmoid_midpoint
        )
        conv_prob: np.ndarray = np.float32(1.0) / (
            np.float32(1.0) + np.exp(conv_exponent)
        )

        # Bernoulli mask: INDEPENDENT draw (D3-INV-009)
        conv_uniform: np.ndarray = self._rng.uniform(
            0.0, 1.0, size=n
        ).astype(np.float32)
        conv_wet: np.ndarray = (conv_uniform < conv_prob).astype(
            np.uint8
        )

        # Apply temperature gate: force zero outside activation range
        conv_wet = conv_wet * conv_gate

        # Gamma volume for convective events
        conv_volume: np.ndarray = self._rng.gamma(
            shape=conv.gamma_shape,
            scale=conv.gamma_scale,
            size=n,
        ).astype(np.float32)

        # Apply wet floor
        np.maximum(conv_volume, np.float32(conv.wet_floor_mm), out=conv_volume)

        # Masked convective rainfall
        convective_rainfall: np.ndarray = (
            conv_wet.astype(np.float32) * conv_volume
        )

        # ── Combination ───────────────────────────────────────────
        # rainfall(t) = strat(t) + conv(t), capped at 150.0 (D3-INV-001)
        combined: np.ndarray = stratiform_rainfall + convective_rainfall
        daily_rainfall_mm: np.ndarray = np.clip(
            combined,
            np.float32(0.0),
            np.float32(params.daily_cap_mm),
        ).astype(np.float32)

        # Overall wet mask: any rainfall > 0
        wet_mask: np.ndarray = (daily_rainfall_mm > np.float32(0.0)).astype(
            np.uint8
        )

        # ── Rounding ──────────────────────────────────────────────
        np.round(daily_rainfall_mm, 2, out=daily_rainfall_mm)
        np.round(stratiform_rainfall, 2, out=stratiform_rainfall)
        np.round(convective_rainfall, 2, out=convective_rainfall)

        # ── Assertions ────────────────────────────────────────────
        if __debug__:
            assert daily_rainfall_mm.shape == (n,), (
                f"D3-INV-010: daily_rainfall_mm shape {daily_rainfall_mm.shape} != ({n},)"
            )
            assert daily_rainfall_mm.dtype == np.float32, (
                f"daily_rainfall_mm dtype {daily_rainfall_mm.dtype} != float32"
            )
            assert daily_rainfall_mm.min() >= 0.0, (
                f"D3-INV-001: negative rainfall detected: {daily_rainfall_mm.min()}"
            )
            assert daily_rainfall_mm.max() <= params.daily_cap_mm, (
                f"D3-INV-001: rainfall exceeds cap: {daily_rainfall_mm.max()}"
            )
            # D3-INV-008: convective only where temp > threshold
            if conv_gate.sum() < n:
                cold_idx = np.where(conv_gate == 0)[0]
                assert np.all(convective_rainfall[cold_idx] == 0.0), (
                    "D3-INV-008: convective rainfall detected below activation temp"
                )

        self._logger.info(
            "Domain 3 | %s | daily_rainfall_mm generated. "
            "mean=%.2f | max=%.2f | wet_days=%d/%d | "
            "strat_events=%d | conv_events=%d",
            params.label,
            daily_rainfall_mm.mean(),
            daily_rainfall_mm.max(),
            int(wet_mask.sum()),
            n,
            int(strat_wet.sum()),
            int(conv_wet.sum()),
        )

        return {
            "daily_rainfall_mm": daily_rainfall_mm,
            "stratiform_rainfall_mm": stratiform_rainfall,
            "convective_rainfall_mm": convective_rainfall,
            "wet_mask": wet_mask,
        }

    # ─── §4.2: Consecutive Dry Days ───────────────────────────────────

    def generate_consecutive_dry_days(
        self,
        daily_rainfall_mm: np.ndarray,
    ) -> np.ndarray:
        n: int = daily_rainfall_mm.shape[0]
        cdd: np.ndarray = np.zeros(n, dtype=np.int16)

        # D3-INV-012: Boundary condition at t=0
        if daily_rainfall_mm[0] == 0.0:
            cdd[0] = np.int16(1)
        else:
            cdd[0] = np.int16(0)

        # Sequential recurrence
        for t in range(1, n):
            if daily_rainfall_mm[t] == 0.0:
                cdd[t] = cdd[t - 1] + np.int16(1)
            else:
                cdd[t] = np.int16(0)

        # ── Assertions ────────────────────────────────────────────
        if __debug__:
            assert cdd.shape == (n,), (
                f"D3-INV-010: CDD shape {cdd.shape} != ({n},)"
            )
            assert cdd.min() >= 0, (
                f"D3-INV-002: negative CDD detected: {cdd.min()}"
            )
            wet_days = np.where(daily_rainfall_mm > 0.0)[0]
            if wet_days.size > 0:
                assert np.all(cdd[wet_days] == 0), (
                    "D3-INV-003: CDD != 0 on wet days"
                )

        self._logger.info(
            "Domain 3 | consecutive_dry_days generated. "
            "max=%d | mean=%.2f",
            int(cdd.max()),
            float(cdd.mean()),
        )

        return cdd

    # ─── §4.3: Rolling 7-Day Rainfall ─────────────────────────────────

    def generate_rolling_7d_rainfall_mm(
        self,
        daily_rainfall_mm: np.ndarray,
    ) -> np.ndarray:
        n: int = daily_rainfall_mm.shape[0]
        window: int = 7
        rolling: np.ndarray = np.empty(n, dtype=np.float32)

        # Cumulative sum approach for O(n) computation
        cumsum: np.ndarray = np.zeros(n + 1, dtype=np.float64)
        cumsum[1:] = np.cumsum(daily_rainfall_mm.astype(np.float64))

        for t in range(n):
            start: int = max(0, t - window + 1)
            window_size: int = t - start + 1
            total: float = cumsum[t + 1] - cumsum[start]
            rolling[t] = np.float32(total / window_size)

        # Safety clip (D3-INV-004)
        np.clip(rolling, np.float32(0.0), np.float32(150.0), out=rolling)

        # Round
        np.round(rolling, 2, out=rolling)

        # ── Assertions ────────────────────────────────────────────
        if __debug__:
            assert rolling.shape == (n,), (
                f"D3-INV-010: rolling_7d shape {rolling.shape} != ({n},)"
            )
            assert rolling.min() >= 0.0, (
                f"D3-INV-004: negative rolling_7d: {rolling.min()}"
            )
            assert rolling.max() <= 150.0, (
                f"D3-INV-004: rolling_7d exceeds 150: {rolling.max()}"
            )

        self._logger.info(
            "Domain 3 | rolling_7d_rainfall_mm generated. "
            "mean=%.2f | max=%.2f",
            float(rolling.mean()),
            float(rolling.max()),
        )

        return rolling

    # ─── §4.4: Cumulative Storm Rainfall ──────────────────────────────

    def generate_cumulative_storm_rainfall_mm(
        self,
        daily_rainfall_mm: np.ndarray,
    ) -> np.ndarray:
        n: int = daily_rainfall_mm.shape[0]
        csr: np.ndarray = np.zeros(n, dtype=np.float32)

        # Boundary condition at t=0
        if daily_rainfall_mm[0] > 0.0:
            csr[0] = daily_rainfall_mm[0]
        else:
            csr[0] = np.float32(0.0)

        # Sequential recurrence
        for t in range(1, n):
            if daily_rainfall_mm[t] > 0.0:
                csr[t] = csr[t - 1] + daily_rainfall_mm[t]
            else:
                csr[t] = np.float32(0.0)

        # Round
        np.round(csr, 2, out=csr)

        # ── Assertions ────────────────────────────────────────────
        if __debug__:
            assert csr.shape == (n,), (
                f"D3-INV-010: CSR shape {csr.shape} != ({n},)"
            )
            dry_days = np.where(daily_rainfall_mm == 0.0)[0]
            if dry_days.size > 0:
                assert np.all(csr[dry_days] == 0.0), (
                    "D3-INV-005: CSR != 0 on dry days"
                )
            assert csr.min() >= 0.0, (
                f"CSR negative: {csr.min()}"
            )

        self._logger.info(
            "Domain 3 | cumulative_storm_rainfall_mm generated. "
            "max=%.2f | mean=%.2f",
            float(csr.max()),
            float(csr.mean()),
        )

        return csr

    # ─── §4.5: Antecedent Moisture Condition ──────────────────────────

    def generate_antecedent_moisture_condition(
        self,
        daily_rainfall_mm: np.ndarray,
        season_label: np.ndarray,
        *,
        hemisphere: Literal["north", "south"],
    ) -> np.ndarray:
        amc_params: AMCParams = AMC_PARAMS[hemisphere]
        n: int = daily_rainfall_mm.shape[0]
        lag: int = amc_params.lag_days  # 5

        # ── Step 1: Compute R5(t) using strict lag window ──────────
        # R5(t) = Σ rainfall[max(0, t-5) : t]  (excludes index t)
        #
        # We use a cumulative sum approach for O(n) computation.
        # cumsum[0] = 0, cumsum[k] = Σ rainfall[0..k-1]
        # R5(t) = cumsum[t] - cumsum[max(0, t - lag)]
        rainfall_f64: np.ndarray = daily_rainfall_mm.astype(np.float64)
        cumsum: np.ndarray = np.zeros(n + 1, dtype=np.float64)
        cumsum[1:] = np.cumsum(rainfall_f64)

        # R5(t) = cumsum[t] - cumsum[max(0, t - lag)]
        # At t=0: R5(0) = cumsum[0] - cumsum[0] = 0 (correct: no prior days)
        # At t=1: R5(1) = cumsum[1] - cumsum[0] = rainfall[0] (correct)
        # At t=5: R5(5) = cumsum[5] - cumsum[0] = sum(rainfall[0:5]) (correct)
        # At t=6: R5(6) = cumsum[6] - cumsum[1] = sum(rainfall[1:6]) (correct)
        upper_indices: np.ndarray = np.arange(n, dtype=np.int32)
        lower_indices: np.ndarray = np.maximum(upper_indices - lag, 0)
        r5: np.ndarray = (
            cumsum[upper_indices] - cumsum[lower_indices]
        ).astype(np.float32)

        # ── Step 2: Season-dependent midpoint ──────────────────────
        r5_mid: np.ndarray = np.empty(n, dtype=np.float32)
        growing_mask: np.ndarray = np.isin(
            season_label, ["spring", "summer"]
        )
        dormant_mask: np.ndarray = ~growing_mask

        r5_mid[growing_mask] = np.float32(amc_params.r5_mid_growing)
        r5_mid[dormant_mask] = np.float32(amc_params.r5_mid_dormant)

        # ── Step 3: Sigmoid transformation ─────────────────────────
        # AMC(t) = 1 / (1 + exp(-k_amc * (R5(t) - R5_mid)))
        k: np.float32 = np.float32(amc_params.k_amc)
        sigmoid_arg: np.ndarray = -k * (r5 - r5_mid)
        amc: np.ndarray = np.float32(1.0) / (
            np.float32(1.0) + np.exp(sigmoid_arg)
        )
        amc = amc.astype(np.float32)

        # ── Safety clip (D3-INV-006) ──────────────────────────────
        np.clip(amc, np.float32(0.0), np.float32(1.0), out=amc)

        # Round
        np.round(amc, 4, out=amc)

        # ── Assertions ────────────────────────────────────────────
        if __debug__:
            assert amc.shape == (n,), (
                f"D3-INV-010: AMC shape {amc.shape} != ({n},)"
            )
            assert amc.dtype == np.float32, (
                f"AMC dtype {amc.dtype} != float32"
            )
            assert amc.min() >= 0.0, (
                f"D3-INV-006: AMC below 0: {amc.min()}"
            )
            assert amc.max() <= 1.0, (
                f"D3-INV-006: AMC above 1: {amc.max()}"
            )

            # D3-INV-011: Cold start validation
            expected_amc_0: float = 1.0 / (
                1.0 + np.exp(k * r5_mid[0])
            )
            assert abs(float(amc[0]) - round(expected_amc_0, 4)) < 1e-3, (
                f"D3-INV-011: AMC(0) = {amc[0]}, "
                f"expected ≈ {expected_amc_0:.4f}"
            )

            # D3-INV-007: Verify today's rainfall is excluded
            # For t >= 1, R5(t) should NOT contain rainfall[t]
            assert r5[0] == 0.0, (
                f"D3-INV-007: R5(0) must be 0.0, got {r5[0]}"
            )

        self._logger.info(
            "Domain 3 | %s | AMC generated. "
            "mean=%.4f | min=%.4f | max=%.4f | R5_mean=%.2f",
            amc_params.label,
            float(amc.mean()),
            float(amc.min()),
            float(amc.max()),
            float(r5.mean()),
        )

        return amc

    def generate_features(
        self,
        daily_temp: np.ndarray,
        season_label: np.ndarray,
        *,
        hemisphere: Literal["north", "south"],
    ) -> dict[str, np.ndarray]:       
        self._logger.info(
            "Domain 3 | %s | Starting full feature generation...",
            hemisphere.upper(),
        )

        # §4.1: Bimodal rainfall
        rainfall_result: dict[str, np.ndarray] = (
            self.generate_daily_rainfall_mm(
                daily_temp=daily_temp,
                hemisphere=hemisphere,
            )
        )
        daily_rainfall_mm: np.ndarray = rainfall_result["daily_rainfall_mm"]

        # §4.2: Consecutive dry days
        consecutive_dry_days: np.ndarray = (
            self.generate_consecutive_dry_days(
                daily_rainfall_mm=daily_rainfall_mm,
            )
        )

        # §4.3: Rolling 7-day rainfall
        rolling_7d_rainfall_mm: np.ndarray = (
            self.generate_rolling_7d_rainfall_mm(
                daily_rainfall_mm=daily_rainfall_mm,
            )
        )

        # §4.4: Cumulative storm rainfall
        cumulative_storm_rainfall_mm: np.ndarray = (
            self.generate_cumulative_storm_rainfall_mm(
                daily_rainfall_mm=daily_rainfall_mm,
            )
        )

        # §4.5: AMC (time-causal)
        antecedent_moisture_condition: np.ndarray = (
            self.generate_antecedent_moisture_condition(
                daily_rainfall_mm=daily_rainfall_mm,
                season_label=season_label,
                hemisphere=hemisphere,
            )
        )

        self._logger.info(
            "Domain 3 | %s | All 5 features generated successfully.",
            hemisphere.upper(),
        )

        return {
            "daily_rainfall_mm": daily_rainfall_mm,
            "consecutive_dry_days": consecutive_dry_days,
            "rolling_7d_rainfall_mm": rolling_7d_rainfall_mm,
            "cumulative_storm_rainfall_mm": cumulative_storm_rainfall_mm,
            "antecedent_moisture_condition": antecedent_moisture_condition,
        }
