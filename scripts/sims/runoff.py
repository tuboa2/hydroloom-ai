from __future__ import annotations
import logging
import numpy as np
from typing import Literal
from params import RUNOFF_PARAMS, RunoffParams

logger = logging.getLogger(__name__)

class RunoffSimulator:
    def __init__(
        self,
        temporal_index: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        self._temporal_index = temporal_index
        self._simulation_days: int = len(temporal_index)
        self._rng = rng
        logger.info(
            "RunoffSimulator initialized. Days: %d",
            self._simulation_days,
        )

    # ─── §5.1: Daily Runoff Volume (SCS Curve Number) ─────────────────

    def generate_daily_runoff_volume_m3(
        self,
        daily_rainfall_mm: np.ndarray,
        antecedent_moisture_condition: np.ndarray,
        *,
        hemisphere: Literal["north", "south"],
    ) -> np.ndarray:
        params: RunoffParams = RUNOFF_PARAMS[hemisphere]
        n: int = self._simulation_days

        # ── Step 1: Effective Curve Number via continuous AMC interpolation ──
        # CN_eff(t) = CN_I + AMC(t) · (CN_III - CN_I)
        # When AMC = 0 (dry): CN_eff = CN_I (maximum infiltration)
        # When AMC = 1 (wet):  CN_eff = CN_III (minimum infiltration)
        cn_range: np.float32 = np.float32(params.cn_iii - params.cn_i)
        cn_eff: np.ndarray = (
            np.float32(params.cn_i)
            + antecedent_moisture_condition * cn_range
        ).astype(np.float32)

        # ── Step 2: SCS Retention Parameter S (mm) ──────────────────────
        # S = (25400 / CN_eff) - 254
        # S represents the maximum potential soil moisture retention.
        # Units: millimeters. S decreases as CN increases (wetter soil).
        s_retention: np.ndarray = (
            np.float32(25400.0) / cn_eff - np.float32(254.0)
        ).astype(np.float32)

        # ── Step 3: Initial Abstraction threshold Ia = 0.2 * S ─────────
        # Ia is the rainfall absorbed before runoff begins (depression
        # storage, interception, initial infiltration).
        ia: np.ndarray = np.float32(0.2) * s_retention

        # ── Step 4: SCS Runoff Depth Q (mm) ─────────────────────────────
        # Q = (P - Ia)² / (P + 0.8·S)  when P > Ia, else Q = 0
        #
        # The denominator (P + 0.8·S) is strictly positive whenever P > 0,
        # because S >= 0 and P > Ia > 0. Guard against division by zero
        # when P = 0 by using np.where.
        p: np.ndarray = daily_rainfall_mm.astype(np.float32)
        excess: np.ndarray = p - ia
        runoff_condition: np.ndarray = excess > np.float32(0.0)

        denominator: np.ndarray = p + np.float32(0.8) * s_retention

        # Safe division: only compute where denominator > 0 AND P > Ia
        safe_mask: np.ndarray = runoff_condition & (
            denominator > np.float32(0.0)
        )
        q_mm: np.ndarray = np.zeros(n, dtype=np.float32)
        q_mm[safe_mask] = (
            excess[safe_mask] ** 2 / denominator[safe_mask]
        ).astype(np.float32)

        # ── Step 5: Convert depth (mm) → volume (m³) ───────────────────
        # runoff_m3 = Q_mm · A_catchment_ha · 10000 m²/ha / 1000 mm/m
        #           = Q_mm · A_catchment_ha · 10
        # Simplification: Q(mm) * A(ha) / 1000 * 10000 = Q * A * 10
        # But spec says: runoff_m3(t) = Q(t) · A_catchment / 1000
        # where A_catchment is in m². Let's use the spec formula exactly.
        catchment_area_m2: np.float32 = np.float32(
            params.catchment_area_ha * 10000.0
        )
        runoff_m3: np.ndarray = (
            q_mm * catchment_area_m2 / np.float32(1000.0)
        ).astype(np.float32)

        # ── Rounding ────────────────────────────────────────────────────
        np.round(runoff_m3, 2, out=runoff_m3)

        # ── Assertions ──────────────────────────────────────────────────
        if __debug__:
            assert runoff_m3.shape == (n,), (
                f"D4-INV-001: runoff shape {runoff_m3.shape} != ({n},)"
            )
            assert runoff_m3.dtype == np.float32, (
                f"D4-INV-001: runoff dtype {runoff_m3.dtype} != float32"
            )
            assert runoff_m3.min() >= 0.0, (
                f"D4-INV-002: negative runoff detected: {runoff_m3.min()}"
            )
            # Dry days must produce zero runoff
            dry_mask = daily_rainfall_mm == 0.0
            if dry_mask.any():
                assert np.all(runoff_m3[dry_mask] == 0.0), (
                    "D4-INV-003: non-zero runoff on dry day"
                )

        logger.info(
            "Domain 4 | %s | daily_runoff_volume_m3 generated. "
            "mean=%.2f | max=%.2f | nonzero_days=%d/%d",
            params.label,
            float(runoff_m3.mean()),
            float(runoff_m3.max()),
            int((runoff_m3 > 0).sum()),
            n,
        )

        return runoff_m3

    # ─── §5.2: Total Suspended Solids ─────────────────────────────────

    def generate_total_suspended_solids_mg_L(
        self,
        daily_rainfall_mm: np.ndarray,
        daily_runoff_volume_m3: np.ndarray,
        consecutive_dry_days: np.ndarray,
        cumulative_storm_rainfall_mm: np.ndarray,
        *,
        hemisphere: Literal["north", "south"],
    ) -> np.ndarray:        
        params: RunoffParams = RUNOFF_PARAMS[hemisphere]
        n: int = self._simulation_days

        # Wet mask: TSS is defined only on days with rainfall > 0
        wet_mask: np.ndarray = daily_rainfall_mm > np.float32(0.0)

        # ── Component 1: Base EMC from LogNormal distribution ───────────
        # TSS_base ~ LogNormal(μ=ln(54.5), σ=0.75)
        # Median EMC ≈ 54.5 mg/L, representing typical urban stormwater.
        tss_base: np.ndarray = self._rng.lognormal(
            mean=np.log(params.tss_lognormal_median),
            sigma=params.tss_lognormal_sigma,
            size=n,
        ).astype(np.float32)

        # ── Component 2: First-flush buildup factor ─────────────────────
        # FF(t) = 1.0 + 0.15 · min(CDD_at_storm_start, 14)
        #
        # Physical rationale: During dry periods, pollutants accumulate on
        # impervious surfaces (vehicle emissions, litter, atmospheric
        # deposition). The first flush of a storm event washes off this
        # accumulated load. Cap at 14 days because surface deposits
        # reach equilibrium with wind/decomposition removal.
        #
        # CDD_at_storm_start: We need the CDD value at the START of the
        # current storm event, not the running CDD. Since CDD resets to 0
        # on wet days, we use a forward-fill of the last non-zero CDD
        # before each wet spell begins.
        cdd_at_storm_start: np.ndarray = np.zeros(n, dtype=np.int16)
        last_dry_cdd: np.int16 = np.int16(0)

        for t in range(n):
            if daily_rainfall_mm[t] == 0.0:
                # Dry day: track the accumulating CDD for the next storm
                last_dry_cdd = consecutive_dry_days[t]
            else:
                # Wet day: use the CDD from the last dry day as the
                # antecedent buildup period for this storm
                cdd_at_storm_start[t] = last_dry_cdd

        ff_factor: np.ndarray = (
            np.float32(1.0)
            + np.float32(params.first_flush_coeff)
            * np.minimum(cdd_at_storm_start, np.int16(params.first_flush_cdd_cap)).astype(
                np.float32
            )
        ).astype(np.float32)

        # ── Component 3: Storm depletion decay ──────────────────────────
        # DD(t) = exp(-λ · CSR_prior(t))
        #
        # CSR_prior is the cumulative storm rainfall PRIOR to today's
        # contribution. Since CSR(t) already includes today's rainfall,
        # we subtract today's rainfall to get the prior accumulation:
        #   CSR_prior(t) = CSR(t) - rainfall(t) if wet else 0
        #
        # Physical rationale: As a storm progresses, the available surface
        # pollutant reservoir is progressively washed off. Exponential
        # decay models this first-order depletion process.
        csr_prior: np.ndarray = np.maximum(
            cumulative_storm_rainfall_mm - daily_rainfall_mm,
            np.float32(0.0),
        ).astype(np.float32)

        dd_raw: np.ndarray = np.exp(
            -np.float32(params.depletion_lambda) * csr_prior
        ).astype(np.float32)

        # ── Component 4: Velocity Scour Override (PHYS-05 Resolution) ───
        # When rainfall(t) > scour_rainfall_threshold_mm (50.0 mm):
        #   DD_eff(t) = max(DD(t), scour_dd_floor (0.40))
        #
        # Physical rationale: During extreme rainfall events (>50mm/day),
        # channel flow velocities exceed the critical shear stress for
        # bed and bank materials. This re-suspends previously deposited
        # sediments, overriding the surface washoff depletion model.
        # The floor of 0.40 ensures TSS remains at least 40% of the
        # first-flush adjusted base, preventing the physically impossible
        # scenario of zero suspended solids during a flood.
        scour_mask: np.ndarray = (
            daily_rainfall_mm > np.float32(params.scour_rainfall_threshold_mm)
        )
        dd_eff: np.ndarray = dd_raw.copy()
        dd_eff[scour_mask] = np.maximum(
            dd_raw[scour_mask],
            np.float32(params.scour_dd_floor),
        )

        # ── Assembly ────────────────────────────────────────────────────
        # TSS(t) = TSS_base(t) · FF(t) · DD_eff(t)  on wet days
        tss: np.ndarray = np.zeros(n, dtype=np.float32)
        tss[wet_mask] = (
            tss_base[wet_mask] * ff_factor[wet_mask] * dd_eff[wet_mask]
        ).astype(np.float32)

        # Physical upper bound: cap at 1000 mg/L
        np.clip(tss, np.float32(0.0), np.float32(1000.0), out=tss)

        # Rounding
        np.round(tss, 2, out=tss)

        # ── Assertions ──────────────────────────────────────────────────
        if __debug__:
            assert tss.shape == (n,), (
                f"D4-INV-004: TSS shape {tss.shape} != ({n},)"
            )
            assert tss.dtype == np.float32, (
                f"D4-INV-004: TSS dtype {tss.dtype} != float32"
            )
            assert tss.min() >= 0.0, (
                f"D4-INV-005: negative TSS: {tss.min()}"
            )
            assert tss.max() <= 1000.0, (
                f"D4-INV-005: TSS exceeds 1000: {tss.max()}"
            )
            # Dry days must have zero TSS
            dry_days = np.where(~wet_mask)[0]
            if dry_days.size > 0:
                assert np.all(tss[dry_days] == 0.0), (
                    "D4-INV-006: non-zero TSS on dry day"
                )
            # PHYS-05: Verify scour override is applied
            scour_days = np.where(scour_mask & wet_mask)[0]
            if scour_days.size > 0:
                assert np.all(
                    dd_eff[scour_days] >= params.scour_dd_floor
                ), (
                    "D4-INV-007: scour override failed — "
                    "DD_eff below floor on heavy rain day"
                )

        logger.info(
            "Domain 4 | %s | total_suspended_solids_mg_L generated. "
            "mean=%.2f | max=%.2f | wet_days=%d | scour_days=%d",
            params.label,
            float(tss[wet_mask].mean()) if wet_mask.any() else 0.0,
            float(tss.max()),
            int(wet_mask.sum()),
            int(scour_mask.sum()),
        )

        return tss

    # ─── §5.3: Nutrient Load Index ────────────────────────────────────

    def generate_nutrient_load_index(
        self,
        daily_runoff_volume_m3: np.ndarray,
        daily_max_temp_celsius: np.ndarray,
        *,
        hemisphere: Literal["north", "south"],
    ) -> np.ndarray:
        params: RunoffParams = RUNOFF_PARAMS[hemisphere]
        n: int = self._simulation_days

        # ── Nitrogen EMC: LogNormal(ln(2.0), 0.5) mg/L ─────────────────
        # Typical total nitrogen in urban runoff: median ~2.0 mg/L
        n_emc: np.ndarray = self._rng.lognormal(
            mean=np.log(params.n_emc_median),
            sigma=params.n_emc_sigma,
            size=n,
        ).astype(np.float32)

        # ── Phosphorus EMC: LogNormal(ln(0.3), 0.6) mg/L ───────────────
        # Typical total phosphorus in urban runoff: median ~0.3 mg/L
        p_emc: np.ndarray = self._rng.lognormal(
            mean=np.log(params.p_emc_median),
            sigma=params.p_emc_sigma,
            size=n,
        ).astype(np.float32)

        # ── Temperature modulation of biological nutrient release ───────
        # Warmer temperatures increase microbial activity, enhancing
        # nutrient leaching from organic debris on impervious surfaces.
        # Multiplicative factor: 1.0 at baseline, up to ~1.5 at peak heat.
        temp_mod: np.ndarray = (
            np.float32(1.0)
            + np.float32(0.02)
            * np.maximum(
                daily_max_temp_celsius - np.float32(params.baseline_temp),
                np.float32(0.0),
            )
        ).astype(np.float32)

        # ── Load computation (kg) ───────────────────────────────────────
        # N_load = runoff_m3 · N_emc_mg/L · temp_mod / 1000
        # (mg/L × m³ = mg × 10⁻³ L/m³ ... but 1 m³ = 1000 L)
        # Actually: mg/L × m³ × 1000 L/m³ / 1e6 mg/kg = g/1000 = kg
        # Simplified: load_kg = runoff_m3 * emc_mg_L / 1000
        n_load: np.ndarray = (
            daily_runoff_volume_m3 * n_emc * temp_mod / np.float32(1000.0)
        ).astype(np.float32)

        p_load: np.ndarray = (
            daily_runoff_volume_m3 * p_emc * temp_mod / np.float32(1000.0)
        ).astype(np.float32)

        # ── Composite Index ─────────────────────────────────────────────
        # NLI(t) = 0.6 · N_load / N_ref + 0.4 · P_load / P_ref
        nli: np.ndarray = (
            np.float32(0.6) * n_load / np.float32(params.n_ref_kg)
            + np.float32(0.4) * p_load / np.float32(params.p_ref_kg)
        ).astype(np.float32)

        # Ensure non-negative
        np.maximum(nli, np.float32(0.0), out=nli)

        # Rounding
        np.round(nli, 4, out=nli)

        # ── Assertions ──────────────────────────────────────────────────
        if __debug__:
            assert nli.shape == (n,), (
                f"D4-INV-008: NLI shape {nli.shape} != ({n},)"
            )
            assert nli.dtype == np.float32, (
                f"D4-INV-008: NLI dtype {nli.dtype} != float32"
            )
            assert nli.min() >= 0.0, (
                f"D4-INV-009: negative NLI: {nli.min()}"
            )
            # Zero runoff days must produce zero NLI
            zero_runoff = daily_runoff_volume_m3 == 0.0
            if zero_runoff.any():
                assert np.all(nli[zero_runoff] == 0.0), (
                    "D4-INV-010: non-zero NLI when runoff is zero"
                )

        logger.info(
            "Domain 4 | %s | nutrient_load_index generated. "
            "mean=%.4f | max=%.4f",
            params.label,
            float(nli.mean()),
            float(nli.max()),
        )

        return nli

    # ─── §5.4: Heat × Nutrient Synergy ────────────────────────────────

    def generate_heat_x_nutrient_synergy(
        self,
        temp_anomaly_celsius: np.ndarray,
        nutrient_load_index: np.ndarray,
    ) -> np.ndarray:       
        n: int = self._simulation_days

        positive_anomaly: np.ndarray = np.maximum(
            temp_anomaly_celsius, np.float32(0.0)
        ).astype(np.float32)

        synergy: np.ndarray = (
            positive_anomaly * nutrient_load_index
        ).astype(np.float32)

        # Rounding
        np.round(synergy, 4, out=synergy)

        # ── Assertions ──────────────────────────────────────────────────
        if __debug__:
            assert synergy.shape == (n,), (
                f"D4-INV-011: synergy shape {synergy.shape} != ({n},)"
            )
            assert synergy.dtype == np.float32, (
                f"D4-INV-011: synergy dtype {synergy.dtype} != float32"
            )
            assert synergy.min() >= 0.0, (
                f"D4-INV-012: negative synergy: {synergy.min()}"
            )

        logger.info(
            "Domain 4 | heat_x_nutrient_synergy generated. "
            "mean=%.4f | max=%.4f",
            float(synergy.mean()),
            float(synergy.max()),
        )

        return synergy

    # ─── Master Orchestrator ──────────────────────────────────────────

    def generate_features(
        self,
        daily_rainfall_mm: np.ndarray,
        antecedent_moisture_condition: np.ndarray,
        consecutive_dry_days: np.ndarray,
        cumulative_storm_rainfall_mm: np.ndarray,
        daily_max_temp_celsius: np.ndarray,
        temp_anomaly_celsius: np.ndarray,
        *,
        hemisphere: Literal["north", "south"],
    ) -> dict[str, np.ndarray]:       
        logger.info(
            "Domain 4 | %s | Starting full feature generation...",
            hemisphere.upper(),
        )

        # §5.1: Runoff volume
        daily_runoff_volume_m3: np.ndarray = (
            self.generate_daily_runoff_volume_m3(
                daily_rainfall_mm=daily_rainfall_mm,
                antecedent_moisture_condition=antecedent_moisture_condition,
                hemisphere=hemisphere,
            )
        )

        # §5.2: Total suspended solids
        total_suspended_solids_mg_L: np.ndarray = (
            self.generate_total_suspended_solids_mg_L(
                daily_rainfall_mm=daily_rainfall_mm,
                daily_runoff_volume_m3=daily_runoff_volume_m3,
                consecutive_dry_days=consecutive_dry_days,
                cumulative_storm_rainfall_mm=cumulative_storm_rainfall_mm,
                hemisphere=hemisphere,
            )
        )

        # §5.3: Nutrient load index
        nutrient_load_index: np.ndarray = (
            self.generate_nutrient_load_index(
                daily_runoff_volume_m3=daily_runoff_volume_m3,
                daily_max_temp_celsius=daily_max_temp_celsius,
                hemisphere=hemisphere,
            )
        )

        # §5.4: Heat × nutrient synergy
        heat_x_nutrient_synergy: np.ndarray = (
            self.generate_heat_x_nutrient_synergy(
                temp_anomaly_celsius=temp_anomaly_celsius,
                nutrient_load_index=nutrient_load_index,
            )
        )

        logger.info(
            "Domain 4 | %s | All 4 features generated successfully.",
            hemisphere.upper(),
        )

        return {
            "daily_runoff_volume_m3": daily_runoff_volume_m3,
            "total_suspended_solids_mg_L": total_suspended_solids_mg_L,
            "nutrient_load_index": nutrient_load_index,
            "heat_x_nutrient_synergy": heat_x_nutrient_synergy,
        }
