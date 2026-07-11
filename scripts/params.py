import numpy as np
from dataclasses import dataclass

__all__= [
    "OccupancyParams",
    "ApplianceEfficiencyParams",
    "LandscapeTypeParams",
    "HemisphereTemperatureParams",
    "PrecipitationParams",
    "AMCParams",
    "RunoffParams",
    "HEMISPHERE_TEMPERATURE_PARAMS",
    "SEASON_BOUNDARIES_NORTH",
    "SOUTH_PHASE_SHIFT",
    "OCCUPANCY_PARAMS",
    "APPLIANCE_EFFICIENCY_PARAMS",
    "LANDSCAPE_TYPE_PARAMS",
    "PRECIPITATION_PARAMS",
    "AMC_PARAMS",
    "RUNOFF_PARAMS"
]

@dataclass(frozen=True)
class OccupancyParams:
    # hemispheric parameters for occupancy_count generation
    r: float
    mu: float
    label: str
    cap: int = 8

    @property
    def p(self) -> float:
        return self.r / (self.r + self.mu)

@dataclass(frozen=True)
class ApplianceEfficiencyParams:
    # immutable beta distribution parameters for appliance_efficiency_score
    alpha: float
    beta: float
    label: str
    lower_bound: float = 0.15
    upper_bound: float = 0.95

    @property
    def theoretical_mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def theoretical_mode(self) -> float:
        # only valid when alpha > 1 and beta > 1
        return self.alpha - 1.0 / (self.alpha + self.beta - 2.0)

@dataclass(frozen=True)
class LandscapeTypeParams:
    # immutable parameters for landscape type categorical sampling
    categories: tuple[str, ...]
    weights: tuple[float, ...]
    label: str

    def __post_init__(self) -> None:
        if len(self.categories) != len(self.weights):
            raise ValueError(
                f"Categories ({len(self.categories)}) and weights "
                f"({len(self.weights)}) must have an equal length."
            )
        w_sum = sum(float(w) for w in self.weights)
        if abs(w_sum - 1.0) > 1e-9:
            raise ValueError(
                f"Weights must sum to 1.0, got {w_sum:.12f}"
            )
        if any(float(w) < 0.0 for w in self.weights):
            raise ValueError("All weights must be non-negative.")
        
    @property
    def weight_array(self) -> np.ndarray:
        # returns weights as float64 ndarray
        return np.array(self.weights, dtype=np.float64)
    
    @property
    def fallback_category(self) -> str:
        # modal category used as FP fallback
        return self.categories[self.weights.index(max(self.weights))]

@dataclass(frozen=True)
class HemisphereTemperatureParams:
    label: str
    annual_mean: float
    amplitude: float
    phase_shift: int
    physical_bounds: tuple[float, float]
    anomaly_bounds: tuple[float, float]
    ou_theta: float
    ou_sigma: float
    baseline_temp: float
    climate_sigma: float
    climate_rho: float

# updated hemispheric temp params
HEMISPHERE_TEMPERATURE_PARAMS: dict[str, HemisphereTemperatureParams] = {
    "north": HemisphereTemperatureParams(
        label="North Hemisphere",
        annual_mean=16.0,
        amplitude=11.5,
        phase_shift=30,
        physical_bounds=(7.0, 35.0),
        anomaly_bounds=(-8.0, 8.0),
        ou_theta=0.25,
        ou_sigma=2.5,
        baseline_temp=18.0,
        climate_sigma=1.2,
        climate_rho=0.45
    ),
    "south": HemisphereTemperatureParams(
        label="South Hemisphere",
        annual_mean=15.0,
        amplitude=5.5,
        phase_shift=40,
        physical_bounds=(9.0, 25.0),
        anomaly_bounds=(-5.0, 5.0),
        ou_theta=0.25,
        ou_sigma=1.8,
        baseline_temp=14.5,
        climate_sigma=0.8,
        climate_rho=0.5
    )
}

# season boundary
SEASON_BOUNDARIES_NORTH: dict[str, list[tuple[int, int]]] = {
    "winter": [(0, 78), (355, 364)],
    "spring": [(79, 171)],
    "summer": [(172, 265)],
    "autumn": [(266, 354)]
}

# south phase shift
SOUTH_PHASE_SHIFT: int = 182 

OCCUPANCY_PARAMS: dict[str, OccupancyParams] = {
    "north": OccupancyParams(r=3, mu=2.44, cap=8, label="North Hemisphere"),
    "south": OccupancyParams(r=5.72, mu=2.85, cap=8, label="South Hemisphere")
}

APPLIANCE_EFFICIENCY_PARAMS: dict[str, ApplianceEfficiencyParams] = {
    "north": ApplianceEfficiencyParams(
        alpha=9.36, beta=3.64, label="North Hemisphere"
    ),
    "south": ApplianceEfficiencyParams(
        alpha=4.22, beta=4.55, label="South Hemisphere"
    )
}

# hemisphereic constants
LANDSCAPE_TYPE_PARAMS: dict[str, LandscapeTypeParams] = {
    "north": LandscapeTypeParams(
        categories=(
            "turfgrass_dominant",
            "hardscape_dominant",
            "container_balcony",
            "xeriscape_native",
            "food_homegarden",
        ),
        weights=(0.35, 0.40, 0.15, 0.07, 0.03),
        label="North Hemisphere"
    ),
    "south": LandscapeTypeParams(
        categories=(
            "turfgrass_dominant",
            "hardscape_dominant",
            "container_balcony",
            "xeriscape_native",
            "food_homegarden",
        ),
        weights=(0.30, 0.45, 0.15, 0.07, 0.03),
        label="South Hemisphere",
    ),
}

@dataclass(frozen=True)
class StratiformParams:
    sigmoid_k: float
    sigmoid_midpoint: float
    gamma_shape: float
    gamma_scale: float
    wet_floor_mm: float
    label: str

@dataclass(frozen=True)
class ConvectiveParams:
    activation_temp: float
    sigmoid_k: float
    sigmoid_midpoint: float
    gamma_shape: float
    gamma_scale: float
    wet_floor_mm: float
    label: str

@dataclass(frozen=True)
class PrecipitationParams:
    stratiform: StratiformParams
    convective: ConvectiveParams
    daily_cap_mm: float
    label: str


@dataclass(frozen=True)
class AMCParams:
    k_amc: float
    r5_mid_dormant: float
    r5_mid_growing: float
    lag_days: int
    label: str

PRECIPITATION_PARAMS: dict[str, PrecipitationParams] = {
    "north": PrecipitationParams(
        stratiform=StratiformParams(
            sigmoid_k=0.20,
            sigmoid_midpoint=11.35,
            gamma_shape=0.80,
            gamma_scale=8.0,
            wet_floor_mm=0.1,
            label="North Stratiform",
        ),
        convective=ConvectiveParams(
            activation_temp=25.0,
            sigmoid_k=-0.30,
            sigmoid_midpoint=28.0,
            gamma_shape=1.50,
            gamma_scale=18.0,
            wet_floor_mm=2.0,
            label="North Convective",
        ),
        daily_cap_mm=150.0,
        label="North Hemisphere",
    ),
    "south": PrecipitationParams(
        stratiform=StratiformParams(
            sigmoid_k=0.39,
            sigmoid_midpoint=11.33,
            gamma_shape=0.65,
            gamma_scale=10.0,
            wet_floor_mm=0.1,
            label="South Stratiform",
        ),
        convective=ConvectiveParams(
            activation_temp=20.0,
            sigmoid_k=-0.35,
            sigmoid_midpoint=23.0,
            gamma_shape=1.20,
            gamma_scale=15.0,
            wet_floor_mm=1.5,
            label="South Convective",
        ),
        daily_cap_mm=150.0,
        label="South Hemisphere",
    ),
}

AMC_PARAMS: dict[str, AMCParams] = {
    "north": AMCParams(
        k_amc=0.12,
        r5_mid_dormant=35.0,
        r5_mid_growing=25.0,
        lag_days=5,
        label="North AMC",
    ),
    "south": AMCParams(
        k_amc=0.10,
        r5_mid_dormant=30.0,
        r5_mid_growing=20.0,
        lag_days=5,
        label="South AMC",
    ),
}


# ─── Domain 4: Urban Runoff & Pollutant Loading Parameters ─────────────


@dataclass(frozen=True)
class RunoffParams:
    label: str

    # SCS Curve Number parameters
    cn_i: float       # Curve Number for dry conditions (AMC-I)
    cn_iii: float     # Curve Number for wet conditions (AMC-III)
    catchment_area_ha: float  # Catchment area in hectares

    # TSS composite function parameters
    tss_lognormal_median: float  # Median of LogNormal base EMC (mg/L)
    tss_lognormal_sigma: float   # Sigma of LogNormal distribution
    first_flush_coeff: float     # Buildup coefficient per dry day
    first_flush_cdd_cap: int     # Maximum effective dry days for buildup
    depletion_lambda: float      # Exponential decay rate per mm CSR

    # Velocity scour override (PHYS-05 resolution)
    scour_rainfall_threshold_mm: float  # Rainfall threshold for scour (mm)
    scour_dd_floor: float               # Minimum DD_eff during scour events

    # Nutrient load parameters
    n_emc_median: float   # Median nitrogen EMC (mg/L)
    n_emc_sigma: float    # Sigma for nitrogen LogNormal
    p_emc_median: float   # Median phosphorus EMC (mg/L)
    p_emc_sigma: float    # Sigma for phosphorus LogNormal
    n_ref_kg: float       # Reference nitrogen load (kg) for normalization
    p_ref_kg: float       # Reference phosphorus load (kg) for normalization
    baseline_temp: float  # Temperature baseline for nutrient modulation (°C)


RUNOFF_PARAMS: dict[str, RunoffParams] = {
    "north": RunoffParams(
        label="North Hemisphere",
        # SCS-CN: Urban residential, ~65% impervious
        cn_i=61.0,
        cn_iii=91.0,
        catchment_area_ha=55.0,   # 55 hectares (§1 Architectural Decisions)
        # TSS: NSQD mixed-use urban
        tss_lognormal_median=54.5,
        tss_lognormal_sigma=0.75,
        first_flush_coeff=0.15,
        first_flush_cdd_cap=14,
        depletion_lambda=0.008,
        # Scour override (PHYS-05)
        scour_rainfall_threshold_mm=50.0,
        scour_dd_floor=0.40,
        # Nutrient loading
        n_emc_median=2.0,
        n_emc_sigma=0.5,
        p_emc_median=0.3,
        p_emc_sigma=0.6,
        n_ref_kg=1.5,
        p_ref_kg=0.25,
        baseline_temp=18.0,
    ),
    "south": RunoffParams(
        label="South Hemisphere",
        # SCS-CN: Urban residential, ~50% impervious
        cn_i=55.0,
        cn_iii=87.0,
        catchment_area_ha=40.0,   # 40 hectares (§1 Architectural Decisions)
        # TSS: slightly lower EMC for subtropical urban
        tss_lognormal_median=54.5,
        tss_lognormal_sigma=0.75,
        first_flush_coeff=0.15,
        first_flush_cdd_cap=14,
        depletion_lambda=0.008,
        # Scour override (PHYS-05)
        scour_rainfall_threshold_mm=50.0,
        scour_dd_floor=0.40,
        # Nutrient loading (similar EMC, smaller reference loads)
        n_emc_median=2.0,
        n_emc_sigma=0.5,
        p_emc_median=0.3,
        p_emc_sigma=0.6,
        n_ref_kg=1.2,
        p_ref_kg=0.20,
        baseline_temp=14.5,
    ),
}

@dataclass(frozen=True)
class MacroBehaviorParams:
    # 7.1 Ban Constants
    ban_cdd_trigger: int = 10
    ban_duration_min: int = 7
    ban_duration_max: int = 21
    ban_compliance_outdoor: float = 0.60
    ban_compliance_base: float = 0.90
    
    # 7.2 Holiday Constants
    holiday_multiplier: float = 1.25
    annual_holidays: tuple[int, ...] = (1, 90, 150, 185, 330, 359)
    
    # 7.3 Pricing Constants
    pricing_lag_days: int = 3
    pricing_tier_1_multiplier: float = 0.95
    pricing_tier_2_multiplier: float = 0.85

class WQIParams:
    wqi_base: float = 85.0
    demand_threshold: float = 1000.0
    recovery_volume_divisor: float = 50.0
    scour_rainfall_threshold_mm: float = 50.0
    scour_dd_floor: float = 0.40
