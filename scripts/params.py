import numpy as np
from dataclasses import dataclass

__all__= [
    "OccupancyParams",
    "ApplianceEfficiencyParams",
    "LandscapeTypeParams",
    "HemisphereTemperatureParams",
    "HEMISPHERE_TEMPERATURE_PARAMS",
    "SEASON_BOUNDARIES_NORTH",
    "SOUTH_PHASE_SHIFT",
    "OCCUPANCY_PARAMS",
    "APPLIANCE_EFFICIENCY_PARAMS",
    "LANDSCAPE_TYPE_PARAMS",
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
        annual_mean=15.2,
        amplitude=7.15,
        phase_shift=30,
        physical_bounds=(7.0, 35.0),
        anomaly_bounds=(-6.0, 6.0),
        ou_theta=0.25,
        ou_sigma=0.88,
        baseline_temp=18.0,
        climate_sigma=0.3,
        climate_rho=0.45
    ),
    "south": HemisphereTemperatureParams(
        label="South Hemisphere",
        annual_mean=13.3,
        amplitude=3.65,
        phase_shift=40,
        physical_bounds=(9.0, 25.0),
        anomaly_bounds=(-3.0, 3.0),
        ou_theta=0.25,
        ou_sigma=0.71,
        baseline_temp=14.5,
        climate_sigma=0.125,
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
