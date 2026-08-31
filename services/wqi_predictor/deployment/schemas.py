from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded"] = "healthy"
    version: str = "0.2.0"
    service: str = "hydroloom-wqi-service"
    uptime_seconds: float
    memory_rss_mb: float
    loaded_models: list[str]
    environment: str
    timestamp: str


class WQIPredictionRequest(BaseModel):
    hemisphere: Literal["north", "south"] = Field(
        default="north", description="Geographic hemisphere partition"
    )
    daily_rainfall_mm: float = Field(
        default=5.0, ge=0.0, le=300.0, description="Daily precipitation (mm)"
    )
    daily_max_temp: float = Field(
        default=26.5, ge=-20.0, le=55.0, description="Daily maximum temperature (°C)"
    )
    antecedent_moisture_condition: float = Field(
        default=25.0, ge=0.0, le=100.0, description="Antecedent 5-day moisture index"
    )
    total_suspended_solids_mg_l: float = Field(
        default=45.0, ge=0.0, le=600.0, description="Total Suspended Solids (mg/L)"
    )
    nutrient_load_index: float = Field(
        default=30.0, ge=0.0, le=150.0, description="Nutrient runoff interaction load"
    )
    consecutive_dry_days: int = Field(
        default=3, ge=0, le=90, description="Consecutive dry days prior to current date"
    )
    consumer_demand_cluster: int = Field(
        default=1, ge=0, le=3, description="Consumer demand archetype (0: Conservation, 1: Avg, 2: Landscape, 3: Heavy)"
    )
    day_of_year: int = Field(
        default=180, ge=1, le=366, description="Day of year (1-366)"
    )
    include_shap: bool = Field(
        default=True, description="Whether to return SHAP feature attributions"
    )
    model_family: Literal["ensemble", "lightgbm", "xgboost", "catboost", "linear"] = Field(
        default="ensemble", description="Model family selector"
    )


class WQIPredictionResponse(BaseModel):
    wqi_score: float = Field(..., description="Predicted Water Quality Index (0-100)")
    classification: Literal["Excellent", "Good", "Moderate", "Poor", "Hazardous"]
    confidence_interval: tuple[float, float]
    model_used: str
    hemisphere: str
    shap_breakdown: dict[str, float]
    advisories: list[str]


class ClusterPredictionRequest(BaseModel):
    hemisphere: Literal["north", "south"] = Field(default="north")
    log_per_capita_usage: float = Field(
        default=4.8, ge=2.0, le=8.0, description="Log per-capita daily volume (L)"
    )
    dry_day_spike_factor: float = Field(
        default=1.25, ge=0.5, le=3.5, description="Spike ratio during dry periods"
    )
    efficiency_penalty_ratio: float = Field(
        default=0.15, ge=0.0, le=1.0, description="Tier violation penalty ratio"
    )
    landscape_demand_index: float = Field(
        default=0.45, ge=0.0, le=1.0, description="Landscape outdoor irrigation demand"
    )


class ClusterPredictionResponse(BaseModel):
    cluster_id: int
    archetype_name: str
    description: str
    behavior_scores: dict[str, float]
    recommendations: list[str]


class SimulationRequest(BaseModel):
    hemisphere: Literal["north", "south"] = Field(default="north")
    days: int = Field(default=90, ge=14, le=365, description="Simulation time horizon in days")
    scenario: Literal["baseline", "drought_heatwave", "intense_storm", "conservation_mandate"] = Field(
        default="baseline", description="Macro simulation scenario preset"
    )


class SimulationResponse(BaseModel):
    scenario: str
    hemisphere: str
    days: int
    timeline: list[dict[str, Any]]
    summary: dict[str, float]


class FeedbackSubmission(BaseModel):
    user_name: str | None = Field(default="Anonymous")
    user_email: str | None = None
    category: Literal["forecast_accuracy", "feature_request", "scenario_preset", "bug_report", "general"] = "general"
    rating: int = Field(default=5, ge=1, le=5)
    comment: str = Field(..., min_length=2, max_length=2000)
    scenario_payload: dict[str, Any] | None = None


class FeedbackResponse(BaseModel):
    id: str
    status: str
    received_at: str
    message: str
