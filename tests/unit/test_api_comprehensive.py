from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from services.wqi_predictor.deployment.api import app

client = TestClient(app)


def test_health_and_ping():
    """Verify health and ping endpoints."""
    res_health = client.get("/health")
    assert res_health.status_code == 200
    data = res_health.json()
    assert data["status"] == "healthy"
    assert data["service"] == "hydroloom-wqi-service"
    assert data["version"] == "0.2.0"
    assert "memory_rss_mb" in data

    res_ping = client.get("/ping")
    assert res_ping.status_code == 200
    assert res_ping.json()["status"] == "healthy"


@pytest.mark.parametrize("model_family", ["ensemble", "lightgbm", "xgboost", "catboost", "linear"])
@pytest.mark.parametrize("hemisphere", ["north", "south"])
def test_predict_wqi_all_families_and_hemispheres(model_family: str, hemisphere: str):
    """Verify WQI predictions across all model families and hemispheres."""
    payload = {
        "hemisphere": hemisphere,
        "daily_rainfall_mm": 25.0,
        "daily_max_temp": 22.0,
        "antecedent_moisture_condition": 30.0,
        "total_suspended_solids_mg_l": 45.0,
        "nutrient_load_index": 35.0,
        "consecutive_dry_days": 4,
        "consumer_demand_cluster": 1,
        "model_family": model_family,
    }
    response = client.post("/api/v1/predict/wqi", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Core response assertions
    assert 0.0 <= data["wqi_score"] <= 100.0
    assert "model_used" in data
    assert data["classification"] in ["Excellent", "Good", "Moderate", "Poor", "Hazardous"]
    assert "confidence_interval" in data
    assert data["confidence_interval"][0] <= data["confidence_interval"][1]
    assert len(data["shap_breakdown"]) >= 3
    assert len(data["advisories"]) >= 1


def test_predict_wqi_edge_cases():
    """Verify boundary conditions and extreme climatic inputs."""
    # Zero rainfall & freezing temperature
    freezing_payload = {
        "hemisphere": "north",
        "daily_rainfall_mm": 0.0,
        "daily_max_temp": -5.0,
        "antecedent_moisture_condition": 5.0,
        "total_suspended_solids_mg_l": 5.0,
        "nutrient_load_index": 5.0,
        "consecutive_dry_days": 30,
        "consumer_demand_cluster": 0,
        "model_family": "ensemble",
    }
    res1 = client.post("/api/v1/predict/wqi", json=freezing_payload)
    assert res1.status_code == 200
    assert 0.0 <= res1.json()["wqi_score"] <= 100.0

    # Extreme storm deluge
    storm_payload = {
        "hemisphere": "south",
        "daily_rainfall_mm": 180.0,
        "daily_max_temp": 38.0,
        "antecedent_moisture_condition": 90.0,
        "total_suspended_solids_mg_l": 350.0,
        "nutrient_load_index": 120.0,
        "consecutive_dry_days": 0,
        "consumer_demand_cluster": 3,
        "model_family": "ensemble",
    }
    res2 = client.post("/api/v1/predict/wqi", json=storm_payload)
    assert res2.status_code == 200
    assert res2.json()["classification"] in ["Poor", "Hazardous", "Moderate", "Good"]


@pytest.mark.parametrize("hemisphere", ["north", "south"])
def test_cluster_predict_archetypes(hemisphere: str):
    """Verify K-Means behavioral clustering predictions."""
    payload = {
        "hemisphere": hemisphere,
        "log_per_capita_usage": 4.5,
        "dry_day_spike_factor": 1.2,
        "efficiency_penalty_ratio": 0.1,
        "landscape_demand_index": 0.3,
    }
    response = client.post("/api/v1/cluster/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["cluster_id"] in [0, 1, 2, 3]
    assert len(data["archetype_name"]) > 0
    assert len(data["description"]) > 0
    assert len(data["recommendations"]) >= 1
    assert "Per-Capita Volume" in data["behavior_scores"]


@pytest.mark.parametrize("scenario", ["baseline", "intense_storm", "drought_heatwave", "conservation_mandate"])
@pytest.mark.parametrize("days", [30, 90, 180, 365])
def test_simulation_scenarios_and_horizons(scenario: str, days: int):
    """Verify hydrological Markov/SCS simulations across all scenarios and durations."""
    payload = {
        "hemisphere": "north",
        "days": days,
        "scenario": scenario,
    }
    response = client.post("/api/v1/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["scenario"] == scenario
    assert data["days"] == days
    assert len(data["timeline"]) == days
    assert "summary" in data
    assert 0.0 <= data["summary"]["avg_wqi"] <= 100.0
    assert data["summary"]["min_wqi"] <= data["summary"]["max_wqi"]


def test_models_metadata_and_leaderboard():
    """Verify model metadata and candidate rankings endpoint."""
    response = client.get("/api/v1/models/metadata")
    assert response.status_code == 200
    data = response.json()

    assert "version" in data
    assert len(data["leaderboard"]) >= 4
    assert len(data["top_features"]) >= 4


def test_feedback_submission_and_retrieval():
    """Verify community feedback persistence and retrieval."""
    feedback_payload = {
        "user_name": "Dr. Hydrologist",
        "category": "forecast_accuracy",
        "rating": 5,
        "comment": "Exceptional physical washoff calibration and low cold-start latency.",
    }
    res_post = client.post("/api/v1/feedback", json=feedback_payload)
    assert res_post.status_code == 200
    data_post = res_post.json()
    assert data_post["status"] == "received"

    # Retrieve feedback list
    res_get = client.get("/api/v1/feedback")
    assert res_get.status_code == 200
    data_get = res_get.json()
    assert isinstance(data_get, list)
    assert any(item["user_name"] == "Dr. Hydrologist" for item in data_get)
