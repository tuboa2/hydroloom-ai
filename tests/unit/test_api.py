from fastapi.testclient import TestClient

from services.wqi_predictor.deployment.api import app

client = TestClient(app)


def test_health_and_ping_endpoints() -> None:
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["service"] == "hydroloom-wqi-service"
    assert "memory_rss_mb" in data

    ping_res = client.get("/ping")
    assert ping_res.status_code == 200


def test_wqi_prediction_valid_request() -> None:
    payload = {
        "hemisphere": "north",
        "daily_rainfall_mm": 12.5,
        "daily_max_temp": 28.0,
        "antecedent_moisture_condition": 30.0,
        "total_suspended_solids_mg_l": 65.0,
        "nutrient_load_index": 45.0,
        "consecutive_dry_days": 4,
        "consumer_demand_cluster": 1,
        "day_of_year": 150,
        "include_shap": True,
        "model_family": "ensemble",
    }
    res = client.post("/api/v1/predict/wqi", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert 0.0 <= data["wqi_score"] <= 100.0
    assert data["classification"] in ["Excellent", "Good", "Moderate", "Poor", "Hazardous"]
    assert "confidence_interval" in data
    assert len(data["shap_breakdown"]) > 0
    assert len(data["advisories"]) > 0


def test_wqi_prediction_boundary_constraints() -> None:
    # Invalid rainfall (< 0)
    invalid_payload = {
        "hemisphere": "north",
        "daily_rainfall_mm": -10.0,
    }
    res = client.post("/api/v1/predict/wqi", json=invalid_payload)
    assert res.status_code == 422


def test_cluster_prediction() -> None:
    payload = {
        "hemisphere": "north",
        "log_per_capita_usage": 4.5,
        "dry_day_spike_factor": 1.1,
        "efficiency_penalty_ratio": 0.08,
        "landscape_demand_index": 0.25,
    }
    res = client.post("/api/v1/cluster/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["cluster_id"] in [0, 1, 2, 3]
    assert len(data["archetype_name"]) > 0
    assert len(data["recommendations"]) > 0


def test_simulation_endpoint() -> None:
    payload = {
        "hemisphere": "north",
        "days": 45,
        "scenario": "intense_storm",
    }
    res = client.post("/api/v1/simulate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert len(data["timeline"]) == 45
    assert "summary" in data
    assert data["summary"]["storm_events"] >= 0


def test_metadata_endpoint() -> None:
    res = client.get("/api/v1/models/metadata")
    assert res.status_code == 200
    data = res.json()
    assert "leaderboard" in data
    assert len(data["leaderboard"]) >= 4


def test_feedback_submission_and_listing() -> None:
    feedback_payload = {
        "user_name": "HydroTester",
        "category": "forecast_accuracy",
        "rating": 5,
        "comment": "Forecast aligned with municipal laboratory benchmarks.",
    }
    res = client.post("/api/v1/feedback", json=feedback_payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "received"

    list_res = client.get("/api/v1/feedback")
    assert list_res.status_code == 200
    entries = list_res.json()
    assert any(e["user_name"] == "HydroTester" for e in entries)
