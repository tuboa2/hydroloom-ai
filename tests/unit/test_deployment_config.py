from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.deploy_to_hf import package_and_upload

ROOT_DIR = Path(__file__).resolve().parents[2]


def test_dockerfile_configuration():
    """Verify Dockerfile meets Render free-tier and security specifications."""
    dockerfile_path = ROOT_DIR / "Dockerfile"
    assert dockerfile_path.exists(), "Dockerfile must exist at repository root."

    content = dockerfile_path.read_text(encoding="utf-8")

    # Multi-stage verification
    assert "AS builder" in content, "Dockerfile should use multi-stage builds."
    assert "AS runtime" in content, "Dockerfile should have a dedicated runtime stage."

    # Non-root unprivileged security hardening
    assert "USER 10001" in content or "USER appuser" in content, "Must switch to non-root user."
    assert "addgroup --gid 10001" in content, "Must define custom group."
    assert "adduser --uid 10001" in content, "Must define unprivileged appuser."

    # Dynamic $PORT binding for Render
    assert "${PORT:-8000}" in content or "$PORT" in content, "Must dynamically bind to $PORT."

    # Python version
    assert "python:3.12-slim" in content, "Must use python:3.12-slim base image."

    # Healthcheck presence
    assert "HEALTHCHECK" in content, "Must declare container HEALTHCHECK."
    assert "/health" in content, "Healthcheck should query /health endpoint."


def test_dockerignore_configuration():
    """Verify .dockerignore excludes heavy dev dependencies and sensitive assets."""
    dockerignore_path = ROOT_DIR / ".dockerignore"
    assert dockerignore_path.exists(), ".dockerignore must exist."

    content = dockerignore_path.read_text(encoding="utf-8")
    lines = [line.strip() for line in content.splitlines() if line.strip() and not line.startswith("#")]

    expected_exclusions = [".git", ".venv", "node_modules", "tests", "__pycache__"]
    for pattern in expected_exclusions:
        assert any(pattern in line for line in lines), f"Pattern '{pattern}' should be in .dockerignore"


def test_render_blueprint_configuration():
    """Verify render.yaml complies with Render Infrastructure-as-Code schema."""
    render_yaml_path = ROOT_DIR / "render.yaml"
    assert render_yaml_path.exists(), "render.yaml blueprint must exist."

    with render_yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert "services" in data, "render.yaml must declare services list."
    services = data["services"]
    assert len(services) >= 1, "Must contain at least 1 service."

    api_service = next((s for s in services if s.get("name") == "hydroloom-api"), None)
    assert api_service is not None, "Service 'hydroloom-api' must be defined."
    assert api_service.get("type") == "web"
    assert api_service.get("runtime") == "docker" or api_service.get("env") == "docker"
    assert api_service.get("plan") == "free"
    assert api_service.get("healthCheckPath") == "/health"


def test_vercel_frontend_configuration():
    """Verify app/vercel.json configures SPA routing rewrites and security headers."""
    vercel_path = ROOT_DIR / "app" / "vercel.json"
    assert vercel_path.exists(), "app/vercel.json must exist."

    with vercel_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("version") == 2
    assert "rewrites" in data, "Must configure SPA rewrites."

    rewrites = data["rewrites"]
    has_spa_rewrite = any(r.get("source") == "/(.*)" and r.get("destination") == "/index.html" for r in rewrites)
    assert has_spa_rewrite, "Must rewrite all paths to /index.html for client-side routing."

    assert "headers" in data, "Must define security headers."
    headers_config = data["headers"]
    assert len(headers_config) >= 1
    header_entries = headers_config[0].get("headers", [])
    header_keys = [h.get("key") for h in header_entries]

    assert "X-Content-Type-Options" in header_keys
    assert "X-Frame-Options" in header_keys


def test_huggingface_packaging_dry_run():
    """Verify Hugging Face deployment packager stages all artifacts cleanly."""
    result = package_and_upload(repo_id="tuboa2/hydroloom-ai", dry_run=True)
    assert result["status"] == "dry_run_success"
    assert result["staged_count"] >= 5, "Should stage Model Card, artifacts, models, and docs."
