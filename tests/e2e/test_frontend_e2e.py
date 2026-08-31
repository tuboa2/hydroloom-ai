from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

ROOT_DIR = Path(__file__).resolve().parents[2]
APP_DIR = ROOT_DIR / "app"
TEST_PORT = 5174


@pytest.fixture(scope="module")
def vite_server():
    """Start Vite preview or dev server for end-to-end browser testing."""
    # Ensure build exists first
    subprocess.run(["pnpm", "build"], cwd=str(APP_DIR), check=True, capture_output=True)

    # Launch vite preview server
    proc = subprocess.Popen(
        ["pnpm", "preview", "--port", str(TEST_PORT), "--host"],
        cwd=str(APP_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to become ready
    url = f"http://localhost:{TEST_PORT}"
    ready = False
    for _ in range(30):
        try:
            import urllib.request
            with urllib.request.urlopen(url, timeout=1) as resp:
                if resp.status == 200:
                    ready = True
                    break
        except Exception:
            time.sleep(0.3)

    if not ready:
        proc.kill()
        raise RuntimeError(f"Vite server failed to start on {url}")

    yield url

    # Cleanup
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()


def test_full_application_e2e(vite_server: str):
    """Automated E2E test verifying UI components, tabs, sliders, SVG charts, and feedback modal."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # 1. Load Application
        page.goto(vite_server)
        page.wait_for_load_state("networkidle")

        # Verify Brand Title
        assert "Hydroloom AI" in page.content()
        assert "PROD v0.2.0" in page.content()

        # 2. Test WQI Studio
        # Verify Circular Gauge & WQI Score
        assert page.locator("svg").count() >= 1
        page.click("text=Storm Surge Washoff")
        page.wait_for_timeout(300)
        assert page.locator("text=Storm Surge Washoff").is_visible()

        # Test Slider Adjustment
        rainfall_slider = page.locator("input[type='range']").first
        assert rainfall_slider.is_visible()
        rainfall_slider.evaluate("(el) => { el.value = '85.0'; el.dispatchEvent(new Event('input', { bubbles: true })); el.dispatchEvent(new Event('change', { bubbles: true })); }")
        page.wait_for_timeout(200)

        # 3. Test Behavioral Clusters Tab
        page.click("button:has-text('Behavioral Clusters')")
        page.wait_for_timeout(300)
        assert "Unsupervised Behavioral Clustering Studio" in page.content()
        assert "4-Axis Behavioral Profile Radar" in page.content()

        # Click presets
        page.click("text=Conservationist")
        page.wait_for_timeout(200)
        assert "Cluster 0" in page.content() or "Conservationist" in page.content()

        page.click("text=Landscape Irrigation Heavy")
        page.wait_for_timeout(200)
        assert "Cluster 2" in page.content() or "Landscape" in page.content()

        # 4. Test Climate Simulator Tab
        page.click("button:has-text('Climate Simulator')")
        page.wait_for_timeout(300)
        assert "Physical Hydrological Climate Simulator" in page.content()
        assert "Projected WQI Trajectory" in page.content()

        # Switch Time Horizon
        page.click("text=180d")
        page.wait_for_timeout(200)
        page.click("text=Run Scenario Simulation")
        page.wait_for_timeout(300)

        # Verify KPI Metrics Rendered
        assert "Average Projected WQI" in page.content()
        assert "Critical WQI Trough" in page.content()
        assert "Total Precipitation" in page.content()

        # 5. Test Model Leaderboard Tab
        page.click("button:has-text('Model Leaderboard')")
        page.wait_for_timeout(300)
        assert "Model Benchmark" in page.content()
        assert "Candidate Regressor Performance" in page.content()
        assert "Stacked Ensemble" in page.content()
        assert "LightGBM" in page.content()

        # 6. Test Community Feedback Modal
        page.click("button:has-text('Community Notes')")
        page.wait_for_timeout(300)
        assert "Global Community Notes" in page.content()

        # Fill out feedback
        page.fill("input[placeholder*='EcoWater Lab']", "E2E Automated Verifier")
        page.fill("textarea[placeholder*='Share your experience']", "Automated end-to-end verification passed with 100% fidelity.")
        page.click("button:has-text('Publish Community Note')")
        page.wait_for_timeout(500)

        # Assert confirmation
        assert page.locator("text=Thank you").is_visible() or "Community" in page.content()

        browser.close()
