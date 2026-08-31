from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REPO_ID = "tuboa2/hydroloom-ai"

MODEL_CARD_TEMPLATE = """---
language:
- en
license: mit
tags:
- water-quality
- time-series
- ensemble
- lightgbm
- xgboost
- catboost
- scikit-learn
- polars
metrics:
- rmse
- mae
- r2
pipeline_tag: tabular-regression
---

# Hydroloom AI: Water Quality Index & Behavioral Clustering Platform

Hydroloom is a production-grade machine learning system for physical hydrological simulation, consumer water-use behavioral clustering, and supervised Water Quality Index (WQI) forecasting.

## Architecture
- **Supervised WQI Service**: Multi-model regressors (LightGBM, XGBoost, CatBoost, Regularized Linear) + Level-1 Out-of-Fold (OOF) Stacking & Constrained Weighted Blending with AR(1) Residual Error Correction.
- **Unsupervised Clustering Service**: 4-archetype K-Means clustering ($k=4$) mapping consumer profiles (Conservationist, Average, Landscape Heavy, High Volume).
- **Physical Hydrological Simulator**: SCS Curve Number runoff, Markov-chain precipitation, and first-flush contaminant washoff kinetics.

## Repository Contents
- `artifacts/`: Checkpoint weights, scalers, preprocessors, Optuna candidate registries, and SHAP explainers for North and South hemispheres.
- `clustering_models/`: Serialized K-Means models (`kmeans_north_k4.joblib`, `kmeans_south_k4.joblib`).
- `metadata/`: Evaluation metrics, lead-lag correlations, and feature importance registries.

## Usage with Python
```python
from huggingface_hub import hf_hub_download
import joblib

# Load clustering model
model_path = hf_hub_download(
    repo_id="tuboa2/hydroloom-ai",
    repo_type="model",
    filename="clustering_models/kmeans_north_k4.joblib"
)
kmeans_north = joblib.load(model_path)
```
"""


def package_and_upload(
    repo_id: str = DEFAULT_REPO_ID,
    repo_type: str = "model",
    token: str | None = None,
    create_repo: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Package artifacts and upload to the Hugging Face Hub."""
    token = token or os.getenv("HF_TOKEN")

    with tempfile.TemporaryDirectory() as staging_dir_str:
        staging_dir = Path(staging_dir_str)

        # 1. Write Model Card README.md
        readme_path = staging_dir / "README.md"
        readme_path.write_text(MODEL_CARD_TEMPLATE, encoding="utf-8")
        print(f"Generated Model Card: {readme_path.name}")

        # 2. Copy artifacts directory if exists
        local_artifacts = ROOT_DIR / "artifacts"
        if local_artifacts.exists():
            target_artifacts = staging_dir / "artifacts"
            shutil.copytree(local_artifacts, target_artifacts, dirs_exist_ok=True)
            print(f"Staged artifacts from {local_artifacts}")

        # 3. Copy clustering models
        local_clustering = ROOT_DIR / "services" / "behavior_clustering" / "models"
        target_clustering = staging_dir / "clustering_models"
        target_clustering.mkdir(parents=True, exist_ok=True)
        if local_clustering.exists():
            for f in local_clustering.glob("*.joblib"):
                shutil.copy2(f, target_clustering / f.name)
                print(f"Staged clustering model: {f.name}")

        # 4. Copy evaluation summary documentation
        docs_dir = ROOT_DIR / "docs"
        if docs_dir.exists():
            target_docs = staging_dir / "docs"
            shutil.copytree(docs_dir, target_docs, dirs_exist_ok=True)
            print(f"Staged documentation from {docs_dir}")

        staged_files = list(staging_dir.rglob("*"))
        print(f"Total staged files for Hugging Face Hub: {len([f for f in staged_files if f.is_file()])}")

        if dry_run:
            print(f"[DRY-RUN] Packaging verified successfully for repo: {repo_id} ({repo_type})")
            return {"repo_id": repo_id, "status": "dry_run_success", "staged_count": len(staged_files)}

        api = HfApi(token=token)
        if create_repo:
            try:
                api.create_repo(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    private=False,
                    exist_ok=True,
                )
                print(f"Verified/Created repository: {repo_id} ({repo_type})")
            except Exception as e:
                print(f"Note on repo creation: {e}")

        print(f"Uploading bundled assets to Hugging Face Hub ({repo_id})...")
        future = api.upload_folder(
            folder_path=str(staging_dir),
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message="feat(hub): deploy model checkpoints, preprocessors, and evaluation artifacts",
        )
        print("Upload completed successfully!")
        return {"repo_id": repo_id, "status": "success", "details": str(future)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deploy Hydroloom assets to Hugging Face Hub")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face repository ID")
    parser.add_argument("--repo-type", default="model", choices=["model", "space", "dataset"], help="Repository type")
    parser.add_argument("--token", "-t", "-token", default=None, help="Hugging Face write token")
    parser.add_argument("--dry-run", action="store_true", help="Stage and validate files without uploading")

    args = parser.parse_args()
    package_and_upload(repo_id=args.repo_id, repo_type=args.repo_type, token=args.token, dry_run=args.dry_run)
