import logging
import joblib
import numpy as np
from pathlib import Path
from typing import Literal

from features.data_loader import DataLoader
from features.extract_features import BehavioralFeatureExtractor
from features.extract_features import scaling_pipeline
from features.select_features import FeatureSelection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%m-%d-%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)

parent_dir = Path(__file__).resolve().parent.parent
data_dir = parent_dir / "data/raw"
data_dir.mkdir(parents=True, exist_ok=True)

output_dir = parent_dir / "data/processed"
output_dir.mkdir(parents=True, exist_ok=True)

def run(hemisphere: Literal["north", "south"]):
    # 1. load
    logger.info(f"Loading {hemisphere} hemisphere data from {data_dir}...")
    loader = DataLoader(data_dir=data_dir, hemisphere=hemisphere)
    loader.load_and_validate()

    X_raw = {
        "household": loader._household,
        "water_usage": loader._water_usage,
        "environment": loader._environment
    }

    # 2. extract features
    logger.info(f"Extracting features for {hemisphere} hemisphere...")
    extractor = BehavioralFeatureExtractor(hemisphere=hemisphere)
    X_extracted = extractor.fit_transform(X_raw)
    logger.info(f"Extracted shape: {X_extracted.shape}")

    # 3. scale
    logger.info(f"Scaling features for the {hemisphere} hemisphere...")
    X_scaled = scaling_pipeline.fit_transform(X_extracted)
    logger.info(f"Scaled shape: {X_scaled.shape}")

    # 4. drop near-zero variance features
    logger.info("Applying VarianceThreshold (0.01)")
    final_selection = FeatureSelection(X_scaled)
    final_selection.variance_threshold()
    X_final = final_selection.features
    logger.info(f"Final shape: {X_final.shape}")

    # 5. save artifacts
    logger.info(f"Saving artifacts to {output_dir}")
    np.save(output_dir / f"{hemisphere}_features.npy", X_final)

    # 6. save fitted components for inference
    joblib.dump({
        "extractor": extractor,
        "scaler": scaling_pipeline,
        "variance_threshold": final_selection.vt
    }, output_dir / f"{hemisphere}_pipeline_components.pkl")

    logger.info(f"{hemisphere.capitalize()} Hemisphere Feature Engineering Done.")


if __name__ == "__main__":
    run(hemisphere="north")
    run(hemisphere="south")
