import logging
import joblib
import polars as pl
from pathlib import Path
from typing import Literal

from features.data_loader import DataLoader
# fixed: import the hemisphere isolated scaling pipeline
from features.extract_features import north_pipeline, south_pipeline
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
    logger.info(f"Loading {hemisphere} hemisphere data...")
    loader = DataLoader(data_dir=data_dir, hemisphere=hemisphere)
    loader.load_and_validate()

    X_raw = {
        "household": loader._household,
        "water_usage": loader._water_usage,
        "environment": loader._environment
    }

    # FIXED: Select the isolated pipeline corresponding to the target hemisphere
    pipeline = north_pipeline if hemisphere == "north" else south_pipeline

    # 1. Run Extraction and Scaling through the safe pipeline
    logger.info(f"Processing extraction and scaling for {hemisphere} hemisphere...")
    X_scaled = pipeline.fit_transform(X_raw)
    
    # Extract feature names dynamically from the ColumnTransformer step
    scaler_step = pipeline.named_steps["scale"]
    scaled_feature_names = [name.split("__")[-1] for name in scaler_step.get_feature_names_out()]

    # 2. Drop near-zero variance features
    logger.info("Applying VarianceThreshold (0.01)...")
    final_selection = FeatureSelection(X_scaled)
    final_selection.variance_threshold()
    X_final_array = final_selection.features
    
    # Determine which feature names survived selection
    supported_mask = final_selection.vt.get_support()
    final_feature_names = [name for keep, name in zip(supported_mask, scaled_feature_names) if keep]

    # 3. FIXED: Wrap back into a Polars DataFrame to keep feature headers intact
    X_final_df = pl.DataFrame(X_final_array, schema=final_feature_names)
    logger.info(f"Final processed shape: {X_final_df.shape}")

    # 4. FIXED: Save as a compressed Parquet file for seamless downstream analysis
    parquet_path = output_dir / f"{hemisphere}_features.parquet"
    X_final_df.write_parquet(parquet_path)
    logger.info(f"Saved features to {parquet_path}")

    # 5. Save fitted components for clean downstream inference pipelines
    joblib.dump({
        "extractor": pipeline.named_steps["extract"],
        "scaler": scaler_step,
        "variance_threshold": final_selection.vt
    }, output_dir / f"{hemisphere}_pipeline_components.pkl")

    # 6. get unscaled dataset
    extractor = pipeline.named_steps["extract"]
    unscaled_df = extractor.fit_transform(X_raw)
    features = [
        "log_per_capita_usage",
        "dry_day_spike_factor",
        "efficiency_penalty_ratio",
        "landscape_demand_index"
    ]
    unscaled_df = unscaled_df[features]
    pl.DataFrame(unscaled_df).write_parquet(output_dir / f"{hemisphere}_unscaled.parquet")

    logger.info(f"{hemisphere.capitalize()} Hemisphere Feature Engineering Done.\n")


if __name__ == "__main__":
    run(hemisphere="north")
    run(hemisphere="south")
