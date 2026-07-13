from __future__ import annotations
import logging
import global_init
import polars as pl
from pathlib import Path
from typing import Literal
from params import HEMISPHERE_TEMPERATURE_PARAMS
from sims.household import HouseholdSimulator
from sims.environment import EnvironmentalSimulator
from sims.precipitation import PrecipitationSimulator
from sims.runoff import RunoffSimulator
from sims.cluster import ClusterSimulator
from sims.macro_behavior import MacroBehavioralSimulator
from sims.interactions import InteractionSimulator
from sims.wqi import WQISimulator

# logging config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%m-%d-%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)

# get data directory
parent_dir = Path(__file__).resolve().parent.parent
data_dir = parent_dir / "data/raw"
data_dir.mkdir(parents=True, exist_ok=True)

service_b_data_dir = str(
    parent_dir / "services/behavior_clustering/data/processed"
)
service_b_models_dir = str(
    parent_dir / "services/behavior_clustering/models"
)

processed_dir = parent_dir / "data/processed"
processed_dir.mkdir(parents=True, exist_ok=True)

def run(hemisphere: Literal["north", "south"]) -> None:
    # run the data gen pipeline
    logger.info("Data Generation Pipeline Execution Started")

    # 1. global initialization
    logger.info("Creating Global Initialization...")
    global_config = global_init.run(
        simulation_days=1825,
        days_per_year=365,
        num_years=5,
        population_size=100_000,
        subsample=5_000,
        random_seed=2032,
    )
    logger.info("Global Initialization Complete.\n")

    # 2. simulator initialization
    household_sim = HouseholdSimulator(
        global_config=global_config
    )
    env_sim = EnvironmentalSimulator(
        temporal_index=global_config.temporal_index,
        rng=global_config.rng
    )
    precipitation_sim = PrecipitationSimulator(
        temporal_index=global_config.temporal_index,
        rng=global_config.rng
    )
    runoff_sim = RunoffSimulator(
        temporal_index=global_config.temporal_index,
        rng=global_config.rng
    )
    cluster_sim = ClusterSimulator(
        rng=global_config.rng,
        n_days=global_config.simulation_days,
        subsample_size=global_config.sim_config.subsample,
        service_b_data_dir=service_b_data_dir,
        service_b_models_dir=service_b_models_dir 
    )
    macro_sim = MacroBehavioralSimulator(
        rng=global_config.rng,
        n_days=global_config.simulation_days
    )
    interaction_sim = InteractionSimulator()
    wqi_sim = WQISimulator(
        rng=global_config.rng,
    )

    # 3. generate temporal framework
    temporal_framework = env_sim.generate_temporal_framework(
        config=global_config,
        hemisphere=hemisphere
    )

    # 4. generate daily max temp
    daily_max_temp = env_sim.generate_daily_max_temp(
        day_index=temporal_framework["day_index"],
        year_index=temporal_framework["year_index"],
        hemisphere=hemisphere
    )

    # 5. generate precipitation features
    precipitation_features = precipitation_sim.generate_features(
        daily_temp=daily_max_temp["daily_max_temp_celsius"],
        season_label=temporal_framework["season_label"],
        hemisphere=hemisphere
    )

    # 6. generate runoff pollutant features
    runoff_features = runoff_sim.generate_features(
        daily_rainfall_mm=precipitation_features["daily_rainfall_mm"],
        antecedent_moisture_condition=precipitation_features["antecedent_moisture_condition"],
        consecutive_dry_days=precipitation_features["consecutive_dry_days"],
        cumulative_storm_rainfall_mm=precipitation_features["cumulative_storm_rainfall_mm"],
        daily_max_temp_celsius=daily_max_temp["daily_max_temp_celsius"],
        temp_anomaly_celsius=daily_max_temp["temp_anomaly_celsius"],
        hemisphere=hemisphere
    )

    # 7. generate household features
    occupancy_count = household_sim.generate_occupancy_count(hemisphere=hemisphere)
    appliance_efficiency = household_sim.generate_appliance_efficiency_score(hemisphere=hemisphere)
    landscape_type = household_sim.generate_landscape_type(hemisphere=hemisphere)
    water_usage = household_sim.generate_daily_water_usage_liters(
        occupancy_count=occupancy_count,
        appliance_efficiency_score=appliance_efficiency,
        landscape_type=landscape_type,
        daily_max_temp_celsius=daily_max_temp["daily_max_temp_celsius"],
        daily_rainfall_mm=precipitation_features["daily_rainfall_mm"],
        hemisphere=hemisphere
    )

    # 8. generate cluster daily means feature
    cluster = cluster_sim.generate_features(
        water_usage_matrix=water_usage,
        hemisphere=hemisphere,
        daily_rainfall_mm=precipitation_features["daily_rainfall_mm"],
        consecutive_dry_days=precipitation_features["consecutive_dry_days"]
    )

    # 9. generate macro behavioral features
    macro = macro_sim.generate_features(
        day_index=temporal_framework["day_index"],
        season_label=temporal_framework["season_label"],
        consecutive_dry_days=precipitation_features["consecutive_dry_days"],
        cluster_means_dict=cluster
    )

    # 10. generate interaction features
    interactions = interaction_sim.generate_features(
        consecutive_dry_days=precipitation_features["consecutive_dry_days"],
        daily_max_temp_celsius=daily_max_temp["daily_max_temp_celsius"],
        cluster_standard_consumers_daily_mean=cluster["cluster_standard_consumers_daily_mean_liters"],
        daily_runoff_volume_m3=runoff_features["daily_runoff_volume_m3"],
        hemisphere=hemisphere,
        baseline_temp=HEMISPHERE_TEMPERATURE_PARAMS[hemisphere].baseline_temp
    )

    # 11. generate target variable
    wqi_features = wqi_sim.generate_target(
        daily_max_temp_celsius=daily_max_temp["daily_max_temp_celsius"],
        daily_rainfall_mm=precipitation_features["daily_rainfall_mm"],
        total_suspended_solids_mg_L=runoff_features["total_suspended_solids_mg_L"],
        daily_runoff_volume_m3=runoff_features["daily_runoff_volume_m3"],
        nutrient_load_index=runoff_features["nutrient_load_index"],
        heat_x_nutrient_synergy=runoff_features["heat_x_nutrient_synergy"],
        consecutive_dry_days=precipitation_features["consecutive_dry_days"],
        cluster_heavy_users_mean=cluster["cluster_heavy_users_daily_mean_liters"]
    )

    logger.info("Assembling final Parquet dataset and verifying invariants...")
        
    # Assemble raw dict
    final_data = {
        "hemisphere": [hemisphere] * global_config.simulation_days,
        "day_index": temporal_framework["day_index"],
        "year_index": temporal_framework["year_index"],
        "is_weekend": temporal_framework["is_weekend"],
        "season_label": temporal_framework["season_label"],
        "daily_max_temp_celsius": daily_max_temp["daily_max_temp_celsius"],
        "temp_anomaly_celsius": daily_max_temp["temp_anomaly_celsius"],
        "cumulative_heat_index": daily_max_temp["cumulative_heat_index"],
        "daily_rainfall_mm": precipitation_features["daily_rainfall_mm"],
        "consecutive_dry_days": precipitation_features["consecutive_dry_days"],
        "rolling_7d_rainfall_mm": precipitation_features["rolling_7d_rainfall_mm"],
        "cumulative_storm_rainfall_mm": precipitation_features["cumulative_storm_rainfall_mm"],
        "antecedent_moisture_condition": precipitation_features["antecedent_moisture_condition"],
        "daily_runoff_volume_m3": runoff_features["daily_runoff_volume_m3"],
        "total_suspended_solids_mg_L": runoff_features["total_suspended_solids_mg_L"],
        "nutrient_load_index": runoff_features["nutrient_load_index"],
        "heat_x_nutrient_synergy": runoff_features["heat_x_nutrient_synergy"],
        "cluster_heavy_users_daily_mean_liters": cluster["cluster_heavy_users_daily_mean_liters"],
        "cluster_conservationists_daily_mean_liters": cluster["cluster_conservationists_daily_mean_liters"],
        "cluster_outdoor_landscape_daily_mean_liters": cluster["cluster_outdoor_landscape_daily_mean_liters"],
        "cluster_standard_consumers_daily_mean_liters": cluster["cluster_standard_consumers_daily_mean_liters"],
        "watering_ban_active": macro["watering_ban_active"],
        "holiday_weekend_flag": macro["holiday_weekend_flag"],
        "tiered_pricing_regime": macro["tiered_pricing_regime"],
        "drought_x_heat_stress": interactions["drought_x_heat_stress"],
        "demand_x_runoff_pressure": interactions["demand_x_runoff_pressure"],
        "water_quality_index": wqi_features["water_quality_index"],
    }
    
    final_df = pl.DataFrame(final_data)
    
    assert "latent_groundwater" not in final_df.columns, "FATAL: Latent leakage!"
    assert "latent_industrial" not in final_df.columns, "FATAL: Latent leakage!"
    
    assert len(final_df.columns) == 27, f"FATAL: Expected 27 columns, got {len(final_df.columns)}"
    assert len(final_df) == 1825, "FATAL: Expected exactly 1825 rows per hemisphere!"
    
    final_df.write_parquet(processed_dir / f"{hemisphere}_raw.parquet")
    logger.info(f"Successfully saved Service A Dataset to: {processed_dir}")

if __name__ == "__main__":
    run(hemisphere="north")
    run(hemisphere="south")
