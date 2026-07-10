from __future__ import annotations
import logging
import polars as pl
import global_init  # global initialization
from pathlib import Path
from typing import Literal
from sims.household import HouseholdSimulator
from sims.environment import EnvironmentalSimulator
from sims.precipitation import PrecipitationSimulator
from sims.runoff import RunoffSimulator
from sims.cluster import ClusterSimulator
from sims.macro_behavior import MacroBehavioralSimulator

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

    temp_df = pl.DataFrame(daily_max_temp) 
    temp_df.write_csv(data_dir / f"{hemisphere}_temp.csv")

    # 5. generate precipitation features
    precipitation_features = precipitation_sim.generate_features(
        daily_temp=daily_max_temp["daily_max_temp_celsius"],
        season_label=temporal_framework["season_label"],
        hemisphere=hemisphere
    )

    precipitation_df = pl.DataFrame(precipitation_features)
    precipitation_df.write_csv(data_dir / f"{hemisphere}_precipitation.csv")

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

    runoff_df = pl.DataFrame(runoff_features)
    runoff_df.write_csv(data_dir / f"{hemisphere}_runoff.csv")

    # 7. generate household features
    household_ids = household_sim.generate_household_ids()
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

    cluster_df = pl.DataFrame(cluster)
    cluster_df.write_csv(data_dir / f"{hemisphere}_cluster.csv")

    # 9. generate macro behavioral features
    macro = macro_sim.generate_features(
        day_index=temporal_framework["day_index"],
        season_label=temporal_framework["season_label"],
        consecutive_dry_days=precipitation_features["consecutive_dry_days"],
        cluster_means_dict=cluster
    )

    macro_df = pl.DataFrame(macro)
    macro_df.write_csv(data_dir / f"{hemisphere}_macro.csv")

if __name__ == "__main__":
    run(hemisphere="north")
    run(hemisphere="south")
