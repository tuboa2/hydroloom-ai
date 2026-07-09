# full data generation pipeline for the ul dataset
from __future__ import annotations
from pathlib import Path
import logging
import polars as pl
import global_init  # global initialization
from sims.household import HouseholdSimulator
from sims.environment import EnvironmentalSimulator
from sims.precipitation import PrecipitationSimulator
from sims.runoff import RunoffSimulator
from sims.cluster import ClusterSimulator

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

def run() -> None:
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

    # 3. generate temporal framework
    north_temporal_framework = env_sim.generate_temporal_framework(
        config=global_config,
        hemisphere="north"
    )
    south_temporal_framework = env_sim.generate_temporal_framework(
        config=global_config,
        hemisphere="south"
    )

    # 4. generate daily max temp
    north_daily_max_temp = env_sim.generate_daily_max_temp(
        day_index=north_temporal_framework["day_index"],
        year_index=north_temporal_framework["year_index"],
        hemisphere="north"
    )
    south_daily_max_temp = env_sim.generate_daily_max_temp(
        day_index=south_temporal_framework["day_index"],
        year_index=south_temporal_framework["year_index"],
        hemisphere="south"
    )

    north_temp_df = pl.DataFrame(north_daily_max_temp)
    south_temp_df = pl.DataFrame(south_daily_max_temp)

    north_temp_df.write_csv(data_dir / "north_temp.csv")
    south_temp_df.write_csv(data_dir / "south_temp.csv")

    # 5. generate precipitation features
    north_precipitation_features = precipitation_sim.generate_features(
        daily_temp=north_daily_max_temp["daily_max_temp_celsius"],
        season_label=north_temporal_framework["season_label"],
        hemisphere="north"
    )
    south_precipitation_features = precipitation_sim.generate_features(
        daily_temp=south_daily_max_temp["daily_max_temp_celsius"],
        season_label=south_temporal_framework["season_label"],
        hemisphere="south"
    )

    north_precipitation_df = pl.DataFrame(north_precipitation_features)
    south_precipitation_df = pl.DataFrame(south_precipitation_features)

    north_precipitation_df.write_csv(data_dir / "north_precipitation.csv")
    south_precipitation_df.write_csv(data_dir / "south_precipitation.csv")

    # 6. generate runoff pollutant features
    north_runoff_features = runoff_sim.generate_features(
        daily_rainfall_mm=north_precipitation_features["daily_rainfall_mm"],
        antecedent_moisture_condition=north_precipitation_features["antecedent_moisture_condition"],
        consecutive_dry_days=north_precipitation_features["consecutive_dry_days"],
        cumulative_storm_rainfall_mm=north_precipitation_features["cumulative_storm_rainfall_mm"],
        daily_max_temp_celsius=north_daily_max_temp["daily_max_temp_celsius"],
        temp_anomaly_celsius=north_daily_max_temp["temp_anomaly_celsius"],
        hemisphere="north"
    )
    south_runoff_features = runoff_sim.generate_features(
        daily_rainfall_mm=south_precipitation_features["daily_rainfall_mm"],
        antecedent_moisture_condition=south_precipitation_features["antecedent_moisture_condition"],
        consecutive_dry_days=south_precipitation_features["consecutive_dry_days"],
        cumulative_storm_rainfall_mm=south_precipitation_features["cumulative_storm_rainfall_mm"],
        daily_max_temp_celsius=south_daily_max_temp["daily_max_temp_celsius"],
        temp_anomaly_celsius=south_daily_max_temp["temp_anomaly_celsius"],
        hemisphere="south"
    )

    north_runoff_df = pl.DataFrame(north_runoff_features)
    south_runoff_df = pl.DataFrame(south_runoff_features)

    north_runoff_df.write_csv(data_dir / "north_runoff.csv")
    south_runoff_df.write_csv(data_dir / "south_runoff.csv")

    # 7. generate household features
    north_household_ids = household_sim.generate_household_ids()
    north_occupancy_count = household_sim.generate_occupancy_count(hemisphere="north")
    north_appliance_efficiency = household_sim.generate_appliance_efficiency_score(hemisphere="north")
    north_landscape_type = household_sim.generate_landscape_type(hemisphere="north")
    north_water_usage = household_sim.generate_daily_water_usage_liters(
        occupancy_count=north_occupancy_count,
        appliance_efficiency_score=north_appliance_efficiency,
        landscape_type=north_landscape_type,
        daily_max_temp_celsius=north_daily_max_temp["daily_max_temp_celsius"],
        daily_rainfall_mm=north_precipitation_features["daily_rainfall_mm"],
        hemisphere="north"
    )
    south_household_ids = household_sim.generate_household_ids()
    south_occupancy_count = household_sim.generate_occupancy_count(hemisphere="south")
    south_appliance_efficiency = household_sim.generate_appliance_efficiency_score(hemisphere="south")
    south_landscape_type = household_sim.generate_landscape_type(hemisphere="south")
    south_water_usage = household_sim.generate_daily_water_usage_liters(
        occupancy_count=south_occupancy_count,
        appliance_efficiency_score=south_appliance_efficiency,
        landscape_type=south_landscape_type,
        daily_max_temp_celsius=south_daily_max_temp["daily_max_temp_celsius"],
        daily_rainfall_mm=south_precipitation_features["daily_rainfall_mm"],
        hemisphere="south"
    )

    # 8. generate cluster daily means feature
    north_cluster = cluster_sim.generate_features(
        water_usage_matrix=north_water_usage,
        hemisphere="north",
        daily_rainfall_mm=north_precipitation_features["daily_rainfall_mm"],
        consecutive_dry_days=north_precipitation_features["consecutive_dry_days"]
    )
    south_cluster = cluster_sim.generate_features(
        water_usage_matrix=south_water_usage,
        hemisphere="south",
        daily_rainfall_mm=south_precipitation_features["daily_rainfall_mm"],
        consecutive_dry_days=south_precipitation_features["consecutive_dry_days"]
    )

    north_cluster_df = pl.DataFrame(north_cluster)
    south_cluster_df = pl.DataFrame(south_cluster)

    north_cluster_df.write_csv(data_dir / "north_cluster.csv")
    south_cluster_df.write_csv(data_dir / "south_cluster.csv")

if __name__ == "__main__":
    run()
