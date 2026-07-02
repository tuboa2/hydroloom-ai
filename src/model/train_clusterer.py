import joblib
import polars as pl
from sklearn.cluster import KMeans
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
data_dir = parent_dir / "data/processed"
data_dir.mkdir(parents=True, exist_ok=True)
model_dir = parent_dir / "model"
model_dir.mkdir(parents=True, exist_ok=True)

def main():
    # get parquets
    print("Loading preprocessed features...")
    north_df = pl.read_parquet(data_dir / "north_final.parquet")
    south_df = pl.read_parquet(data_dir / "south_final.parquet")

    # initialize kmeans
    kmeans = KMeans(n_clusters=4, random_state=2033, n_init="auto")

    # train kmeans on north and south data
    print("Clustering North Hemisphere...")
    north_labels = kmeans.fit_predict(north_df.to_numpy())
    north_clusters = north_df.with_columns(
        pl.Series("cluster_id", north_labels).cast(pl.Utf8)
    )

    print("Clustering South Hemisphere...")
    south_labels = kmeans.fit_predict(south_df.to_numpy())
    south_clusters = south_df.with_columns(
        pl.Series("cluster_id", south_labels).cast(pl.Utf8)
    )

    # save final datasets for service a and c to use
    print("Saving finalized datasets...")
    north_clusters.write_parquet(data_dir / "north_final_with_clusters.parquet")
    south_clusters.write_parquet(data_dir / "south_final_with_clusters.parquet")

    # save the trained models for future use
    print("Exporting model artifacts...")
    joblib.dump(kmeans, model_dir / "kmeans_north_k4.joblib")
    joblib.dump(kmeans, model_dir / "kmeans_south_k4.joblib")
    
    print("Success! Service B is officially complete. Ready for XGBoost.")   


if __name__ == "__main__":
    main()
