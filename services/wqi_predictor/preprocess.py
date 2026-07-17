import polars as pl
from pathlib import Path
from features import (
    lag_engineer,
    preprocessor
)
from data import (
    splitter,
    validator
)

data_path = Path("../../data/processed")

# load datasets
north_df = pl.read_parquet(data_path / "north_raw.parquet")
south_df = pl.read_parquet(data_path / "south_raw.parquet")

# lag engineering
north_lag_df = lag_engineer.apply_lags(df=north_df, hemisphere="north")
south_lag_df = lag_engineer.apply_lags(df=south_df, hemisphere="south")

# time series split
north_train_df, north_val_df, north_test_df = splitter.chronological_split(df=north_lag_df)
south_train_df, south_val_df, south_test_df = splitter.chronological_split(df=south_lag_df)

# validate split
north_split_results = validator.validate_drift(train_df=north_train_df, val_df=north_val_df, test_df=north_test_df)
south_split_results = validator.validate_drift(train_df=south_train_df, val_df=south_val_df, test_df=south_test_df)

# preprocessors
north_preprocessor = preprocessor.build_preprocessor(hemisphere="north")
south_preprocessor = preprocessor.build_preprocessor(hemisphere="south")

# x y split - North
X_north_train = north_train_df.drop("water_quality_index")
y_north_train = north_train_df.select("water_quality_index")

X_north_val = north_val_df.drop("water_quality_index")
y_north_val = north_val_df.select("water_quality_index") 

X_north_test = north_test_df.drop("water_quality_index")
y_north_test = north_test_df.select("water_quality_index")

# x y split - South
X_south_train = south_train_df.drop("water_quality_index")
y_south_train = south_train_df.select("water_quality_index")

X_south_val = south_val_df.drop("water_quality_index")
y_south_val = south_val_df.select("water_quality_index") 

X_south_test = south_test_df.drop("water_quality_index")
y_south_test = south_test_df.select("water_quality_index")

# transform columns
X_north_train_scaled = north_preprocessor.fit_transform(X_north_train)
X_south_train_scaled = south_preprocessor.fit_transform(X_south_train)
X_north_val_scaled = north_preprocessor.transform(X_north_val)
X_south_val_scaled = south_preprocessor.transform(X_south_val)
X_north_test_scaled = north_preprocessor.transform(X_north_test)
X_south_test_scaled = south_preprocessor.transform(X_south_test)
