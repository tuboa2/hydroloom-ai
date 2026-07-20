import polars as pl
from pathlib import Path
from features import (
    lag_engineer,
    preprocessor,
    cascade_selector
)
from data import (
    splitter,
    validator
)

data_path = Path("./data/processed")

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

# transform columns and convert to pandas
X_north_train_scaled_pd = north_preprocessor.fit_transform(X_north_train.to_pandas(), y_north_train.to_pandas())
X_south_train_scaled_pd = south_preprocessor.fit_transform(X_south_train.to_pandas(), y_south_train.to_pandas())
X_north_val_scaled_pd = north_preprocessor.transform(X_north_val.to_pandas())
X_south_val_scaled_pd = south_preprocessor.transform(X_south_val.to_pandas())
X_north_test_scaled_pd = north_preprocessor.transform(X_north_test.to_pandas())
X_south_test_scaled_pd = south_preprocessor.transform(X_south_test.to_pandas())

# get column names
north_columns = north_preprocessor.get_feature_names_out().tolist()
south_columns = south_preprocessor.get_feature_names_out().tolist()

# transform to polars
X_north_train_scaled = pl.DataFrame(X_north_train_scaled_pd, schema=north_columns)
X_south_train_scaled = pl.DataFrame(X_south_train_scaled_pd, schema=south_columns)
X_north_val_scaled = pl.DataFrame(X_north_val_scaled_pd, schema=north_columns)
X_south_val_scaled = pl.DataFrame(X_south_val_scaled_pd, schema=south_columns)
X_north_test_scaled = pl.DataFrame(X_north_test_scaled_pd, schema=north_columns)
X_south_test_scaled = pl.DataFrame(X_south_test_scaled_pd, schema=south_columns)

# cascade features
fsc = cascade_selector.FeatureSelectionCascade(random_state=42)
north_features = fsc.run(X_north_train_scaled, y_north_train, X_north_val_scaled, y_north_val)
south_features = fsc.run(X_south_train_scaled, y_south_train, X_south_val_scaled, y_south_val)

north_drop_list = [col for col, should_drop in north_features.items() if should_drop]
south_drop_list = [col for col, should_drop in south_features.items() if should_drop]

# filter
X_north_train_filtered = X_north_train_scaled.drop(north_drop_list)
X_south_train_filtered = X_south_train_scaled.drop(south_drop_list)
X_north_val_filtered = X_north_val_scaled.drop(north_drop_list)
X_south_val_filtered = X_south_val_scaled.drop(south_drop_list)
X_north_test_filtered = X_north_test_scaled.drop(north_drop_list)
X_south_test_filtered = X_south_test_scaled.drop(south_drop_list)
