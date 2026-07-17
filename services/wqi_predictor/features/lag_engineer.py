import polars as pl
from typing import Literal

def apply_lags(df: pl.DataFrame, hemisphere: Literal["north", "south"]) -> pl.DataFrame:
    # shifts features chronologically based on granger causality profiling
    df_lagged = df.clone()

    if hemisphere == "north":
        # north lags cumulative heat index by 7 days
        df_lagged = df_lagged.with_columns(
            cumulative_heat_index_lag7=pl.col("cumulative_heat_index").shift(7)
        )
        df_lagged = df_lagged.drop("cumulative_heat_index")
        # shift specific cluster features by 1 day
        cluster_lag1 = [
            "cluster_heavy_users_daily_mean_liters",
            "cluster_conservationists_daily_mean_liters",
            "cluster_standard_consumers_daily_mean_liters"
        ]
        for col in cluster_lag1:
            if col in df_lagged.columns:
                df_lagged = df_lagged.with_columns(
                    pl.col(col).shift(1).alias(f"{col}_lag1")
                )
                df_lagged = df_lagged.drop(col)

    elif hemisphere == "south":
        df_lagged = df_lagged.with_columns(
            cumulative_heat_index_lag3=pl.col("cumulative_heat_index").shift(3)
        )
        df_lagged = df_lagged.drop("cumulative_heat_index")

    return df_lagged.drop_nans()
