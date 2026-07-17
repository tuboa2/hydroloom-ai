import polars as pl

def chronological_split(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    # splits 5-year daily time-series dataset without shuffling
    train_df = df[0:1905].clone()
    val_df = df[1095:1460].clone()
    test_df = df[1460:1825].clone()

    return train_df, val_df, test_df
    