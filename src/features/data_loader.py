import polars as pl
import numpy as np
from pathlib import Path

class DataLoader:
    # stateful loader with embedded validation gates
    def __init__(self, data_dir: Path, hemisphere: str) -> None:
        self.hemisphere = hemisphere
        self.data_dir = data_dir
        self._household: pl.DataFrame | None = None
        self._water_usage: np.ndarray | None = None
        self._environment: pl.DataFrame | None = None

    def load_and_validate(self) -> None:
        self._household = pl.read_csv(
            self.data_dir / f"{self.hemisphere}_household.csv",
            schema_overrides={
                "household_id": pl.Utf8,
                "occupancy_count": pl.Int16,
                "appliance_efficiency_score": pl.Float32,
                "landscape_type": pl.Categorical
            }
        )
        # fixed: clipped the massive appliance efficiency score outlier
        self._household = self._household.with_columns(
            pl.col("appliance_efficiency_score").clip(0.15, 1.0)
        )
        self._water_usage = pl.read_parquet(
            self.data_dir / f"{self.hemisphere}_water_usage.parquet"
        ).to_numpy().astype(np.float32)
        self._environment = pl.read_csv(
            self.data_dir / f"{self.hemisphere}_environment.csv",
            schema_overrides={
                "daily_max_temp_celsius": pl.Float32,
                "daily_rainfall_mm": pl.Float32
            }
        )
        self._run_validations()

    def _run_validations(self) -> None:
        assert self._household.height == 100_000
        assert self._water_usage.shape == (100_000, 365)
        assert self._environment.height == 365
        assert np.all(self._water_usage >= 0), "Negative Usage Detected"
        assert self._household["occupancy_count"].min() >= 1
        assert self._household["occupancy_count"].max() <= 8
