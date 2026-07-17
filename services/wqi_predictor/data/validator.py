from dataclasses import dataclass
import polars as pl
from scipy.stats import ks_2samp

@dataclass(frozen=True)
class KSResult:
    # dataclass to hold the result of a kolmogorov-smirnov test and basic stats
    statistic: float
    p_value: float
    dist1_mean: float
    dist1_std: float
    dist2_mean: float
    dist2_std: float

def validate_drift(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame
) -> dict[str, KSResult]:
    # verifies distributional drift accross the splits
    target = "water_quality_index"

    train_target = train_df[target].drop_nans()
    val_target = val_df[target].drop_nans()
    test_target = test_df[target].drop_nans()
    
    results = {}
    
    # train vs validation
    stat, pval = ks_2samp(train_target, val_target)
    results["train_vs_val"] = KSResult(
        statistic=stat,
        p_value=pval,
        dist1_mean=train_target.mean(),
        dist1_std=train_target.std(),
        dist2_mean=val_target.mean(),
        dist2_std=val_target.std(),
    )

    # train vs test
    stat, pval = ks_2samp(train_target, test_target)
    results["train_vs_test"] = KSResult(
        statistic=stat,
        p_value=pval,
        dist1_mean=train_target.mean(),
        dist1_std=train_target.std(),
        dist2_mean=test_target.mean(),
        dist2_std=test_target.std(),
    )

    # validation vs test
    stat, pval = ks_2samp(val_target, test_target)
    results["val_vs_test"] = KSResult(
        statistic=stat,
        p_value=pval,
        dist1_mean=val_target.mean(),
        dist1_std=val_target.std(),
        dist2_mean=test_target.mean(),
        dist2_std=test_target.std(),
    )

    return results
