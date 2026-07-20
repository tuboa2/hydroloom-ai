import numpy as np
import polars as pl
import shap
from typing import List, Tuple, Dict
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit

class FeatureSelectionCascade:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def lasso_screening(self, X_train: pl.DataFrame, y_train: pl.Series) -> Dict[str, bool]:
        lasso = LassoCV(
            cv=TimeSeriesSplit(n_splits=5),
            alphas=np.logspace(-4, 1, 100),
            random_state=self.random_state,
            max_iter=10000
        )
        lasso.fit(X_train, y_train)
        return {
            col: abs(coef) < 1e-6 for col, coef in zip(X_train.columns, lasso.coef_)
        }

    def permutation_importance(self, X_train: pl.DataFrame, y_train: pl.Series, X_val: pl.DataFrame, y_val: pl.Series) -> Tuple[RandomForestRegressor, Dict[str, bool], pl.Series]:
        rf_baseline = RandomForestRegressor(
            n_estimators=300,
            random_state=self.random_state,
            n_jobs=-1
        )
        rf_baseline.fit(X_train, y_train)

        perm_results = permutation_importance(
            rf_baseline, X_val, y_val,
            n_repeats=30,
            random_state=self.random_state,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1
        )

        importance_means = pl.Series("importance", perm_results.importances_mean)
        perm_zero_flags = {
            col: imp <= 0 for col, imp in zip(X_train.columns, importance_means)
        }
        return rf_baseline, perm_zero_flags, importance_means

    def treeshap(self, rf_baseline: RandomForestRegressor, X_val: pl.DataFrame, y_std: float) -> Dict[str, bool]:
        explainer = shap.TreeExplainer(rf_baseline)
        shap_values = explainer.shap_values(X_val.to_numpy())
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_threshold = 0.001 * y_std
        return {
            col: shap_val < shap_threshold for col, shap_val in zip(X_val.columns, mean_abs_shap)
        }

    def run(self, X_train: pl.DataFrame, y_train: pl.Series, X_val: pl.DataFrame, y_val: pl.Series) -> List[str]:
        # stage 1: linear zero check
        l1_zero_flags = self.lasso_screening(X_train, y_train)
        # stage 2: permutation check
        rf_baseline, perm_zero_flags, perm_importances = self.permutation_importance(
            X_train, y_train, X_val, y_val
        )
        # stage 3: exact shap check
        y_std = y_train.std()
        shap_zero_flags = self.treeshap(rf_baseline, X_val, y_std)
        # retained features
        retained_features = []
        for col in X_train.columns:
            # multi-gate exclusion rule
            if l1_zero_flags[col] and perm_zero_flags[col] and shap_zero_flags[col]:
                continue
            retained_features.append(col)
        return retained_features
    