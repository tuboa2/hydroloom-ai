import tempfile
import warnings
from pathlib import Path

import numpy as np
import polars as pl

from services.wqi_predictor.explainability.shap_analyzer import _save_instance_plots
from services.wqi_predictor.models.lightgbm_regressor import (
    _lgbm_train_params,
    train_lightgbm_fold,
)
from services.wqi_predictor.models.linear_regressor import train_linear_model
from services.wqi_predictor.tuning.optuna_orchestrator import run


def test_lgbm_params_feature_starvation_constraint():
    params = {
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 5,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.3,
        "feature_fraction_bynode": 0.3,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "min_gain_to_split": 0.0,
        "path_smooth": 0.0,
    }

    # With 10 features (e.g. North dataset), product must be >= 0.1
    lgbm_params = _lgbm_train_params(params, loss_name="rmse", num_features=10)
    colsample = lgbm_params["feature_fraction"]
    bynode = lgbm_params["feature_fraction_bynode"]

    assert colsample * bynode >= 1.0 / 10.0
    assert colsample <= 1.0
    assert bynode <= 1.0


def test_train_lightgbm_fold_low_colsample_small_features():
    # 10 features, 200 rows (mimicking north dataset)
    rng = np.random.default_rng(42)
    x_train = rng.normal(size=(200, 10)).astype(np.float32)
    y_train = rng.normal(size=200).astype(np.float64)
    x_val = rng.normal(size=(50, 10)).astype(np.float32)
    y_val = rng.normal(size=50).astype(np.float64)

    params = {
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": 4,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.3,
        "feature_fraction_bynode": 0.3,
        "min_child_samples": 5,
        "reg_alpha": 0.01,
        "reg_lambda": 0.01,
        "min_gain_to_split": 0.0,
        "path_smooth": 0.0,
        "n_estimators": 20,
    }

    # Should train without LightGBMError: (train_data->num_features()) > (0)
    booster, best_iter = train_lightgbm_fold(
        params=params,
        loss_name="logcosh",
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
    )
    assert booster is not None
    assert best_iter >= 1


def test_linear_quantile_transformer_scaling():
    # Small sample size (e.g. 185 samples from smaller CV fold)
    n_samples = 185
    rng = np.random.default_rng(42)
    x_train = rng.normal(size=(n_samples, 5))
    y_train = rng.normal(size=n_samples)

    params = {
        "model_type": "ridge",
        "alpha": 1.0,
        "target_transform": "quantile_normal",
    }

    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        model, _ = train_linear_model(params, x_train, y_train)

    # Ensure no UserWarning about n_quantiles > n_samples was raised
    quantile_warnings = [
        w
        for w in recorded_warnings
        if "n_quantiles is greater than the total number of samples" in str(w.message)
    ]
    assert len(quantile_warnings) == 0

    # Ensure n_quantiles was properly set to min(len(x_train), 1000) = 185
    assert hasattr(model, "transformer")
    assert model.transformer.n_quantiles == 185


def test_optuna_orchestrator_prunes_on_lightgbm_error():
    # Verify optuna handles trial pruning cleanly
    rng = np.random.default_rng(42)
    n_samples = 100
    features = [f"f{i}" for i in range(5)]
    x_train = pl.DataFrame({f: rng.normal(size=n_samples) for f in features})
    y_train = pl.Series("target", rng.normal(size=n_samples))

    with tempfile.TemporaryDirectory() as tmp_dir:
        storage = f"sqlite:///{tmp_dir}/test_optuna.db"
        result = run(
            hemisphere="north",
            model_family="lightgbm",
            loss_name="rmse",
            x_train=x_train,
            y_train=y_train,
            feature_columns=features,
            n_trials=2,
            storage=storage,
        )
        assert result.best_trial_number >= 0
        assert len(result.best_params) > 0


def test_shap_force_plot_zero_variance():
    # Test that zero-variance force plot does not raise uncaught warning
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir)

        class MockExplainer:
            expected_value = 50.0

        explainer = MockExplainer()
        shap_values = np.zeros((10, 5))  # Zero variance across all instances
        x = np.zeros((10, 5))
        feature_names = [f"f{i}" for i in range(5)]
        y_true = np.zeros(10)
        y_pred = np.zeros(10)

        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            res = _save_instance_plots(
                output_dir=output_dir,
                explainer=explainer,
                shap_values=shap_values,
                x=x,
                feature_names=feature_names,
                y_true=y_true,
                y_pred=y_pred,
            )

        xlim_warnings = [
            w for w in recorded_warnings if "identical low and high xlims" in str(w.message)
        ]
        assert len(xlim_warnings) == 0
        assert len(res["force_plots"]) > 0
