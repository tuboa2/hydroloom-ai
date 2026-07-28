from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, PowerTransformer, RobustScaler, StandardScaler

from .feature_groups import CATEGORICAL_COLUMNS, assign_numeric_transformer


def build_preprocessor(feature_columns: Sequence[str]) -> ColumnTransformer:
    feature_columns = list(feature_columns)

    if not feature_columns:
        raise ValueError("feature_columns must not be empty.")

    categorical_features = [column for column in feature_columns if column in CATEGORICAL_COLUMNS]

    numeric_features = [column for column in feature_columns if column not in CATEGORICAL_COLUMNS]

    standard_features: list[str] = []
    robust_features: list[str] = []
    power_features: list[str] = []

    for column in numeric_features:
        transformer_name = assign_numeric_transformer(column)

        if transformer_name == "standard":
            standard_features.append(column)
        elif transformer_name == "power":
            power_features.append(column)
        else:
            robust_features.append(column)

    transformers = []

    if standard_features:
        transformers.append(
            (
                "standard",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                standard_features,
            )
        )

    if robust_features:
        transformers.append(
            (
                "robust",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", RobustScaler()),
                    ]
                ),
                robust_features,
            )
        )

    if power_features:
        transformers.append(
            (
                "power",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("transformer", PowerTransformer(method="yeo-johnson", standardize=True)),
                    ]
                ),
                power_features,
            )
        )

    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                dtype=np.float64,
                            ),
                        ),
                    ]
                ),
                categorical_features,
            )
        )

    if not transformers:
        raise ValueError("No valid preprocessing transformers were constructed.")

    preprocessor = ColumnTransformer(
        transformers=transformers, remainder="drop", verbose_feature_names_out=False
    )

    preprocessor.set_output(transform="pandas")

    return preprocessor
