from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OrdinalEncoder,
    PowerTransformer,
    RobustScaler,
    StandardScaler
)
from features.feature_registry import (
    NORTH_REGISTRY,
    SOUTH_REGISTRY
)

def build_preprocessor(hemisphere: str) -> ColumnTransformer:
    # constructs the column transformer pipeline dynamically based on the hemisphere
    if hemisphere.lower() == "north":
        registry = NORTH_REGISTRY
    elif hemisphere.lower() == "south":
        registry = SOUTH_REGISTRY
    else:
        raise ValueError(f"Unknown Hemisphere: {hemisphere}")

    transformers = []

    # standard scaler for normal featires and cluster features
    standard_features = registry.normal + registry.cluster
    if standard_features:
        transformers.append(("standard", StandardScaler(), standard_features))

    # robust scaler for features with moderate skew and bounded values
    if registry.robust:
        transformers.append(("robust", RobustScaler(), registry.robust))

    # power transformer with yeo-johnson method for highly skewed features
    if registry.power_transform:
        transformers.append(
            ("power", PowerTransformer(method="yeo-johnson"), registry.power_transform)
        )

    # ordinal encoder for categorical features
    if registry.categorical:
        transformers.append(
            (
                "ordinal",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                registry.categorical
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )
