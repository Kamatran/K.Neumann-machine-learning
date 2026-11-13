"""W3: what the weights of a fitted model say. Complete weight_summary and top_weights in Z1, choose_alpha in Z3."""

import pandas as pd
from collections import Counter
from problems import NOISE_PREFIX


def weight_summary(weights: pd.Series) -> dict:
    """Counts of weights, for mlflow.log_metrics.

    weights: one weight per feature, indexed by the feature name (as returned by fit_and_log).
    Returns a dict with the keys:
        nonzero        - weights different from 0
        zero           - weights exactly equal to 0
        noise_nonzero  - nonzero weights of the noise columns (names starting with NOISE_PREFIX)
        real_nonzero   - nonzero weights of the other columns
    All values are ints.
    """
        # TODO (Z1)

    nonzero = 0
    zero = 0
    noise_nonzero = 0
    real_nonzero = 0
    if weights is not None:
        for feature_name, weight_value in weights.items():
            if weight_value != 0:
                nonzero += 1
                if feature_name.startswith("NOISE_PREFIX"):
                    noise_nonzero += 1
                else:
                    real_nonzero += 1
            elif weight_value == 0:
                zero += 1
    return {
        "nonzero": nonzero,
        "zero": zero,
        "noise_nonzero": noise_nonzero,
        "real_nonzero": real_nonzero
    }


def top_weights(weights: pd.Series, k: int = 5) -> dict:
    """The k weights with the largest absolute value, for mlflow.log_dict.

    Returns {feature name: weight as a float}, the largest |weight| first.
    """
    # TODO (Z1)
    if weights is not None:
        weights.sort_values(inplace=True, key=lambda x: x.abs(), ascending=False)
    return weights.head(k).to_dict()


def choose_alpha(search: pd.DataFrame) -> float:
    """The alpha with the lowest validation MSE.

    search: a DataFrame with the columns alpha and val_mse (as returned by search_alpha).
    Returns the alpha as a float.
    """
    # TODO (Z3)
    return search.sort_values('val_mse').iloc[0]['alpha']
