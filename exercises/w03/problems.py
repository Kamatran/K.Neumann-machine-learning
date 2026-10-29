"""W3: the two problems, the alpha grids and logging of one fitted model. Complete, do not change.

Every run in w03-noise and w03-cholesterol goes through fit_and_log, so the runs you compare in MLflow
were measured the same way: the same split, the same metrics, the same weight table.
"""

from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
N_NOISE = 100            # problem 1: columns of pure noise added to the 10 real features
NOISE_SEED = 0
NOISE_PREFIX = "noise_"
REAL_FEATURES = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
EXPERIMENTS = {"noise": "w03-noise", "cholesterol": "w03-cholesterol"}

RIDGE_ALPHAS = np.logspace(-2, 4, 13)     # 0.01, 0.0316, ..., 10 000
LASSO_ALPHAS = np.logspace(-2, 2, 17)     # 0.01, 0.0178, ..., 100
LASSO_MAX_ITER = 10_000


class Split(NamedTuple):
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def load_problem(name, seed=SEED):
    """Diabetes split 60/20/20 (264 / 89 / 89 rows).

    "cholesterol": the 10 original features. "noise": the same 10 features plus 100 columns of noise
    drawn once from a normal distribution, for all 442 patients, before the split.
    """
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    if name == "noise":
        rng = np.random.default_rng(NOISE_SEED)
        noise = pd.DataFrame(rng.normal(size=(len(X), N_NOISE)), index=X.index,
                             columns=[f"{NOISE_PREFIX}{i:03d}" for i in range(N_NOISE)])
        X = pd.concat([X, noise], axis=1)
    elif name != "cholesterol":
        raise ValueError(f"unknown problem {name!r}: use 'noise' or 'cholesterol'")
    X_trval, X_test, y_trval, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_trval, y_trval, test_size=0.25, random_state=seed)
    return Split(X_train, X_val, X_test, y_train, y_val, y_test)


def mse(y, pred):
    return float(np.mean((np.asarray(y) - np.asarray(pred)) ** 2))


def standardized(model):
    """StandardScaler fitted on the training part, then the model: the penalty depends on the scale of the features."""
    return make_pipeline(StandardScaler(), model)


def weights_of(model, columns):
    """Weights of the last step of a fitted pipeline, indexed by feature; None if they do not map to features."""
    coef = getattr(model[-1], "coef_", None)
    if coef is None or np.ravel(coef).shape[0] != len(columns):
        return None
    return pd.Series(np.ravel(coef), index=list(columns), name="weight")


def plot_weights(weights):
    """Bar chart of the weights sorted by absolute value; noise columns in grey."""
    order = weights.abs().sort_values(ascending=False).index
    colors = ["#bbbbbb" if c.startswith(NOISE_PREFIX) else "#1f77b4" for c in order]
    fig, ax = plt.subplots(figsize=(max(6, 0.09 * len(order) + 3), 3.6))
    ax.bar(range(len(order)), weights[order].to_numpy(), color=colors)
    if len(order) <= 20:
        ax.set_xticks(range(len(order)), order, rotation=45, ha="right")
    else:
        ax.set_xticks([])
        ax.set_xlabel("features sorted by |weight| (grey: noise)")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("weight")
    fig.tight_layout()
    return fig


def fit_and_log(model, data, columns=None):
    """Fits the model on the training part and logs it into the ACTIVE MLflow run.

    columns: the features to use (default: all). Logs the parameters model and n_features, the metrics
    train_mse and val_mse, a metric w_<name> for every original Diabetes feature that was used,
    the table weights.json and the chart weights.png. Returns (fitted model, weights as a Series or None).
    The test part is never touched here.
    """
    columns = list(data.X_train.columns if columns is None else columns)
    model.fit(data.X_train[columns], data.y_train)
    steps = getattr(model, "steps", [(type(model).__name__, model)])
    mlflow.log_params({"model": "+".join(type(step).__name__ for _, step in steps), "n_features": len(columns)})
    mlflow.log_metrics({"train_mse": mse(data.y_train, model.predict(data.X_train[columns])),
                        "val_mse": mse(data.y_val, model.predict(data.X_val[columns]))})
    if len(columns) < data.X_train.shape[1]:
        mlflow.log_dict({"columns": columns}, "columns.json")

    weights = weights_of(model, columns) if hasattr(model, "steps") else None
    if weights is None:
        mlflow.set_tag("weights", "none: the model has no weight per feature")
        return model, None
    mlflow.log_metrics({f"w_{c}": float(weights[c]) for c in REAL_FEATURES if c in weights.index})
    table = weights.reset_index().rename(columns={"index": "feature"})
    table = table.assign(abs_weight=table["weight"].abs()).sort_values("abs_weight", ascending=False)
    mlflow.log_table(table, "weights.json")
    fig = plot_weights(weights)
    mlflow.log_figure(fig, "weights.png")
    plt.close(fig)
    return model, weights


def search_alpha(make_model, alphas, data):
    """Fits make_model(alpha) for every alpha on the training part.

    Returns (search, path): search is a DataFrame with the columns alpha, train_mse, val_mse, nonzero;
    path is a DataFrame of weights, one row per alpha, one column per feature.
    """
    rows, path = [], []
    for alpha in alphas:
        model = standardized(make_model(alpha)).fit(data.X_train, data.y_train)
        weights = weights_of(model, data.X_train.columns)
        rows.append({"alpha": float(alpha),
                     "train_mse": mse(data.y_train, model.predict(data.X_train)),
                     "val_mse": mse(data.y_val, model.predict(data.X_val)),
                     "nonzero": int((weights != 0).sum())})
        path.append(weights)
    return pd.DataFrame(rows), pd.DataFrame(path, index=[float(a) for a in alphas])


def plot_search(search, chosen):
    """Validation and training MSE against alpha, the chosen alpha marked."""
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(search["alpha"], search["val_mse"], "o-", label="validation")
    ax.plot(search["alpha"], search["train_mse"], "s--", label="training", alpha=0.7)
    ax.axvline(chosen, color="grey", ls=":", label=f"chosen alpha = {chosen:.3g}")
    ax.set_xscale("log")
    ax.set_xlabel("alpha")
    ax.set_ylabel("MSE")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_path(path, chosen):
    """Coefficient path: every weight against alpha; noise columns in grey."""
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for column in path.columns:
        noise = column.startswith(NOISE_PREFIX)
        ax.plot(path.index, path[column], color="#bbbbbb" if noise else None, lw=0.6 if noise else 1.4,
                label=None if noise else column, zorder=1 if noise else 2)
    ax.axvline(chosen, color="grey", ls=":")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("alpha")
    ax.set_ylabel("weight")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return fig
