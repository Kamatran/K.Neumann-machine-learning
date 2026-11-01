"""W4: the W1 and W2 models read back from MLflow, with their data prepared exactly as then.
Prepared by the lecturer; do not change.

Both functions load the scikit-learn model you saved in MLflow and rebuild the split and the preprocessing
it was trained with. The PyTorch model then sees the same numbers the scikit-learn model saw, so its weights
can be compared with the scikit-learn weights one by one.

    from previous import breast_cancer_w2, california_w1
    w1 = california_w1()        # w1.train, w1.val: TensorDataset; w1.coef, w1.intercept: scikit-learn weights
"""

from dataclasses import dataclass

import mlflow
import numpy as np
import pandas as pd
import torch
from mlflow import MlflowClient
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

from mlproject.tracking import setup

SEED = 42
CALIFORNIA_TARGET = "MedHouseVal"
CLIPPED = ["AveRooms", "AveBedrms", "Population", "AveOccup"]   # W1, Z5: variant clip-q99
SKEWED = ["MedInc", "AveRooms", "AveBedrms", "Population", "AveOccup"]   # W1, Z5: variant log1p


@dataclass
class Previous:
    train: TensorDataset            # standardized features (float32) and the target as a column (n, 1)
    val: TensorDataset
    features: list[str]
    coef: np.ndarray                # scikit-learn weights, one per feature, in the order of features
    intercept: float
    reference: dict                 # what the scikit-learn model reached on the same data
    source: str                     # where the model came from in MLflow


def _tensors(X, y):
    return TensorDataset(torch.tensor(np.asarray(X), dtype=torch.float32),
                         torch.tensor(np.asarray(y), dtype=torch.float32).unsqueeze(1))


def _transform(variant, X_train, X_other):
    """The W1, Z5 transformations; every statistic (the clipping quantile) comes from X_train."""
    A, B = X_train.copy(), X_other.copy()
    if variant == "log1p":
        A[SKEWED], B[SKEWED] = np.log1p(A[SKEWED]), np.log1p(B[SKEWED])
    elif variant == "clip-q99":
        upper = X_train[CLIPPED].quantile(0.99)
        A[CLIPPED], B[CLIPPED] = A[CLIPPED].clip(upper=upper, axis=1), B[CLIPPED].clip(upper=upper, axis=1)
    elif variant != "none":
        raise ValueError(f"champion variant {variant!r} is not a single linear neuron; see the task list, Z2")
    return A, B


def california_w1(model_name="california-linear", alias="champion") -> Previous:
    """W1, Z5: the model registered as california-linear@champion and the W1 split (60/20/20, seed 42)."""
    setup("w04-california-torch")
    client = MlflowClient()
    version = client.get_model_version_by_alias(model_name, alias)
    variant = client.get_run(version.run_id).data.params["variant"]
    pipeline = mlflow.sklearn.load_model(f"models:/{model_name}@{alias}")
    scaler, linear = pipeline[0], pipeline[-1]

    df = fetch_california_housing(as_frame=True).frame
    rest, _ = train_test_split(df, test_size=0.2, random_state=SEED)      # the test part is not used in W4
    train, val = train_test_split(rest, test_size=0.25, random_state=SEED)
    features = list(scaler.feature_names_in_)
    A, B = _transform(variant, train[features], val[features])
    mse = lambda X, y: float(np.mean((pipeline.predict(X) - y.to_numpy()) ** 2))
    return Previous(
        train=_tensors(scaler.transform(A), train[CALIFORNIA_TARGET]),
        val=_tensors(scaler.transform(B), val[CALIFORNIA_TARGET]),
        features=features, coef=np.asarray(linear.coef_).ravel(), intercept=float(linear.intercept_),
        reference={"variant": variant, "train_mse": mse(A, train[CALIFORNIA_TARGET]), "val_mse": mse(B, val[CALIFORNIA_TARGET])},
        source=f"models:/{model_name}@{alias} (version {version.version}, variant {variant})",
    )


def breast_cancer_w2(experiment="w02-breast-cancer") -> Previous:
    """W2, Z2: the most recent run of w02-breast-cancer with a logged model, and the W2 split
    (60/20/20, stratified, seed 42; the positive class is malignant, target == 0 in scikit-learn)."""
    setup(experiment)
    runs = mlflow.search_runs(experiment_names=[experiment], order_by=["attributes.start_time DESC"])
    for run_id in runs["run_id"]:
        try:
            pipeline = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f"no run with a model in {experiment}: run your W2 script breast_cancer.py again")
    run = runs.set_index("run_id").loc[run_id]
    scaler, logistic = pipeline[0], pipeline[-1]

    data = load_breast_cancer(as_frame=True)
    X, y = data.data, (data.target == 0).astype(int)
    X_trval, _, y_trval, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_trval, y_trval, test_size=0.25, stratify=y_trval,
                                                      random_state=SEED)
    p_val = pipeline.predict_proba(X_val)[:, 1]
    return Previous(
        train=_tensors(scaler.transform(X_train), y_train), val=_tensors(scaler.transform(X_val), y_val),
        features=list(X.columns), coef=np.asarray(logistic.coef_).ravel(), intercept=float(logistic.intercept_[0]),
        reference={"C": float(logistic.C), "val_auc": float(run.get("metrics.val_auc", np.nan)),
                   "val_loss": float(log_loss(y_val, p_val))},
        source=f"runs:/{run_id}/model ({run.get('tags.mlflow.runName', '')})",
    )


def weights_table(previous: Previous, model: torch.nn.Linear) -> pd.DataFrame:
    """scikit-learn and PyTorch weights side by side, the bias as the last row."""
    torch_w = np.r_[model.weight.detach().numpy().ravel(), model.bias.item()]
    sklearn_w = np.r_[previous.coef, previous.intercept]
    table = pd.DataFrame({"sklearn": sklearn_w, "torch": torch_w}, index=previous.features + ["(bias)"])
    table["diff"] = table["torch"] - table["sklearn"]
    return table
