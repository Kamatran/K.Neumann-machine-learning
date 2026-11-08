"""California Housing MLP experiment (W5, task Z2).

Searches over four MLP architectures on the California Housing dataset,
logs every epoch to MLflow, picks the best one by validation MSE, and
evaluates it once on the test set.
"""

import time

import mlflow
import numpy as np
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mlproject.models.mlp import MLP, count_parameters
from mlproject.tracking import setup


setup("w05-california-mlp")

housing = fetch_california_housing()
X, y = housing.data, housing.target

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

train_ds = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(
    train_ds,
    batch_size=256,
    shuffle=True,
    generator=torch.Generator().manual_seed(42),
)


search_id = time.strftime("%Y%m%d-%H%M%S")
hidden_configs = [(), (64,), (64, 64), (256, 256)]

for hidden in hidden_configs:
    torch.manual_seed(42)
    model = MLP(8, 1, hidden=hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    criterion = nn.MSELoss()

    run_name = f"mlp-hidden-{hidden}"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("stage", "search")
        mlflow.set_tag("search_id", search_id)
        mlflow.log_param("hidden", str(hidden))
        mlflow.log_param("n_params", count_parameters(model))

        for epoch in range(50):
            model.train()
            running_loss = 0.0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * xb.size(0)
            train_mse = running_loss / len(train_ds)

            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_mse = criterion(val_pred, y_val_t).item()

            mlflow.log_metric("train_mse", train_mse, step=epoch)
            mlflow.log_metric("val_mse", val_mse, step=epoch)

        #mlflow.pytorch.log_model(model, "model")
        input_example = torch.zeros(1, 8, dtype=torch.float32)
        mlflow.pytorch.log_model(
            model,
            "model",
            input_example=input_example,
            serialization_format="pickle",
        )


runs_df = mlflow.search_runs(
    filter_string=f"tags.search_id = '{search_id}'",
    order_by=["metrics.val_mse ASC"],
)
best_row = runs_df.iloc[0]
best_run_id = best_row["run_id"]
best_hidden = best_row["params.hidden"]
best_val_mse = best_row["metrics.val_mse"]
print(f"Najlepsza architektura: hidden={best_hidden}, val_mse={best_val_mse:.4f}")

best_model = mlflow.pytorch.load_model(f"runs:/{best_run_id}/model")

with mlflow.start_run(run_name=f"final-test-{best_hidden}"):
    mlflow.set_tag("stage", "final")
    mlflow.set_tag("test_evaluated", "true")
    mlflow.log_param("hidden", best_hidden)

    best_model.eval()
    with torch.no_grad():
        test_pred = best_model(X_test_t)
        test_mse = nn.MSELoss()(test_pred, y_test_t).item()

    mlflow.log_metric("test_mse", test_mse)
    print(f"Test MSE: {test_mse:.4f}")
