"""Covertype MLP classification experiment (W5, task Z3).

Searches over three MLP architectures on the Covertype dataset,
logs every epoch to MLflow, picks the best one by validation accuracy,
and evaluates it once on the test set.
"""
# w ostatnim powinien być wzrost dokładności o około 15%
import time

import mlflow
import numpy as np
import torch
from sklearn.datasets import fetch_covtype
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mlproject.models.mlp import MLP, count_parameters
from mlproject.tracking import setup

setup("w05-covtype-mlp")


covtype = fetch_covtype()
X_full, y_full = covtype.data, covtype.target

np.random.seed(42)
indices = np.random.choice(len(X_full), size=100_000, replace=False)
X, y = X_full[indices], y_full[indices]
y = y - 1

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, train_size=0.6, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

train_ds = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(
    train_ds,
    batch_size=256,
    shuffle=True,
    generator=torch.Generator().manual_seed(42),
)


def multiclass_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    return {
        "accuracy": acc,
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1.tolist(),
    }


search_id = time.strftime("%Y%m%d-%H%M%S")
hidden_configs = [(), (256,), (256, 256)]

for hidden in hidden_configs:
    torch.manual_seed(42)
    model = MLP(54, 7, hidden=hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    run_name = f"mlp-hidden-{hidden}"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("stage", "search")
        mlflow.set_tag("search_id", search_id)
        mlflow.log_param("hidden", str(hidden))
        mlflow.log_param("n_params", count_parameters(model))

        for epoch in range(30):
            # --- Trening ---
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * xb.size(0)

                # Licz poprawne predykcje
                _, predicted = torch.max(pred, 1)
                correct += (predicted == yb).sum().item()
                total += yb.size(0)

            train_loss = running_loss / len(train_ds)
            train_acc = correct / total

            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_loss = criterion(val_pred, y_val_t).item()
                _, val_predicted = torch.max(val_pred, 1)
                val_acc = (val_predicted == y_val_t).float().mean().item()

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            _, val_predicted = torch.max(val_pred, 1)
            metrics = multiclass_metrics(y_val_t.numpy(), val_predicted.numpy())

        mlflow.log_metric("val_accuracy_final", metrics["accuracy"])
        mlflow.log_dict(
            {
                "precision_per_class": metrics["precision_per_class"],
                "recall_per_class": metrics["recall_per_class"],
                "f1_per_class": metrics["f1_per_class"],
            },
            "val_metrics.json",
        )

        #mlflow.pytorch.log_model(model, "model")
        input_example = torch.zeros(1, 54, dtype=torch.float32)
        mlflow.pytorch.log_model(
            model,
            "model",
            input_example=input_example,
            serialization_format="pickle",
        )


runs_df = mlflow.search_runs(
    filter_string=f"tags.search_id = '{search_id}'",
    order_by=["metrics.val_accuracy_final DESC"],
)
best_row = runs_df.iloc[0]
best_run_id = best_row["run_id"]
best_hidden = best_row["params.hidden"]
best_val_acc = best_row["metrics.val_accuracy_final"]
print(f"Najlepsza architektura: hidden={best_hidden}, val_accuracy={best_val_acc:.4f}")

best_model = mlflow.pytorch.load_model(f"runs:/{best_run_id}/model")

logreg_runs = mlflow.search_runs(
    experiment_names=["w02-covtype"],
    filter_string="tags.model_type = 'logreg'",
    order_by=["metrics.val_accuracy DESC"],
)
if not logreg_runs.empty:
    logreg_val_acc = logreg_runs.iloc[0]["metrics.val_accuracy"]
    print(f"Logreg z W2 val_accuracy: {logreg_val_acc:.4f}")
else:
    logreg_val_acc = None
    print("Nie znaleziono przebiegu logreg z W2")


with mlflow.start_run(run_name=f"final-test-{best_hidden}"):
    mlflow.set_tag("stage", "final")
    mlflow.set_tag("test_evaluated", "true")
    mlflow.log_param("hidden", best_hidden)

    if logreg_val_acc is not None:
        mlflow.log_metric("logreg_val_accuracy", logreg_val_acc)

    best_model.eval()
    with torch.no_grad():
        test_pred = best_model(X_test_t)
        _, test_predicted = torch.max(test_pred, 1)
        test_acc = (test_predicted == y_test_t).float().mean().item()
        test_metrics = multiclass_metrics(y_test_t.numpy(), test_predicted.numpy())

    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_dict(
        {
            "precision_per_class": test_metrics["precision_per_class"],
            "recall_per_class": test_metrics["recall_per_class"],
            "f1_per_class": test_metrics["f1_per_class"],
        },
        "test_metrics.json",
    )
    print(f"Test accuracy: {test_acc:.4f}")
