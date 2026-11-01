# exercises/w04/torch_linear.py
import sys
sys.path.insert(0, 'exercises/w04')

import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import previous
from mlproject.tracking import setup
from mlproject.training.trainer import evaluate, train_one_epoch


def main():
    w1 = previous.california_w1()
    
    setup("w04-california-torch")

    with mlflow.start_run(run_name="torch-sgd"):
        torch.manual_seed(42)
        model = nn.Linear(8, 1)

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        gen = torch.Generator().manual_seed(42)
        train_loader = DataLoader(w1.train, batch_size=256, shuffle=True, generator=gen)
        val_loader = DataLoader(w1.val, batch_size=256)

        for epoch in range(100):
            train_mse = train_one_epoch(model, train_loader, loss_fn, optimizer)
            val_mse = evaluate(model, val_loader, loss_fn)

            table = previous.weights_table(w1, model)
            max_abs_weight_diff = float(table["diff"].abs().max())

            mlflow.log_metrics(
                {
                    "train_mse": train_mse,
                    "val_mse": val_mse,
                    "max_abs_weight_diff": max_abs_weight_diff,
                },
                step=epoch,
            )

        mlflow.log_params(
            {
                "model": "nn.Linear(8, 1)",
                "loss": "MSELoss",
                "optimizer": "SGD",
                "lr": 0.01,
                "batch_size": 256,
                "epochs": 100,
                "seed": 42,
            }
        )

        mlflow.log_table(
            data=table.reset_index().rename(columns={"index": "feature"}),
            artifact_file="weights.json",
        )

        mlflow.log_metrics(
            {
                "w1_train_mse": w1.reference["train_mse"],
                "w1_val_mse": w1.reference["val_mse"],
            }
        )


if __name__ == "__main__":
    main()