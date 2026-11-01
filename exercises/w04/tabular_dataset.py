"""W4, Z4: Experiment comparing custom TabularDataset with built-in TensorDataset."""

import sys
from pathlib import Path

# Dodajemy katalog nadrzędny do ścieżki, aby importy mlproject działały
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import mlflow

from mlproject.tracking import setup
from mlproject.training.trainer import train_one_epoch, evaluate
from exercises.w04.tabular import Standardize, TabularDataset

SEED = 42
EPOCHS = 20
BATCH_SIZE = 256
LR = 0.01


def run_experiment(run_name: str, train_ds, val_ds):
    torch.manual_seed(SEED)
    model = nn.Linear(8, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    gen = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    with mlflow.start_run(run_name=run_name):
        for epoch in range(EPOCHS):
            train_mse = train_one_epoch(model, train_loader, loss_fn, optimizer)
            val_mse = evaluate(model, val_loader, loss_fn)
            mlflow.log_metrics({"train_mse": train_mse, "val_mse": val_mse}, step=epoch)
        
        return model.weight.detach().clone(), model.bias.detach().clone()


def main():
    setup("w04-california-dataset")
    
    df = pd.read_csv("data/california_housing.csv")
    features = [c for c in df.columns if c != "MedHouseVal"]
    target = "MedHouseVal"

    rest, test = train_test_split(df, test_size=0.2, random_state=SEED)
    train, val = train_test_split(rest, test_size=0.25, random_state=SEED)
    
    print(f"Rozmiar zbioru treningowego: {len(train)} (oczekiwano 12384)")
    print(f"Rozmiar zbioru walidacyjnego: {len(val)} (oczekiwano 4128)")

    scaler = Standardize(train, features)

    tabular_train = TabularDataset(train, features, target, transform=scaler)
    tabular_val = TabularDataset(val, features, target, transform=scaler)

    X_train_t = scaler(torch.tensor(train[features].values, dtype=torch.float32))
    y_train_t = torch.tensor(train[target].values, dtype=torch.float32).unsqueeze(1)
    X_val_t = scaler(torch.tensor(val[features].values, dtype=torch.float32))
    y_val_t = torch.tensor(val[target].values, dtype=torch.float32).unsqueeze(1)

    tensor_train = TensorDataset(X_train_t, y_train_t)
    tensor_val = TensorDataset(X_val_t, y_val_t)

    print("\nRozpoczynam trening: tabular-dataset...")
    w_tab, b_tab = run_experiment("tabular-dataset", tabular_train, tabular_val)
    
    print("Rozpoczynam trening: tensor-dataset...")
    w_ten, b_ten = run_experiment("tensor-dataset", tensor_train, tensor_val)

    max_w_diff = (w_tab - w_ten).abs().max().item()
    max_b_diff = (b_tab - b_ten).abs().max().item()
    max_diff = max(max_w_diff, max_b_diff)
    
    print(f"\n✅ Największa różnica wag między modelami: {max_diff:.4f}")
    print("Oczekiwany wynik: 0.0000")


if __name__ == "__main__":
    main()