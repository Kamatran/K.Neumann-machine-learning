import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # katalog główny projektu do szukania modułów do importu

from mlproject.data.dataset import ImageDataset, read_split
from mlproject.models.mlp import MLP
from mlproject.models.cnn import SimpleCNN
from mlproject.tracking import setup

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mlp", "cnn"], required=True)
    args = parser.parse_args()
    
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits_dir = "data/eurosat/splits"
    train_items = read_split(f"{splits_dir}/train.csv")
    val_items = read_split(f"{splits_dir}/val.csv")
    test_items = read_split(f"{splits_dir}/test.csv")
    
    with open(f"{splits_dir}/classes.json") as f:
        class_names = json.load(f)
    
    train_ds_raw = ImageDataset(train_items, transform=transforms.Compose([
        transforms.Resize((64, 64)), transforms.ToTensor()
    ]))
    
    loader = DataLoader(train_ds_raw, batch_size=64, num_workers=4)
    mean, std = torch.zeros(3), torch.zeros(3)
    count = 0
    for x, _ in loader:
        mean += x.mean(dim=(0, 2, 3)) * x.size(0)
        std += x.std(dim=(0, 2, 3)) * x.size(0)
        count += x.size(0)
    mean /= count
    std /= count
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean.tolist(), std.tolist()),
    ])
    
    train_ds = ImageDataset(train_items, transform=transform)
    val_ds = ImageDataset(val_items, transform=transform)
    test_ds = ImageDataset(test_items, transform=transform)
    
    generator = torch.Generator().manual_seed(42)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=64, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=64, num_workers=4)
    
    torch.manual_seed(42)
    if args.model == "mlp":
        model = MLP(3 * 64 * 64, 10, hidden=(256,)).to(device)
    else:
        model = SimpleCNN(n_classes=10).to(device)
    
    n_params = count_parameters(model)
    print(f"Number of parameters: {n_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # MLflow
    setup("w06-eurosat")
    with mlflow.start_run(run_name=args.model):
        # Logowanie parametrów i wartości kontrolnych
        mlflow.log_params({
            "model": args.model,
            "batch_size": 64,
            "lr": 0.001,
            "epochs": 15,
            "n_params": n_params,
        })
        
        # Logowanie statystyk kanałów
        for i, (m, s) in enumerate(zip(mean.tolist(), std.tolist())):
            mlflow.log_param(f"channel_{i}_mean", round(m, 4))
            mlflow.log_param(f"channel_{i}_std", round(s, 4))
        
        print(f"Channel means: {[round(m, 4) for m in mean.tolist()]}")
        print(f"Channel stds: {[round(s, 4) for s in std.tolist()]}")
        
        for epoch in range(1, 16):
            model.train()
            train_loss = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * x.size(0)
            train_loss /= len(train_ds)
            
            model.eval()
            val_loss, correct = 0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    out = model(x)
                    val_loss += criterion(out, y).item() * x.size(0)
                    correct += (out.argmax(1) == y).sum().item()
            val_loss /= len(val_ds)
            val_acc = correct / len(val_ds)
            
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            }, step=epoch)


        def eval_with_aug(loader, aug_fn):
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in loader:
                    x = aug_fn(x).to(device)
                    y = y.to(device)
                    preds = model(x).argmax(1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)
            return correct / total

        acc_shift16 = eval_with_aug(
            val_loader,
            lambda x: torch.roll(x, (16, 16), dims=(2, 3)),
        )
        acc_rot90 = eval_with_aug(
            val_loader,
            lambda x: torch.rot90(x, 1, dims=(2, 3)),
        )

        print(f"val_accuracy_shift16: {acc_shift16:.4f}")
        print(f"val_accuracy_rot90:   {acc_rot90:.4f}")
        mlflow.log_metric("val_accuracy_shift16", acc_shift16)
        mlflow.log_metric("val_accuracy_rot90", acc_rot90)
        
        model.eval()
        preds, true = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                preds.extend(model(x).argmax(1).cpu().numpy())
                true.extend(y.numpy())
        
        test_acc = sum(p == t for p, t in zip(preds, true)) / len(true)
        mlflow.log_metric("test_accuracy", test_acc)
        
        cm = confusion_matrix(true, preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=150)
        mlflow.log_artifact("confusion_matrix.png")
        
        print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
