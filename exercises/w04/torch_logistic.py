# exercises/w04/torch_logistic.py
import sys
sys.path.insert(0, 'exercises/w04')

import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

import previous
from mlproject.tracking import setup
from mlproject.training.trainer import evaluate, train_one_epoch


def run_experiment(run_name, weight_decay=None):
    """Run one training experiment with optional L2 regularization."""
    w2 = previous.breast_cancer_w2()
    
    torch.manual_seed(42)
    model = nn.Linear(30, 1)
    
    loss_fn = nn.BCEWithLogitsLoss()
    
    if weight_decay is not None and weight_decay > 0:
        optimizer = torch.optim.SGD([
            {'params': model.weight, 'weight_decay': weight_decay},
            {'params': model.bias, 'weight_decay': 0.0}  # scikit-learn nie karze biasu
        ], lr=0.1)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    gen = torch.Generator().manual_seed(42)
    train_loader = DataLoader(w2.train, batch_size=32, shuffle=True, generator=gen)
    val_loader = DataLoader(w2.val, batch_size=32)
    
    with mlflow.start_run(run_name=run_name):
        best_val_loss = float('inf')
        best_epoch = 0
        
        for epoch in range(1000):
            train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer)
            val_loss = evaluate(model, val_loader, loss_fn)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
            
            if epoch % 10 == 0:
                mlflow.log_metrics({
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, step=epoch)
        
        model.eval()
        with torch.no_grad():
            X_val, y_val = w2.val.tensors
            logits = model(X_val).squeeze()
            probs = torch.sigmoid(logits).numpy()
            val_auc = roc_auc_score(y_val.numpy(), probs)
        
        max_abs_weight = float(model.weight.detach().abs().max())
        table = previous.weights_table(w2, model)
        max_abs_weight_diff = float(table['diff'].abs().max())
        
        mlflow.log_metrics({
            'val_auc': val_auc,
            'max_abs_weight': max_abs_weight,
            'max_abs_weight_diff': max_abs_weight_diff,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
        })
        
        mlflow.log_table(
            data=table.reset_index().rename(columns={'index': 'feature'}),
            artifact_file='weights.json'
        )
        
        mlflow.log_params({
            'model': 'nn.Linear(30, 1)',
            'loss': 'BCEWithLogitsLoss',
            'optimizer': 'SGD',
            'lr': 0.1,
            'batch_size': 32,
            'epochs': 1000,
            'weight_decay': weight_decay if weight_decay else 0.0,
        })


def main():
    setup("w04-breast-cancer-torch")
    
    run_experiment("no-penalty", weight_decay=0.0)
    
    
    w2_temp = previous.breast_cancer_w2()
    n_train = len(w2_temp.train.tensors[0])
    C = 1.0 
    weight_decay = 1.0 / (n_train * C)
    
    run_experiment("l2-as-in-w2", weight_decay=weight_decay)


if __name__ == "__main__":
    main()