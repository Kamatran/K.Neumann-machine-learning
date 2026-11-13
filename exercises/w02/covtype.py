import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import mlflow
from mlproject.evaluation.metrics import multiclass_metrics
from mlproject.tracking import setup


def main():
    setup('w02-covtype')
    
    # Fetch dataset
    print("Fetching covtype dataset...")
    covtype = fetch_covtype()
    X = pd.DataFrame(covtype.data, columns=covtype.feature_names)
    y = covtype.target
    
    # Sample 100,000 rows with stratification
    print("Sampling 100,000 rows...")
    X_sample, _, y_sample, _ = train_test_split(
        X, y, train_size=100_000, stratify=y, random_state=42
    )
    
    # Split 60/20/20 (train/val/test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_sample, y_sample, train_size=0.6, stratify=y_sample, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, train_size=0.5, stratify=y_temp, random_state=42
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Run 1: Baseline-majority
    print("Running baseline-majority...")
    with mlflow.start_run(run_name="baseline-majority"):
        unique, counts = np.unique(y_train, return_counts=True)
        majority_class = unique[np.argmax(counts)]
        y_val_pred_baseline = np.full(len(y_val), majority_class)
        
        baseline_metrics = multiclass_metrics(y_val, y_val_pred_baseline)
        print(f"Baseline validation accuracy: {baseline_metrics['accuracy']:.4f}")
        
        mlflow.log_metrics({
            "val_accuracy": baseline_metrics["accuracy"],
            "val_f1_macro": baseline_metrics["f1_macro"],
            "val_f1_micro": baseline_metrics["f1_micro"],
            "val_f1_weighted": baseline_metrics["f1_weighted"]
        })
    
    # Run 2: Logistic Regression
    print("Running LogisticRegression...")
    with mlflow.start_run(run_name="logreg"):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        logreg = LogisticRegression(max_iter=2000, random_state=42)
        logreg.fit(X_train_scaled, y_train)
        
        # Validation metrics
        y_val_pred = logreg.predict(X_val_scaled)
        val_metrics = multiclass_metrics(y_val, y_val_pred)
        print(f"LogReg validation accuracy: {val_metrics['accuracy']:.4f}")
        
        mlflow.log_metrics({
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_macro": val_metrics["f1_macro"],
            "val_f1_micro": val_metrics["f1_micro"],
            "val_f1_weighted": val_metrics["f1_weighted"]
        })
        
        # Log recall per class as JSON
        classes = sorted(set(y_val) | set(y_val_pred))
        recall_dict = {str(cls): recall for cls, recall in zip(classes, val_metrics["recall_per_class"])}
        mlflow.log_dict(recall_dict, "recall_per_class.json")
        
        # Confusion matrix normalized by rows
        fig, ax = plt.subplots(figsize=(10, 8))
        ConfusionMatrixDisplay.from_predictions(
            y_val, y_val_pred, 
            normalize="true",
            ax=ax,
            cmap="Blues"
        )
        plt.title("Confusion Matrix (normalized by row)")
        plt.tight_layout()
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close()
        
        # Test evaluation
        print("Evaluating on test set...")
        y_test_pred = logreg.predict(X_test_scaled)
        test_metrics = multiclass_metrics(y_test, y_test_pred)
        
        mlflow.log_metrics({
            "test_accuracy": test_metrics["accuracy"],
            "test_f1_macro": test_metrics["f1_macro"],
            "test_f1_micro": test_metrics["f1_micro"],
            "test_f1_weighted": test_metrics["f1_weighted"]
        })
        
        mlflow.set_tag("test_evaluated", "true")
        
        print("Done!")

if __name__ == "__main__":
    main()