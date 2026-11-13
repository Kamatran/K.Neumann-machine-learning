import mlflow
import mlflow.data
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from mlproject.tracking import setup
from mlproject.evaluation.metrics import binary_metrics, choose_threshold

if __name__ == "__main__":
    setup("w02-breast-cancer")

    # Load data
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target  # 0 = malignant, 1 = benign

    # Invert labels so that malignant (0) becomes positive class (1)
    y = (y == 0).astype(int)

    # Split: 60% train, 20% val, 20% test with stratification
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    with mlflow.start_run(run_name="breast-cancer-classifier"):
        # Log data description
        data_description = {
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
            "train_malignant": int(y_train.sum()),
            "val_malignant": int(y_val.sum()),
            "test_malignant": int(y_test.sum()),
            "n_features": X.shape[1],
            "feature_types": "numeric",
            "missing_values": int(X.isnull().sum().sum())
        }
        mlflow.log_dict(data_description, "data_description.json")
        
        # Log datasets
        train_df = X_train.copy()
        train_df["target"] = y_train
        train_dataset = mlflow.data.from_pandas(train_df, targets="target", name="train")
        mlflow.log_input(train_dataset, context="training")
        
        val_df = X_val.copy()
        val_df["target"] = y_val
        val_dataset = mlflow.data.from_pandas(val_df, targets="target", name="validation")
        mlflow.log_input(val_dataset, context="validation")
        
        # Model: StandardScaler + LogisticRegression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Get probabilities for positive class (malignant)
        p_val = model.predict_proba(X_val_scaled)[:, 1]
        p_test = model.predict_proba(X_test_scaled)[:, 1]
        
        # Choose threshold on validation set
        # Justification: For cancer detection, we want high recall (sensitivity)
        # to avoid missing malignant cases. We choose min_recall=0.95.
        thresholds = np.arange(0.05, 1.0, 0.05)
        min_recall = 0.95
        
        chosen_threshold = choose_threshold(y_val, p_val, thresholds, min_recall)
        
        if chosen_threshold is None:
            # Fallback if no threshold meets min_recall
            chosen_threshold = 0.5
            print(f"Warning: No threshold achieved recall >= {min_recall}. Using default 0.5")
        
        mlflow.log_param("min_recall", min_recall)
        mlflow.log_param("threshold", chosen_threshold)
        
        # Validation metrics at chosen threshold
        val_metrics = binary_metrics(y_val, p_val, chosen_threshold)
        for metric_name, metric_value in val_metrics.items():
            mlflow.log_metric(f"val_{metric_name}", metric_value)
        
        # Validation AUC
        val_auc = roc_auc_score(y_val, p_val)
        mlflow.log_metric("val_auc", val_auc)
        
        # ROC curve for validation
        fpr, tpr, _ = roc_curve(y_val, p_val)
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f'ROC curve (AUC = {val_auc:.3f})')
        ax_roc.plot([0, 1], [0, 1], 'k--', label='Random')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('Validation ROC Curve')
        ax_roc.legend()
        ax_roc.grid(alpha=0.3)
        plt.tight_layout()
        mlflow.log_figure(fig_roc, "val_roc_curve.png")
        plt.close(fig_roc)
        
        # Test evaluation (only once at the end)
        test_metrics = binary_metrics(y_test, p_test, chosen_threshold)
        for metric_name, metric_value in test_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", metric_value)
        
        mlflow.set_tag("test_evaluated", "true")
        
        # Test confusion matrix
        cm = confusion_matrix(y_test, (p_test >= chosen_threshold).astype(int))
        fig_cm, ax_cm = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["benign", "malignant"])
        disp.plot(ax=ax_cm, cmap=plt.cm.Blues)
        ax_cm.set_title('Test Confusion Matrix')
        plt.tight_layout()
        mlflow.log_figure(fig_cm, "test_confusion_matrix.png")
        plt.close(fig_cm)
        
        # Log model
        mlflow.sklearn.log_model(model, "model", input_example=X_train.head(5))
        
        print(f"Chosen threshold: {chosen_threshold:.2f}")
        print(f"Validation recall: {val_metrics['recall']:.3f}")
        print(f"Test recall: {test_metrics['recall']:.3f}")