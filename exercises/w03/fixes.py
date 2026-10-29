import pandas as pd
import numpy as np
from problems import load_problem, fit_and_log, standardized
from sklearn.linear_model import LinearRegression
import mlflow
from mlproject.tracking import setup


def select_by_correlation(X_train, y_train, threshold=0.1):
    """Wybiera cechy o |korelacji z celem| > threshold.
    
    NIE używa nazw kolumn - tylko statystyki.
    """
    correlations = X_train.corrwith(y_train).abs()
    selected = correlations[correlations > threshold].index.tolist()
    return selected


def remove_correlated_features(X_train, threshold=0.7):
    """Usuwa cechy, które są mocno skorelowane ze sobą nawzajem.
    
    Zostawia jedną z każdej pary skorelowanych cech.
    """
    corr_matrix = X_train.corr().abs()
    
    # Znajdź pary cech o korelacji > threshold
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    # Zwróć listę cech do zachowania
    selected = [col for col in X_train.columns if col not in to_drop]
    return selected


def run_fix_noise():
    """Naprawa problemu 1: selekcja cech przez korelację z celem."""
    data = load_problem("noise")
    setup("w03-noise")
    
    with mlflow.start_run(run_name="fix-correlation"):
        # Wybierz cechy o |korelacji| > 0.1
        selected_columns = select_by_correlation(data.X_train, data.y_train, threshold=0.1)
        
        print(f"Wybrano {len(selected_columns)} cech")
        
        # Przekaż wybrane cechy do fit_and_log
        model, w = fit_and_log(
            standardized(LinearRegression()), 
            data, 
            columns=selected_columns
        )
        
        mlflow.log_param("selection_method", "correlation_with_target")
        mlflow.log_param("correlation_threshold", 0.1)
        mlflow.log_param("n_selected_features", len(selected_columns))


def run_fix_cholesterol():
    """Naprawa problemu 2: usunięcie skorelowanych cech."""
    data = load_problem("cholesterol")
    setup("w03-cholesterol")
    
    with mlflow.start_run(run_name="fix-multicollinearity"):
        # Usuń cechy skorelowane ze sobą (> 0.7)
        selected_columns = remove_correlated_features(data.X_train, threshold=0.7)
        
        print(f"Wybrano {len(selected_columns)} cech po usunięciu skorelowanych")
        
        # Przekaż wybrane cechy do fit_and_log
        model, w = fit_and_log(
            standardized(LinearRegression()), 
            data, 
            columns=selected_columns
        )
        
        mlflow.log_param("selection_method", "remove_correlated_features")
        mlflow.log_param("correlation_threshold", 0.7)
        mlflow.log_param("n_selected_features", len(selected_columns))


if __name__ == "__main__":
    # Uruchom naprawy
    print("=== Naprawa problemu 1 (noise) ===")
    run_fix_noise()
    
    print("\n=== Naprawa problemu 2 (cholesterol) ===")
    run_fix_cholesterol()