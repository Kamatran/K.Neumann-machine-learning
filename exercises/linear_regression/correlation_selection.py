from data import load_frame, split_data
from models import make_pipeline
from mlproject.tracking import setup
import mlflow
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error


df = load_frame()
train, val, test = split_data(df)

y_train = train['MedHouseVal']
y_val = val['MedHouseVal']

setup("w01-correlation")

with mlflow.start_run(run_name="correlation-selection"):
    X_train = train[["MedInc", "Latitude", "Longitude", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup"]]
    X_val = val[["MedInc", "Latitude", "Longitude", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup"]]
    connected_train = pd.concat([X_train, y_train], axis = 1)
    corelation = connected_train.corr()
    corr_result = corelation["MedHouseVal"].to_dict()

    mlflow.log_dict(corr_result, "correlation_with_target.json")

    fig, ax = plt.subplots(figsize=(10, 8))
    seaborn.heatmap(corelation, cmap="coolwarm", annot=True, ax=ax)
    plt.close(fig)

    mlflow.log_figure(fig, "correlation_train.png")

    thresholds = [0.0, 0.05, 0.10, 0.15, 0.30]
    cols = ["MedInc", "Latitude", "Longitude", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup"]
    kept_cols = []
    for t in thresholds:
        for col in cols:
            if abs(corr_result[col]) >= t:
                kept_cols.append(col)

        with mlflow.start_run(run_name=f"threshold+{t}", nested=True):

            X_train = train[kept_cols]
            X_val = val[kept_cols]
            prediction = make_pipeline(X_train, y_train, X_val)
            val_mse = mean_squared_error(y_val, prediction)
            n_features = len(kept_cols)

            mlflow.log_param("threshold", t)
            mlflow.log_param("n_features", n_features)
            mlflow.log_param("features", kept_cols)
            mlflow.log_metric("val_mse", val_mse)
            kept_cols = []