from data import load_frame, split_data
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import mlflow
from mlproject.tracking import setup
from sklearn.metrics import mean_squared_error


df = load_frame()
train, val, test = split_data(df)

y_train = train['MedHouseVal']
y_val = val['MedHouseVal']

def make_pipeline(X_train, y_train, val):
    if X_train.empty:
        return np.repeat(np.mean(y_train), len(val))
    
    scaler = StandardScaler().fit(X_train)
    transformed_X = scaler.transform(X_train)
    transformed_val = scaler.transform(val)
    model = LinearRegression().fit(transformed_X, y_train)
    return model.predict(transformed_val)

runs = {
    "baseline-mean": [], 
    "medinc": ["MedInc"],
    "no-location": ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup"],
    "medinc-location": ["MedInc", "Latitude", "Longitude"],
    "all": ["MedInc", "Latitude", "Longitude", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup"]
    }

setup("w01-models")

for (run, cols) in runs.items():
    with mlflow.start_run(run_name=run):
        X_train = train[cols]
        X_val = val[cols]
        mlflow_train = pd.concat([X_train, y_train], axis=1)
        mlflow_val = pd.concat([X_val, y_val], axis = 1)

        train_ds = mlflow.data.from_pandas(mlflow_train, name="california-train", targets="MedHouseVal")
        mlflow.log_input(train_ds, context="training")

        val_ds = mlflow.data.from_pandas(mlflow_val, name="california-val", targets="MedHouseVal")
        mlflow.log_input(val_ds, context="validation")

        
        prediction = make_pipeline(X_train, y_train, X_val)
        val_mse = mean_squared_error(y_val, prediction)
        features = len(cols)

        mlflow.log_param("n_features", features)
        mlflow.log_metric("val_mse", val_mse)