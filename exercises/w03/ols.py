from problems import load_problem, fit_and_log, standardized
from sklearn.linear_model import LinearRegression
import mlflow
from mlproject.tracking import setup
from weights import weight_summary, top_weights
import pandas as pd
import matplotlib.pyplot as plt
import seaborn

data = load_problem("noise")
data_not = load_problem("cholesterol")
setup("w03-noise")
with mlflow.start_run(run_name="ols"):
    model, w = fit_and_log(standardized(LinearRegression()), data)
    w_summary = weight_summary(w)
    top_w = top_weights(w, k=10)
    print(top_w)

    train = data.X_train
    target = data.y_train
    connected = pd.concat([train,target], axis = 1)
    out = connected.corr()['target'].to_dict()

    mlflow.log_metrics(w_summary)
    mlflow.log_dict(top_w, "top_weights.json")
    mlflow.log_dict(out, "correlation_with_target.json")

setup("w03-cholesterol")
with mlflow.start_run(run_name="ols"):
    model, w = fit_and_log(standardized(LinearRegression()), data_not)
    w_summary = weight_summary(w)
    top_w = top_weights(w, k=10)
    print(top_w)

    train = data.X_train
    target = data.y_train
    connected = pd.concat([train,target], axis = 1)
    out = connected.corr()['target'].to_dict()

    mlflow.log_metrics(w_summary)
    mlflow.log_dict(top_w, "top_weights.json")
    mlflow.log_dict(out, "correlation_with_target.json")
