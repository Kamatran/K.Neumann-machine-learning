import data as d
import mlflow
import mlflow.data
from mlflow.tracking import MlflowClient
from mlproject.tracking import setup
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    setup("w01-transforms")
    
    df = d.load_frame()
    train, val, test = d.split_data(df)
    
    base_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                     'Population', 'AveOccup', 'Latitude', 'Longitude']
    
    y_train = train['MedHouseVal']
    y_val = val['MedHouseVal']
    
    variants = ["none", "log1p", "clip-q99", "poly2", "poly3"]
    
    for variant in variants:
        with mlflow.start_run(run_name=variant, tags={"stage": "selection"}):
            mlflow.log_param("variant", variant)
            
            X_tr = train[base_features].copy()
            X_va = val[base_features].copy()
            
            if variant == "log1p":
                log_cols = ['MedInc', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']
                X_tr[log_cols] = np.log1p(X_tr[log_cols])
                X_va[log_cols] = np.log1p(X_va[log_cols])
                
            elif variant == "clip-q99":
                clip_cols = ['AveRooms', 'AveBedrms', 'Population', 'AveOccup']
                # Próg liczony TYLKO z części treningowej
                thresholds = X_tr[clip_cols].quantile(0.99)
                
                X_tr[clip_cols] = X_tr[clip_cols].clip(upper=thresholds, axis=1)
                X_va[clip_cols] = X_va[clip_cols].clip(upper=thresholds, axis=1)
                
            elif variant in ["poly2", "poly3"]:
                degree = 2 if variant == "poly2" else 3

                scaler = StandardScaler()
                X_tr = pd.DataFrame(scaler.fit_transform(X_tr), columns=base_features, index=X_tr.index)
                X_va = pd.DataFrame(scaler.transform(X_va), columns=base_features, index=X_va.index)
                
                poly = PolynomialFeatures(degree=degree, include_bias=False)
                X_tr = pd.DataFrame(poly.fit_transform(X_tr), columns=poly.get_feature_names_out(base_features), index=X_tr.index)
                X_va = pd.DataFrame(poly.transform(X_va), columns=poly.get_feature_names_out(base_features), index=X_va.index)
            
            if variant in ["none", "log1p", "clip-q99"]:
                scaler = StandardScaler()
                X_tr = pd.DataFrame(scaler.fit_transform(X_tr), columns=base_features, index=X_tr.index)
                X_va = pd.DataFrame(scaler.transform(X_va), columns=base_features, index=X_va.index)
            
            model = LinearRegression().fit(X_tr, y_train)
            
            preds_va = model.predict(X_va)
            va_mse = mean_squared_error(y_val, preds_va)
            mlflow.log_metric("val_mse", va_mse)
            
            preds_tr = model.predict(X_tr)
            train_mse = mean_squared_error(y_train, preds_tr)
            mlflow.log_metric("train_mse", train_mse)

    runs_df = mlflow.search_runs(
        experiment_names=["w01-transforms"],
        filter_string="tags.stage = 'selection'",
        order_by=["metrics.val_mse ASC"],
        max_results=1
    )
    
    best_variant = runs_df.iloc[0]["params.variant"]
    print(f"Wybrany najlepszy wariant: {best_variant} (val_mse: {runs_df.iloc[0]['metrics.val_mse']:.4f})")
    
    with mlflow.start_run(run_name=f"final-test-{best_variant}", tags={"stage": "final", "test_evaluated": "true"}):
        mlflow.log_param("variant", best_variant)
        
        X_tr = train[base_features].copy()
        X_te = test[base_features].copy()
        
        if best_variant == "log1p":
            log_cols = ['MedInc', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']
            X_tr[log_cols] = np.log1p(X_tr[log_cols])
            X_te[log_cols] = np.log1p(X_te[log_cols])
            
        elif best_variant == "clip-q99":
            clip_cols = ['AveRooms', 'AveBedrms', 'Population', 'AveOccup']
            thresholds = X_tr[clip_cols].quantile(0.99)
            
            X_tr[clip_cols] = X_tr[clip_cols].clip(upper=thresholds, axis=1)
            X_te[clip_cols] = X_te[clip_cols].clip(upper=thresholds, axis=1)
            
        elif best_variant in ["poly2", "poly3"]:
            degree = 2 if best_variant == "poly2" else 3
            scaler = StandardScaler()
            X_tr = pd.DataFrame(scaler.fit_transform(X_tr), columns=base_features, index=X_tr.index)
            X_te = pd.DataFrame(scaler.transform(X_te), columns=base_features, index=X_te.index)
            
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            X_tr = pd.DataFrame(poly.fit_transform(X_tr), columns=poly.get_feature_names_out(base_features), index=X_tr.index)
            X_te = pd.DataFrame(poly.transform(X_te), columns=poly.get_feature_names_out(base_features), index=X_te.index)
            
        if best_variant in ["none", "log1p", "clip-q99"]:
            scaler = StandardScaler()
            X_tr = pd.DataFrame(scaler.fit_transform(X_tr), columns=base_features, index=X_tr.index)
            X_te = pd.DataFrame(scaler.transform(X_te), columns=base_features, index=X_te.index)
        
        final_model = LinearRegression().fit(X_tr, y_train)
        
        info = mlflow.sklearn.log_model(
            final_model, 
            name="model", 
            input_example=X_tr.head(5),
            registered_model_name="california-linear"
        )
        
        client = MlflowClient()
        client.set_registered_model_alias("california-linear", "champion", info.registered_model_version)
        
        loaded_model = mlflow.pyfunc.load_model("models:/california-linear@champion")
        
        preds_te = loaded_model.predict(X_te)
        test_mse = mean_squared_error(y_val, preds_te)
        mlflow.log_metric("test_mse", test_mse)
        
        test_df = X_te.copy()
        test_df["target"] = y_val.values
        dataset = mlflow.data.from_pandas(test_df, targets="target", name="test_dataset")
        mlflow.log_input(dataset, context="testing")