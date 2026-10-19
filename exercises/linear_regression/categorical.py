import data as d
import mlflow
from mlproject.tracking import setup
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def calculate_rank(df_subset, category_columns):
    one_hot_matrix = df_subset[category_columns].values
    
    ones_column = np.ones((one_hot_matrix.shape[0], 1))
    design_matrix = np.hstack((ones_column, one_hot_matrix))
    
    return np.linalg.matrix_rank(design_matrix)

def evaluate_model(X_train, y_train, X_val, y_val, numeric_cols):

    X_tr = X_train.copy()
    X_va = X_val.copy()
    
    if len(numeric_cols) > 0:
        scaler = StandardScaler()
        X_tr[numeric_cols] = scaler.fit_transform(X_tr[numeric_cols])
        X_va[numeric_cols] = scaler.transform(X_va[numeric_cols])
    
    model = LinearRegression().fit(X_tr, y_train)
    preds = model.predict(X_va)
    
    return mean_squared_error(y_val, preds)

if __name__ == "__main__":
    setup("w01-categorical")

    runs = {
        "complete": {
            "all": ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude'],
            "all+region-full": ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude', 'region_LA', 'region_SAC', 'region_SD', 'region_SJ', 'region_other']
        },
        "droped": {
            "region-only": ['region_SAC', 'region_SD', 'region_SJ', 'region_other'],
            "no-location+region": ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'region_SAC', 'region_SD', 'region_SJ', 'region_other'],
            "all+region": ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude', 'region_SAC', 'region_SD', 'region_SJ', 'region_other'],
        }
    }

    with mlflow.start_run(run_name="region-onehot"):
        df = d.load_frame()
        df_add = d.add_region(df)
        train, val, test = d.split_data(df_add)
        train_all, val_all, _ = d.one_hot(train, val, "region", False)
        train_drop, val_drop, _ = d.one_hot(train, val, "region")

        y_train = train['MedHouseVal']
        y_val = val['MedHouseVal']

        num_LA = train_all["region_LA"].sum()
        num_SAC = train_all["region_SAC"].sum()
        num_SD = train_all["region_SD"].sum()
        num_SJ = train_all["region_SJ"].sum()
        num_other = train_all["region_other"].sum()
        dict = {
            "LA": num_LA,
            "SAC": num_SAC,
            "SD": num_SD,
            "SJ": num_SJ,
            "other": num_other
        }
        mlflow.log_dict(dict, "categories_cardinality.json")
            
        all_cat_cols = [col for col in train_all.columns if col.startswith("region_")]
        rank_with_all = calculate_rank(train_all, all_cat_cols)

        drop_cat_cols = [col for col in train_drop.columns if col.startswith("region_")]
        rank_without_first = calculate_rank(train_drop, drop_cat_cols)
        
        mlflow.log_metric("rank_with_all_categories", rank_with_all)
        mlflow.log_metric("rank_without_first_category", rank_without_first)


        for (encode, dict) in runs.items():
            for (run, cols) in dict.items():
                with mlflow.start_run(run_name=run, nested=True):
                    if encode == "complete":
                        X_train = train_all[cols]
                        X_val = val_all[cols]
                    else:
                        X_train = train_drop[cols]
                        X_val = val_drop[cols]

                    numeric_cols = [c for c in cols if not c.startswith("region_")]
                    
                    mse = evaluate_model(X_train, y_train, X_val, y_val, numeric_cols)
                    mlflow.log_param("encoding", encode)
                    mlflow.log_param("features", cols)
                    mlflow.log_param("n_features", len(cols))
                    mlflow.log_metric("val_mse", mse)
