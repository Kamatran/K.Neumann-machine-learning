import mlflow
import data
from mlproject.tracking import setup
from sklearn.linear_model import LinearRegression
import joblib
from pathlib import Path
from sklearn.metrics import mean_squared_error

setup("w01-mlflow-basics")

df = data.load_frame()
train, val, test = data.split_data(df)

dictionary = {"train_length": len(train), "val_length": len(val), "test_length": len(test)}

# Z1B/1
def files_to_artefact():
    mlflow.log_artifact("exercises/linear_regression/data.py", artifact_path="code")
    mlflow.log_dict(dictionary, "split.json")

#Z1B/2
def log_dataset(train, name, target, context="training"):
    dataset = mlflow.data.from_pandas(train, name=name, targets=target)
    mlflow.log_input(dataset, context=context)
    print(f"dataset.digest: {dataset.digest}")

with mlflow.start_run():
    files_to_artefact()
    log_dataset(train, "california-train", "MedHouseVal")
    log_dataset(train.iloc[1:], "california-train-no1row", "MedHouseVal")

    X = train.loc[:, 'MedInc'].to_frame()
    y = train.loc[:, 'MedHouseVal'].to_frame()

    reg = LinearRegression().fit(X,y)

    joblib.dump(reg, "data/medinc.joblib") # zapis modelu na dysku
    loaded_model = joblib.load("data/medinc.joblib")
    print(f"Original prediction: {reg.predict(val.loc[:, 'MedInc'].to_frame())}")
    print(f"Loaded model prediction: {loaded_model.predict(val.loc[:, 'MedInc'].to_frame())}")

    mlflow.log_artifact("data/medinc.joblib", artifact_path="model-file") # zapis modelu jako artefakt na mlflow

    info = mlflow.sklearn.log_model(reg, name="model", input_example=X.head(5),
                                registered_model_name="california-medinc") # wczytanie modelu z mlflow
    model_dir = Path(mlflow.artifacts.download_artifacts(info.model_uri)) # stworzony tymczasowy folder modelu na komputerze


    # drukowanie informacji o wczytanym z mlflow modelu
    print("\n--- PLIKI W MODELU ---")
    for file_path in model_dir.iterdir():
        print(file_path.name)

    print("\n--- ZAWARTOŚĆ MLmodel ---")
    with open(model_dir / "MLmodel", "r", encoding="utf-8") as f:
        print(f.read())

    print("\n--- ZAWARTOŚĆ requirements.txt ---")
    with open(model_dir / "requirements.txt", "r", encoding="utf-8") as f:
        print(f.read())

    model = mlflow.pyfunc.load_model("models:/california-medinc/1")
    prediction = model.predict(val.loc[:, 'MedInc'].head(3).to_frame())
    print(f"\n\n Prediction: {prediction}")


    print(f"Digest zbioru treningowego: {mlflow.data.from_pandas(train, name='california-train', targets='MedHouseVal').digest}")
    print(f"MSE walidacyjne: {mean_squared_error(val['MedHouseVal'], reg.predict(val[['MedInc']])):.4f}")
