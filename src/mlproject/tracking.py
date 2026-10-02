"""MLflow setup shared by every script in this project."""

import os

import mlflow

# With an S3-compatible artifact store (MinIO), the MLflow 3.16 server advertises presigned
# downloads. The client then fetches files straight from MinIO's address inside the Docker
# network (minio:9000), which your scripts cannot reach, and the download retries for minutes
# without an error. Forcing transfers through the MLflow server avoids this.
os.environ.setdefault("MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD", "false")
os.environ.setdefault("MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD", "false")


def setup(experiment: str) -> str:
    """Points MLflow at the tracking server from MLFLOW_TRACKING_URI and selects the experiment."""
    # The default matches scripts/mlflow_local.sh and compose.yaml; "uv run" does not read .env,
    # so a different address has to come from the environment (uv run --env-file .env ...).
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    return tracking_uri
