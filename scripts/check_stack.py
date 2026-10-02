"""Checks MLflow: logs a run with an artifact and reads the artifact back.

Works with the local server from scripts/mlflow_local.sh (W1) and with the container stack (from W2).
"""

import tempfile
from pathlib import Path

import mlflow

from mlproject.tracking import setup

tracking_uri = setup("stack-check")

with mlflow.start_run(run_name="stack-check") as run:
    mlflow.log_param("check", "ok")
    mlflow.log_metric("value", 1.0)
    mlflow.log_text("If you can read this, artifacts reach the store.", "check.txt")

with tempfile.TemporaryDirectory() as tmp:
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run.info.run_id, artifact_path="check.txt", dst_path=tmp
    )
    content = Path(local_path).read_text()

print(f"tracking URI : {tracking_uri}")
print(f"run id       : {run.info.run_id}")
print(f"artifact URI : {run.info.artifact_uri}")
print(f"artifact     : {content}")
