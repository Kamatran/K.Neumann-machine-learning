#!/usr/bin/env bash
# W1: MLflow as a local service, without Docker. Run it in a second terminal and leave it running.
# Runs are kept in state/mlflow.db, files of the runs in state/mlartifacts/.
# From W2 the same server runs in a container with PostgreSQL and MinIO instead; your code stays the same.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p state/mlartifacts
root="$(pwd)"

echo "MLflow: http://127.0.0.1:5000   (stop with Ctrl+C)"
exec uv run mlflow server \
  --host 127.0.0.1 --port 5000 --workers 1 \
  --backend-store-uri "sqlite:///$root/state/mlflow.db" \
  --artifacts-destination "$root/state/mlartifacts"
