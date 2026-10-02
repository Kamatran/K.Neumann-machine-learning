#!/usr/bin/env bash
# Restores state/ from an archive made by save_state.sh and starts the local stack.
#
# Usage: scripts/restore_state.sh path/to/ml-state_YYYY-MM-DD_HHMM.tar.gz [--force]
#   --force  overwrite existing state/ on this machine
set -euo pipefail
cd "$(dirname "$0")/.."

archive=""
force=false
for arg in "$@"; do
  case "$arg" in
    --force) force=true ;;
    *) archive="$(realpath "$arg")" ;;
  esac
done
[ -n "$archive" ] || { echo "Usage: $0 archive.tar.gz [--force]" >&2; exit 1; }

if [ -f "$archive.sha256" ]; then
  (cd "$(dirname "$archive")" && sha256sum -c "$(basename "$archive").sha256")
else
  echo "Warning: no checksum file next to the archive; integrity not verified." >&2
fi

if [ -n "$(find state -mindepth 1 ! -name .gitkeep -print -quit)" ] && ! $force; then
  echo "state/ is not empty. Use --force to overwrite it." >&2
  exit 1
fi

# W1 keeps the runs in state/mlflow.db with files in state/mlartifacts/; from W2 in containers.
if command -v docker >/dev/null 2>&1 && [ -n "$(docker compose ps -q 2>/dev/null)" ]; then
  docker compose down
elif curl -sf http://127.0.0.1:5000/health >/dev/null 2>&1; then
  echo "The local MLflow server is still running. Stop it with Ctrl+C in its terminal, then run this script again." >&2
  exit 1
fi
find state -mindepth 1 -maxdepth 1 ! -name .gitkeep ! -name minio ! -name mlflow -exec rm -rf {} +
find state/minio state/mlflow -mindepth 1 ! -name .gitkeep -exec rm -rf {} + 2>/dev/null || true
tar -xzf "$archive"

if [ -n "$(find state/minio -mindepth 1 ! -name .gitkeep -print -quit 2>/dev/null)" ]; then
  docker compose up -d          # the archive comes from the container stack (W2 and later)
  echo "Restored from $archive; the stack is starting."
else
  echo "Restored from $archive. Start MLflow with: scripts/mlflow_local.sh"
fi
