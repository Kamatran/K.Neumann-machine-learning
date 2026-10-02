#!/usr/bin/env bash
# Stops MLflow and archives state/ (the runs database and the files of the runs)
# so it can be moved to another machine.
#
# Usage: scripts/save_state.sh [destination_dir] [--wipe]
#   destination_dir  where to copy the archive, e.g. a USB drive (default: only $HOME)
#   --wipe           after a verified copy, delete state/ from this machine (shared accounts)
set -euo pipefail
cd "$(dirname "$0")/.."

# W1 runs MLflow as a local service, from W2 in containers: stop whatever is running.
stop_services() {
  if command -v docker >/dev/null 2>&1 && [ -n "$(docker compose ps -q 2>/dev/null)" ]; then
    docker compose "$@"
  elif curl -sf http://127.0.0.1:5000/health >/dev/null 2>&1; then
    echo "The local MLflow server is still running. Stop it with Ctrl+C in its terminal, then run this script again." >&2
    exit 1
  fi
}

dest=""
wipe=false
for arg in "$@"; do
  case "$arg" in
    --wipe) wipe=true ;;
    *) dest="$arg" ;;
  esac
done

if [ -n "$(find state -not -user "$(id -u)" -print -quit)" ]; then
  echo "Some files in state/ belong to another user; check LOCAL_UID in .env." >&2
  exit 1
fi

name="ml-state_$(date +%Y-%m-%d_%H%M).tar.gz"
archive="$HOME/$name"   # always written to the Linux disk first, never directly to the USB drive

stop_services stop
tar -czf "$archive" state
(cd "$HOME" && sha256sum "$name" > "$name.sha256")
echo "Saved: $archive ($(du -h "$archive" | cut -f1))"

if [ -n "$dest" ]; then
  cp "$archive" "$archive.sha256" "$dest/"
  (cd "$dest" && sha256sum -c "$name.sha256")
  echo "Copied and verified: $dest/$name"
fi

if $wipe; then
  if [ -z "$dest" ]; then
    echo "--wipe needs a destination_dir, otherwise the only copy would stay on this machine." >&2
    exit 1
  fi
  stop_services down
  find state -mindepth 1 -maxdepth 1 ! -name .gitkeep ! -name minio ! -name mlflow -exec rm -rf {} +
  find state/minio state/mlflow -mindepth 1 ! -name .gitkeep -exec rm -rf {} + 2>/dev/null || true
  echo "Removed state/ from this machine."
fi
