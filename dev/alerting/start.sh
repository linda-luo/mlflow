#!/usr/bin/env bash
# Bring up the database schema, the Timescale tiers, and the tracking server.
set -euo pipefail

# mlflow itself is not installed into the image -- the repo is mounted and put on
# PYTHONPATH so the container always runs the working tree. But parts of mlflow
# resolve their own version through importlib.metadata, which needs *distribution
# metadata* to exist; PYTHONPATH alone does not provide it. Without this, importing
# mlflow aborts partway and the failure surfaces as a misleading
# "cannot import name 'MlflowClient' from 'mlflow'".
#
# Registering the metadata (no dependencies, no build isolation, no reinstall of
# anything) is cheap and makes the stack independent of whether the host checkout
# happens to carry a stale mlflow.egg-info.
if ! python -c "import importlib.metadata as m; m.version('mlflow')" 2>/dev/null; then
    echo "[start] registering mlflow distribution metadata"
    pip install --no-deps --no-build-isolation --quiet -e /mlflow/home
fi

echo "[start] applying migrations and Timescale setup"
python dev/alerting/bootstrap.py

echo "[start] launching tracking server (job execution enabled)"
# On Linux this spawns the huey consumer, which registers the alerting periodic
# tasks -- the thing that cannot happen on Windows.
exec python -m mlflow server \
    --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" \
    --default-artifact-root /tmp/mlflow-artifacts \
    --host 0.0.0.0 \
    --port 5000
