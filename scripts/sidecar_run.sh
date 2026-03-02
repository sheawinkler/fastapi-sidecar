#!/usr/bin/env bash
set -euo pipefail

# Ensure port 8288 is free before starting (ignore errors if nothing is listening).
lsof -ti tcp:8288 | xargs kill || true

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

: "${API_HOST:=*********}"
: "${API_PORT:=8288}"
: "${API_RELOAD:=false}"
: "${SIDECAR_INFERENCE_BACKEND:=coreml}"
: "${SIDECAR_COREML_MODEL_PATH:=models/saved/coreml/ensemble_proxy.mlpackage}"
: "${SIDECAR_CUSTOM_EXPORT_PATH:=models/saved/custom_proxy/ensemble_proxy_v1.json}"

# Backward/typo-compatible alias: some shells export ENABLE_STUB instead of ENSEMBLE_STUB.
# Only apply if ENSEMBLE_STUB is unset.
if [[ -z "${ENSEMBLE_STUB+x}" && -n "${ENABLE_STUB:-}" ]]; then
  export ENSEMBLE_STUB="$ENABLE_STUB"
fi

# Default to full ensemble mode (set ENSEMBLE_STUB=true for lightweight stub mode).
: "${ENSEMBLE_STUB:=false}"

echo "[sidecar-run] root: $ROOT_DIR"
echo "[sidecar-run] API_HOST=$API_HOST API_PORT=$API_PORT API_RELOAD=$API_RELOAD ENSEMBLE_STUB=$ENSEMBLE_STUB"
echo "[sidecar-run] SIDECAR_INFERENCE_BACKEND=$SIDECAR_INFERENCE_BACKEND SIDECAR_COREML_MODEL_PATH=$SIDECAR_COREML_MODEL_PATH SIDECAR_CUSTOM_EXPORT_PATH=$SIDECAR_CUSTOM_EXPORT_PATH"

echo "[sidecar-run] Ensuring virtualenv + deps"
if command -v uv >/dev/null 2>&1; then
  uv venv --allow-existing .venv >/dev/null
  # shellcheck disable=SC1091
  source .venv/bin/activate
  uv pip install -r requirements.txt
else
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install -r requirements.txt
fi

echo "[sidecar-run] Starting sidecar (Ctrl+C to stop)"
exec python -m src.api.fastapi_server
