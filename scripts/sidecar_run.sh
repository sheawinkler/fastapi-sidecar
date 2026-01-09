#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

: "${API_HOST:=*********}"
: "${API_PORT:=8288}"
: "${API_RELOAD:=false}"

# Backward/typo-compatible alias: some shells export ENABLE_STUB instead of ENSEMBLE_STUB.
# Only apply if ENSEMBLE_STUB is unset.
if [[ -z "${ENSEMBLE_STUB+x}" && -n "${ENABLE_STUB:-}" ]]; then
  export ENSEMBLE_STUB="$ENABLE_STUB"
fi

# Default to full ensemble mode (set ENSEMBLE_STUB=true for lightweight stub mode).
: "${ENSEMBLE_STUB:=false}"

echo "[sidecar-run] root: $ROOT_DIR"
echo "[sidecar-run] API_HOST=$API_HOST API_PORT=$API_PORT API_RELOAD=$API_RELOAD ENSEMBLE_STUB=$ENSEMBLE_STUB"

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
