#!/usr/bin/env bash
set -euo pipefail

# Ensure port 8288 is free before starting (ignore errors if nothing is listening).
lsof -ti tcp:8288 | xargs kill || true

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

: "${API_HOST:=127.0.0.1}"
: "${API_PORT:=8288}"
: "${API_RELOAD:=false}"
: "${SIDECAR_INFERENCE_BACKEND:=custom_export}"
: "${SIDECAR_COREML_MODEL_PATH:=models/saved/coreml/ensemble_proxy.mlpackage}"
: "${SIDECAR_CUSTOM_EXPORT_PATH:=models/saved/custom_proxy/ensemble_proxy_v1.json}"
: "${SIDECAR_PYTHON_VERSION:=3.12}"
: "${SIDECAR_FORCE_INSTALL_DEPS:=false}"
: "${SIDECAR_BOOTSTRAP_VENV:=false}"
: "${SIDECAR_PYTHON_BIN:=python3.12}"

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

BOOTSTRAP_VENV_FLAG="$(printf '%s' "$SIDECAR_BOOTSTRAP_VENV" | tr '[:upper:]' '[:lower:]')"
if [[ "$BOOTSTRAP_VENV_FLAG" == "true" ]]; then
  echo "[sidecar-run] Ensuring virtualenv + deps"
  REQ_HASH_FILE=".venv/.requirements.sha256"
  REQ_HASH="$(shasum -a 256 requirements.txt | awk '{print $1}')"
  INSTALL_DEPS=false
  FORCE_INSTALL_FLAG="$(printf '%s' "$SIDECAR_FORCE_INSTALL_DEPS" | tr '[:upper:]' '[:lower:]')"
  if [[ "$FORCE_INSTALL_FLAG" == "true" ]]; then
    INSTALL_DEPS=true
  elif [[ ! -f "$REQ_HASH_FILE" ]]; then
    INSTALL_DEPS=true
  elif [[ "$(cat "$REQ_HASH_FILE" 2>/dev/null || true)" != "$REQ_HASH" ]]; then
    INSTALL_DEPS=true
  fi

  if command -v uv >/dev/null 2>&1; then
    uv venv --allow-existing --python "$SIDECAR_PYTHON_VERSION" .venv >/dev/null
    # shellcheck disable=SC1091
    source .venv/bin/activate
    SIDECAR_PYTHON_BIN=python
    if [[ "$INSTALL_DEPS" == "true" ]]; then
      echo "[sidecar-run] Installing requirements via uv (force=$SIDECAR_FORCE_INSTALL_DEPS)"
      uv pip install -r requirements.txt
      printf '%s\n' "$REQ_HASH" > "$REQ_HASH_FILE"
    else
      echo "[sidecar-run] Requirements unchanged; skipping dependency install"
    fi
  else
    python3 -m venv .venv
    # shellcheck disable=SC1091
    source .venv/bin/activate
    SIDECAR_PYTHON_BIN=python
    if [[ "$INSTALL_DEPS" == "true" ]]; then
      echo "[sidecar-run] Installing requirements via pip (force=$SIDECAR_FORCE_INSTALL_DEPS)"
      pip install -r requirements.txt
      printf '%s\n' "$REQ_HASH" > "$REQ_HASH_FILE"
    else
      echo "[sidecar-run] Requirements unchanged; skipping dependency install"
    fi
  fi
else
  echo "[sidecar-run] Using current Python environment (set SIDECAR_BOOTSTRAP_VENV=true to rebuild .venv)"
fi

if ! command -v "$SIDECAR_PYTHON_BIN" >/dev/null 2>&1; then
  echo "[sidecar-run] ERROR: Python interpreter not found: $SIDECAR_PYTHON_BIN" >&2
  exit 1
fi
echo "[sidecar-run] Python interpreter: $(command -v "$SIDECAR_PYTHON_BIN")"

echo "[sidecar-run] Starting sidecar (Ctrl+C to stop)"
exec "$SIDECAR_PYTHON_BIN" -m src.api.fastapi_server
