#!/usr/bin/env bash
set -euo pipefail

SIDECAR_URL="${SIDECAR_URL:-http://127.0.0.1:8288}"
HEALTH_ENDPOINT="${SIDECAR_URL%/}/health"
SCHEMA_ENDPOINT="${SIDECAR_URL%/}/schema/features"
PREDICT_ENDPOINT="${SIDECAR_URL%/}/predict"
TOKEN="${SIDECAR_SMOKE_TOKEN:-SOL/USDC}"
EXPECTED_MODE="${SIDECAR_EXPECTED_MODE:-}"

if [[ -n "${SIDECAR_SMOKE_PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="$SIDECAR_SMOKE_PYTHON_BIN"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "[sidecar-smoke] ERROR: no Python interpreter found (python/python3)" >&2
  exit 2
fi

echo "[sidecar-smoke] Checking health at $HEALTH_ENDPOINT"
health_response=$(curl -fsS "$HEALTH_ENDPOINT")
mode=$(echo "$health_response" | "$PYTHON_BIN" -c 'import json,sys;print(json.load(sys.stdin)["mode"])')
models=$(echo "$health_response" | "$PYTHON_BIN" -c 'import json,sys;print(json.load(sys.stdin)["models_loaded"])')

echo "[sidecar-smoke] Mode: $mode | Models loaded: $models"

if [[ -n "$EXPECTED_MODE" && "$mode" != "$EXPECTED_MODE" ]]; then
  echo "[sidecar-smoke] ERROR: expected mode '$EXPECTED_MODE' but got '$mode'" >&2
  exit 3
fi

echo "[sidecar-smoke] Checking feature schema at $SCHEMA_ENDPOINT"
schema_response=$(curl -fsS "$SCHEMA_ENDPOINT")
server_schema_version=$(echo "$schema_response" | "$PYTHON_BIN" -c 'import json,sys;print(json.load(sys.stdin)["schema_version"])')
server_expected_dim=$(echo "$schema_response" | "$PYTHON_BIN" -c 'import json,sys;print(json.load(sys.stdin)["expected_dim"])')

if [[ "$server_schema_version" != "v1" ]]; then
  echo "[sidecar-smoke] ERROR: expected schema_version 'v1' but got '$server_schema_version'" >&2
  exit 4
fi
if [[ "$server_expected_dim" != "29" ]]; then
  echo "[sidecar-smoke] ERROR: expected expected_dim '29' but got '$server_expected_dim'" >&2
  exit 5
fi

tmp_payload=$(mktemp)
"$PYTHON_BIN" - <<'PY' "$tmp_payload" "$TOKEN" > /dev/null
import json, sys
path, token = sys.argv[1:3]
features = [round(i * 0.1, 2) for i in range(1, 30)]
payload = {"token": token, "schema_version": "v1", "features": features}
with open(path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh)
PY

echo "[sidecar-smoke] Posting sample payload to $PREDICT_ENDPOINT"
predict_response=$(curl -fsS -X POST "$PREDICT_ENDPOINT" -H 'Content-Type: application/json' --data-binary "@${tmp_payload}")
runtime_ms=$(echo "$predict_response" | "$PYTHON_BIN" -c 'import json,sys;print(round(json.load(sys.stdin)["latency_ms"], 2))')
prediction=$(echo "$predict_response" | "$PYTHON_BIN" -c 'import json,sys;print(json.load(sys.stdin)["prediction"])')
score=$(echo "$predict_response" | "$PYTHON_BIN" -c 'import json,sys;print(json.load(sys.stdin)["score"])')
inference_id=$(echo "$predict_response" | "$PYTHON_BIN" -c 'import json,sys;print(json.load(sys.stdin)["inference_id"])')
resp_schema_version=$(echo "$predict_response" | "$PYTHON_BIN" -c 'import json,sys;print(json.load(sys.stdin)["schema_version"])')

rm -f "$tmp_payload"

if [[ -z "$inference_id" ]]; then
  echo "[sidecar-smoke] ERROR: missing inference_id" >&2
  exit 6
fi
if [[ "$resp_schema_version" != "v1" ]]; then
  echo "[sidecar-smoke] ERROR: expected response schema_version 'v1' but got '$resp_schema_version'" >&2
  exit 7
fi

"$PYTHON_BIN" - <<'PY' "$score" > /dev/null
import math, sys
v = float(sys.argv[1])
if not (math.isfinite(v) and 0.0 <= v <= 1.0):
    raise SystemExit(1)
PY

echo "[sidecar-smoke] Prediction: $prediction | Score: $score | Latency: ${runtime_ms}ms | inference_id: $inference_id"
echo "$predict_response" | "$PYTHON_BIN" -m json.tool
