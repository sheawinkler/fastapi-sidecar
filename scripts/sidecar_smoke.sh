#!/usr/bin/env bash
set -euo pipefail

SIDECAR_URL="${SIDECAR_URL:-http://127.0.0.1:8288}"
HEALTH_ENDPOINT="${SIDECAR_URL%/}/health"
PREDICT_ENDPOINT="${SIDECAR_URL%/}/predict"
TOKEN="${SIDECAR_SMOKE_TOKEN:-SOL/USDC}"
EXPECTED_MODE="${SIDECAR_EXPECTED_MODE:-}"

echo "[sidecar-smoke] Checking health at $HEALTH_ENDPOINT"
health_response=$(curl -fsS "$HEALTH_ENDPOINT")
mode=$(echo "$health_response" | python -c 'import json,sys;print(json.load(sys.stdin)["mode"])')
models=$(echo "$health_response" | python -c 'import json,sys;print(json.load(sys.stdin)["models_loaded"])')

echo "[sidecar-smoke] Mode: $mode | Models loaded: $models"

if [[ -n "$EXPECTED_MODE" && "$mode" != "$EXPECTED_MODE" ]]; then
  echo "[sidecar-smoke] ERROR: expected mode '$EXPECTED_MODE' but got '$mode'" >&2
  exit 3
fi

tmp_payload=$(mktemp)
python - <<'PY' "$tmp_payload" "$TOKEN" > /dev/null
import json, sys
path, token = sys.argv[1:3]
features = [round(i * 0.1, 2) for i in range(1, 30)]
payload = {"token": token, "features": features}
with open(path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh)
PY

echo "[sidecar-smoke] Posting sample payload to $PREDICT_ENDPOINT"
predict_response=$(curl -fsS -X POST "$PREDICT_ENDPOINT" -H 'Content-Type: application/json' --data-binary "@${tmp_payload}")
runtime_ms=$(echo "$predict_response" | python -c 'import json,sys;print(round(json.load(sys.stdin)["latency_ms"], 2))')
prediction=$(echo "$predict_response" | python -c 'import json,sys;print(json.load(sys.stdin)["prediction"])')

rm -f "$tmp_payload"

echo "[sidecar-smoke] Prediction: $prediction | Latency: ${runtime_ms}ms"
echo "$predict_response" | python -m json.tool
