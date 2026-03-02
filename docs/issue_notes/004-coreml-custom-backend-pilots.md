# 004: CoreML-Default + Custom Export Backend Pilots

Date: 2026-03-02

## Goal
Build two exploratory backend pilots for sidecar inference:

1. Core ML backend pilot (optional acceleration backend).
2. Custom JSON-export backend pilot with replay benchmark harness.

## What Changed

- Added backend runtime selector:
  - `src/api/inference_backends.py`
  - Backends: `torch`, `coreml`, `custom_export`, and `stub`.
  - Default requested backend: `custom_export`.
  - Fallback chain:
    - `custom_export` -> `coreml` -> `torch`
    - `coreml` -> `custom_export` -> `torch`
  - `/health` now includes `inference_backend` status/provenance.

- Sidecar API integration:
  - `src/api/fastapi_server.py` now delegates inference to backend runtime.
  - Predict metadata includes backend status snapshot.

- Sidecar launcher defaults:
  - `scripts/sidecar_run.sh` now defaults:
    - `SIDECAR_INFERENCE_BACKEND=custom_export`
    - `SIDECAR_COREML_MODEL_PATH=models/saved/coreml/ensemble_proxy.mlpackage`
    - `SIDECAR_CUSTOM_EXPORT_PATH=models/saved/custom_proxy/ensemble_proxy_v1.json`

- Pilot build/export scripts:
  - `scripts/build_coreml_pilot.py`
    - Distills ensemble probabilities into a compact surrogate.
    - Exports:
      - checkpoint: `models/saved/proxy/ensemble_proxy_surrogate.pt`
      - Core ML: `models/saved/coreml/ensemble_proxy.mlpackage`
      - manifest: `models/saved/coreml/ensemble_proxy_manifest.json`
  - `scripts/export_custom_proxy.py`
    - Converts surrogate checkpoint into custom JSON:
      - `models/saved/custom_proxy/ensemble_proxy_v1.json`
  - `scripts/replay_backend_benchmark.py`
    - Replays feature vectors and benchmarks `torch` vs `coreml` vs `custom_export`.
    - Emits report:
      - `reports/backend_replay_benchmark.json`

## Test Coverage

- Added backend runtime tests:
  - `tests/sidecar/test_inference_backends.py`
- Updated health test for backend visibility:
  - `tests/sidecar/test_fastapi_server.py`

## Runbook

1. Install `coremltools` in the active sidecar environment (macOS Apple Silicon):

```bash
pip install coremltools
```

2. Build Core ML pilot artifacts:

```bash
python scripts/build_coreml_pilot.py --samples 800 --epochs 35
```

3. Export custom JSON backend:

```bash
python scripts/export_custom_proxy.py
```

4. Benchmark replay parity and latency:

```bash
python scripts/replay_backend_benchmark.py --limit 300
```

5. Start sidecar (default requests custom export):

```bash
scripts/sidecar_run.sh
```

## Notes

- If custom export or Core ML artifacts are missing, runtime falls back automatically.
- Backend status and fallback reason are visible at `/health`.
- This is an exploratory surrogate path; production routing should be validated with replay + shadow checks before hard dependency on any non-torch backend.
