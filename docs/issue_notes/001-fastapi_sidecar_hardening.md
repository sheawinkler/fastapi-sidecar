# Issue Proposal: FastAPI Sidecar Hardening

## Context
- FastAPI bridge previously defaulted to stub mode because several ensemble factories raised exceptions during import/initialization.
- Torch models expected tensor inputs, but `/predict` forwarded raw dicts, triggering runtime errors once real mode was enabled.
- Lack of automated smoke testing made regressions difficult to catch before running the full Solana smoke suite.

## Work Completed
- Added lazy imports + feature flattening to `project/src/api/fastapi_server.py`, enabling real-mode tensor inference and stub normalization.
- Provided default `train_model` in `BaseModel` and per-model fallbacks in `EnsembleOrchestrator` so the sidecar no longer collapses when a single model fails to load.
- Repaired RL, VAE, and quantile model factory signatures so they satisfy `forward`/`train_model` requirements.
- Introduced `scripts/sidecar_smoke.sh` and documented it in the README.
- Verified `/health` now reports `mode:"full"` and `/predict` returns ensemble metadata.

## Remaining Tasks
- Remove fallback logs by fixing DynamicPortfolioOptimizer initialization (batch norm weights) and any other models still triggering fallbacks.
- Add unit tests around feature tensorization + stub mapping in `fastapi_server`.
- Wire the smoke script into CI once the repo is established.
