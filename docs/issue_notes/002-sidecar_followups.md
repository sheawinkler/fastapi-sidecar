# Issue Proposal: Sidecar Telemetry & Regression Coverage

## Motivation
Now that the FastAPI bridge runs in full mode, we need higher confidence that future changes (Rust client, ensemble tweaks, LM Studio swaps) do not regress the HTTP contract.

## Proposed Scope
1. **Telemetry surfacing**
   - Expose per-model fallback status via `/health` so dashboards can alert when a model silently downgrades to a stub.
   - Emit structured logs to Langfuse/MCP when fallbacks occur.
2. **Smoke automation**
   - Extend `scripts/sidecar_smoke.sh` with JSON schema checks, confidence thresholds, and exit codes for CI failures.
   - Add a GitHub Action (once the repo exists) that runs the script against both stub and full modes on every PR.
3. **Performance sampling**
   - Record latency histogram in telemetry and optionally push to Prometheus/StatsD for long-running ops.
4. **Documentation**
   - Convert the current README notes into a dedicated `SIDECAR.md` with troubleshooting steps, fallback matrix, and curl examples.

## Acceptance Criteria
- CI job fails when `/health` does not match the expected mode or when `/predict` deviates from the schema.
- Operators can tell from `/health` which models (if any) are running as fallbacks.
- Documentation covers end-to-end bring-up (FastAPI + Rust client). 
