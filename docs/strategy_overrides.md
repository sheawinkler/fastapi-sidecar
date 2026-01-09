# Real-Time Strategy Overrides via Sidecar

The FastAPI sidecar now consumes the `sol_scaler_signals` project from memMCP and
turns those entries into actionable override directives.  Each signal carries a
momentum/risk/liquidity profile so the sidecar can derive an `override_strength`
and confidence boost for the trading runtime.

## How it works

1. **Signal logging** – `MemeAIAgent` writes normalized discoveries to the
   memMCP orchestrator.  Entries include price, momentum, liquidity, and risk
   scores.
2. **Orchestrator caching** – `services/orchestrator/app.py` exposes
   `GET /signals/latest`, returning the most recent discoveries.
3. **Sidecar ingestion** – the FastAPI server polls `SIGNAL_FEED_URL`
   (defaults to `http://127.0.0.1:8075/signals/latest?limit=200`) every
   `SIGNAL_REFRESH_SECONDS` seconds.  New rows are scored and stored in a local
   cache.
4. **Override exposure**
   - `POST /predict` responses now contain a `metadata.signal_override` block
     whenever the requested token matches a cached signal.  The block carries
     the score, derived multiplier, and recommended confidence floor.
   - `GET /strategy/overrides?limit=15` returns the strongest entries so other
     processes (Rust runtime, dashboards, etc.) can pull high-priority actions
     without performing their own memMCP lookups.

## Environment knobs

| Variable | Default | Description |
| --- | --- | --- |
| `SIGNAL_FEED_URL` | `http://127.0.0.1:8075/signals/latest?limit=200` | Source endpoint (memMCP orchestrator). |
| `SIGNAL_REFRESH_SECONDS` | `20` | Polling cadence for new entries. |
| `SIGNAL_HISTORY_LIMIT` | `200` | Maximum overrides retained locally. |
| `SIGNAL_MOMENTUM_FLOOR` | `1.0` | Baseline momentum required before scoring. |
| `SIGNAL_MOMENTUM_CEILING` | `3.0` | Momentum level treated as full strength. |
| `SIGNAL_LIQUIDITY_SWEETSPOT` | `250000` | Liquidity target (USD) for score normalization. |
| `SIGNAL_OVERRIDE_WEIGHT` | `0.45` | Multiplier applied to expected return when an override hits. |

## Next steps

- Teach the Rust `SidecarClient` to call `/strategy/overrides` so the execution
  engine can prioritize urgent entries even before a prediction request occurs.
- Thread the `metadata.signal_override` payload into `SignalExecutor` so Kelly
  sizing and stop logic can honor `priority="HIGH"` directives.
- Add Playwright/API tests for `/strategy/overrides` and a smoke test verifying
  that a memMCP entry propagates end-to-end.
