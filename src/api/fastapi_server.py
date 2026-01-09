"""FastAPI bridge exposing the trained ensemble to external agents.

This service wraps the existing ensemble orchestrator so tools like the Rust
trader can obtain predictions over HTTP.  It also hosts lightweight telemetry
endpoints so we can inspect model call volumes and latencies.

It also supports a low-latency WebSocket guidance stream (`/ws/guidance`) so
execution engines can subscribe to real-time model hints without polling.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import deque
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Union

import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.cors import CORSMiddleware

torch = None  # type: ignore[assignment]


def _require_torch():
    """Lazy-import torch only when full-mode tensorization is needed."""

    global torch
    if torch is None:
        try:  # pragma: no cover - torch optional in stub mode
            import torch as _torch  # type: ignore
        except Exception as exc:  # pragma: no cover - fallback when torch missing
            raise RuntimeError("PyTorch is not available in this environment") from exc
        torch = _torch
    return torch

EnsembleOrchestrator = None  # type: ignore
STUB_REASON = ""


class StubEnsemble:
    """Fallback ensemble when heavy dependencies are unavailable."""

    def __init__(self, reason: str = "missing dependencies") -> None:
        self.reason = reason
        self._calls = 0
        self.models: Dict[str, Any] = {"stub": "synthetic"}

    def predict_ensemble(self, features: Dict[str, Any], model: Optional[str] = None) -> Dict[str, Any]:
        self._calls += 1
        score = float(sum(hash(k) % 100 for k in features.keys()) % 100) / 100.0
        return {
            "prediction": score,
            "confidence": 0.35 + (score % 0.5),
            "expected_return": (score - 0.5) / 4,
            "model": model or "stub",
            "reason": self.reason,
            "raw_features": features,
        }

    def load_all(self) -> None:  # parity with real orchestrator
        return None


def load_ensemble():
    stub_requested = os.getenv("ENSEMBLE_STUB", "false").lower() == "true"
    if stub_requested:
        logging.getLogger("ensemble.api").warning("Using StubEnsemble: stub mode enabled")
        return StubEnsemble(reason="stub mode enabled")

    global EnsembleOrchestrator, STUB_REASON
    if EnsembleOrchestrator is None and not STUB_REASON:
        try:
            from src.ai.ensemble_orchestrator import EnsembleOrchestrator as _EnsembleOrchestrator

            EnsembleOrchestrator = _EnsembleOrchestrator
        except Exception as exc:  # pragma: no cover - guard for missing module/deps
            STUB_REASON = str(exc)

    if EnsembleOrchestrator is None:
        reason = STUB_REASON or "orchestrator unavailable"
        logging.getLogger("ensemble.api").warning("Using StubEnsemble: %s", reason)
        return StubEnsemble(reason=reason)

    try:
        init_kwargs = {}
        if os.getenv("ENSEMBLE_INPUT_DIM"):
            init_kwargs["input_dim"] = int(os.getenv("ENSEMBLE_INPUT_DIM"))
        orchestrator = EnsembleOrchestrator(**init_kwargs)
        load_all = getattr(orchestrator, "load_all", None)
        if callable(load_all):
            load_all()
        return orchestrator
    except Exception as exc:  # pragma: no cover - heavy deps missing
        logging.getLogger("ensemble.api").warning(
            "Falling back to StubEnsemble: %s", exc, exc_info=True
        )
        return StubEnsemble(reason=str(exc))


FeaturePayload = Union[Dict[str, Any], List[float]]


class PredictRequest(BaseModel):
    """Request model for /predict.

    We intentionally accept multiple shapes:

    - Preferred schema:
      {"token": "BONK", "features": [..], "model": "ensemble"}

    - Legacy/Rust-friendly schema (flat dict):
      {"symbol": "BONK", "price": 0.01, "confidence": 0.7, ...}

    FastAPI/Pydantic will store unknown keys in `model_extra`.
    """

    model_config = ConfigDict(extra="allow")

    token: Optional[str] = Field(None, description="Token symbol")
    features: Optional[FeaturePayload] = Field(None, description="Feature payload")
    model: Optional[str] = Field(None, description="Optional model override")


class PredictResponse(BaseModel):
    prediction: float
    confidence: float
    expected_return: float
    latency_ms: float
    model: str
    metadata: Dict[str, Any]


class TelemetryState(BaseModel):
    total_calls: int = 0
    avg_latency_ms: float = 0.0


class GuidancePayload(BaseModel):
    """Payload broadcast over /ws/guidance.

    This matches the Rust `SidecarGuidance::from_value` contract:
    - required: symbol
    - optional numeric fields: score/confidence/size_multiplier/risk_multiplier/kelly_fraction/expected_return
    """

    symbol: str
    confidence: float = 0.6
    score: float = 0.55
    expected_return: float = 0.0
    size_multiplier: float = 1.0
    risk_multiplier: float = 1.0
    kelly_fraction: float = 0.02
    notes: Optional[str] = None
    memory_ref: Optional[str] = None


# In-memory pub/sub so websocket clients receive low-latency guidance pushes.
_guidance_subscribers: set[asyncio.Queue] = set()
_guidance_lock = asyncio.Lock()


async def _register_guidance_queue(queue: asyncio.Queue) -> None:
    async with _guidance_lock:
        _guidance_subscribers.add(queue)


async def _unregister_guidance_queue(queue: asyncio.Queue) -> None:
    async with _guidance_lock:
        _guidance_subscribers.discard(queue)


async def _broadcast_guidance(event: GuidancePayload) -> None:
    message = {"type": "guidance", "data": event.model_dump()}
    async with _guidance_lock:
        queues = list(_guidance_subscribers)
    for queue in queues:
        try:
            queue.put_nowait(message)
        except asyncio.QueueFull:
            continue


app = FastAPI(title="Trading Ensemble API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("ensemble.api")
logger.setLevel(logging.INFO)

_ensemble = load_ensemble()
_telemetry = TelemetryState()

SIGNAL_FEED_URL = os.getenv(
    "SIGNAL_FEED_URL", "http://127.0.0.1:8075/signals/latest?limit=200"
)
SIGNAL_REFRESH_SECONDS = int(os.getenv("SIGNAL_REFRESH_SECONDS", "20"))
SIGNAL_HISTORY_LIMIT = int(os.getenv("SIGNAL_HISTORY_LIMIT", "200"))
SIGNAL_MOMENTUM_FLOOR = float(os.getenv("SIGNAL_MOMENTUM_FLOOR", "1.0"))
SIGNAL_MOMENTUM_CEILING = float(os.getenv("SIGNAL_MOMENTUM_CEILING", "3.0"))
SIGNAL_LIQUIDITY_SWEETSPOT = float(os.getenv("SIGNAL_LIQUIDITY_SWEETSPOT", "250000"))
SIGNAL_OVERRIDE_WEIGHT = float(os.getenv("SIGNAL_OVERRIDE_WEIGHT", "0.45"))


class OverrideEntry(BaseModel):
    symbol: str
    address: str
    price_usd: float
    volume_24h_usd: float
    liquidity_usd: float
    momentum_score: float
    risk_score: float
    verified: bool = False
    created_at: Optional[datetime] = None
    file: Optional[str] = None
    override_strength: float
    priority: str
    reason: str
    suggested_confidence: float
    multiplier: float


class SignalCache:
    def __init__(self) -> None:
        self._entries: Deque[OverrideEntry] = deque(maxlen=SIGNAL_HISTORY_LIMIT)
        self._by_symbol: Dict[str, OverrideEntry] = {}
        self._by_address: Dict[str, OverrideEntry] = {}
        self._lock = asyncio.Lock()
        self._last_error: Optional[str] = None
        self._known_files: set[str] = set()

    async def run(self) -> None:
        while True:
            await self.refresh()
            await asyncio.sleep(SIGNAL_REFRESH_SECONDS)

    async def refresh(self) -> None:
        try:
            entries = await self._fetch_feed()
        except Exception as exc:  # pragma: no cover - network
            self._last_error = str(exc)
            logger.warning("Signal feed refresh failed: %s", exc)
            return

        async with self._lock:
            for entry in entries:
                key = entry.file or f"{entry.symbol}:{entry.address}:{entry.created_at}"
                if key in self._known_files:
                    continue
                self._known_files.add(key)
                self._entries.append(entry)
                self._by_symbol[entry.symbol.upper()] = entry
                self._by_address[entry.address.lower()] = entry
            while len(self._entries) > SIGNAL_HISTORY_LIMIT:
                removed = self._entries.popleft()
                self._by_symbol.pop(removed.symbol.upper(), None)
                self._by_address.pop(removed.address.lower(), None)
                key = removed.file or f"{removed.symbol}:{removed.address}:{removed.created_at}"
                self._known_files.discard(key)

    async def _fetch_feed(self) -> List[OverrideEntry]:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(SIGNAL_FEED_URL)
        resp.raise_for_status()
        body = resp.json()
        rows = body.get("signals", []) if isinstance(body, dict) else []
        entries: List[OverrideEntry] = []
        for raw in rows:
            normalized = self._normalize_entry(raw)
            if normalized:
                entries.append(normalized)
        return entries

    def _normalize_entry(self, raw: Dict[str, Any]) -> Optional[OverrideEntry]:
        try:
            symbol = str(raw.get("symbol", "")).upper()
            address = str(raw.get("address", ""))
            if not symbol or not address:
                return None
            created_at = raw.get("created_at")
            created_dt = None
            if isinstance(created_at, str):
                try:
                    created_dt = datetime.fromisoformat(created_at)
                except ValueError:
                    created_dt = None
            momentum = float(raw.get("momentum_score") or 0.0)
            risk = float(raw.get("risk_score") or 0.0)
            liquidity = float(raw.get("liquidity_usd") or 0.0)
            volume = float(raw.get("volume_24h_usd") or 0.0)
            price = float(raw.get("price_usd") or 0.0)
            verified = bool(raw.get("verified", False))
        except (TypeError, ValueError):
            return None

        strength = _score_override(momentum, risk, liquidity, verified)
        priority, reason = _classify_override(strength, momentum, risk, liquidity, volume)
        suggested_confidence = min(0.99, 0.55 + strength * 0.4)
        multiplier = 1.0 + (strength * SIGNAL_OVERRIDE_WEIGHT)

        return OverrideEntry(
            symbol=symbol,
            address=address,
            price_usd=price,
            volume_24h_usd=volume,
            liquidity_usd=liquidity,
            momentum_score=momentum,
            risk_score=risk,
            verified=verified,
            created_at=created_dt,
            file=raw.get("file"),
            override_strength=strength,
            priority=priority,
            reason=reason,
            suggested_confidence=suggested_confidence,
            multiplier=multiplier,
        )

    async def override_for(self, token: str) -> Optional[OverrideEntry]:
        key = token.upper()
        async with self._lock:
            if key in self._by_symbol:
                return self._by_symbol[key]
            if token.startswith("0x"):
                return self._by_address.get(token.lower())
            return None

    async def top(self, limit: int) -> List[OverrideEntry]:
        limit = max(1, min(limit, SIGNAL_HISTORY_LIMIT))
        async with self._lock:
            entries = list(self._entries)
        entries.sort(key=lambda e: e.override_strength, reverse=True)
        return entries[:limit]


signal_cache = SignalCache()


def _score_override(momentum: float, risk: float, liquidity: float, verified: bool) -> float:
    momentum_span = max(0.1, SIGNAL_MOMENTUM_CEILING - SIGNAL_MOMENTUM_FLOOR)
    momentum_norm = max(
        0.0, min(1.0, (momentum - SIGNAL_MOMENTUM_FLOOR) / momentum_span)
    )
    safety = max(0.0, min(1.0, 1.0 - risk))
    liquidity_norm = max(0.0, min(1.0, liquidity / SIGNAL_LIQUIDITY_SWEETSPOT))
    verified_bonus = 0.05 if verified else 0.0
    raw_score = (
        (0.6 * momentum_norm)
        + (0.25 * safety)
        + (0.15 * liquidity_norm)
        + verified_bonus
    )
    return max(0.0, min(1.0, raw_score))


def _classify_override(
    strength: float, momentum: float, risk: float, liquidity: float, volume: float
) -> tuple[str, str]:
    if strength >= 0.8:
        priority = "HIGH"
    elif strength >= 0.55:
        priority = "MEDIUM"
    else:
        priority = "LOW"
    reason = (
        f"momentum={momentum:.2f} risk={risk:.2f} liq=${liquidity:,.0f} vol=${volume:,.0f}"
    )
    return priority, reason


def _coerce_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as exc:  # pragma: no cover - validation path
            raise ValueError(f"Feature '{label}' must be numeric") from exc
    raise ValueError(f"Unsupported feature type for '{label}': {type(value).__name__}")


def _filter_numeric_features(payload: Dict[str, Any]) -> Dict[str, float]:
    """Best-effort filter that drops non-numeric fields.

    This is useful for legacy callers that include metadata keys like `symbol` or `side`.
    """

    numeric: Dict[str, float] = {}
    for key, value in payload.items():
        try:
            numeric[str(key)] = _coerce_float(value, str(key))
        except ValueError:
            continue
    return numeric


def _derive_predict_inputs(req: PredictRequest) -> tuple[str, FeaturePayload, Optional[str]]:
    """Normalize /predict inputs across supported request shapes."""

    extras = dict(getattr(req, "model_extra", None) or {})

    token = req.token
    features = req.features

    if features is None:
        # Treat a flat payload (common for Rust SidecarClient) as a feature dict.
        excluded = {"token", "model", "features"}
        features = {k: v for k, v in extras.items() if k not in excluded}

    inferred = None
    if isinstance(features, dict):
        inferred = features.get("token") or features.get("symbol")

    token = token or extras.get("token") or extras.get("symbol") or inferred
    token_str = str(token).strip() if token is not None else ""
    if not token_str:
        token_str = "UNKNOWN"

    if features is None:
        raise ValueError("Missing features")
    if isinstance(features, dict) and not features:
        raise ValueError("Feature object is empty")
    if isinstance(features, list) and not features:
        raise ValueError("Feature array is empty")

    return token_str, features, req.model


def _flatten_features(payload: FeaturePayload) -> List[float]:
    if isinstance(payload, dict):
        return [_coerce_float(payload[key], str(key)) for key in sorted(payload.keys())]
    if isinstance(payload, list):
        return [_coerce_float(val, f"idx_{idx}") for idx, val in enumerate(payload)]
    raise ValueError("Features must be provided as an object or array of numbers")


def _resize_vector(vector: List[float], expected_dim: Optional[int]) -> List[float]:
    if not expected_dim or expected_dim <= 0:
        return vector
    if len(vector) == expected_dim:
        return vector
    if len(vector) < expected_dim:
        return vector + [0.0] * (expected_dim - len(vector))
    return vector[:expected_dim]


def _tensorize_features(payload: FeaturePayload, expected_dim: Optional[int]):
    torch_mod = _require_torch()
    vector = _resize_vector(_flatten_features(payload), expected_dim)
    if not vector:
        raise ValueError("Feature vector is empty")
    tensor = torch_mod.tensor(vector, dtype=torch_mod.float32)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def _mapping_for_stub(payload: FeaturePayload) -> Dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    values = _flatten_features(payload)
    return {f"f{i}": val for i, val in enumerate(values)}


def _model_inventory_size() -> int:
    models = getattr(_ensemble, "models", None)
    if isinstance(models, dict):
        return len(models)
    try:
        return len(_ensemble)  # type: ignore[arg-type]
    except Exception:  # pragma: no cover - handle objects without __len__
        return 0


def _model_status() -> List[Dict[str, Any]]:
    models = getattr(_ensemble, "models", None)
    if not isinstance(models, dict):
        return []

    statuses: List[Dict[str, Any]] = []
    for name, model in models.items():
        cls = model.__class__.__name__
        name_attr = getattr(model, "name", "")
        is_fallback = bool(
            cls == "LightweightFallbackModel"
            or (isinstance(name_attr, str) and name_attr.endswith("_fallback"))
        )
        statuses.append(
            {
                "name": str(name),
                "class": cls,
                "fallback": is_fallback,
            }
        )
    return statuses


def record_latency(latency_ms: float) -> None:
    total = _telemetry.total_calls + 1
    # simple running average
    _telemetry.avg_latency_ms = (
        (_telemetry.avg_latency_ms * _telemetry.total_calls) + latency_ms
    ) / total
    _telemetry.total_calls = total


@app.get("/health")
async def health() -> Dict[str, Any]:
    mode = "stub" if isinstance(_ensemble, StubEnsemble) else "full"
    payload: Dict[str, Any] = {
        "status": "ok",
        "mode": mode,
        "models_loaded": _model_inventory_size(),
        "telemetry": _telemetry.model_dump(),
        "guidance_subscribers": len(_guidance_subscribers),
    }

    if mode == "stub":
        payload["stub_reason"] = getattr(_ensemble, "reason", "")
    else:
        payload["models"] = _model_status()

    return payload


@app.get("/ping")
async def ping() -> Dict[str, Any]:
    """Lightweight readiness probe compatible with legacy clients."""

    return await health()


@app.get("/strategy/overrides")
async def strategy_overrides(limit: int = 15) -> Dict[str, Any]:
    overrides = await signal_cache.top(limit)
    return {
        "overrides": [entry.model_dump() for entry in overrides],
        "source": SIGNAL_FEED_URL,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
    start = time.perf_counter()

    token, features, model = _derive_predict_inputs(req)

    try:
        if isinstance(_ensemble, StubEnsemble):
            features_payload = _mapping_for_stub(features)
            result = _ensemble.predict_ensemble(features_payload, model=model)
        else:
            expected_dim = getattr(_ensemble, "input_dim", None)
            try:
                tensor_input = _tensorize_features(features, expected_dim)
            except ValueError:
                # If callers include non-numeric metadata keys, drop them and retry.
                if isinstance(features, dict):
                    tensor_input = _tensorize_features(
                        _filter_numeric_features(features), expected_dim
                    )
                else:
                    raise
            result = _ensemble.predict_ensemble(tensor_input)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:  # pragma: no cover - torch unavailable
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - runtime errors
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    latency_ms = (time.perf_counter() - start) * 1000.0
    record_latency(latency_ms)

    prediction_value = float(result.get("prediction", 0.0))
    base_confidence = float(result.get("confidence", 0.0))
    base_expected_return = float(result.get("expected_return", 0.0))

    confidence_value = base_confidence
    expected_return = base_expected_return

    override = None
    if token != "UNKNOWN":
        override = await signal_cache.override_for(token)

    metadata = dict(result)
    if override:
        override_payload = override.model_dump()
        confidence_value = max(confidence_value, override.suggested_confidence)
        expected_return *= override.multiplier
        override_payload["applied_multiplier"] = override.multiplier
        metadata["signal_override"] = override_payload

    # Push a low-latency guidance event (pre-override) for subscribers.
    if token != "UNKNOWN":
        guidance = GuidancePayload(
            symbol=token,
            confidence=max(0.0, min(0.999, base_confidence)),
            score=max(0.0, min(1.0, prediction_value)),
            expected_return=base_expected_return,
            notes="predict",
        )
        asyncio.create_task(_broadcast_guidance(guidance))

    payload = PredictResponse(
        prediction=prediction_value,
        confidence=confidence_value,
        expected_return=expected_return,
        latency_ms=latency_ms,
        model=result.get("model", model or "ensemble"),
        metadata=metadata,
    )
    return payload


@app.websocket("/ws/guidance")
async def guidance_stream(websocket: WebSocket):
    """Stream guidance events to subscribers.

    Payloads are JSON objects like:
      {"type": "guidance", "data": {...}}
    """

    await websocket.accept()
    queue: asyncio.Queue = asyncio.Queue(maxsize=250)
    await _register_guidance_queue(queue)
    try:
        await websocket.send_json({"type": "ready"})
        while True:
            message = await queue.get()
            await websocket.send_json(message)
    except WebSocketDisconnect:
        return
    finally:
        await _unregister_guidance_queue(queue)


@app.post("/guidance/publish")
async def publish_guidance(event: GuidancePayload) -> Dict[str, str]:
    """Publish a guidance event to all connected websocket subscribers."""

    await _broadcast_guidance(event)
    return {"status": "queued"}


@app.get("/telemetry", response_model=TelemetryState)
async def telemetry() -> TelemetryState:
    return _telemetry


@app.on_event("startup")
async def bootstrap_signal_cache() -> None:
    asyncio.create_task(signal_cache.run())


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "src.api.fastapi_server:app",
        host=os.getenv("API_HOST", "127.0.0.1"),
        port=int(os.getenv("API_PORT", "8288")),
        reload=os.getenv("API_RELOAD", "false").lower() == "true",
    )
