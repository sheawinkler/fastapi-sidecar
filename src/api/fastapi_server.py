"""FastAPI bridge exposing the trained ensemble to external agents.

This service wraps the existing ensemble orchestrator so tools like the Rust
trader can obtain predictions over HTTP.  It also hosts lightweight telemetry
endpoints so we can inspect model call volumes and latencies.

It also supports a low-latency WebSocket guidance stream (`/ws/guidance`) so
execution engines can subscribe to real-time model hints without polling.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Union

import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.cors import CORSMiddleware

from ..training.calibrator import CalibratorBundle, load_latest_calibrator

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

# Canonical feature schema (v1)
FEATURE_SCHEMA_VERSION = "v1"
FEATURE_DIM = 29
FEATURE_NAMES = [f"x{i}" for i in range(FEATURE_DIM)]


def _data_dir() -> Path:
    """Base directory for sidecar local persistence (feedback, calibrators, etc.)."""

    root = os.getenv("SIDECAR_DATA_DIR", "data")
    return Path(root)


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    """Append a single JSON object as one line (JSONL)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    with path.open("a", encoding="utf-8") as fh:
        fh.write(payload)
        fh.write("\n")


async def _append_jsonl_async(path: Path, record: Dict[str, Any]) -> None:
    await asyncio.to_thread(_append_jsonl, path, record)


class StubEnsemble:
    """Fallback ensemble when heavy dependencies are unavailable."""

    def __init__(self, reason: str = "missing dependencies") -> None:
        self.reason = reason
        self._calls = 0
        self.models: Dict[str, Any] = {"stub": "synthetic"}

    def predict_ensemble(self, features: Dict[str, Any], model: Optional[str] = None) -> Dict[str, Any]:
        self._calls += 1
        try:
            ordered = sorted(features.items(), key=lambda kv: str(kv[0]))
            payload = json.dumps(
                ordered, ensure_ascii=False, separators=(",", ":"), default=str
            )
            digest = hashlib.sha256(payload.encode("utf-8")).digest()
            score = int.from_bytes(digest[:4], "big") / float(2**32)
        except Exception:
            score = 0.5

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
      {"token": "BONK", "features": [..], "schema_version": "v1", "model": "ensemble"}

    - Legacy/Rust-friendly schema (flat dict):
      {"symbol": "BONK", "price": 0.01, "confidence": 0.7, ...}

    FastAPI/Pydantic will store unknown keys in `model_extra`.
    """

    model_config = ConfigDict(extra="allow")

    token: Optional[str] = Field(None, description="Token symbol")
    features: Optional[FeaturePayload] = Field(None, description="Feature payload")
    schema_version: Optional[str] = Field(
        None, description="Feature schema version (default: v1)"
    )
    model: Optional[str] = Field(None, description="Optional model override")


class PredictResponse(BaseModel):
    inference_id: str
    schema_version: str

    # Backward-compatible field: historically returned a scalar; in full mode it may be a
    # class index (0-4). Use `class_prediction` + `score` for explicit semantics.
    prediction: float
    class_prediction: Optional[int] = None

    # Guidance-friendly score in [0,1].
    score: float

    confidence: float
    expected_return: float
    latency_ms: float
    model: str
    metadata: Dict[str, Any]


class FeedbackRequest(BaseModel):
    """Outcome feedback for lightweight calibration/training."""

    token: str
    schema_version: str = Field(default=FEATURE_SCHEMA_VERSION)
    features: List[float]

    # Optional linkage to a prior /predict response.
    inference_id: Optional[str] = None
    prediction_timestamp: Optional[str] = None

    # How long after prediction this label applies.
    horizon_seconds: int = Field(default=900, ge=1, le=60 * 60 * 24 * 14)

    # Provide either a realized return (preferred) or a discrete class label.
    realized_return: Optional[float] = None
    realized_class: Optional[int] = Field(default=None, ge=0, le=4)

    # Optional structured extras.
    model_output: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class FeedbackResponse(BaseModel):
    status: str
    inference_id: Optional[str] = None
    feature_hash: str
    stored_at: str


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

# Optional lightweight calibrator trained from /feedback logs.
_CALIBRATOR_ROOT = Path(os.getenv("CALIBRATOR_ROOT", "models/saved/calibrator"))
_calibrator: Optional[CalibratorBundle]
_calibrator, _calibrator_reason = load_latest_calibrator(_CALIBRATOR_ROOT)

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


def _derive_predict_inputs(
    req: PredictRequest,
) -> tuple[str, FeaturePayload, Optional[str], str]:
    """Normalize /predict inputs across supported request shapes."""

    extras = dict(getattr(req, "model_extra", None) or {})

    token = req.token
    features = req.features
    schema_version = req.schema_version or extras.get("schema_version")

    if features is None:
        # Treat a flat payload (common for Rust SidecarClient) as a feature dict.
        excluded = {"token", "model", "features", "schema_version"}
        features = {k: v for k, v in extras.items() if k not in excluded}

    inferred = None
    if isinstance(features, dict):
        inferred = features.get("token") or features.get("symbol")
        schema_version = schema_version or features.get("schema_version")

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

    schema_str = str(schema_version).strip() if schema_version is not None else ""
    if not schema_str:
        schema_str = FEATURE_SCHEMA_VERSION

    return token_str, features, req.model, schema_str


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


def _hash_vector(vector: List[float]) -> str:
    """Stable hash of a feature vector for tracing/debugging."""

    payload = json.dumps(vector, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _prepare_feature_vector(
    payload: FeaturePayload, expected_dim: Optional[int]
) -> tuple[List[float], Dict[str, Any]]:
    """Normalize payload into a float vector and collect metadata/warnings."""

    warnings: List[str] = []
    meta: Dict[str, Any] = {}

    if isinstance(payload, dict):
        numeric = _filter_numeric_features(payload)
        dropped = len(payload) - len(numeric)
        if dropped:
            warnings.append(f"dropped_non_numeric_fields:{dropped}")
        if not numeric:
            raise ValueError("Feature object contains no numeric values")

        # NOTE: legacy path — dicts are flattened by sorted keys, not a semantic schema.
        warnings.append("legacy_object_features_sorted_by_key")
        keys_sorted = sorted(numeric.keys())
        keys_hash = hashlib.sha256(
            json.dumps(keys_sorted, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

        raw = _flatten_features(numeric)
        provided_dim = len(raw)
        vector = _resize_vector(raw, expected_dim)

        meta.update(
            {
                "input_type": "object",
                "provided_dim": provided_dim,
                "key_count": len(keys_sorted),
                "keys_hash": keys_hash,
            }
        )

    elif isinstance(payload, list):
        raw = _flatten_features(payload)
        provided_dim = len(raw)
        vector = _resize_vector(raw, expected_dim)
        meta.update({"input_type": "array", "provided_dim": provided_dim})

    else:
        raise ValueError("Features must be provided as an object or array of numbers")

    expected = int(expected_dim) if expected_dim else len(vector)
    if len(vector) != expected:
        # Should not happen given _resize_vector, but keep the warning just in case.
        warnings.append(f"feature_dim_mismatch:vector={len(vector)} expected={expected}")

    if expected_dim and meta.get("provided_dim") != expected_dim:
        warnings.append(
            f"feature_dim_mismatch:provided={meta.get('provided_dim')} expected={expected_dim} (padded/truncated)"
        )

    meta["expected_dim"] = expected
    meta["vector_hash"] = _hash_vector(vector)
    if warnings:
        meta["warnings"] = warnings

    return vector, meta


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
        "feature_schema": {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "expected_dim": FEATURE_DIM,
        },
        "calibrator": {
            "active": _calibrator is not None,
            "reason": _calibrator_reason,
            "trained_at": getattr(_calibrator, "trained_at", None),
            "schema_version": getattr(_calibrator, "schema_version", None),
            "expected_dim": getattr(_calibrator, "expected_dim", None),
        },
    }

    if mode == "stub":
        payload["stub_reason"] = getattr(_ensemble, "reason", "")
    else:
        payload["models"] = _model_status()

    return payload


@app.get("/schema/features")
async def feature_schema() -> Dict[str, Any]:
    """Return the canonical feature schema used by the sidecar.

    Clients should send a 29-float vector in this exact order.
    """

    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "expected_dim": FEATURE_DIM,
        "feature_names": FEATURE_NAMES,
        "notes": "Send features as a fixed-order float array. Legacy dict payloads are supported but flattened by sorted keys and will include warnings.",
    }


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
    inference_id = uuid.uuid4().hex

    token, features, model, schema_version = _derive_predict_inputs(req)

    # Determine expected feature dimension.
    expected_dim = FEATURE_DIM
    if not isinstance(_ensemble, StubEnsemble):
        expected_dim = int(getattr(_ensemble, "input_dim", FEATURE_DIM) or FEATURE_DIM)

    # Normalize features and collect warnings/trace metadata.
    feature_vector_meta: Dict[str, Any] = {}
    feature_vector: Optional[List[float]] = None
    try:
        feature_vector, feature_vector_meta = _prepare_feature_vector(features, expected_dim)
    except ValueError as exc:
        if isinstance(_ensemble, StubEnsemble):
            # Stub mode can still run with non-numeric keys, but warn operators.
            feature_vector_meta = {
                "input_type": "unknown",
                "warnings": [f"feature_vector_unavailable:{exc}"],
            }
        else:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    warnings = list(feature_vector_meta.get("warnings", []))
    if schema_version != FEATURE_SCHEMA_VERSION:
        warnings.append(
            f"schema_version_mismatch:provided={schema_version} expected={FEATURE_SCHEMA_VERSION}"
        )
    if expected_dim != FEATURE_DIM:
        warnings.append(
            f"expected_dim_mismatch:model_expected={expected_dim} schema_expected={FEATURE_DIM}"
        )
    if warnings:
        feature_vector_meta["warnings"] = warnings

    # Run inference.
    try:
        if isinstance(_ensemble, StubEnsemble):
            features_payload = _mapping_for_stub(features)
            result = _ensemble.predict_ensemble(features_payload, model=model)
        else:
            if feature_vector is None:
                raise ValueError("Feature vector is empty")
            torch_mod = _require_torch()
            tensor_input = torch_mod.tensor(feature_vector, dtype=torch_mod.float32).unsqueeze(0)
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

    metadata: Dict[str, Any] = dict(result)
    metadata["inference_id"] = inference_id
    metadata["token"] = token
    metadata["feature_schema_version"] = schema_version
    metadata["feature"] = feature_vector_meta

    class_prediction: Optional[int] = None
    probabilities: Optional[List[float]] = None

    if not isinstance(_ensemble, StubEnsemble):
        try:
            class_prediction = int(result.get("prediction", 2))
        except (TypeError, ValueError):
            class_prediction = None

        raw_probs = result.get("probabilities")
        if isinstance(raw_probs, list) and len(raw_probs) >= 5:
            try:
                probabilities = [float(p) for p in raw_probs[:5]]
            except (TypeError, ValueError):
                probabilities = None

    prediction_value = float(result.get("prediction", 0.0))
    base_confidence = float(result.get("confidence", 0.0))
    base_expected_return = float(result.get("expected_return", 0.0))

    # Guidance-friendly score.
    if probabilities is not None:
        raw_score = max(0.0, min(1.0, probabilities[3] + probabilities[4]))
        metadata["score_definition"] = "P(buy)+P(strong_buy)"
    elif class_prediction is not None:
        # Defensive fallback if probabilities are missing for some reason.
        if class_prediction >= 3:
            raw_score = 1.0
        elif class_prediction <= 1:
            raw_score = 0.0
        else:
            raw_score = 0.5
        metadata["score_definition"] = "class_to_score_fallback"
    else:
        # Stub mode (or scalar-only models).
        raw_score = max(0.0, min(1.0, prediction_value))
        metadata["score_definition"] = "stub_or_scalar_prediction"

    score_value = raw_score
    expected_return_base = base_expected_return

    if _calibrator is not None and feature_vector is not None:
        calibration: Dict[str, Any] = {
            "active": True,
            "trained_at": _calibrator.trained_at,
            "schema_version": _calibrator.schema_version,
            "expected_dim": _calibrator.expected_dim,
            "score_raw": raw_score,
            "expected_return_raw": base_expected_return,
        }

        compatible = True
        if _calibrator.schema_version and _calibrator.schema_version != schema_version:
            compatible = False
            calibration["reason"] = (
                f"schema_mismatch:{_calibrator.schema_version}!={schema_version}"
            )
        if _calibrator.expected_dim and _calibrator.expected_dim != expected_dim:
            compatible = False
            calibration["reason"] = (
                f"dim_mismatch:{_calibrator.expected_dim}!={expected_dim}"
            )

        if compatible:
            try:
                cal_score, cal_ret = _calibrator.predict(feature_vector)
                calibration["score_calibrated"] = cal_score
                calibration["expected_return_calibrated"] = cal_ret
                score_value = cal_score
                expected_return_base = cal_ret
            except Exception as exc:  # pragma: no cover
                calibration["active"] = False
                calibration["reason"] = f"predict_failed:{exc}"

        metadata["calibration"] = calibration
    else:
        metadata["calibration"] = {
            "active": False,
            "reason": _calibrator_reason,
            "score_raw": raw_score,
            "expected_return_raw": base_expected_return,
        }

    confidence_value = base_confidence
    expected_return = expected_return_base

    override = None
    if token != "UNKNOWN":
        override = await signal_cache.override_for(token)

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
            score=score_value,
            expected_return=expected_return_base,
            notes="predict",
        )
        asyncio.create_task(_broadcast_guidance(guidance))

    payload = PredictResponse(
        inference_id=inference_id,
        schema_version=schema_version,
        prediction=prediction_value,
        class_prediction=class_prediction,
        score=score_value,
        confidence=confidence_value,
        expected_return=expected_return,
        latency_ms=latency_ms,
        model=str(result.get("model", model or "ensemble")),
        metadata=metadata,
    )
    return payload


@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(req: FeedbackRequest) -> FeedbackResponse:
    """Persist outcome feedback for nightly training/calibration."""

    token = str(req.token).strip()
    if not token:
        raise HTTPException(status_code=400, detail="token must be non-empty")

    if req.schema_version != FEATURE_SCHEMA_VERSION:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported schema_version '{req.schema_version}'. Expected '{FEATURE_SCHEMA_VERSION}'."
            ),
        )

    if len(req.features) != FEATURE_DIM:
        raise HTTPException(
            status_code=400,
            detail=f"features must be length {FEATURE_DIM} for schema {FEATURE_SCHEMA_VERSION}",
        )

    if req.realized_return is None and req.realized_class is None:
        raise HTTPException(
            status_code=400,
            detail="Provide either realized_return or realized_class",
        )

    realized_return = None
    if req.realized_return is not None:
        realized_return = float(req.realized_return)
        if not math.isfinite(realized_return):
            raise HTTPException(status_code=400, detail="realized_return must be finite")

    feature_hash = _hash_vector(req.features)
    stored_at = datetime.utcnow().isoformat()
    day = stored_at.split("T", 1)[0]

    record: Dict[str, Any] = {
        "stored_at": stored_at,
        "token": token,
        "schema_version": req.schema_version,
        "features": req.features,
        "feature_hash": feature_hash,
        "inference_id": req.inference_id,
        "prediction_timestamp": req.prediction_timestamp,
        "horizon_seconds": int(req.horizon_seconds),
        "realized_return": realized_return,
        "realized_class": req.realized_class,
        "model_output": req.model_output,
        "metadata": req.metadata,
    }

    path = _data_dir() / "feedback" / f"{day}.jsonl"
    await _append_jsonl_async(path, record)

    return FeedbackResponse(
        status="ok",
        inference_id=req.inference_id,
        feature_hash=feature_hash,
        stored_at=stored_at,
    )


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
    if os.getenv("SIGNAL_CACHE_ENABLED", "true").lower() != "true":
        logger.info("Signal cache disabled")
        return
    asyncio.create_task(signal_cache.run())


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "src.api.fastapi_server:app",
        host=os.getenv("API_HOST", "127.0.0.1"),
        port=int(os.getenv("API_PORT", "8288")),
        reload=os.getenv("API_RELOAD", "false").lower() == "true",
    )
