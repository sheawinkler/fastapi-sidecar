"""FastAPI bridge exposing the trained ensemble to external agents.

This service wraps the existing ensemble orchestrator so tools like the Rust
trader can obtain predictions over HTTP.  It also hosts lightweight telemetry
endpoints so we can inspect model call volumes and latencies.

It also supports a low-latency WebSocket guidance stream (`/ws/guidance`) so
execution engines can subscribe to real-time model hints without polling.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import math
import os
import time
import uuid
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.cors import CORSMiddleware

from .inference_backends import InferenceBackendRuntime
from .predictive_trainer import PredictiveTrainerConfig, PredictiveTrainerManager
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

RUST_SEMANTIC_FEATURE_VERSION = "rust_sidecar_semantic_v1"


def _data_dir() -> Path:
    """Base directory for sidecar local persistence (feedback, calibrators, etc.)."""

    root = os.getenv("SIDECAR_DATA_DIR", "data")
    return Path(root)


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(str(os.getenv(name, default)).strip())
    except Exception:
        parsed = default
    return max(minimum, min(maximum, parsed))


_INFERENCE_CACHE_MAX = _env_int("SIDECAR_INFERENCE_CACHE_MAX", 20_000, 100, 500_000)
_INFERENCE_CACHE_TTL_SECONDS = _env_int(
    "SIDECAR_INFERENCE_CACHE_TTL_SECONDS",
    60 * 60 * 24,
    60,
    60 * 60 * 24 * 14,
)
_inference_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_inference_cache_lock = asyncio.Lock()
_INFERENCE_EXECUTOR = ThreadPoolExecutor(
    max_workers=_env_int("SIDECAR_INFERENCE_EXECUTOR_WORKERS", 4, 1, 32),
    thread_name_prefix="sidecar-infer",
)


def _prune_inference_cache_locked(now_ts: float) -> None:
    expire_before = now_ts - float(_INFERENCE_CACHE_TTL_SECONDS)

    while _inference_cache:
        oldest_key, oldest = next(iter(_inference_cache.items()))
        cached_at = float(oldest.get("_cached_at_unix", 0.0))
        if cached_at >= expire_before:
            break
        _inference_cache.pop(oldest_key, None)

    while len(_inference_cache) > _INFERENCE_CACHE_MAX:
        _inference_cache.popitem(last=False)


async def _cache_inference_record(record: Dict[str, Any]) -> None:
    inference_id = str(record.get("inference_id", "")).strip()
    if not inference_id:
        return
    now_ts = time.time()
    entry = dict(record)
    entry["_cached_at_unix"] = now_ts
    async with _inference_cache_lock:
        _prune_inference_cache_locked(now_ts)
        _inference_cache[inference_id] = entry
        _inference_cache.move_to_end(inference_id)
        _prune_inference_cache_locked(now_ts)


async def _get_cached_inference(inference_id: str) -> Optional[Dict[str, Any]]:
    key = str(inference_id).strip()
    if not key:
        return None

    now_ts = time.time()
    async with _inference_cache_lock:
        _prune_inference_cache_locked(now_ts)
        entry = _inference_cache.get(key)
        if entry is None:
            return None
        cached_at = float(entry.get("_cached_at_unix", 0.0))
        if cached_at < now_ts - float(_INFERENCE_CACHE_TTL_SECONDS):
            _inference_cache.pop(key, None)
            return None
        return dict(entry)


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    """Append a single JSON object as one line (JSONL)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    with path.open("a", encoding="utf-8") as fh:
        fh.write(payload)
        fh.write("\n")


async def _append_jsonl_async(path: Path, record: Dict[str, Any]) -> None:
    await asyncio.to_thread(_append_jsonl, path, record)


def _extract_predictive_authority(features: FeaturePayload | None) -> Dict[str, Any]:
    authority: Dict[str, Any] = {
        "buy_authority_mode": "pass",
        "buy_blocker_stage": None,
        "buy_blocker_reason": None,
        "size_zero_reason": None,
    }
    if not isinstance(features, dict):
        return authority

    metadata = features.get("metadata")
    if not isinstance(metadata, dict):
        return authority

    contract = metadata.get("predictive_execution_contract")
    if not isinstance(contract, dict):
        contract = {}

    predictive = metadata.get("predictive")
    trace = predictive.get("selected_selector_trace") if isinstance(predictive, dict) else None
    if not isinstance(trace, dict):
        trace = contract.get("selected_selector_trace")
    if not isinstance(trace, dict):
        trace = {}

    def pick_str(*values: Any) -> Optional[str]:
        for value in values:
            if isinstance(value, str):
                trimmed = value.strip()
                if trimmed:
                    return trimmed
        return None

    authority["buy_authority_mode"] = (
        pick_str(contract.get("buy_authority_mode"), trace.get("buy_authority_mode")) or "pass"
    )
    authority["buy_blocker_stage"] = pick_str(
        contract.get("buy_blocker_stage"),
        trace.get("buy_blocker_stage"),
    )
    authority["buy_blocker_reason"] = pick_str(
        contract.get("buy_blocker_reason"),
        trace.get("buy_blocker_reason"),
    )
    authority["size_zero_reason"] = pick_str(
        contract.get("size_zero_reason"),
        trace.get("size_zero_reason"),
    )
    return authority


async def _predict_with_executor(
    *,
    feature_vector: Optional[List[float]],
    raw_features: Dict[str, Any],
    model: Optional[str],
) -> Dict[str, Any]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _INFERENCE_EXECUTOR,
        partial(
            _inference_backend.predict,
            feature_vector=feature_vector,
            raw_features=raw_features,
            model=model,
        ),
    )


async def _calibrator_predict_with_executor(feature_vector: List[float]) -> Tuple[float, float]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _INFERENCE_EXECUTOR,
        partial(_calibrator.predict, feature_vector),
    )


def _shutdown_inference_executor() -> None:
    _INFERENCE_EXECUTOR.shutdown(wait=False, cancel_futures=True)


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

    # Optional guidance knobs (used by Rust SidecarGuidance). These are kept lightweight and
    # bounded so clients can safely apply them.
    size_multiplier: float = 1.0
    risk_multiplier: float = 1.0
    kelly_fraction: float = 0.02
    buy_authority_mode: str = "pass"
    buy_blocker_stage: Optional[str] = None
    buy_blocker_reason: Optional[str] = None
    size_zero_reason: Optional[str] = None
    notes: Optional[str] = None

    latency_ms: float
    model: str
    metadata: Dict[str, Any]


class FeedbackRequest(BaseModel):
    """Outcome feedback for lightweight calibration/training."""

    token: str
    schema_version: Optional[str] = None
    features: Optional[List[float]] = None

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
    buy_authority_mode: str = "pass"
    buy_blocker_stage: Optional[str] = None
    buy_blocker_reason: Optional[str] = None
    size_zero_reason: Optional[str] = None
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

def _initialize_runtime() -> tuple[Any, InferenceBackendRuntime]:
    stub_requested = os.getenv("ENSEMBLE_STUB", "false").lower() == "true"
    requested_backend = str(
        os.getenv("SIDECAR_INFERENCE_BACKEND", "custom_export")
    ).strip().lower() or "custom_export"
    non_torch_requested = requested_backend in {"auto", "custom", "custom_export", "coreml"}

    if stub_requested:
        ensemble: Any = load_ensemble()
        backend = InferenceBackendRuntime(ensemble, StubEnsemble)
        return ensemble, backend

    if non_torch_requested:
        ensemble = StubEnsemble(reason="torch ensemble deferred")
        backend = InferenceBackendRuntime(ensemble, StubEnsemble)
        if not backend.is_stub_backend():
            return ensemble, backend

    ensemble = load_ensemble()
    backend = InferenceBackendRuntime(ensemble, StubEnsemble)
    return ensemble, backend


def _sidecar_mode() -> str:
    return "stub" if _inference_backend.is_stub_backend() else "full"


def _backend_is_full() -> bool:
    return not _inference_backend.is_stub_backend()


_ensemble, _inference_backend = _initialize_runtime()
_telemetry = TelemetryState()

# Optional lightweight calibrator trained from /feedback logs.
_CALIBRATOR_ROOT = Path(os.getenv("CALIBRATOR_ROOT", "models/saved/calibrator"))
_calibrator: Optional[CalibratorBundle]
_calibrator, _calibrator_reason = load_latest_calibrator(_CALIBRATOR_ROOT)
_trainer_manager = PredictiveTrainerManager(PredictiveTrainerConfig.from_env(_data_dir()))
_trainer_manager.set_guidance_subscriber_count_provider(lambda: len(_guidance_subscribers))

DEFAULT_SIGNAL_FEED_URL = "http://127.0.0.1:8075/signals/latest?limit=200"
SIGNAL_FEED_URL = os.getenv("SIGNAL_FEED_URL", DEFAULT_SIGNAL_FEED_URL)
SIGNAL_FEED_API_KEY = str(os.getenv("SIGNAL_FEED_API_KEY", "")).strip()
SIGNAL_FEED_API_HEADER = str(os.getenv("SIGNAL_FEED_API_HEADER", "x-api-key")).strip()
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
        self._disabled_reason: Optional[str] = None

    async def run(self) -> None:
        while True:
            await self.refresh()
            if self._disabled_reason is not None:
                return
            await asyncio.sleep(SIGNAL_REFRESH_SECONDS)

    async def refresh(self) -> None:
        if self._disabled_reason is not None:
            return
        try:
            entries = await self._fetch_feed()
        except httpx.HTTPStatusError as exc:  # pragma: no cover - network
            if self._should_disable_default_feed(exc):
                self._last_error = str(exc)
                self._disabled_reason = "default_signal_feed_not_found"
                logger.info(
                    "Signal cache disabled: default feed endpoint unavailable at %s",
                    SIGNAL_FEED_URL,
                )
                return
            self._last_error = str(exc)
            logger.warning("Signal feed refresh failed: %s", exc)
            return
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

    def _should_disable_default_feed(self, exc: httpx.HTTPStatusError) -> bool:
        return (
            SIGNAL_FEED_URL == DEFAULT_SIGNAL_FEED_URL
            and exc.response is not None
            and exc.response.status_code == 404
        )

    async def _fetch_feed(self) -> List[OverrideEntry]:
        headers: Optional[Dict[str, str]] = None
        if SIGNAL_FEED_API_KEY and SIGNAL_FEED_API_HEADER:
            headers = {SIGNAL_FEED_API_HEADER: SIGNAL_FEED_API_KEY}
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(SIGNAL_FEED_URL, headers=headers)
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


def _dig_path(payload: Any, path: str) -> Any:
    current = payload
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def _to_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = _coerce_float(value, "value")
    except ValueError:
        return None
    if not math.isfinite(out):
        return None
    return float(out)


def _to_optional_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return None


def _clamp01(value: Optional[float]) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _log_norm(value: Optional[float], scale: float) -> float:
    if value is None or value <= 0.0:
        return 0.0
    safe_scale = max(scale, 1e-6)
    return float(max(0.0, min(5.0, math.log1p(value) / safe_scale)))


def _side_to_signal_value(payload: Dict[str, Any]) -> float:
    raw_side = payload.get("side") or payload.get("signal_type") or payload.get("action")
    if not isinstance(raw_side, str):
        return 0.0
    lowered = raw_side.strip().lower()
    if "buy" in lowered:
        return 1.0
    if "sell" in lowered:
        return -1.0
    return 0.0


def _looks_like_rust_sidecar_payload(payload: Dict[str, Any]) -> bool:
    """Detect rich object payloads sent by AlgoTraderV2 Rust sidecar client."""

    if not isinstance(payload, dict):
        return False
    signature = {"symbol", "price", "confidence", "metadata", "ws_flow", "signal_type"}
    hits = sum(1 for key in signature if key in payload)
    return hits >= 2 and "metadata" in payload


def _numeric_scalar_coverage(
    payload: Any, max_depth: int = 6, max_nodes: int = 3000
) -> Tuple[int, int]:
    """Count numeric-compatible scalar coverage in nested JSON payloads."""

    stack: List[Tuple[Any, int]] = [(payload, 0)]
    seen = 0
    numeric = 0
    total_scalars = 0

    while stack and seen < max_nodes:
        node, depth = stack.pop()
        seen += 1
        if depth > max_depth:
            continue

        if isinstance(node, dict):
            for value in node.values():
                stack.append((value, depth + 1))
            continue
        if isinstance(node, list):
            for value in node:
                stack.append((value, depth + 1))
            continue

        total_scalars += 1
        if _to_optional_float(node) is not None:
            numeric += 1

    return numeric, total_scalars


def _arena_opinion_stats(metadata: Dict[str, Any]) -> Dict[str, float]:
    arena = metadata.get("arena")
    if not isinstance(arena, dict):
        return {
            "buy_ratio": 0.0,
            "sell_or_veto_ratio": 0.0,
            "avg_score": 0.0,
            "avg_confidence": 0.0,
            "opinion_count": 0.0,
        }

    opinions = arena.get("opinions")
    if not isinstance(opinions, list):
        return {
            "buy_ratio": 0.0,
            "sell_or_veto_ratio": 0.0,
            "avg_score": 0.0,
            "avg_confidence": 0.0,
            "opinion_count": 0.0,
        }

    buy = 0
    sell_or_veto = 0
    scored: List[float] = []
    confident: List[float] = []

    for item in opinions:
        if not isinstance(item, dict):
            continue
        action = str(item.get("action", "")).strip().lower()
        if action == "buy":
            buy += 1
        elif action in {"sell", "veto"}:
            sell_or_veto += 1

        score = _to_optional_float(item.get("score"))
        if score is not None:
            scored.append(score)

        conf = _to_optional_float(item.get("confidence"))
        if conf is not None:
            confident.append(conf)

    total = len(opinions)
    buy_ratio = (buy / total) if total else 0.0
    sell_or_veto_ratio = (sell_or_veto / total) if total else 0.0
    avg_score = (sum(scored) / len(scored)) if scored else 0.0
    avg_conf = (sum(confident) / len(confident)) if confident else 0.0

    return {
        "buy_ratio": float(max(0.0, min(1.0, buy_ratio))),
        "sell_or_veto_ratio": float(max(0.0, min(1.0, sell_or_veto_ratio))),
        "avg_score": float(avg_score),
        "avg_confidence": float(max(0.0, min(1.0, avg_conf))),
        "opinion_count": float(total),
    }


def _correlation_action_value(metadata: Dict[str, Any]) -> float:
    action = _dig_path(metadata, "correlation.action")
    if not isinstance(action, str):
        return 0.0
    lowered = action.strip().lower()
    if lowered == "skip":
        return -1.0
    if lowered == "downsize":
        return -0.5
    if lowered == "none":
        return 0.0
    return 0.0


def _build_rust_semantic_feature_vector(payload: Dict[str, Any]) -> Tuple[List[float], Dict[str, Any]]:
    """Project rich Rust sidecar objects into a stable 29-float feature vector."""

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    def pick_float(paths: List[str]) -> Optional[float]:
        for path in paths:
            value = _to_optional_float(_dig_path(payload, path))
            if value is not None:
                return value
        return None

    raw_fields: Dict[str, Optional[float]] = {
        "price": pick_float(["price"]),
        "size": pick_float(["size"]),
        "base_confidence": pick_float(["confidence"]),
        "safety_score": pick_float(["metadata.safety_score"]),
        "model_confidence": pick_float(["metadata.confidence_model.confidence"]),
        "legacy_confidence": pick_float(["metadata.confidence_model.legacy_confidence"]),
        "expected_edge_bps": pick_float(["metadata.expected_edge_bps"]),
        "confidence_edge_bps": pick_float(["metadata.confidence_edge_bps"]),
        "ws_age_secs": pick_float(["ws_flow.age_secs", "metadata.ws_flow_gate.stats.age_secs"]),
        "ws_notional_1m_usd": pick_float(
            ["ws_flow.notional_1m_usd", "metadata.ws_flow_gate.stats.notional_1m_usd"]
        ),
        "ws_notional_5m_usd": pick_float(
            ["ws_flow.notional_5m_usd", "metadata.ws_flow_gate.stats.notional_5m_usd"]
        ),
        "ws_volume_1m": pick_float(["ws_flow.volume_1m", "metadata.ws_flow_gate.stats.volume_1m"]),
        "ws_volume_5m": pick_float(["ws_flow.volume_5m", "metadata.ws_flow_gate.stats.volume_5m"]),
        "ws_stale_penalty": pick_float(["metadata.confidence_model.flow.ws_stale_penalty"]),
        "concentration_penalty": pick_float(
            ["metadata.confidence_model.concentration.combined_penalty"]
        ),
        "top1_holder_frac": pick_float(
            ["metadata.tier1.enrichment.onchain_top1_holder_frac"]
        ),
        "top10_holder_frac": pick_float(
            ["metadata.tier1.enrichment.onchain_top10_holder_frac"]
        ),
        "market_cap_usd": pick_float(["metadata.tier1.market_cap_usd"]),
        "tvl_usd": pick_float(["metadata.tier1.tvl_usd"]),
        "volume_24h_usd": pick_float(["metadata.tier1.volume_24h_usd"]),
        "age_seconds": pick_float(["metadata.tier1.age_seconds"]),
        "trade_fraction": pick_float(["metadata.trade_fraction"]),
        "suggested_trade_sol": pick_float(["metadata.suggested_trade_sol"]),
        "arena_size_multiplier_applied": pick_float(
            ["metadata.arena_size_multiplier_applied", "metadata.arena.applied_size_multiplier"]
        ),
        "correlation_score": pick_float(["metadata.correlation.score"]),
    }

    ws_pass_bool = _to_optional_bool(_dig_path(metadata, "ws_flow_gate.pass"))
    buy_gate_pass_bool = _to_optional_bool(_dig_path(metadata, "birdeye_ws_buy_gate.pass"))
    grinder_active_bool = _to_optional_bool(_dig_path(metadata, "grinder_scalp.active"))
    bucket0_active_bool = _to_optional_bool(_dig_path(metadata, "trade_sizing.bucket0_active"))

    arena_stats = _arena_opinion_stats(metadata)
    mapped_hits = sum(1 for value in raw_fields.values() if value is not None)
    mapped_total = len(raw_fields)
    mapped_coverage = (mapped_hits / mapped_total) if mapped_total else 0.0

    numeric_scalars, total_scalars = _numeric_scalar_coverage(payload)
    numeric_ratio = (numeric_scalars / total_scalars) if total_scalars else 0.0

    side_signal = _side_to_signal_value(payload)
    correlation_action_value = _correlation_action_value(metadata)

    features = [
        _log_norm(raw_fields["price"], scale=2.0),
        _log_norm(raw_fields["size"], scale=2.0),
        _clamp01(raw_fields["base_confidence"]),
        side_signal,
        _clamp01(raw_fields["safety_score"]),
        _clamp01(raw_fields["model_confidence"]),
        _clamp01(raw_fields["legacy_confidence"]),
        float(max(-5.0, min(5.0, (raw_fields["expected_edge_bps"] or 0.0) / 1000.0))),
        float(max(-5.0, min(5.0, (raw_fields["confidence_edge_bps"] or 0.0) / 1000.0))),
        float(1.0 / (1.0 + max(0.0, (raw_fields["ws_age_secs"] or 0.0) / 60.0))),
        _log_norm(raw_fields["ws_notional_1m_usd"], scale=8.0),
        _log_norm(raw_fields["ws_notional_5m_usd"], scale=8.0),
        _log_norm(raw_fields["ws_volume_1m"], scale=6.0),
        _log_norm(raw_fields["ws_volume_5m"], scale=6.0),
        _clamp01(raw_fields["ws_stale_penalty"]),
        _clamp01(raw_fields["concentration_penalty"]),
        _clamp01(raw_fields["top1_holder_frac"]),
        _clamp01(raw_fields["top10_holder_frac"]),
        _log_norm(raw_fields["market_cap_usd"], scale=12.0),
        _log_norm(raw_fields["tvl_usd"], scale=10.0),
        _log_norm(raw_fields["volume_24h_usd"], scale=11.0),
        float(
            max(
                0.0,
                min(1.0, ((raw_fields["age_seconds"] or 0.0) / 86_400.0) / 30.0),
            )
        ),
        _clamp01(raw_fields["trade_fraction"]),
        _log_norm(raw_fields["suggested_trade_sol"], scale=1.5),
        float((raw_fields["arena_size_multiplier_applied"] or 1.0) - 1.0),
        float(arena_stats["buy_ratio"] - arena_stats["sell_or_veto_ratio"]),
        float(arena_stats["avg_confidence"]),
        float(max(-1.0, min(1.0, raw_fields["correlation_score"] or 0.0))),
        float(max(0.0, min(1.0, 0.5 * mapped_coverage + 0.5 * numeric_ratio))),
    ]

    diagnostics: Dict[str, Any] = {
        "input_type": "object",
        "provided_dim": len(features),
        "extraction_mode": RUST_SEMANTIC_FEATURE_VERSION,
        "mapped_field_hits": mapped_hits,
        "mapped_field_total": mapped_total,
        "mapped_field_coverage": round(mapped_coverage, 6),
        "source_numeric_scalars": numeric_scalars,
        "source_scalar_values": total_scalars,
        "source_numeric_ratio": round(numeric_ratio, 6),
        "semantic_flags": {
            "ws_flow_gate_pass": bool(ws_pass_bool),
            "birdeye_ws_buy_gate_pass": bool(buy_gate_pass_bool),
            "grinder_scalp_active": bool(grinder_active_bool),
            "bucket0_active": bool(bucket0_active_bool),
            "correlation_action": correlation_action_value,
        },
        "arena": {
            "opinion_count": int(arena_stats["opinion_count"]),
            "buy_ratio": arena_stats["buy_ratio"],
            "sell_or_veto_ratio": arena_stats["sell_or_veto_ratio"],
            "avg_score": arena_stats["avg_score"],
            "avg_confidence": arena_stats["avg_confidence"],
        },
    }

    return features, diagnostics


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
        #
        # NOTE: `timestamp` is usually transport metadata (often epoch seconds) and can dominate
        # model inputs when unnormalized, leading to near-constant predictions. We exclude it
        # from the legacy feature object by default.
        excluded = {"token", "model", "features", "schema_version", "timestamp"}
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
        if _looks_like_rust_sidecar_payload(payload):
            raw, semantic_meta = _build_rust_semantic_feature_vector(payload)
            provided_dim = len(raw)
            vector = _resize_vector(raw, expected_dim)
            warnings.append("semantic_object_features_rust_sidecar_v1")
            meta.update(semantic_meta)
            meta["provided_dim"] = provided_dim
        else:
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
                    "extraction_mode": "legacy_sorted_numeric_keys_v1",
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
    if _inference_backend.active_backend in {"custom_export", "coreml"}:
        return 1
    models = getattr(_ensemble, "models", None)
    if isinstance(models, dict):
        return len(models)
    try:
        return len(_ensemble)  # type: ignore[arg-type]
    except Exception:  # pragma: no cover - handle objects without __len__
        return 0


def _model_status() -> List[Dict[str, Any]]:
    if _inference_backend.active_backend == "custom_export":
        return [{"name": "custom_export", "class": "CustomProxyModel", "fallback": False}]
    if _inference_backend.active_backend == "coreml":
        return [{"name": "coreml", "class": "CoreMLProxyModel", "fallback": False}]

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
    mode = _sidecar_mode()
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
        "feedback": {
            "inference_cache_entries": len(_inference_cache),
            "inference_cache_max": _INFERENCE_CACHE_MAX,
            "inference_cache_ttl_seconds": _INFERENCE_CACHE_TTL_SECONDS,
        },
        "inference_backend": _inference_backend.status_dict(),
        "trainer": _trainer_manager.health_payload(),
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
        "notes": "Send features as a fixed-order float array. Legacy dict payloads are flattened by sorted numeric keys; rich Rust sidecar payloads are semantically projected into a 29-float vector.",
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
        "auth_header_configured": bool(SIGNAL_FEED_API_KEY and SIGNAL_FEED_API_HEADER),
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
    start = time.perf_counter()
    inference_id = uuid.uuid4().hex

    try:
        token, features, model, schema_version = _derive_predict_inputs(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Determine expected feature dimension.
    expected_dim = FEATURE_DIM
    if _backend_is_full():
        expected_dim = int(getattr(_ensemble, "input_dim", FEATURE_DIM) or FEATURE_DIM)
        if expected_dim == FEATURE_DIM:
            expected_dim = _inference_backend.expected_input_dim(FEATURE_DIM)

    # Normalize features and collect warnings/trace metadata.
    feature_vector_meta: Dict[str, Any] = {}
    feature_vector: Optional[List[float]] = None
    try:
        feature_vector, feature_vector_meta = _prepare_feature_vector(features, expected_dim)
    except ValueError as exc:
        if _inference_backend.is_stub_backend():
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
        result = await _predict_with_executor(
            feature_vector=feature_vector,
            raw_features=features,
            model=model,
        )
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
    if feature_vector is not None:
        feature_vector_meta["vector"] = feature_vector
    metadata["inference_id"] = inference_id
    metadata["token"] = token
    metadata["feature_schema_version"] = schema_version
    metadata["inference_backend"] = _inference_backend.status_dict()
    metadata["feature"] = feature_vector_meta

    class_prediction: Optional[int] = None
    probabilities: Optional[List[float]] = None

    if _backend_is_full():
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
                cal_score, cal_ret = await _calibrator_predict_with_executor(feature_vector)
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

    predictive_authority = _extract_predictive_authority(features)

    # Derive lightweight guidance multipliers. These are intentionally conservative; the Rust
    # executor applies them *after* its own risk engine.
    side_hint: Optional[str] = None
    if isinstance(features, dict):
        raw_side = features.get("side") or features.get("signal_type") or features.get("action")
        if isinstance(raw_side, str):
            side_hint = raw_side.strip().lower()

    score_clamped = max(0.0, min(1.0, float(score_value)))
    conf_clamped = max(0.0, min(0.999, float(base_confidence)))

    guidance_size_multiplier = 1.0
    guidance_risk_multiplier = 1.0
    guidance_kelly_fraction = 0.02

    if side_hint != "sell":
        # Size multiplier: modest +/- 20% around 1.0.
        guidance_size_multiplier = 1.0 + (0.4 * (score_clamped - 0.5))
        guidance_size_multiplier = max(0.8, min(1.2, guidance_size_multiplier))

        # Risk multiplier: modest +/- 20% around 1.0 (blends model confidence + score).
        guidance_risk_multiplier = 1.0 + (0.3 * (score_clamped - 0.5)) + (0.3 * (conf_clamped - 0.5))
        guidance_risk_multiplier = max(0.8, min(1.2, guidance_risk_multiplier))

        # Kelly fraction: small [0.0, 0.25] hint, blended in Rust with its own sizing.
        guidance_kelly_fraction = 0.02 + (0.1 * score_clamped * conf_clamped)
        guidance_kelly_fraction = max(0.0, min(0.25, guidance_kelly_fraction))

    metadata["guidance"] = {
        "basis": "pre_override",
        "side_hint": side_hint,
        "size_multiplier": guidance_size_multiplier,
        "risk_multiplier": guidance_risk_multiplier,
        "kelly_fraction": guidance_kelly_fraction,
        **predictive_authority,
    }

    # Push a low-latency guidance event (pre-override) for subscribers.
    if token != "UNKNOWN":
        guidance = GuidancePayload(
            symbol=token,
            confidence=max(0.0, min(0.999, base_confidence)),
            score=score_value,
            expected_return=expected_return_base,
            size_multiplier=guidance_size_multiplier,
            risk_multiplier=guidance_risk_multiplier,
            kelly_fraction=guidance_kelly_fraction,
            buy_authority_mode=str(predictive_authority.get("buy_authority_mode") or "pass"),
            buy_blocker_stage=predictive_authority.get("buy_blocker_stage"),
            buy_blocker_reason=predictive_authority.get("buy_blocker_reason"),
            size_zero_reason=predictive_authority.get("size_zero_reason"),
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
        size_multiplier=guidance_size_multiplier,
        risk_multiplier=guidance_risk_multiplier,
        kelly_fraction=guidance_kelly_fraction,
        buy_authority_mode=str(predictive_authority.get("buy_authority_mode") or "pass"),
        buy_blocker_stage=predictive_authority.get("buy_blocker_stage"),
        buy_blocker_reason=predictive_authority.get("buy_blocker_reason"),
        size_zero_reason=predictive_authority.get("size_zero_reason"),
        notes="predict",
        latency_ms=latency_ms,
        model=str(result.get("model", model or "ensemble")),
        metadata=metadata,
    )
    await _cache_inference_record(
        {
            "inference_id": inference_id,
            "token": token,
            "schema_version": schema_version,
            "features": feature_vector,
            "prediction_timestamp": datetime.utcnow().isoformat(),
            "model_output": {
                "prediction": prediction_value,
                "class_prediction": class_prediction,
                "score": score_value,
                "confidence": confidence_value,
                "expected_return": expected_return,
                "model": str(result.get("model", model or "ensemble")),
            },
        }
    )
    return payload


@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(req: FeedbackRequest) -> FeedbackResponse:
    """Persist outcome feedback for nightly training/calibration."""

    token = str(req.token).strip()
    if not token:
        raise HTTPException(status_code=400, detail="token must be non-empty")

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

    schema_version = str(req.schema_version or "").strip() or FEATURE_SCHEMA_VERSION
    prediction_timestamp = req.prediction_timestamp
    model_output = req.model_output

    feature_vector: Optional[List[float]] = None
    if req.features is not None:
        try:
            feature_vector = [float(v) for v in req.features]
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="features must be numeric") from exc
    elif req.inference_id:
        cached = await _get_cached_inference(req.inference_id)
        if cached is not None:
            cached_schema = str(cached.get("schema_version", "")).strip()
            if not req.schema_version and cached_schema:
                schema_version = cached_schema

            cached_features = cached.get("features")
            if isinstance(cached_features, list):
                try:
                    feature_vector = [float(v) for v in cached_features]
                except (TypeError, ValueError):
                    feature_vector = None

            if prediction_timestamp is None:
                prediction_timestamp = cached.get("prediction_timestamp")
            if model_output is None and isinstance(cached.get("model_output"), dict):
                model_output = cached.get("model_output")

    if schema_version != FEATURE_SCHEMA_VERSION:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported schema_version '{schema_version}'. Expected '{FEATURE_SCHEMA_VERSION}'."
            ),
        )

    if feature_vector is None:
        if req.inference_id:
            detail = "features missing and inference_id not found (or expired) in cache"
        else:
            detail = "Provide features or inference_id"
        raise HTTPException(status_code=400, detail=detail)

    if len(feature_vector) != FEATURE_DIM:
        raise HTTPException(
            status_code=400,
            detail=f"features must be length {FEATURE_DIM} for schema {FEATURE_SCHEMA_VERSION}",
        )
    if not all(math.isfinite(v) for v in feature_vector):
        raise HTTPException(status_code=400, detail="features must be finite")

    feature_hash = _hash_vector(feature_vector)
    stored_at = datetime.utcnow().isoformat()
    day = stored_at.split("T", 1)[0]

    record: Dict[str, Any] = {
        "stored_at": stored_at,
        "token": token,
        "schema_version": schema_version,
        "features": feature_vector,
        "feature_hash": feature_hash,
        "inference_id": req.inference_id,
        "prediction_timestamp": prediction_timestamp,
        "horizon_seconds": int(req.horizon_seconds),
        "realized_return": realized_return,
        "realized_class": req.realized_class,
        "model_output": model_output,
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
            send_task = asyncio.create_task(queue.get())
            recv_task = asyncio.create_task(websocket.receive())
            done, pending = await asyncio.wait(
                {send_task, recv_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            for task in pending:
                with contextlib.suppress(asyncio.CancelledError):
                    await task

            if recv_task in done:
                incoming = recv_task.result()
                if incoming.get("type") == "websocket.disconnect":
                    break
                continue

            message = send_task.result()
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


@app.get("/trainer/health")
async def trainer_health() -> Dict[str, Any]:
    return _trainer_manager.health_payload()


@app.get("/trainer/health/deep")
async def trainer_health_deep() -> Dict[str, Any]:
    return await _trainer_manager.health_payload_deep_async()


@app.get("/trainer/status")
async def trainer_status() -> Dict[str, Any]:
    return _trainer_manager.status_payload()


@app.get("/trainer/status/deep")
async def trainer_status_deep() -> Dict[str, Any]:
    return await _trainer_manager.status_payload_deep_async()


@app.get("/trainer/history")
async def trainer_history(limit: int = 25) -> Dict[str, Any]:
    return _trainer_manager.history_payload(limit=max(1, min(limit, 200)))


@app.post("/trainer/run")
async def trainer_run() -> Dict[str, Any]:
    return await _trainer_manager.start_run(requested_by="manual", auto_promote=None)


@app.post("/trainer/promote/{run_id}")
async def trainer_promote(run_id: str) -> Dict[str, Any]:
    try:
        return await _trainer_manager.promote_run(run_id, forced=True)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/trainer/active-model")
async def trainer_active_model() -> Dict[str, Any]:
    return _trainer_manager.active_model_payload()


@app.get("/trainer/candidate-model/{run_id}")
async def trainer_candidate_model(run_id: str) -> Dict[str, Any]:
    try:
        return _trainer_manager.candidate_model_payload(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


async def _start_trainer_manager_background() -> None:
    try:
        await _trainer_manager.start()
    except Exception:
        logger.exception("trainer manager startup failed")


@app.on_event("startup")
async def bootstrap_signal_cache() -> None:
    if os.getenv("SIGNAL_CACHE_ENABLED", "true").lower() != "true":
        logger.info("Signal cache disabled")
    else:
        asyncio.create_task(signal_cache.run())
    # Availability-first: keep API startup non-blocking while trainer snapshot/scheduler warms.
    asyncio.create_task(_start_trainer_manager_background())


@app.on_event("shutdown")
async def shutdown_trainer_manager() -> None:
    await _trainer_manager.shutdown()
    _shutdown_inference_executor()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "src.api.fastapi_server:app",
        host=os.getenv("API_HOST", "127.0.0.1"),
        port=int(os.getenv("API_PORT", "8288")),
        reload=os.getenv("API_RELOAD", "false").lower() == "true",
    )
