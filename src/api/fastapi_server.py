"""FastAPI bridge exposing the trained ensemble to external agents.

This service wraps the existing ensemble orchestrator so tools like the Rust
trader can obtain predictions over HTTP.  It also hosts lightweight telemetry
endpoints so we can inspect model call volumes and latencies.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

try:  # pragma: no cover - torch optional in stub mode
    import torch  # type: ignore
except Exception:  # pragma: no cover - fallback when torch missing
    torch = None  # type: ignore

try:
    from project.src.ai.ensemble_orchestrator import EnsembleOrchestrator
except Exception as exc:  # pragma: no cover - guard for missing module/deps
    EnsembleOrchestrator = None  # type: ignore
    STUB_REASON = str(exc)
else:
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
    if stub_requested or EnsembleOrchestrator is None:
        reason = STUB_REASON or ("stub mode enabled" if stub_requested else "orchestrator unavailable")
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
    token: str = Field(..., description="Token symbol")
    features: FeaturePayload
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
    if torch is None:  # pragma: no cover - handled by stub mode
        raise RuntimeError("PyTorch is not available in this environment")
    vector = _resize_vector(_flatten_features(payload), expected_dim)
    if not vector:
        raise ValueError("Feature vector is empty")
    tensor = torch.tensor(vector, dtype=torch.float32)
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


def record_latency(latency_ms: float) -> None:
    total = _telemetry.total_calls + 1
    # simple running average
    _telemetry.avg_latency_ms = (
        (_telemetry.avg_latency_ms * _telemetry.total_calls) + latency_ms
    ) / total
    _telemetry.total_calls = total


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "mode": "stub" if isinstance(_ensemble, StubEnsemble) else "full",
        "models_loaded": _model_inventory_size(),
        "telemetry": _telemetry.model_dump(),
    }


@app.get("/ping")
async def ping() -> Dict[str, Any]:
    """Lightweight readiness probe compatible with legacy clients."""

    return await health()


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
    start = time.perf_counter()
    try:
        if isinstance(_ensemble, StubEnsemble):
            features_payload = _mapping_for_stub(req.features)
            result = _ensemble.predict_ensemble(features_payload, model=req.model)
        else:
            expected_dim = getattr(_ensemble, "input_dim", None)
            tensor_input = _tensorize_features(req.features, expected_dim)
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

    payload = PredictResponse(
        prediction=float(result["prediction"]),
        confidence=float(result.get("confidence", 0.0)),
        expected_return=float(result.get("expected_return", 0.0)),
        latency_ms=latency_ms,
        model=result.get("model", req.model or "ensemble"),
        metadata=result,
    )
    return payload


@app.get("/telemetry", response_model=TelemetryState)
async def telemetry() -> TelemetryState:
    return _telemetry


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "project.src.api.fastapi_server:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8288")),
        reload=os.getenv("API_RELOAD", "false").lower() == "true",
    )
