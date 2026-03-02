import importlib
import asyncio
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Pytest 8 defaults to importlib mode, which does not prepend the project root to sys.path.
# Ensure `import src...` works for this repo.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _fresh_import_fastapi_server(monkeypatch: pytest.MonkeyPatch) -> object:
    """Import src.api.fastapi_server fresh (respecting current env vars)."""

    module_name = "src.api.fastapi_server"
    if module_name in sys.modules:
        del sys.modules[module_name]
    return importlib.import_module(module_name)


@pytest.fixture()
def sidecar_server_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ENSEMBLE_STUB", "true")
    monkeypatch.setenv("SIGNAL_CACHE_ENABLED", "false")
    monkeypatch.setenv("CALIBRATOR_ROOT", str(tmp_path / "calibrator"))
    return _fresh_import_fastapi_server(monkeypatch)


def test_health_includes_schema_and_calibrator_status(sidecar_server_module):
    with TestClient(sidecar_server_module.app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()

    assert body["status"] == "ok"
    assert "feature_schema" in body
    assert body["feature_schema"]["schema_version"] == "v1"
    assert body["feature_schema"]["expected_dim"] == 29

    assert "calibrator" in body
    assert body["calibrator"]["active"] is False


def test_schema_features_v1(sidecar_server_module):
    with TestClient(sidecar_server_module.app) as client:
        resp = client.get("/schema/features")
        assert resp.status_code == 200
        body = resp.json()

    assert body["schema_version"] == "v1"
    assert body["expected_dim"] == 29
    assert body["feature_names"][0] == "x0"
    assert body["feature_names"][-1] == "x28"
    assert len(body["feature_names"]) == 29


def test_signal_feed_uses_api_key_header(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ENSEMBLE_STUB", "true")
    monkeypatch.setenv("SIGNAL_CACHE_ENABLED", "false")
    monkeypatch.setenv("CALIBRATOR_ROOT", str(tmp_path / "calibrator"))
    monkeypatch.setenv("SIGNAL_FEED_URL", "http://127.0.0.1:8075/signals/latest?limit=2")
    monkeypatch.setenv("SIGNAL_FEED_API_KEY", "test-key-123")
    monkeypatch.setenv("SIGNAL_FEED_API_HEADER", "x-api-key")

    module = _fresh_import_fastapi_server(monkeypatch)
    observed = {}

    class _DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "signals": [
                    {
                        "symbol": "SOL",
                        "address": "So11111111111111111111111111111111111111112",
                        "price_usd": 100.0,
                        "volume_24h_usd": 500000.0,
                        "liquidity_usd": 350000.0,
                        "momentum_score": 1.5,
                        "risk_score": 0.2,
                        "verified": True,
                        "created_at": "2026-03-02T00:00:00Z",
                    }
                ]
            }

    class _DummyAsyncClient:
        def __init__(self, *args, **kwargs):
            return None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url, headers=None):
            observed["url"] = url
            observed["headers"] = headers or {}
            return _DummyResponse()

    monkeypatch.setattr(module.httpx, "AsyncClient", _DummyAsyncClient)
    entries = asyncio.run(module.signal_cache._fetch_feed())

    assert observed["url"].startswith("http://127.0.0.1:8075/signals/latest")
    assert observed["headers"].get("x-api-key") == "test-key-123"
    assert len(entries) == 1
    assert entries[0].symbol == "SOL"


def test_predict_vector_features_returns_score_and_schema(sidecar_server_module):
    payload = {
        "token": "SOL/USDC",
        "schema_version": "v1",
        "features": [round(i * 0.1, 2) for i in range(1, 30)],
    }

    with TestClient(sidecar_server_module.app) as client:
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        body = resp.json()

    assert body["schema_version"] == "v1"
    assert isinstance(body["inference_id"], str) and body["inference_id"]
    assert 0.0 <= float(body["score"]) <= 1.0
    assert isinstance(body["metadata"], dict)
    assert body["metadata"].get("calibration", {}).get("active") is False
    assert body["metadata"].get("feature", {}).get("input_type") == "array"


def test_predict_dict_features_warns_legacy(sidecar_server_module):
    payload = {
        "token": "SOL/USDC",
        "schema_version": "v1",
        "features": {
            "x": 1.0,
            "y": 2.0,
            "symbol": "SOL",  # non-numeric, should be dropped from vectorization
            "side": "buy",  # non-numeric, should be dropped from vectorization
        },
    }

    with TestClient(sidecar_server_module.app) as client:
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        body = resp.json()

    warnings = body["metadata"].get("feature", {}).get("warnings", [])
    assert any("legacy_object_features_sorted_by_key" in w for w in warnings)
    assert any("dropped_non_numeric_fields" in w for w in warnings)


def test_predict_rust_sidecar_payload_uses_semantic_extraction(sidecar_server_module):
    payload = {
        "token": "BONK",
        "schema_version": "v1",
        "features": {
            "symbol": "BONK",
            "price": 0.00042,
            "size": 0.8,
            "confidence": 0.69,
            "side": "buy",
            "signal_type": "buy",
            "metadata": {
                "safety_score": 0.73,
                "expected_edge_bps": 420.0,
                "confidence_edge_bps": 360.0,
                "confidence_model": {
                    "confidence": 0.72,
                    "legacy_confidence": 0.66,
                    "flow": {"ws_stale_penalty": 0.11},
                    "concentration": {"combined_penalty": 0.18},
                },
                "tier1": {
                    "market_cap_usd": 25_000_000,
                    "tvl_usd": 1_500_000,
                    "volume_24h_usd": 4_500_000,
                    "age_seconds": 7200,
                    "enrichment": {
                        "onchain_top1_holder_frac": 0.09,
                        "onchain_top10_holder_frac": 0.41,
                    },
                },
                "trade_fraction": 0.14,
                "suggested_trade_sol": 0.45,
                "arena_size_multiplier_applied": 1.12,
                "ws_flow_gate": {
                    "pass": True,
                    "stats": {
                        "age_secs": 14,
                        "notional_1m_usd": 55000,
                        "notional_5m_usd": 225000,
                        "volume_1m": 1400,
                        "volume_5m": 7200,
                    },
                },
                "arena": {
                    "opinions": [
                        {"action": "buy", "score": 0.82, "confidence": 0.78},
                        {"action": "hold", "score": 0.43, "confidence": 0.55},
                    ]
                },
                "correlation": {"score": 0.17, "action": "downsize"},
                "grinder_scalp": {"active": False},
            },
            "ws_flow": {
                "age_secs": 12,
                "notional_1m_usd": 62000,
                "notional_5m_usd": 260000,
                "volume_1m": 1600,
                "volume_5m": 7600,
            },
        },
    }

    with TestClient(sidecar_server_module.app) as client:
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        body = resp.json()

    feature_meta = body["metadata"].get("feature", {})
    warnings = feature_meta.get("warnings", [])
    assert feature_meta.get("input_type") == "object"
    assert feature_meta.get("extraction_mode") == "rust_sidecar_semantic_v1"
    assert feature_meta.get("provided_dim") == 29
    assert feature_meta.get("mapped_field_hits", 0) > 0
    assert any("semantic_object_features_rust_sidecar_v1" in w for w in warnings)
    assert not any("legacy_object_features_sorted_by_key" in w for w in warnings)


def test_predict_flat_rust_payload_without_features_key(sidecar_server_module):
    payload = {
        "symbol": "BONK",
        "price": 0.00041,
        "size": 0.75,
        "confidence": 0.65,
        "side": "buy",
        "signal_type": "buy",
        "metadata": {
            "safety_score": 0.7,
            "expected_edge_bps": 300.0,
            "confidence_model": {"confidence": 0.71, "legacy_confidence": 0.64},
            "arena": {"opinions": [{"action": "buy", "score": 0.8, "confidence": 0.75}]},
        },
        "ws_flow": {"age_secs": 10, "notional_5m_usd": 180000},
    }

    with TestClient(sidecar_server_module.app) as client:
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        body = resp.json()

    assert body["metadata"].get("token") == "BONK"
    feature_meta = body["metadata"].get("feature", {})
    assert feature_meta.get("extraction_mode") == "rust_sidecar_semantic_v1"


def test_feedback_writes_jsonl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ENSEMBLE_STUB", "true")
    monkeypatch.setenv("SIGNAL_CACHE_ENABLED", "false")
    monkeypatch.setenv("CALIBRATOR_ROOT", str(tmp_path / "calibrator"))
    monkeypatch.setenv("SIDECAR_DATA_DIR", str(tmp_path / "sidecar_data"))

    module = _fresh_import_fastapi_server(monkeypatch)

    payload = {
        "token": "SOL/USDC",
        "schema_version": "v1",
        "features": [0.0] * 29,
        "inference_id": "abc123",
        "prediction_timestamp": "2026-01-09T00:00:00Z",
        "horizon_seconds": 900,
        "realized_return": 0.123,
        "metadata": {"source": "pytest"},
    }

    with TestClient(module.app) as client:
        resp = client.post("/feedback", json=payload)
        assert resp.status_code == 200
        body = resp.json()

    assert body["status"] == "ok"
    assert body["inference_id"] == "abc123"
    assert isinstance(body["feature_hash"], str) and len(body["feature_hash"]) > 10

    stored_at = body["stored_at"]
    day = stored_at.split("T", 1)[0]
    out_path = tmp_path / "sidecar_data" / "feedback" / f"{day}.jsonl"
    assert out_path.exists()

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert lines, "feedback file should contain at least one JSONL line"
    last = json.loads(lines[-1])
    assert last["token"] == "SOL/USDC"
    assert last["schema_version"] == "v1"
    assert last["features"] == [0.0] * 29
    assert last["realized_return"] == 0.123


def test_predict_applies_calibrator_when_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Build a tiny calibrator artifact tree.
    calibrator_root = tmp_path / "calibrator"
    artifact_dir = calibrator_root / "run1"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    from sklearn.dummy import DummyClassifier, DummyRegressor

    score_model = DummyClassifier(strategy="constant", constant=1)
    return_model = DummyRegressor(strategy="constant", constant=0.25)

    # Fit to establish expected sklearn attributes/classes.
    import numpy as np

    X = np.zeros((2, 29), dtype=float)
    y = np.asarray([0, 1], dtype=int)
    score_model.fit(X, y)
    return_model.fit(X, np.asarray([0.0, 0.0], dtype=float))

    payload = {
        "schema_version": "v1",
        "expected_dim": 29,
        "trained_at": "2026-01-09T00:00:00Z",
        "score_model": score_model,
        "return_model": return_model,
    }

    import joblib

    joblib.dump(payload, artifact_dir / "calibrator.joblib")
    (artifact_dir / "manifest.json").write_text(
        json.dumps({"schema_version": "v1", "expected_dim": 29}, indent=2) + "\n",
        encoding="utf-8",
    )
    (calibrator_root / "latest.json").write_text(
        json.dumps({"artifact_dir": "run1"}, indent=2) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("ENSEMBLE_STUB", "true")
    monkeypatch.setenv("SIGNAL_CACHE_ENABLED", "false")
    monkeypatch.setenv("CALIBRATOR_ROOT", str(calibrator_root))

    module = _fresh_import_fastapi_server(monkeypatch)

    payload = {
        "token": "SOL/USDC",
        "schema_version": "v1",
        "features": [1.0] * 29,
    }

    with TestClient(module.app) as client:
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        body = resp.json()

    calib = body["metadata"].get("calibration", {})
    assert calib.get("active") is True
    assert calib.get("schema_version") == "v1"
    assert calib.get("expected_dim") == 29

    assert 0.99 <= float(body["score"]) <= 1.0
    assert abs(float(body["expected_return"]) - 0.25) < 1e-6
