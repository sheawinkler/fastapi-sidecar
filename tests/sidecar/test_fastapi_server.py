import importlib
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
