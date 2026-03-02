import json
from pathlib import Path

import pytest
import torch


def _runtime_module():
    from src.api import inference_backends as mod

    return mod


class DummyEnsemble:
    input_dim = 29

    def predict_ensemble(self, x, model=None):
        if isinstance(x, dict):
            return {
                "prediction": 0.4,
                "confidence": 0.6,
                "expected_return": 0.05,
                "model": model or "stub",
            }
        tensor = x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
        total = float(tensor.sum().item())
        probs = [0.1, 0.2, 0.3, 0.25, 0.15]
        return {
            "prediction": 2 if total < 1 else 3,
            "confidence": 0.7,
            "probabilities": probs,
            "expected_return": 0.1,
            "model": "ensemble",
        }


class DummyStubEnsemble:
    reason = "stub mode"

    def predict_ensemble(self, features, model=None):
        return {
            "prediction": 0.5,
            "confidence": 0.5,
            "expected_return": 0.0,
            "model": model or "stub",
            "echo": features,
        }


def test_coreml_requested_falls_back_to_torch_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    mod = _runtime_module()
    missing = tmp_path / "no_model.mlpackage"
    missing_custom = tmp_path / "no_custom.json"
    monkeypatch.setenv("SIDECAR_INFERENCE_BACKEND", "coreml")
    monkeypatch.setenv("SIDECAR_COREML_MODEL_PATH", str(missing))
    monkeypatch.setenv("SIDECAR_CUSTOM_EXPORT_PATH", str(missing_custom))

    runtime = mod.InferenceBackendRuntime(DummyEnsemble(), DummyStubEnsemble)
    status = runtime.status_dict()

    assert status["requested"] == "coreml"
    assert status["active"] == "torch"
    assert "coreml_unavailable" in (status.get("fallback_reason") or "")


def test_default_requested_backend_is_custom_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    mod = _runtime_module()
    missing_coreml = tmp_path / "no_model.mlpackage"
    missing_custom = tmp_path / "no_custom.json"
    monkeypatch.delenv("SIDECAR_INFERENCE_BACKEND", raising=False)
    monkeypatch.setenv("SIDECAR_COREML_MODEL_PATH", str(missing_coreml))
    monkeypatch.setenv("SIDECAR_CUSTOM_EXPORT_PATH", str(missing_custom))

    runtime = mod.InferenceBackendRuntime(DummyEnsemble(), DummyStubEnsemble)
    status = runtime.status_dict()

    assert status["requested"] == "custom_export"
    assert status["active"] == "torch"
    fallback_reason = status.get("fallback_reason") or ""
    assert "custom_export_unavailable" in fallback_reason
    assert "coreml_unavailable" in fallback_reason
    assert fallback_reason.index("custom_export_unavailable") < fallback_reason.index("coreml_unavailable")


def test_custom_export_backend_predicts_from_json(tmp_path: Path):
    mod = _runtime_module()
    payload = {
        "schema_version": "custom_proxy_v1",
        "model_name": "custom_export_proxy",
        "input_dim": 29,
        "layers": [
            {
                "name": "linear",
                "activation": "none",
                "weight": [[0.0] * 29 for _ in range(5)],
                "bias": [-2.0, -1.0, 0.0, 1.0, 2.0],
            }
        ],
        "metadata": {"note": "test"},
    }
    export_path = tmp_path / "proxy.json"
    export_path.write_text(json.dumps(payload), encoding="utf-8")

    runtime = mod.InferenceBackendRuntime(
        DummyEnsemble(),
        DummyStubEnsemble,
        requested_backend="custom_export",
        custom_export_path=export_path,
    )

    status = runtime.status_dict()
    assert status["active"] == "custom_export"

    out = runtime.predict(feature_vector=[0.0] * 29, raw_features=[0.0] * 29, model=None)
    assert out["backend"] == "custom_export"
    assert out["prediction"] == 4
    assert isinstance(out["probabilities"], list)
    assert len(out["probabilities"]) == 5


def test_stub_runtime_routes_to_stub_predictor():
    mod = _runtime_module()
    runtime = mod.InferenceBackendRuntime(
        DummyStubEnsemble(),
        DummyStubEnsemble,
        requested_backend="coreml",
    )

    status = runtime.status_dict()
    assert status["active"] == "stub"

    out = runtime.predict(feature_vector=None, raw_features={"x": 1.0}, model="stub")
    assert out["backend"] == "stub"
    assert out["model"] == "stub"
