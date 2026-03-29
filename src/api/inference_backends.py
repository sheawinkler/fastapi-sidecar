"""Inference backend runtime for sidecar prediction paths.

This module provides a small runtime abstraction over three backend modes:

- `torch`: existing ensemble orchestrator path
- `coreml`: Core ML surrogate model path (macOS-only, optional dependency)
- `custom_export`: custom JSON-exported surrogate runtime (NumPy)

The default requested backend is `custom_export`. Runtime falls back to supported
alternatives when artifacts/dependencies are missing.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

FeaturePayload = Union[Dict[str, Any], List[float]]

LOGGER = logging.getLogger("ensemble.api.backends")


def _softmax(logits: np.ndarray) -> np.ndarray:
    arr = np.asarray(logits, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return np.asarray([], dtype=np.float64)
    shifted = arr - np.max(arr)
    exp = np.exp(shifted)
    denom = float(np.sum(exp))
    if denom <= 0.0 or not np.isfinite(denom):
        return np.ones_like(arr, dtype=np.float64) / float(arr.size)
    return exp / denom


def _expected_return_from_probs(probabilities: Sequence[float]) -> float:
    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if probs.size < 5:
        return 0.0
    class_values = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    return float(np.dot(probs[:5], class_values) / 2.0)


def _mapping_for_stub(payload: FeaturePayload) -> Dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    return {f"f{i}": float(val) for i, val in enumerate(payload)}


def _coerce_coreml_vector(value: Any) -> Optional[np.ndarray]:
    if isinstance(value, np.ndarray):
        arr = value.astype(np.float64, copy=False).reshape(-1)
        return arr if arr.size > 0 else None
    if isinstance(value, (list, tuple)):
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        return arr if arr.size > 0 else None
    if isinstance(value, dict):
        parsed: List[Tuple[int, float]] = []
        for key, raw in value.items():
            try:
                idx = int(key)
                fv = float(raw)
            except Exception:
                continue
            parsed.append((idx, fv))
        if parsed:
            parsed.sort(key=lambda item: item[0])
            return np.asarray([val for _, val in parsed], dtype=np.float64)
    return None


class CustomProxyModel:
    """Minimal NumPy runtime for custom-exported surrogate models."""

    def __init__(self, path: Path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        schema = str(payload.get("schema_version", "")).strip()
        if schema != "custom_proxy_v1":
            raise ValueError(f"Unsupported custom proxy schema: {schema!r}")

        input_dim = int(payload.get("input_dim", 0))
        if input_dim <= 0:
            raise ValueError("custom proxy input_dim must be > 0")
        self.input_dim = input_dim

        layers = payload.get("layers")
        if not isinstance(layers, list) or not layers:
            raise ValueError("custom proxy layers must be a non-empty list")

        parsed_layers: List[Tuple[np.ndarray, np.ndarray, str]] = []
        for idx, layer in enumerate(layers):
            if not isinstance(layer, dict):
                raise ValueError(f"Layer {idx} must be an object")
            weight = np.asarray(layer.get("weight"), dtype=np.float64)
            bias = np.asarray(layer.get("bias"), dtype=np.float64).reshape(-1)
            activation = str(layer.get("activation", "none")).strip().lower()
            if weight.ndim != 2:
                raise ValueError(f"Layer {idx} weight must be 2D")
            if bias.size != weight.shape[0]:
                raise ValueError(
                    f"Layer {idx} bias size {bias.size} does not match weight rows {weight.shape[0]}"
                )
            if activation not in {"none", "relu", "tanh"}:
                raise ValueError(f"Unsupported activation in layer {idx}: {activation!r}")
            parsed_layers.append((weight, bias, activation))

        if parsed_layers[0][1].size <= 0 or parsed_layers[-1][1].size < 5:
            raise ValueError("custom proxy final layer must produce >= 5 logits")

        if parsed_layers[0][0].shape[1] != self.input_dim:
            raise ValueError(
                f"custom proxy first layer expects {parsed_layers[0][0].shape[1]} inputs, "
                f"but input_dim={self.input_dim}"
            )

        self.layers = parsed_layers
        self.model_name = str(payload.get("model_name", "custom_export_proxy")).strip() or "custom_export_proxy"
        self.metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}

    def predict_logits(self, feature_vector: Sequence[float]) -> np.ndarray:
        x = np.asarray(feature_vector, dtype=np.float64).reshape(1, -1)
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"custom proxy feature length mismatch: got {x.shape[1]}, expected {self.input_dim}"
            )
        for weight, bias, activation in self.layers:
            x = np.matmul(x, weight.T) + bias.reshape(1, -1)
            if activation == "relu":
                x = np.maximum(x, 0.0)
            elif activation == "tanh":
                x = np.tanh(x)
        return x.reshape(-1)


class InferenceBackendRuntime:
    """Select and run inference backends with deterministic fallback behavior."""

    def __init__(
        self,
        ensemble: Any,
        stub_cls: type,
        *,
        requested_backend: Optional[str] = None,
        coreml_model_path: Optional[Union[str, Path]] = None,
        custom_export_path: Optional[Union[str, Path]] = None,
        coreml_input_name: Optional[str] = None,
        coreml_output_name: Optional[str] = None,
    ) -> None:
        self._ensemble = ensemble
        self._stub_cls = stub_cls

        self.requested_backend = (
            str(requested_backend or os.getenv("SIDECAR_INFERENCE_BACKEND", "custom_export"))
            .strip()
            .lower()
        )
        if not self.requested_backend:
            self.requested_backend = "custom_export"

        self.coreml_model_path = Path(
            coreml_model_path
            or os.getenv("SIDECAR_COREML_MODEL_PATH", "models/saved/coreml/ensemble_proxy.mlpackage")
        )
        self.custom_export_path = Path(
            custom_export_path
            or os.getenv(
                "SIDECAR_CUSTOM_EXPORT_PATH",
                "models/saved/custom_proxy/ensemble_proxy_v1.json",
            )
        )
        self.coreml_input_name = str(
            coreml_input_name or os.getenv("SIDECAR_COREML_INPUT_NAME", "features")
        ).strip()
        self.coreml_output_name = str(
            coreml_output_name or os.getenv("SIDECAR_COREML_OUTPUT_NAME", "logits")
        ).strip()

        self.active_backend = "torch"
        self.fallback_reason: Optional[str] = None
        self.available_backends: List[str] = []
        self._coreml_model: Any = None
        self._custom_model: Optional[CustomProxyModel] = None
        self._coreml_inputs: List[str] = []
        self._coreml_outputs: List[str] = []

        self._initialize()

    def _initialize(self) -> None:
        chain = self._resolve_chain(self.requested_backend)
        reasons: List[str] = []
        stub_ensemble = isinstance(self._ensemble, self._stub_cls)
        for candidate in chain:
            if candidate == "torch":
                if stub_ensemble:
                    reasons.append("torch_unavailable:ensemble_not_loaded")
                    continue
                self.active_backend = "torch"
                self.available_backends = self._discover_available_backends()
                if reasons:
                    self.fallback_reason = "; ".join(reasons)
                return
            if candidate == "coreml":
                ok, reason = self._try_activate_coreml()
                if ok:
                    self.active_backend = "coreml"
                    self.available_backends = self._discover_available_backends()
                    if reasons:
                        self.fallback_reason = "; ".join(reasons)
                    return
                if reason:
                    reasons.append(f"coreml_unavailable:{reason}")
                continue
            if candidate == "custom_export":
                ok, reason = self._try_activate_custom_export()
                if ok:
                    self.active_backend = "custom_export"
                    self.available_backends = self._discover_available_backends()
                    if reasons:
                        self.fallback_reason = "; ".join(reasons)
                    return
                if reason:
                    reasons.append(f"custom_export_unavailable:{reason}")

        if stub_ensemble:
            self.active_backend = "stub"
            self.available_backends = ["stub"]
            stub_reason = str(getattr(self._ensemble, "reason", "")).strip()
            reason_parts = [part for part in reasons if part]
            if stub_reason:
                reason_parts.insert(0, f"stub_ensemble:{stub_reason}")
            self.fallback_reason = "; ".join(reason_parts) if reason_parts else "stub ensemble is active"
            return

        self.active_backend = "torch"
        self.available_backends = self._discover_available_backends()
        if reasons:
            self.fallback_reason = "; ".join(reasons)

    def _resolve_chain(self, requested: str) -> List[str]:
        mapping = {
            "coreml": ["coreml", "custom_export", "torch"],
            "custom_export": ["custom_export", "coreml", "torch"],
            "custom": ["custom_export", "coreml", "torch"],
            "torch": ["torch"],
            "auto": ["custom_export", "coreml", "torch"],
        }
        return mapping.get(requested, ["custom_export", "coreml", "torch"])

    def _discover_available_backends(self) -> List[str]:
        available: List[str] = []
        if not isinstance(self._ensemble, self._stub_cls):
            available.append("torch")
        if self._coreml_model is not None:
            available.append("coreml")
        if self._custom_model is not None:
            available.append("custom_export")
        return available

    def is_stub_backend(self) -> bool:
        return self.active_backend == "stub"

    def expected_input_dim(self, default_dim: int) -> int:
        if self.active_backend == "custom_export" and self._custom_model is not None:
            return int(self._custom_model.input_dim)
        return int(default_dim)

    def _try_activate_coreml(self) -> Tuple[bool, Optional[str]]:
        if not self.coreml_model_path.exists():
            return False, f"model_path_missing:{self.coreml_model_path}"
        try:
            import coremltools as ct  # type: ignore
        except Exception as exc:
            return False, f"coremltools_import_failed:{exc}"

        compute_units_name = str(os.getenv("SIDECAR_COREML_COMPUTE_UNITS", "ALL")).strip().upper()
        compute_unit = {
            "ALL": ct.ComputeUnit.ALL,
            "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
            "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
            "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        }.get(compute_units_name, ct.ComputeUnit.ALL)

        try:
            model = ct.models.MLModel(str(self.coreml_model_path), compute_units=compute_unit)
            spec = model.get_spec()
            self._coreml_inputs = [item.name for item in spec.description.input]
            self._coreml_outputs = [item.name for item in spec.description.output]
            if not self.coreml_input_name:
                self.coreml_input_name = self._coreml_inputs[0] if self._coreml_inputs else "features"
            if not self.coreml_output_name:
                if "logits" in self._coreml_outputs:
                    self.coreml_output_name = "logits"
                elif self._coreml_outputs:
                    self.coreml_output_name = self._coreml_outputs[0]
                else:
                    self.coreml_output_name = "logits"
            self._coreml_model = model
            return True, None
        except Exception as exc:
            return False, f"coreml_model_load_failed:{exc}"

    def _try_activate_custom_export(self) -> Tuple[bool, Optional[str]]:
        if not self.custom_export_path.exists():
            return False, f"export_path_missing:{self.custom_export_path}"
        try:
            self._custom_model = CustomProxyModel(self.custom_export_path)
            return True, None
        except Exception as exc:
            return False, f"custom_export_load_failed:{exc}"

    def status_dict(self) -> Dict[str, Any]:
        return {
            "requested": self.requested_backend,
            "active": self.active_backend,
            "available": self.available_backends or self._discover_available_backends(),
            "fallback_reason": self.fallback_reason,
            "artifacts": {
                "coreml_model_path": str(self.coreml_model_path),
                "custom_export_path": str(self.custom_export_path),
            },
            "coreml": {
                "input_name": self.coreml_input_name,
                "output_name": self.coreml_output_name,
                "inputs": self._coreml_inputs,
                "outputs": self._coreml_outputs,
            },
        }

    def _predict_stub(self, raw_features: FeaturePayload, model: Optional[str]) -> Dict[str, Any]:
        payload = _mapping_for_stub(raw_features)
        result = self._ensemble.predict_ensemble(payload, model=model)
        if not isinstance(result, dict):
            raise RuntimeError("stub backend returned non-dict result")
        out = dict(result)
        out["backend"] = "stub"
        return out

    def _predict_torch(self, feature_vector: Sequence[float]) -> Dict[str, Any]:
        import torch

        tensor = torch.tensor(list(feature_vector), dtype=torch.float32).unsqueeze(0)
        result = self._ensemble.predict_ensemble(tensor)
        if not isinstance(result, dict):
            raise RuntimeError("torch backend returned non-dict result")
        out = dict(result)
        out["backend"] = "torch"
        return out

    def _extract_coreml_logits(self, raw_output: Dict[str, Any]) -> np.ndarray:
        if self.coreml_output_name and self.coreml_output_name in raw_output:
            vec = _coerce_coreml_vector(raw_output.get(self.coreml_output_name))
            if vec is not None and vec.size >= 5:
                return vec[:5]

        for key in ("logits", "output", "probabilities"):
            if key in raw_output:
                vec = _coerce_coreml_vector(raw_output.get(key))
                if vec is not None and vec.size >= 5:
                    return vec[:5]

        for value in raw_output.values():
            vec = _coerce_coreml_vector(value)
            if vec is not None and vec.size >= 5:
                return vec[:5]

        raise ValueError("Unable to extract 5-class output from Core ML prediction")

    def _predict_coreml(self, feature_vector: Sequence[float]) -> Dict[str, Any]:
        if self._coreml_model is None:
            raise RuntimeError("coreml backend is not initialized")
        x = np.asarray(feature_vector, dtype=np.float32).reshape(1, -1)
        payload = {self.coreml_input_name: x}
        raw = self._coreml_model.predict(payload)
        logits = self._extract_coreml_logits(raw)
        probabilities = _softmax(logits)
        prediction = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities))
        expected_return = _expected_return_from_probs(probabilities)
        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probabilities.tolist(),
            "expected_return": expected_return,
            "model": "coreml_proxy",
            "backend": "coreml",
            "coreml_raw_output_keys": sorted(raw.keys()),
        }

    def _predict_custom_export(self, feature_vector: Sequence[float]) -> Dict[str, Any]:
        if self._custom_model is None:
            raise RuntimeError("custom_export backend is not initialized")
        logits = self._custom_model.predict_logits(feature_vector)[:5]
        probabilities = _softmax(logits)
        prediction = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities))
        expected_return = _expected_return_from_probs(probabilities)
        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probabilities.tolist(),
            "expected_return": expected_return,
            "model": self._custom_model.model_name,
            "backend": "custom_export",
            "custom_metadata": self._custom_model.metadata,
        }

    def predict(
        self,
        *,
        feature_vector: Optional[Sequence[float]],
        raw_features: FeaturePayload,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self.active_backend == "stub":
            return self._predict_stub(raw_features, model)

        if feature_vector is None:
            raise ValueError("feature_vector required for non-stub inference backend")
        if len(feature_vector) == 0:
            raise ValueError("feature_vector cannot be empty")

        if self.active_backend == "coreml":
            return self._predict_coreml(feature_vector)
        if self.active_backend == "custom_export":
            return self._predict_custom_export(feature_vector)
        return self._predict_torch(feature_vector)
