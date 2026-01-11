"""Lightweight calibration bundle loader.

We keep the sidecar's primary ML ensemble intact and optionally apply a small
calibrator trained from logged feedback.

Artifacts are stored under `models/saved/calibrator/<timestamp>/`.
A `models/saved/calibrator/latest.json` pointer indicates the last known-good run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import json


@dataclass
class CalibratorBundle:
    schema_version: str
    expected_dim: int
    trained_at: str
    score_model: Any
    return_model: Any
    manifest: Dict[str, Any]

    def predict(self, features: Sequence[float]) -> Tuple[float, float]:
        """Return (score, expected_return) for a single feature vector."""

        import numpy as np  # local import

        x = np.asarray(list(features), dtype=float).reshape(1, -1)

        score = 0.5
        try:
            if hasattr(self.score_model, "predict_proba"):
                score = float(self.score_model.predict_proba(x)[0][1])
            else:
                score = float(self.score_model.predict(x)[0])
        except Exception:
            score = 0.5

        expected_return = 0.0
        try:
            expected_return = float(self.return_model.predict(x)[0])
        except Exception:
            expected_return = 0.0

        # Guardrails.
        if not (0.0 <= score <= 1.0):
            score = max(0.0, min(1.0, score))

        # Expected return can be noisy; clamp to a sane range.
        if expected_return != expected_return:  # NaN
            expected_return = 0.0
        expected_return = max(-0.5, min(0.5, expected_return))

        return score, expected_return


def load_latest_calibrator(root: Path) -> Tuple[Optional[CalibratorBundle], str]:
    """Load the latest calibrator bundle.

    Returns (bundle, reason). If bundle is None, reason is a human-readable string.
    """

    latest_path = root / "latest.json"
    if not latest_path.exists():
        return None, "no_latest_pointer"

    try:
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"invalid_latest_json:{exc}"

    artifact_dir = latest.get("artifact_dir")
    if not artifact_dir:
        return None, "latest_missing_artifact_dir"

    artifact_path = Path(artifact_dir)
    if not artifact_path.is_absolute():
        artifact_path = root / artifact_dir

    manifest_path = artifact_path / "manifest.json"
    model_path = artifact_path / "calibrator.joblib"

    if not model_path.exists():
        return None, f"missing_artifact:{model_path}"

    try:
        manifest: Dict[str, Any] = {}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        import joblib  # local import

        payload = joblib.load(model_path)
        bundle = CalibratorBundle(
            schema_version=str(payload.get("schema_version", "")),
            expected_dim=int(payload.get("expected_dim", 0) or 0),
            trained_at=str(payload.get("trained_at", "")),
            score_model=payload.get("score_model"),
            return_model=payload.get("return_model"),
            manifest=manifest,
        )
        if not bundle.schema_version:
            bundle.schema_version = str(manifest.get("schema_version", ""))
        if bundle.expected_dim <= 0:
            bundle.expected_dim = int(manifest.get("expected_dim", 0) or 0)
        return bundle, "ok"
    except Exception as exc:
        return None, f"load_failed:{exc}"
