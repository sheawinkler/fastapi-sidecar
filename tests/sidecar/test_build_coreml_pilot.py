import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "build_coreml_pilot.py"


def _load_build_module():
    spec = importlib.util.spec_from_file_location("build_coreml_pilot_for_test", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_skip_coreml_builds_checkpoint_without_export(tmp_path: Path, monkeypatch):
    mod = _load_build_module()

    def fake_teacher_probabilities(_ensemble, vectors):
        probs = np.zeros((vectors.shape[0], 5), dtype=np.float32)
        probs[:, 4] = 1.0
        return probs

    def fail_coreml_export(*_args, **_kwargs):
        raise AssertionError("CoreML export should not run with --skip-coreml")

    monkeypatch.setattr(mod, "EnsembleOrchestrator", lambda input_dim: object())
    monkeypatch.setattr(mod, "_teacher_probabilities", fake_teacher_probabilities)
    monkeypatch.setattr(mod, "_export_coreml", fail_coreml_export)

    checkpoint = tmp_path / "proxy" / "surrogate.pt"
    coreml_path = tmp_path / "coreml" / "proxy.mlpackage"
    manifest = tmp_path / "coreml" / "manifest.json"
    feedback_dir = tmp_path / "feedback"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_coreml_pilot.py",
            "--samples",
            "8",
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--feedback-dir",
            str(feedback_dir),
            "--checkpoint-path",
            str(checkpoint),
            "--coreml-path",
            str(coreml_path),
            "--manifest-path",
            str(manifest),
            "--skip-coreml",
        ],
    )

    assert mod.main() == 0
    assert checkpoint.exists()
    assert manifest.exists()
    assert not coreml_path.exists()

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["coreml_export"] == {
        "status": "skipped",
        "reason": "skip_coreml_requested",
        "coreml_path": str(coreml_path),
    }
