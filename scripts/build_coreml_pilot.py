#!/usr/bin/env python3
"""Build Core ML pilot backend by distilling ensemble outputs into a surrogate.

The script trains a compact MLP proxy against live ensemble probabilities and
exports:

- Torch checkpoint (for reproducibility / custom export path)
- Core ML model package (for sidecar runtime backend)
- Manifest with fit and parity metrics
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ai.ensemble_orchestrator import EnsembleOrchestrator  # noqa: E402


@dataclass
class DistillationMetrics:
    train_kl: float
    val_kl: float
    val_class_agreement: float
    val_prob_mae: float


class SurrogateMLP(nn.Module):
    """Small proxy network that maps 29 features -> 5 class logits."""

    def __init__(self, input_dim: int = 29, hidden_dim: int = 64):
        super().__init__()
        hidden_mid = max(16, hidden_dim // 2)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_mid)
        self.fc3 = nn.Linear(hidden_mid, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def _iter_feedback_vectors(path: Path, input_dim: int) -> List[List[float]]:
    vectors: List[List[float]] = []
    if not path.exists():
        return vectors
    for file in sorted(path.glob("*.jsonl")):
        for line in file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            raw = row.get("features")
            if not isinstance(raw, list) or len(raw) != input_dim:
                continue
            try:
                vec = [float(v) for v in raw]
            except (TypeError, ValueError):
                continue
            if not all(np.isfinite(vec)):
                continue
            vectors.append(vec)
    return vectors


def _sample_vectors(
    feedback_vectors: List[List[float]],
    *,
    sample_count: int,
    input_dim: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out: List[List[float]] = []

    if feedback_vectors:
        random.Random(seed).shuffle(feedback_vectors)
        out.extend(feedback_vectors[: min(len(feedback_vectors), sample_count)])

    while len(out) < sample_count:
        synthetic = rng.normal(loc=0.0, scale=1.0, size=(input_dim,)).astype(np.float32)
        out.append(synthetic.tolist())

    arr = np.asarray(out[:sample_count], dtype=np.float32)
    return arr


def _set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _teacher_probabilities(
    ensemble: EnsembleOrchestrator, vectors: np.ndarray
) -> np.ndarray:
    targets = np.zeros((vectors.shape[0], 5), dtype=np.float32)
    for i in range(vectors.shape[0]):
        tensor = torch.tensor(vectors[i], dtype=torch.float32).unsqueeze(0)
        result = ensemble.predict_ensemble(tensor, update_history=False)
        probs = result.get("probabilities", [0.2] * 5)
        arr = np.asarray(probs, dtype=np.float32).reshape(-1)
        if arr.size < 5:
            arr = np.pad(arr, (0, 5 - arr.size), mode="constant", constant_values=0.0)
        arr = arr[:5]
        total = float(np.sum(arr))
        if total <= 0.0 or not np.isfinite(total):
            arr = np.full((5,), 0.2, dtype=np.float32)
        else:
            arr = arr / total
        targets[i] = arr
    return targets


def _train_surrogate(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    input_dim: int,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> Tuple[SurrogateMLP, DistillationMetrics]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = SurrogateMLP(input_dim=input_dim, hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    x_val_t = torch.tensor(x_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    model.train()
    for _ in range(max(1, epochs)):
        perm = torch.randperm(x_train_t.size(0))
        for start in range(0, x_train_t.size(0), max(1, batch_size)):
            idx = perm[start : start + batch_size]
            xb = x_train_t[idx]
            yb = y_train_t[idx]
            logits = model(xb)
            loss = F.kl_div(
                F.log_softmax(logits, dim=-1),
                yb,
                reduction="batchmean",
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        train_logits = model(x_train_t)
        val_logits = model(x_val_t)
        train_kl = float(
            F.kl_div(F.log_softmax(train_logits, dim=-1), y_train_t, reduction="batchmean").item()
        )
        val_kl = float(
            F.kl_div(F.log_softmax(val_logits, dim=-1), y_val_t, reduction="batchmean").item()
        )
        val_probs = F.softmax(val_logits, dim=-1).cpu().numpy()
        val_target = y_val_t.cpu().numpy()
        val_prob_mae = float(np.mean(np.abs(val_probs - val_target)))
        val_class_agreement = float(
            np.mean(np.argmax(val_probs, axis=1) == np.argmax(val_target, axis=1))
        )

    return model, DistillationMetrics(
        train_kl=train_kl,
        val_kl=val_kl,
        val_class_agreement=val_class_agreement,
        val_prob_mae=val_prob_mae,
    )


def _export_coreml(
    model: SurrogateMLP,
    *,
    input_dim: int,
    output_path: Path,
    input_name: str,
) -> None:
    try:
        import coremltools as ct  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "coremltools is required to export the Core ML pilot model"
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    example = torch.zeros((1, input_dim), dtype=torch.float32)
    traced = torch.jit.trace(model.eval(), example)

    minimum_target = getattr(ct.target, "macOS13", None)
    convert_kwargs: Dict[str, Any] = {
        "convert_to": "mlprogram",
        "inputs": [ct.TensorType(name=input_name, shape=example.shape, dtype=np.float32)],
        "outputs": [ct.TensorType(name="logits", dtype=np.float32)],
    }
    if minimum_target is not None:
        convert_kwargs["minimum_deployment_target"] = minimum_target

    mlmodel = ct.convert(traced, **convert_kwargs)
    mlmodel.save(str(output_path))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Core ML sidecar backend pilot")
    parser.add_argument("--samples", type=int, default=800, help="Distillation sample count")
    parser.add_argument("--epochs", type=int, default=35, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--input-dim", type=int, default=29, help="Feature vector size")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Surrogate hidden layer size")
    parser.add_argument(
        "--feedback-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "feedback",
        help="Feedback directory containing JSONL feature logs",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=PROJECT_ROOT / "models" / "saved" / "proxy" / "ensemble_proxy_surrogate.pt",
        help="Torch checkpoint output path",
    )
    parser.add_argument(
        "--coreml-path",
        type=Path,
        default=PROJECT_ROOT / "models" / "saved" / "coreml" / "ensemble_proxy.mlpackage",
        help="Core ML output path (.mlpackage)",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=PROJECT_ROOT / "models" / "saved" / "coreml" / "ensemble_proxy_manifest.json",
        help="Manifest output path",
    )
    parser.add_argument(
        "--input-name",
        type=str,
        default="features",
        help="Core ML input tensor name",
    )
    args = parser.parse_args()
    _set_global_seeds(args.seed)

    feedback_vectors = _iter_feedback_vectors(args.feedback_dir, args.input_dim)
    sampled = _sample_vectors(
        feedback_vectors,
        sample_count=max(50, args.samples),
        input_dim=args.input_dim,
        seed=args.seed,
    )

    ensemble = EnsembleOrchestrator(input_dim=args.input_dim)
    targets = _teacher_probabilities(ensemble, sampled)

    split = max(1, int(0.8 * sampled.shape[0]))
    x_train = sampled[:split]
    y_train = targets[:split]
    x_val = sampled[split:]
    y_val = targets[split:]
    if x_val.shape[0] == 0:
        x_val = x_train[: min(32, x_train.shape[0])]
        y_val = y_train[: min(32, y_train.shape[0])]

    model, metrics = _train_surrogate(
        x_train,
        y_train,
        x_val,
        y_val,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": "ensemble_proxy_surrogate_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "input_dim": args.input_dim,
        "hidden_dim": args.hidden_dim,
        "state_dict": model.state_dict(),
        "metrics": asdict(metrics),
        "teacher": "EnsembleOrchestrator",
        "sample_count": int(sampled.shape[0]),
        "feedback_vector_count": int(len(feedback_vectors)),
    }
    torch.save(checkpoint, args.checkpoint_path)

    _export_coreml(
        model,
        input_dim=args.input_dim,
        output_path=args.coreml_path,
        input_name=args.input_name,
    )

    manifest = {
        "status": "ok",
        "schema_version": "coreml_proxy_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "coreml_path": str(args.coreml_path),
        "checkpoint_path": str(args.checkpoint_path),
        "input_name": args.input_name,
        "input_dim": args.input_dim,
        "hidden_dim": args.hidden_dim,
        "distillation_metrics": asdict(metrics),
        "sample_count": int(sampled.shape[0]),
        "feedback_vector_count": int(len(feedback_vectors)),
    }
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
