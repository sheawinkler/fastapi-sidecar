#!/usr/bin/env python3
"""Replay benchmark harness for sidecar inference backends."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from datetime import datetime, UTC
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ai.ensemble_orchestrator import EnsembleOrchestrator  # noqa: E402
from src.api.inference_backends import InferenceBackendRuntime  # noqa: E402


class _NeverStub:
    pass


def _extract_feature_vector(row: Dict[str, Any], input_dim: int) -> List[float] | None:
    raw = row.get("features")
    if not isinstance(raw, list) or len(raw) != input_dim:
        return None
    try:
        vec = [float(v) for v in raw]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in vec):
        return None
    return vec


def _load_vectors_from_feedback(feedback_dir: Path, input_dim: int, limit: int) -> List[List[float]]:
    vectors: List[List[float]] = []
    if not feedback_dir.exists():
        return vectors
    for file in sorted(feedback_dir.glob("*.jsonl")):
        for line in file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            vec = _extract_feature_vector(row, input_dim)
            if vec is None:
                continue
            vectors.append(vec)
            if len(vectors) >= limit:
                return vectors
    return vectors


def _load_vectors_from_jsonl(path: Path, input_dim: int, limit: int) -> List[List[float]]:
    vectors: List[List[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, list):
            raw = row
            row = {"features": raw}
        if not isinstance(row, dict):
            continue
        vec = _extract_feature_vector(row, input_dim)
        if vec is None:
            continue
        vectors.append(vec)
        if len(vectors) >= limit:
            break
    return vectors


def _extract_score(result: Dict[str, Any]) -> float:
    probs = result.get("probabilities")
    if isinstance(probs, list) and len(probs) >= 5:
        try:
            return float(probs[3]) + float(probs[4])
        except (TypeError, ValueError):
            pass
    prediction = result.get("prediction", 0.0)
    try:
        pred = float(prediction)
    except (TypeError, ValueError):
        pred = 0.0
    # Convert class-ish prediction to score-ish fallback.
    if pred >= 3.0:
        return 1.0
    if pred <= 1.0:
        return 0.0
    return max(0.0, min(1.0, pred))


def _benchmark_runtime(
    runtime: InferenceBackendRuntime,
    vectors: Sequence[Sequence[float]],
) -> Dict[str, Any]:
    latencies_ms: List[float] = []
    predictions: List[int] = []
    scores: List[float] = []
    confidences: List[float] = []

    for vec in vectors:
        start = time.perf_counter()
        out = runtime.predict(feature_vector=vec, raw_features=list(vec), model=None)
        latency_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(latency_ms)
        predictions.append(int(out.get("prediction", 0)))
        scores.append(_extract_score(out))
        confidences.append(float(out.get("confidence", 0.0)))

    return {
        "samples": len(vectors),
        "latency_ms": {
            "p50": float(np.percentile(latencies_ms, 50)),
            "p95": float(np.percentile(latencies_ms, 95)),
            "mean": float(np.mean(latencies_ms)),
            "median": float(median(latencies_ms)),
            "max": float(np.max(latencies_ms)),
        },
        "predictions": predictions,
        "scores": scores,
        "confidences": confidences,
    }


def _compare_to_baseline(candidate: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, float]:
    c_preds = np.asarray(candidate["predictions"], dtype=np.int64)
    b_preds = np.asarray(baseline["predictions"], dtype=np.int64)
    c_scores = np.asarray(candidate["scores"], dtype=np.float64)
    b_scores = np.asarray(baseline["scores"], dtype=np.float64)
    c_conf = np.asarray(candidate["confidences"], dtype=np.float64)
    b_conf = np.asarray(baseline["confidences"], dtype=np.float64)
    return {
        "class_agreement": float(np.mean(c_preds == b_preds)),
        "score_mae": float(np.mean(np.abs(c_scores - b_scores))),
        "confidence_mae": float(np.mean(np.abs(c_conf - b_conf))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay benchmark for sidecar backends")
    parser.add_argument("--input-dim", type=int, default=29)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument(
        "--feedback-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "feedback",
        help="Directory containing feedback JSONL with features vectors",
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=None,
        help="Optional explicit replay input JSONL",
    )
    parser.add_argument(
        "--coreml-path",
        type=Path,
        default=PROJECT_ROOT / "models" / "saved" / "coreml" / "ensemble_proxy.mlpackage",
    )
    parser.add_argument(
        "--custom-export-path",
        type=Path,
        default=PROJECT_ROOT / "models" / "saved" / "custom_proxy" / "ensemble_proxy_v1.json",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=PROJECT_ROOT / "reports" / "backend_replay_benchmark.json",
    )
    parser.add_argument("--seed", type=int, default=42, help="Teacher ensemble seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    vectors: List[List[float]]
    if args.input_jsonl is not None:
        vectors = _load_vectors_from_jsonl(args.input_jsonl, args.input_dim, args.limit)
    else:
        vectors = _load_vectors_from_feedback(args.feedback_dir, args.input_dim, args.limit)

    if not vectors:
        raise RuntimeError("No replay vectors found. Provide --input-jsonl or populate data/feedback.")

    ensemble = EnsembleOrchestrator(input_dim=args.input_dim)

    torch_runtime = InferenceBackendRuntime(
        ensemble,
        _NeverStub,
        requested_backend="torch",
        coreml_model_path=args.coreml_path,
        custom_export_path=args.custom_export_path,
    )
    coreml_runtime = InferenceBackendRuntime(
        ensemble,
        _NeverStub,
        requested_backend="coreml",
        coreml_model_path=args.coreml_path,
        custom_export_path=args.custom_export_path,
    )
    custom_runtime = InferenceBackendRuntime(
        ensemble,
        _NeverStub,
        requested_backend="custom_export",
        coreml_model_path=args.coreml_path,
        custom_export_path=args.custom_export_path,
    )

    torch_stats = _benchmark_runtime(torch_runtime, vectors)
    report: Dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "sample_count": len(vectors),
        "backends": {
            "torch": {
                "status": torch_runtime.status_dict(),
                "metrics": torch_stats,
            }
        },
    }

    for name, runtime in (("coreml", coreml_runtime), ("custom_export", custom_runtime)):
        status = runtime.status_dict()
        if status.get("active") != name:
            report["backends"][name] = {
                "status": status,
                "available": False,
                "reason": status.get("fallback_reason") or "requested backend unavailable",
            }
            continue

        metrics = _benchmark_runtime(runtime, vectors)
        comparison = _compare_to_baseline(metrics, torch_stats)
        speedup = (
            torch_stats["latency_ms"]["mean"] / metrics["latency_ms"]["mean"]
            if metrics["latency_ms"]["mean"] > 0
            else 0.0
        )
        report["backends"][name] = {
            "status": status,
            "available": True,
            "metrics": metrics,
            "comparison_to_torch": comparison,
            "mean_latency_speedup_vs_torch": float(speedup),
        }

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
