#!/usr/bin/env python3
"""Train a lightweight calibrator from /feedback logs.

This is intentionally conservative: we start with simple sklearn models that are
fast to train on a laptop.

Artifacts:
- models/saved/calibrator/<timestamp>/calibrator.joblib
- models/saved/calibrator/<timestamp>/manifest.json
- models/saved/calibrator/latest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _iter_feedback_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for file in sorted(path.glob("*.jsonl")):
        for line in file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _build_dataset(
    rows: List[Dict[str, Any]], expected_dim: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    X: List[List[float]] = []
    y_cls: List[int] = []
    y_ret: List[float] = []

    used = 0
    skipped = 0

    for row in rows:
        features = row.get("features")
        if not isinstance(features, list) or len(features) != expected_dim:
            skipped += 1
            continue

        realized_return = row.get("realized_return")
        realized_class = row.get("realized_class")

        if realized_return is None and realized_class is None:
            skipped += 1
            continue

        try:
            vec = [float(x) for x in features]
        except (TypeError, ValueError):
            skipped += 1
            continue

        if realized_return is not None:
            try:
                ret = float(realized_return)
            except (TypeError, ValueError):
                skipped += 1
                continue
            cls = 1 if ret > 0 else 0
        else:
            try:
                cls_raw = int(realized_class)
            except (TypeError, ValueError):
                skipped += 1
                continue
            cls = 1 if cls_raw >= 3 else 0
            ret = 0.0

        X.append(vec)
        y_cls.append(cls)
        y_ret.append(ret)
        used += 1

    meta = {"used": used, "skipped": skipped}
    return (
        np.asarray(X, dtype=float),
        np.asarray(y_cls, dtype=int),
        np.asarray(y_ret, dtype=float),
        meta,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train sidecar calibrator from feedback logs")
    parser.add_argument(
        "--data-dir",
        default="data/feedback",
        help="Directory containing feedback YYYY-MM-DD.jsonl files",
    )
    parser.add_argument(
        "--out-root",
        default="models/saved/calibrator",
        help="Output root for versioned calibrator artifacts",
    )
    parser.add_argument(
        "--schema-version",
        default="v1",
        help="Feature schema version expected in feedback logs",
    )
    parser.add_argument(
        "--expected-dim",
        type=int,
        default=29,
        help="Expected feature vector dimension",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=20,
        help="Minimum samples required to train",
    )
    args = parser.parse_args()

    feedback_dir = Path(args.data_dir)
    rows = _iter_feedback_rows(feedback_dir)

    X, y_cls, y_ret, ds_meta = _build_dataset(rows, args.expected_dim)

    if X.shape[0] < args.min_samples:
        print(
            f"Not enough samples to train: have {X.shape[0]}, need {args.min_samples}",
            file=sys.stderr,
        )
        return 2

    # Train simple, fast models.
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    score_model = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
                ),
            ),
        ]
    )

    return_model = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("reg", Ridge(alpha=1.0)),
        ]
    )

    score_model.fit(X, y_cls)
    return_model.fit(X, y_ret)

    # Basic metrics (in-sample; better evaluation can be added later).
    cls_acc = float(score_model.score(X, y_cls))
    pred_ret = return_model.predict(X)
    mae = float(np.mean(np.abs(pred_ret - y_ret)))

    trained_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    out_root = Path(args.out_root)
    artifact_dir = out_root / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": args.schema_version,
        "expected_dim": int(args.expected_dim),
        "trained_at": trained_at,
        "score_model": score_model,
        "return_model": return_model,
    }

    import joblib

    joblib.dump(payload, artifact_dir / "calibrator.joblib")

    manifest = {
        "status": "ok",
        "schema_version": args.schema_version,
        "expected_dim": int(args.expected_dim),
        "trained_at": trained_at,
        "dataset": {
            "total_rows": len(rows),
            **ds_meta,
        },
        "metrics": {
            "score_model_accuracy_in_sample": cls_acc,
            "return_model_mae_in_sample": mae,
        },
    }

    (artifact_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    out_root.mkdir(parents=True, exist_ok=True)
    latest = {
        "artifact_dir": run_id,
        "trained_at": trained_at,
        "schema_version": args.schema_version,
        "expected_dim": int(args.expected_dim),
    }
    (out_root / "latest.json").write_text(
        json.dumps(latest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(json.dumps({"ok": True, "artifact_dir": str(artifact_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
