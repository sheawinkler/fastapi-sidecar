#!/usr/bin/env python3
"""Export custom JSON surrogate backend from distilled torch checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, List

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _tensor_to_list(value: Any) -> List:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return torch.tensor(value).detach().cpu().tolist()


def main() -> int:
    parser = argparse.ArgumentParser(description="Export custom JSON proxy model")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=PROJECT_ROOT / "models" / "saved" / "proxy" / "ensemble_proxy_surrogate.pt",
        help="Input distilled surrogate checkpoint",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=PROJECT_ROOT / "models" / "saved" / "custom_proxy" / "ensemble_proxy_v1.json",
        help="Output custom export JSON",
    )
    args = parser.parse_args()

    if not args.checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")

    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", {})
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("Checkpoint missing state_dict")

    required = [
        "fc1.weight",
        "fc1.bias",
        "fc2.weight",
        "fc2.bias",
        "fc3.weight",
        "fc3.bias",
    ]
    for key in required:
        if key not in state_dict:
            raise ValueError(f"Checkpoint missing required key: {key}")

    input_dim = int(checkpoint.get("input_dim", 29))
    payload: Dict[str, Any] = {
        "schema_version": "custom_proxy_v1",
        "model_name": "custom_export_proxy",
        "created_at": datetime.now(UTC).isoformat(),
        "source_checkpoint": str(args.checkpoint_path),
        "input_dim": input_dim,
        "layers": [
            {
                "name": "fc1",
                "activation": "relu",
                "weight": _tensor_to_list(state_dict["fc1.weight"]),
                "bias": _tensor_to_list(state_dict["fc1.bias"]),
            },
            {
                "name": "fc2",
                "activation": "relu",
                "weight": _tensor_to_list(state_dict["fc2.weight"]),
                "bias": _tensor_to_list(state_dict["fc2.bias"]),
            },
            {
                "name": "fc3",
                "activation": "none",
                "weight": _tensor_to_list(state_dict["fc3.weight"]),
                "bias": _tensor_to_list(state_dict["fc3.bias"]),
            },
        ],
        "metadata": {
            "distillation_metrics": checkpoint.get("metrics", {}),
            "sample_count": checkpoint.get("sample_count"),
            "feedback_vector_count": checkpoint.get("feedback_vector_count"),
            "teacher": checkpoint.get("teacher", "EnsembleOrchestrator"),
        },
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "output_path": str(args.output_path),
                "source_checkpoint": str(args.checkpoint_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
