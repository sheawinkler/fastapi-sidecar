from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_iso(ts: datetime | None = None) -> str:
    value = ts or _utc_now()
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except Exception:
        return default


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, separators=(",", ":"), ensure_ascii=True))
        fh.write("\n")


def _atomic_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp-sidecar")
    shutil.copy2(src, tmp)
    os.replace(tmp, dest)


def _count_json_array_entries(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        payload = _read_json(path)
    except Exception:
        return 0
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        for key in ("entries", "items", "rows", "data"):
            maybe = payload.get(key)
            if isinstance(maybe, list):
                return len(maybe)
    return 0


def _load_history(path: Path, limit: int = 25) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows[-limit:]


@dataclass(frozen=True)
class PredictiveTrainerConfig:
    algo_repo_dir: Path
    data_dir: Path
    train_interval_secs: int
    scheduler_enabled: bool
    auto_promote: bool
    relaunch_enabled: bool
    python_bin: str
    train_timeout_secs: int
    positive_share_collapse_tolerance: float
    calibration_mae_degradation_factor: float
    p_positive_brier_degradation_factor: float

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "trainer" / "runs"

    @property
    def history_path(self) -> Path:
        return self.data_dir / "trainer" / "history.jsonl"

    @property
    def lock_path(self) -> Path:
        return self.data_dir / "trainer" / "lock" / "train.lock"

    @property
    def model_path(self) -> Path:
        return self.algo_repo_dir / "logs/analysis/predictive_entry/model_latest.json"

    @property
    def training_path(self) -> Path:
        return self.algo_repo_dir / "logs/analysis/predictive_entry/prelaunch_training_latest.json"

    @property
    def calibration_path(self) -> Path:
        return self.algo_repo_dir / "logs/analysis/predictive_entry/calibration_latest.json"

    @property
    def dataset_path(self) -> Path:
        return self.algo_repo_dir / "logs/analysis/monte_carlo/dataset_latest.jsonl"

    @property
    def dataset_summary_path(self) -> Path:
        return self.algo_repo_dir / "logs/analysis/monte_carlo/dataset_latest_summary.json"

    @property
    def shadow_index_path(self) -> Path:
        return self.algo_repo_dir / "logs/index/predictive_candidate_firehose_shadow_outcomes.json"

    @property
    def run_context_dir(self) -> Path:
        return self.algo_repo_dir / "logs/run_context"

    @property
    def log_path(self) -> Path:
        return self.algo_repo_dir / "logs/real_algotrader_continuous.log"

    @property
    def deploy_script_path(self) -> Path:
        return self.algo_repo_dir / "scripts/deploy_live.sh"

    @property
    def train_script_path(self) -> Path:
        return self.algo_repo_dir / "scripts/train_predictive_entry_model.py"

    @property
    def calibration_script_path(self) -> Path:
        return self.algo_repo_dir / "scripts/update_predictive_next_state_ledger.py"

    @property
    def eval_pack_script_path(self) -> Path:
        return self.algo_repo_dir / "scripts/report_model_eval_pack.py"

    @property
    def expectancy_script_path(self) -> Path:
        return self.algo_repo_dir / "scripts/analyze_expectancy_from_log.py"

    @classmethod
    def from_env(cls, base_data_dir: Path) -> "PredictiveTrainerConfig":
        algo_repo_dir = Path(
            os.getenv(
                "SIDECAR_PREDICTIVE_TRAINER_ALGO_REPO_DIR",
                "/Users/sheawinkler/Documents/Projects/algotraderv2_rust",
            )
        ).expanduser()
        data_dir = Path(
            os.getenv("SIDECAR_PREDICTIVE_TRAINER_DATA_DIR", str(base_data_dir))
        ).expanduser()
        return cls(
            algo_repo_dir=algo_repo_dir,
            data_dir=data_dir,
            train_interval_secs=max(
                60, _safe_int(os.getenv("SIDECAR_PREDICTIVE_TRAINER_INTERVAL_SECS"), 600)
            ),
            scheduler_enabled=os.getenv(
                "SIDECAR_PREDICTIVE_TRAINER_ENABLED", "false"
            ).strip().lower()
            == "true",
            auto_promote=os.getenv(
                "SIDECAR_PREDICTIVE_TRAINER_AUTO_PROMOTE", "true"
            ).strip().lower()
            == "true",
            relaunch_enabled=os.getenv(
                "SIDECAR_PREDICTIVE_TRAINER_RELAUNCH_ENABLED", "true"
            ).strip().lower()
            == "true",
            python_bin=str(
                os.getenv("SIDECAR_PREDICTIVE_TRAINER_PYTHON", sys.executable)
            ).strip()
            or sys.executable,
            train_timeout_secs=max(
                300, _safe_int(os.getenv("SIDECAR_PREDICTIVE_TRAINER_TIMEOUT_SECS"), 1800)
            ),
            positive_share_collapse_tolerance=max(
                0.0,
                min(
                    0.5,
                    _safe_float(
                        os.getenv("SIDECAR_PREDICTIVE_TRAINER_POSITIVE_SHARE_TOLERANCE"), 0.05
                    )
                    or 0.05,
                ),
            ),
            calibration_mae_degradation_factor=max(
                1.0,
                _safe_float(
                    os.getenv(
                        "SIDECAR_PREDICTIVE_TRAINER_CALIBRATION_MAE_DEGRADATION_FACTOR"
                    ),
                    1.25,
                )
                or 1.25,
            ),
            p_positive_brier_degradation_factor=max(
                1.0,
                _safe_float(
                    os.getenv(
                        "SIDECAR_PREDICTIVE_TRAINER_P_POSITIVE_BRIER_DEGRADATION_FACTOR"
                    ),
                    1.25,
                )
                or 1.25,
            ),
        )


class PredictiveTrainerManager:
    def __init__(self, config: PredictiveTrainerConfig) -> None:
        self.config = config
        self._lock = asyncio.Lock()
        self._current_task: asyncio.Task | None = None
        self._active_run_id: str | None = None
        self._scheduler_task: asyncio.Task | None = None
        self._pending_restart: dict[str, Any] | None = None

    async def start(self) -> None:
        if not self.config.scheduler_enabled:
            return
        if self._scheduler_task and not self._scheduler_task.done():
            return
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())

    async def shutdown(self) -> None:
        if self._scheduler_task:
            self._scheduler_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._scheduler_task
            self._scheduler_task = None

    async def _scheduler_loop(self) -> None:
        while True:
            try:
                if not self.is_running():
                    await self.start_run(requested_by="scheduler", auto_promote=None)
            except Exception:
                self._append_history(
                    {
                        "timestamp": _utc_iso(),
                        "event": "scheduler_error",
                        "message": "trainer scheduler cycle failed",
                    }
                )
            await asyncio.sleep(self.config.train_interval_secs)

    def is_running(self) -> bool:
        return self._current_task is not None and not self._current_task.done()

    def _append_history(self, payload: dict[str, Any]) -> None:
        _append_jsonl(self.config.history_path, payload)

    def _run_dir(self, run_id: str) -> Path:
        return self.config.runs_dir / run_id

    def _manifest_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "manifest.json"

    def _read_manifest(self, run_id: str) -> dict[str, Any]:
        path = self._manifest_path(run_id)
        if not path.exists():
            raise FileNotFoundError(f"missing manifest for run {run_id}")
        return _read_json(path)

    def _write_manifest(self, run_id: str, payload: dict[str, Any]) -> None:
        _write_json(self._manifest_path(run_id), payload)

    def _latest_run_context_started_at(self) -> str | None:
        files = sorted(
            self.config.run_context_dir.glob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for path in files:
            try:
                payload = _read_json(path)
            except Exception:
                continue
            started_at = str(payload.get("started_at") or "").strip()
            if started_at:
                return started_at
        return None

    def _latest_run_context_path(self) -> str | None:
        files = sorted(
            self.config.run_context_dir.glob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return str(files[0]) if files else None

    def _current_open_positions(self, work_dir: Path) -> dict[str, Any]:
        started_at = self._latest_run_context_started_at()
        if not started_at:
            return {
                "ok": False,
                "reason": "missing_run_context",
                "open_positions_remaining": None,
                "positions_closed_total": None,
                "run_start": None,
            }
        out_path = work_dir / "live_expectancy.json"
        cmd = [
            self.config.python_bin,
            str(self.config.expectancy_script_path),
            "--log",
            str(self.config.log_path),
            "--since",
            started_at,
            "--json-out",
            str(out_path),
        ]
        subprocess.run(
            cmd,
            cwd=self.config.algo_repo_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = _read_json(out_path)
        return {
            "ok": True,
            "reason": "analyzed",
            "open_positions_remaining": _safe_int(payload.get("open_positions_remaining"), 0),
            "positions_closed_total": _safe_int(payload.get("positions_closed_total"), 0),
            "total_net_sol": _safe_float(payload.get("summary", {}).get("total_net_sol")),
            "run_start": started_at,
        }

    def _repo_launch_ready(self) -> dict[str, Any]:
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=self.config.algo_repo_dir,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=self.config.algo_repo_dir,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.config.algo_repo_dir,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        origin_main = subprocess.run(
            ["git", "rev-parse", "origin/main"],
            cwd=self.config.algo_repo_dir,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return {
            "branch": branch,
            "status_clean": not status,
            "head": head,
            "origin_main": origin_main,
            "head_matches_origin_main": bool(head and origin_main and head == origin_main),
        }

    def _active_artifacts(self) -> dict[str, Any]:
        model = _read_json(self.config.model_path) if self.config.model_path.exists() else {}
        training = _read_json(self.config.training_path) if self.config.training_path.exists() else {}
        calibration = (
            _read_json(self.config.calibration_path) if self.config.calibration_path.exists() else {}
        )
        training_section = training.get("training") or {}
        validation_section = training.get("validation") or {}
        data_quality = model.get("data_quality") or {}
        return {
            "model_path": str(self.config.model_path),
            "training_path": str(self.config.training_path),
            "calibration_path": str(self.config.calibration_path),
            "training_rows": _safe_int(
                training_section.get("rows"), _safe_int(data_quality.get("row_count"), 0)
            ),
            "shadow_rows": _safe_int(training_section.get("shadow_rows"), 0),
            "executed_rows": _safe_int(training_section.get("executed_rows"), 0),
            "positive_rows": _safe_int(
                validation_section.get("positive_rows"),
                _safe_int(data_quality.get("positive_net_sol_count"), 0),
            ),
            "negative_rows": _safe_int(
                validation_section.get("negative_rows"),
                _safe_int(data_quality.get("negative_net_sol_count"), 0),
            ),
            "trained_at": validation_section.get("trained_at"),
            "version": training_section.get("version") or model.get("version"),
            "calibration_global_mae_sol": _safe_float(
                model.get("calibration", {}).get("global_mae_sol")
            ),
            "p_positive_after_cost_brier": _safe_float(
                model.get("calibration", {})
                .get("tradeability_head_brier", {})
                .get("p_positive_after_cost")
            ),
            "calibration_rows": _safe_int(calibration.get("rows"), 0),
        }

    def _dataset_summary(self) -> dict[str, Any]:
        if self.config.dataset_summary_path.exists():
            try:
                return _read_json(self.config.dataset_summary_path)
            except Exception:
                pass
        rows = 0
        if self.config.dataset_path.exists():
            with self.config.dataset_path.open("r", encoding="utf-8", errors="ignore") as fh:
                rows = sum(1 for line in fh if line.strip())
        return {
            "rows": rows,
            "ok": False,
            "reason": "dataset_summary_unavailable",
            "unknown_sleeve_ratio": None,
            "summary": str(self.config.dataset_summary_path),
        }

    def _build_attestation(
        self,
        candidate_model_path: Path,
        dataset_summary: dict[str, Any],
        candidate_output_hint: Path,
    ) -> dict[str, Any]:
        model = _read_json(candidate_model_path)
        data_quality = model.get("data_quality") or {}
        executed_quality = model.get("executed_data_quality") or {}
        shadow_quality = model.get("shadow_data_quality") or {}
        training_window = model.get("training_window") or {}
        version = model.get("version")
        return {
            "artifacts": {
                "dataset_path": str(self.config.dataset_path.relative_to(self.config.algo_repo_dir)),
                "dataset_summary_path": str(
                    self.config.dataset_summary_path.relative_to(self.config.algo_repo_dir)
                ),
                "model_path": str(self.config.model_path.relative_to(self.config.algo_repo_dir)),
            },
            "dataset": {
                "ok": bool(dataset_summary.get("ok", False)),
                "quality_gates": dict(dataset_summary.get("quality_gates") or {}),
                "reason": dataset_summary.get("reason"),
                "rows": _safe_int(dataset_summary.get("rows"), 0),
                "summary": str(dataset_summary.get("summary") or self.config.dataset_summary_path),
                "unknown_sleeve_ratio": dataset_summary.get("unknown_sleeve_ratio"),
            },
            "generated_at": _utc_iso(),
            "mode": "trainer_sidecar_auto",
            "sources": {
                "log_path": "logs/real_algotrader_continuous.log",
                "run_context_dir": "logs/run_context",
                "trade_index": "index/trade_outcomes.json",
                "position_index": "index/position_outcomes.json",
                "cycles_path": "logs/auto_tuner/cycles.jsonl",
                "log_glob": "logs/*.log",
            },
            "training": {
                "excluded_invalid_target_rows": dict(
                    training_window.get("excluded_invalid_target_rows")
                    or {"executed": 0, "shadow": 0}
                ),
                "executed_rows": _safe_int(
                    training_window.get("executed_rows"),
                    _safe_int(executed_quality.get("row_count"), 0),
                ),
                "output": str(candidate_output_hint),
                "reason_groups": ["dead_timeout", "trailing_stop", "liquidity_collapse", "stop_loss", "other"],
                "rows": _safe_int(training_window.get("rows"), _safe_int(data_quality.get("row_count"), 0)),
                "shadow_rows": _safe_int(
                    training_window.get("shadow_rows"),
                    _safe_int(shadow_quality.get("row_count"), 0),
                ),
                "version": version,
            },
            "validation": {
                "issues": [],
                "model_version": version,
                "negative_rows": _safe_int(data_quality.get("negative_net_sol_count"), 0),
                "positive_rows": _safe_int(data_quality.get("positive_net_sol_count"), 0),
                "trained_at": model.get("trained_at"),
                "training_rows": _safe_int(data_quality.get("row_count"), 0),
            },
        }

    def _evaluate_candidate(
        self,
        *,
        active: dict[str, Any],
        candidate_model: dict[str, Any],
        candidate_attestation: dict[str, Any],
    ) -> dict[str, Any]:
        issues: list[str] = []
        training = candidate_attestation.get("training") or {}
        validation = candidate_attestation.get("validation") or {}
        candidate_rows = _safe_int(training.get("rows"), 0)
        candidate_shadow_rows = _safe_int(training.get("shadow_rows"), 0)
        candidate_positive = _safe_int(validation.get("positive_rows"), 0)
        candidate_negative = _safe_int(validation.get("negative_rows"), 0)
        candidate_total = max(1, candidate_positive + candidate_negative)
        candidate_positive_share = candidate_positive / float(candidate_total)

        active_positive = _safe_int(active.get("positive_rows"), 0)
        active_negative = _safe_int(active.get("negative_rows"), 0)
        active_total = max(1, active_positive + active_negative)
        active_positive_share = active_positive / float(active_total)

        if candidate_rows <= _safe_int(active.get("training_rows"), 0):
            issues.append("training_rows_not_improved")
        if candidate_shadow_rows <= _safe_int(active.get("shadow_rows"), 0):
            issues.append("shadow_rows_not_improved")
        if candidate_positive_share + self.config.positive_share_collapse_tolerance < active_positive_share:
            issues.append("positive_share_collapsed")

        candidate_calibration = candidate_model.get("calibration", {})
        candidate_mae = _safe_float(candidate_calibration.get("global_mae_sol"))
        active_mae = _safe_float(active.get("calibration_global_mae_sol"))
        if (
            candidate_mae is not None
            and active_mae is not None
            and candidate_mae > active_mae * self.config.calibration_mae_degradation_factor
        ):
            issues.append("global_mae_degraded")

        candidate_brier = _safe_float(
            candidate_calibration.get("tradeability_head_brier", {}).get("p_positive_after_cost")
        )
        active_brier = _safe_float(active.get("p_positive_after_cost_brier"))
        if (
            candidate_brier is not None
            and active_brier is not None
            and candidate_brier > active_brier * self.config.p_positive_brier_degradation_factor
        ):
            issues.append("p_positive_brier_degraded")

        return {
            "ok": not issues,
            "issues": issues,
            "candidate_positive_share": round(candidate_positive_share, 6),
            "active_positive_share": round(active_positive_share, 6),
            "candidate_global_mae_sol": candidate_mae,
            "active_global_mae_sol": active_mae,
            "candidate_p_positive_after_cost_brier": candidate_brier,
            "active_p_positive_after_cost_brier": active_brier,
        }

    async def start_run(
        self,
        *,
        requested_by: str,
        auto_promote: bool | None,
    ) -> dict[str, Any]:
        async with self._lock:
            if self.is_running():
                return {
                    "status": "already_running",
                    "run_id": self._active_run_id,
                }
            run_id = _utc_now().strftime("%Y%m%dT%H%M%SZ") + "-" + os.urandom(4).hex()
            self._active_run_id = run_id
            self._current_task = asyncio.create_task(
                self._run_cycle(run_id=run_id, requested_by=requested_by, auto_promote=auto_promote)
            )
            return {"status": "started", "run_id": run_id}

    async def _run_cycle(self, *, run_id: str, requested_by: str, auto_promote: bool | None) -> None:
        run_dir = self._run_dir(run_id)
        logs_dir = run_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        lock_payload = {
            "run_id": run_id,
            "requested_by": requested_by,
            "started_at": _utc_iso(),
        }
        _write_json(self.config.lock_path, lock_payload)

        manifest: dict[str, Any] = {
            "run_id": run_id,
            "requested_by": requested_by,
            "started_at": lock_payload["started_at"],
            "status": "running",
            "config": {
                "algo_repo_dir": str(self.config.algo_repo_dir),
                "train_interval_secs": self.config.train_interval_secs,
                "scheduler_enabled": self.config.scheduler_enabled,
                "auto_promote": self.config.auto_promote if auto_promote is None else bool(auto_promote),
                "relaunch_enabled": self.config.relaunch_enabled,
                "train_timeout_secs": self.config.train_timeout_secs,
            },
            "paths": {
                "run_dir": str(run_dir),
                "model_candidate": str(run_dir / "model_candidate.json"),
                "training_candidate": str(run_dir / "prelaunch_training_candidate.json"),
                "calibration_candidate": str(run_dir / "calibration_candidate.json"),
                "ledger_candidate": str(run_dir / "next_state_ledger_candidate.jsonl"),
                "eval_pack_json": str(run_dir / "model_eval_pack.json"),
                "eval_pack_md": str(run_dir / "model_eval_pack.md"),
            },
            "artifacts": {},
            "promotion": {
                "state": "not_attempted",
                "auto_promote_requested": self.config.auto_promote if auto_promote is None else bool(auto_promote),
            },
            "raw_shadow_entry_count": _count_json_array_entries(self.config.shadow_index_path),
            "active_model": self._active_artifacts(),
        }
        self._write_manifest(run_id, manifest)

        try:
            dataset_summary = self._dataset_summary()
            model_candidate = run_dir / "model_candidate.json"
            training_candidate = run_dir / "prelaunch_training_candidate.json"
            calibration_candidate = run_dir / "calibration_candidate.json"
            ledger_candidate = run_dir / "next_state_ledger_candidate.jsonl"
            eval_pack_json = run_dir / "model_eval_pack.json"
            eval_pack_md = run_dir / "model_eval_pack.md"

            train_cmd = [
                self.config.python_bin,
                str(self.config.train_script_path),
                "--input",
                str(self.config.dataset_path),
                "--output",
                str(model_candidate),
                "--shadow-index",
                str(self.config.shadow_index_path),
            ]
            completed = subprocess.run(
                train_cmd,
                cwd=self.config.algo_repo_dir,
                check=True,
                capture_output=True,
                text=True,
                timeout=self.config.train_timeout_secs,
            )
            (logs_dir / "train.stdout.log").write_text(completed.stdout, encoding="utf-8")
            (logs_dir / "train.stderr.log").write_text(completed.stderr, encoding="utf-8")

            candidate_attestation = self._build_attestation(
                model_candidate, dataset_summary, model_candidate
            )
            _write_json(training_candidate, candidate_attestation)

            calibration_cmd = [
                self.config.python_bin,
                str(self.config.calibration_script_path),
                "--dataset",
                str(self.config.dataset_path),
                "--ledger",
                str(ledger_candidate),
                "--snapshot",
                str(calibration_candidate),
            ]
            calibration_completed = subprocess.run(
                calibration_cmd,
                cwd=self.config.algo_repo_dir,
                check=True,
                capture_output=True,
                text=True,
            )
            (logs_dir / "calibration.stdout.log").write_text(
                calibration_completed.stdout, encoding="utf-8"
            )
            (logs_dir / "calibration.stderr.log").write_text(
                calibration_completed.stderr, encoding="utf-8"
            )

            eval_cmd = [
                self.config.python_bin,
                str(self.config.eval_pack_script_path),
                "--model-artifact",
                str(model_candidate),
                "--training-artifact",
                str(training_candidate),
                "--calibration-artifact",
                str(calibration_candidate),
                "--out-json",
                str(eval_pack_json),
                "--out-md",
                str(eval_pack_md),
            ]
            eval_completed = subprocess.run(
                eval_cmd,
                cwd=self.config.algo_repo_dir,
                check=True,
                capture_output=True,
                text=True,
            )
            (logs_dir / "eval.stdout.log").write_text(eval_completed.stdout, encoding="utf-8")
            (logs_dir / "eval.stderr.log").write_text(eval_completed.stderr, encoding="utf-8")

            candidate_model = _read_json(model_candidate)
            gate_result = self._evaluate_candidate(
                active=manifest["active_model"],
                candidate_model=candidate_model,
                candidate_attestation=candidate_attestation,
            )
            candidate_eval_pack = _read_json(eval_pack_json)

            manifest["status"] = "completed"
            manifest["completed_at"] = _utc_iso()
            manifest["dataset"] = dataset_summary
            manifest["artifacts"] = {
                "candidate_model": str(model_candidate),
                "candidate_training": str(training_candidate),
                "candidate_calibration": str(calibration_candidate),
                "candidate_ledger": str(ledger_candidate),
                "candidate_eval_pack_json": str(eval_pack_json),
                "candidate_eval_pack_md": str(eval_pack_md),
            }
            manifest["candidate_model"] = {
                "training_rows": _safe_int(candidate_attestation.get("training", {}).get("rows"), 0),
                "executed_rows": _safe_int(
                    candidate_attestation.get("training", {}).get("executed_rows"), 0
                ),
                "shadow_rows": _safe_int(
                    candidate_attestation.get("training", {}).get("shadow_rows"), 0
                ),
                "positive_rows": _safe_int(
                    candidate_attestation.get("validation", {}).get("positive_rows"), 0
                ),
                "negative_rows": _safe_int(
                    candidate_attestation.get("validation", {}).get("negative_rows"), 0
                ),
                "trained_at": candidate_attestation.get("validation", {}).get("trained_at"),
                "version": candidate_attestation.get("training", {}).get("version"),
                "calibration_rows": _safe_int(
                    _read_json(calibration_candidate).get("rows"), 0
                )
                if calibration_candidate.exists()
                else 0,
            }
            manifest["evaluation_pack"] = {
                "window": candidate_eval_pack.get("window"),
                "calibration_and_sample_sufficiency": candidate_eval_pack.get(
                    "calibration_and_sample_sufficiency"
                ),
                "decision_gaps": candidate_eval_pack.get("decision_gaps"),
            }
            manifest["promotion"]["gate_result"] = gate_result

            should_promote = bool(
                (self.config.auto_promote if auto_promote is None else auto_promote)
                and gate_result.get("ok")
            )
            if should_promote:
                promotion = self._promote_candidate(
                    run_id=run_id,
                    model_candidate=model_candidate,
                    training_candidate=training_candidate,
                    calibration_candidate=calibration_candidate,
                    forced=False,
                )
                manifest["promotion"] = {
                    **manifest["promotion"],
                    **promotion,
                }
            else:
                manifest["promotion"]["state"] = (
                    "gated_off" if not gate_result.get("ok") else "auto_promote_disabled"
                )
            self._write_manifest(run_id, manifest)
            self._append_history(
                {
                    "timestamp": _utc_iso(),
                    "event": "trainer_run_completed",
                    "run_id": run_id,
                    "status": manifest["status"],
                    "promotion_state": manifest["promotion"].get("state"),
                    "training_rows": manifest["candidate_model"].get("training_rows"),
                    "shadow_rows": manifest["candidate_model"].get("shadow_rows"),
                }
            )
        except Exception as exc:
            manifest["status"] = "failed"
            manifest["completed_at"] = _utc_iso()
            manifest["error"] = {
                "type": exc.__class__.__name__,
                "message": str(exc),
            }
            self._write_manifest(run_id, manifest)
            self._append_history(
                {
                    "timestamp": _utc_iso(),
                    "event": "trainer_run_failed",
                    "run_id": run_id,
                    "message": str(exc),
                }
            )
        finally:
            with contextlib.suppress(FileNotFoundError):
                self.config.lock_path.unlink()
            self._active_run_id = None
            self._current_task = None

    def _promote_candidate(
        self,
        *,
        run_id: str,
        model_candidate: Path,
        training_candidate: Path,
        calibration_candidate: Path,
        forced: bool,
    ) -> dict[str, Any]:
        _atomic_copy(model_candidate, self.config.model_path)
        _atomic_copy(training_candidate, self.config.training_path)
        _atomic_copy(calibration_candidate, self.config.calibration_path)

        live_state = self._current_open_positions(self._run_dir(run_id))
        repo_state = self._repo_launch_ready()
        promotion: dict[str, Any] = {
            "state": "promoted",
            "forced": forced,
            "promoted_at": _utc_iso(),
            "live_state": live_state,
            "repo_state": repo_state,
        }

        if not self.config.relaunch_enabled:
            promotion["state"] = "promoted_no_restart_disabled"
            return promotion

        open_positions = live_state.get("open_positions_remaining")
        if open_positions != 0:
            self._pending_restart = {
                "run_id": run_id,
                "reason": "open_positions_remaining",
                "open_positions_remaining": open_positions,
            }
            promotion["state"] = "promoted_pending_restart"
            return promotion

        if not repo_state.get("status_clean") or repo_state.get("branch") != "main" or not repo_state.get(
            "head_matches_origin_main"
        ):
            self._pending_restart = {
                "run_id": run_id,
                "reason": "repo_not_launch_ready",
                "repo_state": repo_state,
            }
            promotion["state"] = "promoted_pending_restart"
            return promotion

        restart_log = self._run_dir(run_id) / "logs" / "restart.stdout.log"
        restart_err = self._run_dir(run_id) / "logs" / "restart.stderr.log"
        try:
            deploy_completed = subprocess.run(
                [str(self.config.deploy_script_path), "--live", "--skip-build"],
                cwd=self.config.algo_repo_dir,
                check=True,
                capture_output=True,
                text=True,
            )
            restart_log.write_text(deploy_completed.stdout, encoding="utf-8")
            restart_err.write_text(deploy_completed.stderr, encoding="utf-8")
            promotion["state"] = "promoted_and_relaunched"
            self._pending_restart = None
        except subprocess.CalledProcessError as exc:
            restart_log.write_text(exc.stdout or "", encoding="utf-8")
            restart_err.write_text(exc.stderr or "", encoding="utf-8")
            promotion["state"] = "promoted_restart_failed"
            promotion["restart_error"] = str(exc)
            self._pending_restart = {
                "run_id": run_id,
                "reason": "restart_failed",
                "error": str(exc),
            }
        promotion["restart_log"] = str(restart_log)
        promotion["restart_err_log"] = str(restart_err)
        return promotion

    async def promote_run(self, run_id: str, *, forced: bool = True) -> dict[str, Any]:
        manifest = self._read_manifest(run_id)
        artifacts = manifest.get("artifacts") or {}
        model_candidate = Path(str(artifacts.get("candidate_model") or ""))
        training_candidate = Path(str(artifacts.get("candidate_training") or ""))
        calibration_candidate = Path(str(artifacts.get("candidate_calibration") or ""))
        if not model_candidate.exists() or not training_candidate.exists() or not calibration_candidate.exists():
            raise FileNotFoundError(f"run {run_id} is missing candidate artifacts")
        promotion = await asyncio.to_thread(
            self._promote_candidate,
            run_id=run_id,
            model_candidate=model_candidate,
            training_candidate=training_candidate,
            calibration_candidate=calibration_candidate,
            forced=forced,
        )
        manifest["promotion"] = {
            **(manifest.get("promotion") or {}),
            **promotion,
        }
        self._write_manifest(run_id, manifest)
        self._append_history(
            {
                "timestamp": _utc_iso(),
                "event": "trainer_run_promoted",
                "run_id": run_id,
                "state": promotion.get("state"),
                "forced": forced,
            }
        )
        return promotion

    def health_payload(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "scheduler_enabled": self.config.scheduler_enabled,
            "auto_promote": self.config.auto_promote,
            "relaunch_enabled": self.config.relaunch_enabled,
            "is_running": self.is_running(),
            "algo_repo_dir": str(self.config.algo_repo_dir),
            "algo_repo_exists": self.config.algo_repo_dir.exists(),
            "active_model_path": str(self.config.model_path),
            "active_training_path": str(self.config.training_path),
            "active_calibration_path": str(self.config.calibration_path),
            "shadow_index_path": str(self.config.shadow_index_path),
            "train_interval_secs": self.config.train_interval_secs,
            "train_timeout_secs": self.config.train_timeout_secs,
            "pending_restart": self._pending_restart,
        }

    def status_payload(self) -> dict[str, Any]:
        manifest = self._read_manifest(self._active_run_id) if self._active_run_id else None
        return {
            "status": "running" if self.is_running() else "idle",
            "active_run_id": self._active_run_id,
            "active_run": manifest,
            "pending_restart": self._pending_restart,
            "active_model": self._active_artifacts(),
            "latest_run_context": self._latest_run_context_path(),
        }

    def history_payload(self, limit: int = 25) -> dict[str, Any]:
        return {
            "history": _load_history(self.config.history_path, limit=limit),
            "pending_restart": self._pending_restart,
        }

    def active_model_payload(self) -> dict[str, Any]:
        return self._active_artifacts()

    def candidate_model_payload(self, run_id: str) -> dict[str, Any]:
        manifest = self._read_manifest(run_id)
        return {
            "run_id": run_id,
            "status": manifest.get("status"),
            "candidate_model": manifest.get("candidate_model"),
            "promotion": manifest.get("promotion"),
            "dataset": manifest.get("dataset"),
            "evaluation_pack": manifest.get("evaluation_pack"),
        }
