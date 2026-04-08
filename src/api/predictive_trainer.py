from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any


PREDICTIVE_CANDIDATE_FIREHOSE_TABLE = "predictive_candidate_firehose_shadow_outcomes"
PREDICTIVE_CANDIDATE_FIREHOSE_STATS_TABLE = (
    "predictive_candidate_firehose_shadow_outcomes_stats"
)


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


def _parse_utc_ts(raw: str | None) -> datetime | None:
    if not raw:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(
        f"{path.name}.tmp-{os.getpid()}-{threading.get_ident()}-{os.urandom(4).hex()}"
    )
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()


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


def _path_exists_text(path: Path) -> bool:
    try:
        return path.exists()
    except Exception:
        return False


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


def _default_shadow_index_stats_cache() -> dict[str, Any]:
    return {
        "current_shadow_entry_count": 0,
        "shadow_index_size_bytes": 0,
        "shadow_index_mtime_ns": 0,
        "shadow_index_count_refreshed_at": None,
        "shadow_index_count_error": None,
        "shadow_index_count_cached": True,
        "shadow_index_source": "sqlite_stats",
        "shadow_index_last_seq": 0,
        "shadow_index_last_write_at": None,
    }


def _read_shadow_sqlite_stats(path: Path) -> dict[str, Any]:
    payload = _default_shadow_index_stats_cache()
    if not path.exists():
        payload["shadow_index_count_error"] = "sqlite_missing"
        return payload
    try:
        stat_result = path.stat()
        payload["shadow_index_size_bytes"] = int(stat_result.st_size)
        payload["shadow_index_mtime_ns"] = int(getattr(stat_result, "st_mtime_ns", 0))
    except Exception:
        pass
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(
            f"""
            CREATE TABLE IF NOT EXISTS {PREDICTIVE_CANDIDATE_FIREHOSE_STATS_TABLE} (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                entry_count INTEGER NOT NULL DEFAULT 0,
                last_seq INTEGER NOT NULL DEFAULT 0,
                last_write_at TEXT
            );
            INSERT OR IGNORE INTO {PREDICTIVE_CANDIDATE_FIREHOSE_STATS_TABLE}
                (singleton, entry_count, last_seq, last_write_at)
            VALUES (1, 0, 0, NULL);
            """
        )
        row = conn.execute(
            f"""
            SELECT entry_count, last_seq, last_write_at
            FROM {PREDICTIVE_CANDIDATE_FIREHOSE_STATS_TABLE}
            WHERE singleton = 1
            """
        ).fetchone()
        if row:
            payload["current_shadow_entry_count"] = _safe_int(row[0], 0)
            payload["shadow_index_last_seq"] = _safe_int(row[1], 0)
            payload["shadow_index_last_write_at"] = row[2]
        payload["shadow_index_count_refreshed_at"] = _utc_iso()
        payload["shadow_index_count_error"] = None
        return payload
    except Exception as exc:
        payload["shadow_index_count_error"] = str(exc)
        payload["shadow_index_count_refreshed_at"] = _utc_iso()
        return payload
    finally:
        conn.close()


def _path_text(path: Path, relative_to: Path | None = None) -> str:
    if relative_to is not None:
        try:
            return str(path.relative_to(relative_to))
        except Exception:
            pass
    return str(path)


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
    scheduler_poll_secs: int
    scheduler_enabled: bool
    scheduler_config_source: str
    scheduler_disabled_reason: str | None
    auto_promote: bool
    relaunch_enabled: bool
    python_bin: str
    train_timeout_secs: int
    min_new_shadow_rows_to_trigger: int
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
    def scheduler_state_path(self) -> Path:
        return self.data_dir / "trainer" / "scheduler_state.json"

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
    def next_state_ledger_path(self) -> Path:
        return self.algo_repo_dir / "logs/analysis/predictive_entry/next_state_ledger.jsonl"

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
    def shadow_sqlite_path(self) -> Path:
        return self.algo_repo_dir / "logs/index/predictive_candidate_firehose_shadow_outcomes.sqlite3"

    @property
    def shadow_duckdb_path(self) -> Path:
        return self.algo_repo_dir / "logs/index/predictive_candidate_firehose_shadow_outcomes.duckdb"

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
    def wallet_path(self) -> Path:
        return self.algo_repo_dir / "wallet_mainnet.json"

    @property
    def build_dataset_script_path(self) -> Path:
        return self.algo_repo_dir / "scripts/build_mc_dataset.py"

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
        ).expanduser().resolve()
        data_dir = Path(
            os.getenv("SIDECAR_PREDICTIVE_TRAINER_DATA_DIR", str(base_data_dir))
        ).expanduser()
        if not data_dir.is_absolute():
            data_dir = (Path.cwd() / data_dir).resolve()
        else:
            data_dir = data_dir.resolve()
        scheduler_env_raw = os.getenv("SIDECAR_PREDICTIVE_TRAINER_ENABLED")
        scheduler_enabled = _env_bool("SIDECAR_PREDICTIVE_TRAINER_ENABLED", True)
        guidance_enabled = _env_bool("SIDECAR_GUIDANCE", True)
        if scheduler_env_raw is None:
            scheduler_config_source = "default_enabled"
        else:
            scheduler_config_source = "env_enabled" if scheduler_enabled else "env_disabled"
        scheduler_disabled_reason = None
        if not scheduler_enabled:
            if guidance_enabled:
                scheduler_disabled_reason = (
                    "scheduler_explicitly_disabled_while_guidance_enabled"
                )
            else:
                scheduler_disabled_reason = "scheduler_explicitly_disabled"
        return cls(
            algo_repo_dir=algo_repo_dir,
            data_dir=data_dir,
            train_interval_secs=max(
                60, _safe_int(os.getenv("SIDECAR_PREDICTIVE_TRAINER_INTERVAL_SECS"), 600)
            ),
            scheduler_poll_secs=max(
                15, _safe_int(os.getenv("SIDECAR_PREDICTIVE_TRAINER_POLL_SECS"), 30)
            ),
            scheduler_enabled=scheduler_enabled,
            scheduler_config_source=scheduler_config_source,
            scheduler_disabled_reason=scheduler_disabled_reason,
            auto_promote=_env_bool("SIDECAR_PREDICTIVE_TRAINER_AUTO_PROMOTE", True),
            relaunch_enabled=_env_bool("SIDECAR_PREDICTIVE_TRAINER_RELAUNCH_ENABLED", True),
            python_bin=str(
                os.getenv("SIDECAR_PREDICTIVE_TRAINER_PYTHON", sys.executable)
            ).strip()
            or sys.executable,
            train_timeout_secs=max(
                300, _safe_int(os.getenv("SIDECAR_PREDICTIVE_TRAINER_TIMEOUT_SECS"), 1800)
            ),
            min_new_shadow_rows_to_trigger=max(
                1,
                _safe_int(
                    os.getenv("SIDECAR_PREDICTIVE_TRAINER_MIN_NEW_SHADOW_ROWS"), 100
                ),
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
        self._state_io_lock = threading.RLock()
        self._current_task: asyncio.Task | None = None
        self._active_run_id: str | None = None
        self._scheduler_task: asyncio.Task | None = None
        self._pending_restart: dict[str, Any] | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sidecar-trainer")
        self._shadow_index_stats_cache = self._bootstrap_shadow_index_stats_cache()
        if self.config.scheduler_disabled_reason:
            print(
                "WARNING: predictive trainer scheduler disabled "
                f"(reason={self.config.scheduler_disabled_reason}, "
                f"source={self.config.scheduler_config_source})",
                file=sys.stderr,
                flush=True,
            )

    def _bootstrap_shadow_index_stats_cache(self) -> dict[str, Any]:
        state_path = self.config.scheduler_state_path
        if state_path.exists():
            try:
                payload = _read_json(state_path)
            except Exception:
                payload = None
            if isinstance(payload, dict):
                cached = {
                    **_default_shadow_index_stats_cache(),
                    "current_shadow_entry_count": _safe_int(
                        payload.get("current_shadow_entry_count"), 0
                    ),
                    "shadow_index_size_bytes": _safe_int(
                        payload.get("shadow_index_size_bytes"), 0
                    ),
                    "shadow_index_mtime_ns": _safe_int(
                        payload.get("shadow_index_mtime_ns"), 0
                    ),
                    "shadow_index_count_refreshed_at": payload.get(
                        "shadow_index_count_refreshed_at"
                    ),
                    "shadow_index_count_error": payload.get("shadow_index_count_error"),
                    "shadow_index_count_cached": bool(
                        payload.get("shadow_index_count_cached", True)
                    ),
                    "shadow_index_source": payload.get("shadow_index_source")
                    or "sqlite_stats",
                    "shadow_index_last_seq": _safe_int(
                        payload.get("shadow_index_last_seq"), 0
                    ),
                    "shadow_index_last_write_at": payload.get(
                        "shadow_index_last_write_at"
                    ),
                }
                if cached["shadow_index_count_refreshed_at"] is not None or cached[
                    "current_shadow_entry_count"
                ] > 0:
                    return cached
        return _read_shadow_sqlite_stats(self.config.shadow_sqlite_path)

    def _persist_shadow_index_stats_cache(self) -> None:
        state = self._read_scheduler_state()
        state.update(self._shadow_index_stats_cache)
        _write_json(self.config.scheduler_state_path, state)

    def _refresh_shadow_index_stats_cache_sync(self) -> dict[str, Any]:
        stats = _read_shadow_sqlite_stats(self.config.shadow_sqlite_path)
        self._shadow_index_stats_cache = stats
        self._persist_shadow_index_stats_cache()
        return dict(stats)

    async def _refresh_shadow_index_stats_cache(self) -> dict[str, Any]:
        return await self._run_sync_on_executor(self._refresh_shadow_index_stats_cache_sync)

    async def start(self) -> None:
        if not self.config.scheduler_enabled:
            return
        if self._scheduler_task and not self._scheduler_task.done():
            return
        await self._refresh_shadow_index_stats_cache()
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())

    async def shutdown(self) -> None:
        if self._scheduler_task:
            self._scheduler_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._scheduler_task
            self._scheduler_task = None
        if self._current_task:
            active_manifest = self._read_manifest_if_exists(self._active_run_id)
            if (
                self._active_run_id
                and active_manifest is not None
                and not self._terminal_manifest_status(active_manifest.get("status"))
            ):
                active_manifest["status"] = "cancelled"
                active_manifest["stage"] = "cancelled"
                active_manifest["completed_at"] = _utc_iso()
                active_manifest["stage_updated_at"] = active_manifest["completed_at"]
                active_manifest["stage_message"] = (
                    "sidecar shutdown cancelled in-progress trainer run"
                )
                active_manifest = self._update_manifest_stage(
                    self._active_run_id,
                    active_manifest,
                    stage="cancelled",
                    stage_message="sidecar shutdown cancelled in-progress trainer run",
                    dataset_rows=active_manifest.get("dataset_rows"),
                    dataset_summary_path=Path(active_manifest["dataset_summary_path"])
                    if active_manifest.get("dataset_summary_path")
                    else None,
                )
                active_manifest["error"] = {
                    "type": "TrainerShutdown",
                    "message": "sidecar shutdown cancelled in-progress trainer run",
                }
                self._write_manifest(self._active_run_id, active_manifest)
                self._append_history(
                    {
                        "timestamp": _utc_iso(),
                        "event": "trainer_run_cancelled_shutdown",
                        "run_id": self._active_run_id,
                    }
                )
            self._current_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._current_task
            self._current_task = None
        self._executor.shutdown(wait=False, cancel_futures=True)

    async def _scheduler_loop(self) -> None:
        while True:
            try:
                await self._refresh_shadow_index_stats_cache()
                trigger = self._scheduler_trigger_payload()
                if trigger.get("should_start"):
                    await self.start_run(
                        requested_by=str(trigger.get("requested_by") or "scheduler_interval"),
                        auto_promote=None,
                    )
            except Exception:
                self._append_history(
                    {
                        "timestamp": _utc_iso(),
                        "event": "scheduler_error",
                        "message": "trainer scheduler cycle failed",
                    }
                )
            await asyncio.sleep(self.config.scheduler_poll_secs)

    def is_running(self) -> bool:
        return self._current_task is not None and not self._current_task.done()

    def _terminal_manifest_status(self, status: Any) -> bool:
        return str(status or "").strip().lower() in {"completed", "failed", "cancelled"}

    def _read_manifest_if_exists(self, run_id: str | None) -> dict[str, Any] | None:
        if not run_id:
            return None
        path = self._manifest_path(run_id)
        with self._state_io_lock:
            if not path.exists():
                return None
            try:
                payload = _read_json(path)
            except Exception:
                return None
        return payload if isinstance(payload, dict) else None

    def _reconcile_run_state(self) -> dict[str, Any]:
        reconciliation = {
            "stale_running_state": False,
            "stale_reason": None,
        }

        if self._current_task is not None and self._current_task.done():
            self._current_task = None

        active_manifest = self._read_manifest_if_exists(self._active_run_id)
        if (
            self._current_task is not None
            and not self._current_task.done()
            and active_manifest is not None
            and self._terminal_manifest_status(active_manifest.get("status"))
            and not self.config.lock_path.exists()
        ):
            self._current_task = None
            self._active_run_id = None
            reconciliation["stale_running_state"] = True
            reconciliation["stale_reason"] = "terminal_manifest_without_lock"

        latest_manifest = self._latest_manifest_summary()
        if (
            self._current_task is None
            and latest_manifest is not None
            and str(latest_manifest.get("status") or "").strip().lower() == "running"
            and not self.config.lock_path.exists()
        ):
            latest_run_id = str(latest_manifest.get("run_id") or "").strip()
            latest_manifest["status"] = "failed"
            latest_manifest["completed_at"] = _utc_iso()
            latest_manifest["stage"] = "failed"
            latest_manifest["stage_updated_at"] = latest_manifest["completed_at"]
            latest_manifest["stage_message"] = "reconciled running manifest without live task or lock"
            latest_manifest["error"] = {
                "type": "StaleRunState",
                "message": "reconciled running manifest without live task or lock",
            }
            if latest_run_id:
                self._write_manifest(latest_run_id, latest_manifest)
            self._append_history(
                {
                    "timestamp": _utc_iso(),
                    "event": "trainer_run_reconciled_stale_state",
                    "run_id": latest_run_id or None,
                    "reason": "running_manifest_without_lock",
                }
            )
            if self._active_run_id == latest_run_id:
                self._active_run_id = None
            reconciliation["stale_running_state"] = True
            reconciliation["stale_reason"] = "running_manifest_without_lock"

        return reconciliation

    def _append_history(self, payload: dict[str, Any]) -> None:
        with self._state_io_lock:
            _append_jsonl(self.config.history_path, payload)

    async def _run_sync_on_executor(self, func: Any, /, *args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, partial(func, *args, **kwargs))

    def _run_dir(self, run_id: str) -> Path:
        return self.config.runs_dir / run_id

    def _manifest_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "manifest.json"

    def _read_manifest(self, run_id: str) -> dict[str, Any]:
        path = self._manifest_path(run_id)
        with self._state_io_lock:
            if not path.exists():
                raise FileNotFoundError(f"missing manifest for run {run_id}")
            return _read_json(path)

    def _write_manifest(self, run_id: str, payload: dict[str, Any]) -> None:
        with self._state_io_lock:
            _write_json(self._manifest_path(run_id), payload)

    def _run_log_paths(self, run_id: str) -> dict[str, str]:
        logs_dir = self._run_dir(run_id) / "logs"
        return {
            "dataset_stdout": str(logs_dir / "dataset.stdout.log"),
            "dataset_stderr": str(logs_dir / "dataset.stderr.log"),
            "train_stdout": str(logs_dir / "train.stdout.log"),
            "train_stderr": str(logs_dir / "train.stderr.log"),
            "calibration_stdout": str(logs_dir / "calibration.stdout.log"),
            "calibration_stderr": str(logs_dir / "calibration.stderr.log"),
            "eval_stdout": str(logs_dir / "eval.stdout.log"),
            "eval_stderr": str(logs_dir / "eval.stderr.log"),
            "restart_stdout": str(logs_dir / "restart.stdout.log"),
            "restart_stderr": str(logs_dir / "restart.stderr.log"),
        }

    def _run_artifact_paths(self, run_id: str) -> dict[str, Path]:
        run_dir = self._run_dir(run_id)
        return {
            "model_candidate": run_dir / "model_candidate.json",
            "training_candidate": run_dir / "prelaunch_training_candidate.json",
            "calibration_candidate": run_dir / "calibration_candidate.json",
            "ledger_candidate": run_dir / "next_state_ledger_candidate.jsonl",
            "eval_pack_json": run_dir / "model_eval_pack.json",
            "eval_pack_md": run_dir / "model_eval_pack.md",
            "promotion_record": run_dir / "promotion.json",
        }

    def _build_run_manifest(
        self,
        *,
        run_id: str,
        requested_by: str,
        auto_promote: bool | None,
        started_at: str,
    ) -> dict[str, Any]:
        artifact_paths = self._run_artifact_paths(run_id)
        return {
            "run_id": run_id,
            "requested_by": requested_by,
            "started_at": started_at,
            "status": "running",
            "config": {
                "algo_repo_dir": str(self.config.algo_repo_dir),
                "train_interval_secs": self.config.train_interval_secs,
                "scheduler_poll_secs": self.config.scheduler_poll_secs,
                "scheduler_enabled": self.config.scheduler_enabled,
                "auto_promote": self.config.auto_promote if auto_promote is None else bool(auto_promote),
                "relaunch_enabled": self.config.relaunch_enabled,
                "train_timeout_secs": self.config.train_timeout_secs,
                "min_new_shadow_rows_to_trigger": self.config.min_new_shadow_rows_to_trigger,
            },
            "paths": {
                "run_dir": str(self._run_dir(run_id)),
                "model_candidate": str(artifact_paths["model_candidate"]),
                "training_candidate": str(artifact_paths["training_candidate"]),
                "calibration_candidate": str(artifact_paths["calibration_candidate"]),
                "ledger_candidate": str(artifact_paths["ledger_candidate"]),
                "eval_pack_json": str(artifact_paths["eval_pack_json"]),
                "eval_pack_md": str(artifact_paths["eval_pack_md"]),
                "promotion_record": str(artifact_paths["promotion_record"]),
            },
            "artifacts": {},
            "promotion": {
                "state": "not_attempted",
                "auto_promote_requested": self.config.auto_promote if auto_promote is None else bool(auto_promote),
            },
            "raw_shadow_entry_count": _safe_int(
                self._shadow_index_stats_cache.get("current_shadow_entry_count"), 0
            ),
            "active_model": self._active_artifacts(),
        }

    def _seed_run_state(
        self,
        *,
        run_id: str,
        requested_by: str,
        auto_promote: bool | None,
    ) -> None:
        started_at = _utc_iso()
        run_dir = self._run_dir(run_id)
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        manifest = self._build_run_manifest(
            run_id=run_id,
            requested_by=requested_by,
            auto_promote=auto_promote,
            started_at=started_at,
        )
        self._update_manifest_stage(
            run_id,
            manifest,
            stage="dataset",
            stage_message="queued for dataset build",
        )
        self._write_manifest(run_id, manifest)
        with self._state_io_lock:
            _write_json(
                self.config.lock_path,
                {
                    "run_id": run_id,
                    "requested_by": requested_by,
                    "started_at": started_at,
                },
            )

    def _artifact_existence_flags(self, run_id: str) -> dict[str, bool]:
        paths = self._run_artifact_paths(run_id)
        return {
            "model_candidate_exists": _path_exists_text(paths["model_candidate"]),
            "training_candidate_exists": _path_exists_text(paths["training_candidate"]),
            "calibration_candidate_exists": _path_exists_text(paths["calibration_candidate"]),
            "eval_pack_exists": _path_exists_text(paths["eval_pack_json"]),
            "promotion_record_exists": _path_exists_text(paths["promotion_record"]),
        }

    def _update_manifest_stage(
        self,
        run_id: str,
        manifest: dict[str, Any],
        *,
        stage: str,
        stage_message: str,
        dataset_rows: int | None = None,
        dataset_summary_path: Path | None = None,
    ) -> dict[str, Any]:
        now = _utc_iso()
        if str(manifest.get("stage") or "").strip() != stage:
            manifest["stage"] = stage
            manifest["stage_started_at"] = now
        else:
            manifest.setdefault("stage_started_at", now)
        manifest["stage_updated_at"] = now
        manifest["stage_message"] = stage_message
        if dataset_rows is not None:
            manifest["dataset_rows"] = _safe_int(dataset_rows, 0)
        if dataset_summary_path is not None:
            manifest["dataset_summary_path"] = str(dataset_summary_path)
        manifest["log_paths"] = self._run_log_paths(run_id)
        manifest.update(self._artifact_existence_flags(run_id))
        return manifest

    def _manifest_for_status(self, run_id: str, manifest: dict[str, Any] | None) -> dict[str, Any] | None:
        if manifest is None:
            return None
        payload = dict(manifest)
        payload.setdefault("log_paths", self._run_log_paths(run_id))
        payload.update(self._artifact_existence_flags(run_id))
        return payload

    def _run_logged_subprocess(
        self,
        *,
        cmd: list[str],
        cwd: Path,
        stdout_path: Path,
        stderr_path: Path,
        timeout: int | None = None,
    ) -> tuple[int, str, str]:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        with stdout_path.open("w", encoding="utf-8", buffering=1) as stdout_fh, stderr_path.open(
            "w", encoding="utf-8", buffering=1
        ) as stderr_fh:
            completed = subprocess.run(
                cmd,
                cwd=cwd,
                check=False,
                stdout=stdout_fh,
                stderr=stderr_fh,
                text=True,
                timeout=timeout,
                env=env,
            )
        stdout_text = stdout_path.read_text(encoding="utf-8") if stdout_path.exists() else ""
        stderr_text = stderr_path.read_text(encoding="utf-8") if stderr_path.exists() else ""
        return completed.returncode, stdout_text, stderr_text

    def _python_cmd(self, script_path: Path, *args: str) -> list[str]:
        return [self.config.python_bin, "-u", str(script_path), *args]

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

    def _latest_manifest_summary(self, *, status: str | None = None) -> dict[str, Any] | None:
        with self._state_io_lock:
            manifests = sorted(
                self.config.runs_dir.glob("*/manifest.json"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            for path in manifests:
                try:
                    payload = _read_json(path)
                except Exception:
                    continue
                if not isinstance(payload, dict):
                    continue
                if status is not None and str(payload.get("status") or "").strip().lower() != status:
                    continue
                return payload
        return None

    def _default_scheduler_state(self) -> dict[str, Any]:
        shadow_stats = _read_shadow_sqlite_stats(self.config.shadow_sqlite_path)
        current_shadow_entries = _safe_int(shadow_stats.get("current_shadow_entry_count"), 0)
        latest_manifest = self._latest_manifest_summary()
        if latest_manifest:
            return {
                "last_run_id": latest_manifest.get("run_id"),
                "last_requested_by": latest_manifest.get("requested_by"),
                "last_run_started_at": latest_manifest.get("started_at") or _utc_iso(),
                "last_trigger_reason": "bootstrap_latest_manifest",
                "last_trigger_shadow_entry_count": _safe_int(
                    latest_manifest.get("raw_shadow_entry_count"), current_shadow_entries
                ),
                **shadow_stats,
            }
        return {
            "last_run_id": None,
            "last_requested_by": None,
            "last_run_started_at": _utc_iso(),
            "last_trigger_reason": "bootstrap_current_shadow_index",
            "last_trigger_shadow_entry_count": current_shadow_entries,
            **shadow_stats,
        }

    def _read_scheduler_state(self) -> dict[str, Any]:
        path = self.config.scheduler_state_path
        with self._state_io_lock:
            if path.exists():
                try:
                    payload = _read_json(path)
                except Exception:
                    payload = self._default_scheduler_state()
                    _write_json(path, payload)
                    return payload
                if isinstance(payload, dict):
                    return payload
            payload = self._default_scheduler_state()
            _write_json(path, payload)
            return payload

    def _write_scheduler_state(self, payload: dict[str, Any]) -> None:
        with self._state_io_lock:
            _write_json(self.config.scheduler_state_path, payload)

    def _scheduler_trigger_payload(self) -> dict[str, Any]:
        reconciliation = self._reconcile_run_state()
        state = self._read_scheduler_state()
        active_model = self._active_artifacts()
        current_shadow_entry_count = _safe_int(
            self._shadow_index_stats_cache.get("current_shadow_entry_count"), 0
        )
        active_model_raw_shadow_entry_count = _safe_int(
            active_model.get("raw_shadow_entry_count"), 0
        )
        last_trigger_shadow_entry_count = _safe_int(
            state.get("last_trigger_shadow_entry_count"), current_shadow_entry_count
        )
        effective_trigger_shadow_entry_count = max(
            last_trigger_shadow_entry_count, active_model_raw_shadow_entry_count
        )
        new_shadow_rows_since_trigger = max(
            0, current_shadow_entry_count - last_trigger_shadow_entry_count
        )
        new_shadow_rows_since_effective_baseline = max(
            0, current_shadow_entry_count - effective_trigger_shadow_entry_count
        )
        shadow_row_lag_vs_active_model = max(
            0, current_shadow_entry_count - active_model_raw_shadow_entry_count
        )
        corpus_advanced_beyond_active_model = (
            active_model_raw_shadow_entry_count <= 0 and current_shadow_entry_count > 0
        ) or shadow_row_lag_vs_active_model > 0
        last_run_started_at = str(state.get("last_run_started_at") or "").strip() or None
        last_started_dt = _parse_utc_ts(last_run_started_at)
        seconds_since_last_run_started: float | None = None
        if last_started_dt is not None:
            seconds_since_last_run_started = max(
                0.0, (_utc_now() - last_started_dt).total_seconds()
            )
        interval_elapsed = (
            seconds_since_last_run_started is None
            or seconds_since_last_run_started >= self.config.train_interval_secs
        )
        row_threshold_reached = (
            new_shadow_rows_since_effective_baseline >= self.config.min_new_shadow_rows_to_trigger
        )
        requested_by: str | None = None
        reason = "waiting"
        should_start = False
        if self.is_running():
            reason = "already_running"
        elif not corpus_advanced_beyond_active_model:
            reason = "current_corpus_already_modeled"
        elif row_threshold_reached:
            reason = "row_threshold"
            requested_by = "scheduler_row_threshold"
            should_start = True
        elif interval_elapsed:
            reason = "interval"
            requested_by = "scheduler_interval"
            should_start = True
        return {
            "should_start": should_start,
            "requested_by": requested_by,
            "reason": reason,
            "stale_running_state": reconciliation["stale_running_state"],
            "stale_reason": reconciliation["stale_reason"],
            "current_shadow_entry_count": current_shadow_entry_count,
            "active_model_raw_shadow_entry_count": active_model_raw_shadow_entry_count,
            "last_trigger_shadow_entry_count": last_trigger_shadow_entry_count,
            "effective_trigger_shadow_entry_count": effective_trigger_shadow_entry_count,
            "new_shadow_rows_since_trigger": new_shadow_rows_since_trigger,
            "new_shadow_rows_since_effective_baseline": new_shadow_rows_since_effective_baseline,
            "shadow_row_lag_vs_active_model": shadow_row_lag_vs_active_model,
            "corpus_advanced_beyond_active_model": corpus_advanced_beyond_active_model,
            "min_new_shadow_rows_to_trigger": self.config.min_new_shadow_rows_to_trigger,
            "interval_elapsed": interval_elapsed,
            "row_threshold_reached": row_threshold_reached,
            "seconds_since_last_run_started": seconds_since_last_run_started,
            "last_run_id": state.get("last_run_id"),
            "last_requested_by": state.get("last_requested_by"),
            "last_trigger_reason": state.get("last_trigger_reason"),
            "shadow_index_count_cached": bool(
                self._shadow_index_stats_cache.get("shadow_index_count_cached", True)
            ),
            "shadow_index_count_refreshed_at": self._shadow_index_stats_cache.get(
                "shadow_index_count_refreshed_at"
            ),
            "shadow_index_count_error": self._shadow_index_stats_cache.get(
                "shadow_index_count_error"
            ),
            "shadow_index_size_bytes": _safe_int(
                self._shadow_index_stats_cache.get("shadow_index_size_bytes"), 0
            ),
            "shadow_index_mtime_ns": _safe_int(
                self._shadow_index_stats_cache.get("shadow_index_mtime_ns"), 0
            ),
            "shadow_index_last_seq": _safe_int(
                self._shadow_index_stats_cache.get("shadow_index_last_seq"), 0
            ),
            "shadow_index_last_write_at": self._shadow_index_stats_cache.get(
                "shadow_index_last_write_at"
            ),
        }

    def _mark_scheduler_trigger(self, *, run_id: str, requested_by: str) -> None:
        trigger = self._scheduler_trigger_payload()
        self._write_scheduler_state(
            {
                "last_run_id": run_id,
                "last_requested_by": requested_by,
                "last_run_started_at": _utc_iso(),
                "last_trigger_reason": requested_by,
                "last_trigger_shadow_entry_count": trigger.get("current_shadow_entry_count"),
            }
        )

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
            "next_state_ledger_path": str(self.config.next_state_ledger_path),
            "training_rows": _safe_int(
                training_section.get("rows"), _safe_int(data_quality.get("row_count"), 0)
            ),
            "shadow_rows": _safe_int(training_section.get("shadow_rows"), 0),
            "raw_shadow_entry_count": _safe_int(training_section.get("raw_shadow_entry_count"), 0),
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
        dataset_path: Path,
        dataset_summary: dict[str, Any],
        dataset_summary_path: Path,
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
                "dataset_path": _path_text(dataset_path, self.config.algo_repo_dir),
                "dataset_summary_path": _path_text(dataset_summary_path, self.config.algo_repo_dir),
                "model_path": _path_text(self.config.model_path, self.config.algo_repo_dir),
            },
            "dataset": {
                "ok": bool(dataset_summary.get("ok", False)),
                "quality_gates": dict(dataset_summary.get("quality_gates") or {}),
                "reason": dataset_summary.get("reason"),
                "rows": _safe_int(dataset_summary.get("rows"), 0),
                "summary": str(dataset_summary.get("summary") or dataset_summary_path),
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
                "raw_shadow_entry_count": _safe_int(
                    self._shadow_index_stats_cache.get("current_shadow_entry_count"), 0
                ),
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
        await self._refresh_shadow_index_stats_cache()
        async with self._lock:
            self._reconcile_run_state()
            if self.is_running():
                return {
                    "status": "already_running",
                    "run_id": self._active_run_id,
                }
            run_id = _utc_now().strftime("%Y%m%dT%H%M%SZ") + "-" + os.urandom(4).hex()
            self._mark_scheduler_trigger(run_id=run_id, requested_by=requested_by)
            self._seed_run_state(
                run_id=run_id,
                requested_by=requested_by,
                auto_promote=auto_promote,
            )
            self._active_run_id = run_id
            self._current_task = asyncio.create_task(
                self._run_cycle(run_id=run_id, requested_by=requested_by, auto_promote=auto_promote)
            )
            return {"status": "started", "run_id": run_id}

    async def _run_cycle(self, *, run_id: str, requested_by: str, auto_promote: bool | None) -> None:
        try:
            await self._run_sync_on_executor(
                self._run_cycle_sync,
                run_id=run_id,
                requested_by=requested_by,
                auto_promote=auto_promote,
            )
        finally:
            with contextlib.suppress(FileNotFoundError):
                self.config.lock_path.unlink()
            self._active_run_id = None
            self._current_task = None

    def _run_cycle_sync(self, *, run_id: str, requested_by: str, auto_promote: bool | None) -> None:
        self._refresh_shadow_index_stats_cache_sync()
        run_dir = self._run_dir(run_id)
        logs_dir = run_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        artifact_paths = self._run_artifact_paths(run_id)
        manifest = self._read_manifest_if_exists(run_id) or self._build_run_manifest(
            run_id=run_id,
            requested_by=requested_by,
            auto_promote=auto_promote,
            started_at=_utc_iso(),
        )
        lock_payload = {
            "run_id": run_id,
            "requested_by": requested_by,
            "started_at": str(manifest.get("started_at") or _utc_iso()),
        }
        with self._state_io_lock:
            _write_json(self.config.lock_path, lock_payload)

        self._update_manifest_stage(
            run_id,
            manifest,
            stage="dataset",
            stage_message="building training dataset",
        )
        self._write_manifest(run_id, manifest)

        try:
            dataset_summary = self._dataset_summary()
            model_candidate = artifact_paths["model_candidate"]
            training_candidate = artifact_paths["training_candidate"]
            calibration_candidate = artifact_paths["calibration_candidate"]
            ledger_candidate = artifact_paths["ledger_candidate"]
            eval_pack_json = artifact_paths["eval_pack_json"]
            eval_pack_md = artifact_paths["eval_pack_md"]
            promotion_record = artifact_paths["promotion_record"]
            dataset_dir = run_dir / "dataset"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            dataset_path = dataset_dir / "dataset_latest.jsonl"
            dataset_summary_path = dataset_dir / "dataset_latest_summary.json"
            dataset_stdout_path = logs_dir / "dataset.stdout.log"
            dataset_stderr_path = logs_dir / "dataset.stderr.log"
            train_stdout_path = logs_dir / "train.stdout.log"
            train_stderr_path = logs_dir / "train.stderr.log"
            calibration_stdout_path = logs_dir / "calibration.stdout.log"
            calibration_stderr_path = logs_dir / "calibration.stderr.log"
            eval_stdout_path = logs_dir / "eval.stdout.log"
            eval_stderr_path = logs_dir / "eval.stderr.log"

            dataset_cmd = self._python_cmd(
                self.config.build_dataset_script_path,
                "--out-dir",
                str(dataset_dir),
                "--log",
                str(self.config.log_path),
                "--log-glob",
                str(self.config.algo_repo_dir / "logs" / "*.log"),
                "--cycles",
                str(self.config.algo_repo_dir / "logs" / "auto_tuner" / "cycles.jsonl"),
                "--run-context-dir",
                str(self.config.run_context_dir),
                "--position-index",
                str(self.config.algo_repo_dir / "index" / "position_outcomes.json"),
                "--trade-index",
                str(self.config.algo_repo_dir / "index" / "trade_outcomes.json"),
                "--min-rows",
                "10",
                "--max-unknown-sleeve-ratio",
                "1.0",
                "--allow-low-quality-dataset",
            )
            dataset_returncode, dataset_stdout, dataset_stderr = self._run_logged_subprocess(
                cmd=dataset_cmd,
                cwd=self.config.algo_repo_dir,
                stdout_path=dataset_stdout_path,
                stderr_path=dataset_stderr_path,
                timeout=self.config.train_timeout_secs,
            )
            if dataset_returncode != 0:
                combined = f"{dataset_stdout}\n{dataset_stderr}"
                quality_gate_failed = False
                for blob in (dataset_stdout, dataset_stderr):
                    blob = (blob or "").strip()
                    if not blob:
                        continue
                    try:
                        payload = json.loads(blob)
                    except Exception:
                        continue
                    if str(payload.get("reason") or "").strip() == "dataset_quality_gate_failed":
                        quality_gate_failed = True
                        break
                if not (quality_gate_failed and dataset_path.exists()):
                    raise RuntimeError(
                        f"dataset build failed ({dataset_returncode}): {combined.strip()}"
                    )
            if dataset_summary_path.exists():
                dataset_summary = _read_json(dataset_summary_path)
            manifest["dataset"] = dataset_summary
            self._update_manifest_stage(
                run_id,
                manifest,
                stage="train",
                stage_message="training candidate model",
                dataset_rows=_safe_int(dataset_summary.get("rows"), 0),
                dataset_summary_path=dataset_summary_path,
            )
            self._write_manifest(run_id, manifest)

            train_cmd = self._python_cmd(
                self.config.train_script_path,
                "--input",
                str(dataset_path),
                "--output",
                str(model_candidate),
                "--shadow-index",
                str(self.config.shadow_sqlite_path),
            )
            train_returncode, train_stdout, train_stderr = self._run_logged_subprocess(
                cmd=train_cmd,
                cwd=self.config.algo_repo_dir,
                stdout_path=train_stdout_path,
                stderr_path=train_stderr_path,
                timeout=self.config.train_timeout_secs,
            )
            if train_returncode != 0:
                raise RuntimeError(
                    f"train step failed ({train_returncode}): {(train_stdout + chr(10) + train_stderr).strip()}"
                )

            self._update_manifest_stage(
                run_id,
                manifest,
                stage="attest",
                stage_message="building candidate attestation",
                dataset_rows=_safe_int(dataset_summary.get("rows"), 0),
                dataset_summary_path=dataset_summary_path,
            )
            self._write_manifest(run_id, manifest)
            candidate_attestation = self._build_attestation(
                model_candidate,
                dataset_path,
                dataset_summary,
                dataset_summary_path,
                model_candidate,
            )
            _write_json(training_candidate, candidate_attestation)

            self._update_manifest_stage(
                run_id,
                manifest,
                stage="calibrate",
                stage_message="building calibration snapshot",
                dataset_rows=_safe_int(dataset_summary.get("rows"), 0),
                dataset_summary_path=dataset_summary_path,
            )
            self._write_manifest(run_id, manifest)
            calibration_cmd = self._python_cmd(
                self.config.calibration_script_path,
                "--dataset",
                str(dataset_path),
                "--ledger",
                str(ledger_candidate),
                "--snapshot",
                str(calibration_candidate),
            )
            calibration_returncode, calibration_stdout, calibration_stderr = self._run_logged_subprocess(
                cmd=calibration_cmd,
                cwd=self.config.algo_repo_dir,
                stdout_path=calibration_stdout_path,
                stderr_path=calibration_stderr_path,
            )
            if calibration_returncode != 0:
                raise RuntimeError(
                    "calibration step failed "
                    f"({calibration_returncode}): {(calibration_stdout + chr(10) + calibration_stderr).strip()}"
                )

            self._update_manifest_stage(
                run_id,
                manifest,
                stage="eval",
                stage_message="building evaluation pack",
                dataset_rows=_safe_int(dataset_summary.get("rows"), 0),
                dataset_summary_path=dataset_summary_path,
            )
            self._write_manifest(run_id, manifest)
            eval_cmd = self._python_cmd(
                self.config.eval_pack_script_path,
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
            )
            eval_returncode, eval_stdout, eval_stderr = self._run_logged_subprocess(
                cmd=eval_cmd,
                cwd=self.config.algo_repo_dir,
                stdout_path=eval_stdout_path,
                stderr_path=eval_stderr_path,
            )
            if eval_returncode != 0:
                raise RuntimeError(
                    f"eval step failed ({eval_returncode}): {(eval_stdout + chr(10) + eval_stderr).strip()}"
                )

            self._update_manifest_stage(
                run_id,
                manifest,
                stage="gate",
                stage_message="evaluating promotion gate",
                dataset_rows=_safe_int(dataset_summary.get("rows"), 0),
                dataset_summary_path=dataset_summary_path,
            )
            self._write_manifest(run_id, manifest)
            candidate_model = _read_json(model_candidate)
            gate_result = self._evaluate_candidate(
                active=manifest["active_model"],
                candidate_model=candidate_model,
                candidate_attestation=candidate_attestation,
            )
            candidate_eval_pack = _read_json(eval_pack_json)

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
                self._update_manifest_stage(
                    run_id,
                    manifest,
                    stage="promote",
                    stage_message="promoting candidate artifacts",
                    dataset_rows=_safe_int(dataset_summary.get("rows"), 0),
                    dataset_summary_path=dataset_summary_path,
                )
                self._write_manifest(run_id, manifest)
                promotion = self._promote_candidate(
                    run_id=run_id,
                    model_candidate=model_candidate,
                    training_candidate=training_candidate,
                    calibration_candidate=calibration_candidate,
                    ledger_candidate=ledger_candidate,
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
            _write_json(promotion_record, manifest["promotion"])
            self._update_manifest_stage(
                run_id,
                manifest,
                stage="completed",
                stage_message="trainer run completed",
                dataset_rows=_safe_int(dataset_summary.get("rows"), 0),
                dataset_summary_path=dataset_summary_path,
            )
            manifest["status"] = "completed"
            manifest["completed_at"] = _utc_iso()
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
            self._update_manifest_stage(
                run_id,
                manifest,
                stage="failed",
                stage_message=f"trainer run failed: {exc}",
                dataset_rows=manifest.get("dataset_rows"),
                dataset_summary_path=Path(manifest["dataset_summary_path"])
                if manifest.get("dataset_summary_path")
                else None,
            )
            manifest["error"] = {
                "type": exc.__class__.__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
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

    def _promote_candidate(
        self,
        *,
        run_id: str,
        model_candidate: Path,
        training_candidate: Path,
        calibration_candidate: Path,
        ledger_candidate: Path,
        forced: bool,
    ) -> dict[str, Any]:
        _atomic_copy(model_candidate, self.config.model_path)
        _atomic_copy(training_candidate, self.config.training_path)
        _atomic_copy(calibration_candidate, self.config.calibration_path)
        if ledger_candidate.exists():
            _atomic_copy(ledger_candidate, self.config.next_state_ledger_path)

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
            restart_env = os.environ.copy()
            restart_env["DEPLOY_WRAPPER_REPLACE_EXISTING"] = "1"
            restart_env["ALGOTRADER_WALLET"] = str(self.config.wallet_path)
            # A sidecar-owned relaunch should not block on curling the same sidecar,
            # and it should not immediately retrigger the trainer on success.
            restart_env["SIDECAR_REQUIRE_HEALTH"] = "0"
            restart_env["SIDECAR_SKIP_POST_LAUNCH_TRAINER_TRIGGER"] = "1"
            restart_env.pop("REAL_ALGOTRADER_WALLET", None)
            restart_log.parent.mkdir(parents=True, exist_ok=True)
            with restart_log.open("w", encoding="utf-8") as stdout_fh, restart_err.open(
                "w", encoding="utf-8"
            ) as stderr_fh:
                restart_process = subprocess.Popen(
                    [str(self.config.deploy_script_path), "--live", "--skip-build"],
                    cwd=self.config.algo_repo_dir,
                    stdout=stdout_fh,
                    stderr=stderr_fh,
                    text=True,
                    env=restart_env,
                    start_new_session=True,
                )
            promotion["state"] = "promoted_and_relaunch_requested"
            promotion["restart_pid"] = restart_process.pid
            promotion["restart_dispatch"] = "detached_wrapper"
            self._pending_restart = None
        except Exception as exc:
            restart_log.write_text("", encoding="utf-8")
            restart_err.write_text(str(exc), encoding="utf-8")
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
        ledger_candidate = Path(str(artifacts.get("candidate_ledger") or ""))
        if (
            not model_candidate.exists()
            or not training_candidate.exists()
            or not calibration_candidate.exists()
            or not ledger_candidate.exists()
        ):
            raise FileNotFoundError(f"run {run_id} is missing candidate artifacts")
        promotion = await self._run_sync_on_executor(
            self._promote_candidate,
            run_id=run_id,
            model_candidate=model_candidate,
            training_candidate=training_candidate,
            calibration_candidate=calibration_candidate,
            ledger_candidate=ledger_candidate,
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

    def _freshness_payload(self) -> dict[str, Any]:
        reconciliation = self._reconcile_run_state()
        active_model = self._active_artifacts()
        active_model_shadow_rows = _safe_int(active_model.get("shadow_rows"), 0)
        active_model_raw_shadow_entry_count = _safe_int(active_model.get("raw_shadow_entry_count"), 0)
        trigger = self._scheduler_trigger_payload()
        current_shadow_entry_count = _safe_int(trigger.get("current_shadow_entry_count"), 0)
        new_shadow_rows_since_trigger = _safe_int(trigger.get("new_shadow_rows_since_trigger"), 0)
        latest_manifest = self._latest_manifest_summary(status="completed") or {}
        latest_completed_run_id = str(latest_manifest.get("run_id") or "").strip() or None
        latest_completed_run_status = str(latest_manifest.get("status") or "").strip() or None
        latest_completed_run_raw_shadow_entry_count = _safe_int(
            latest_manifest.get("raw_shadow_entry_count"), 0
        )
        latest_completed_promotion = latest_manifest.get("promotion") or {}
        latest_completed_promotion_state = (
            str(latest_completed_promotion.get("state") or "").strip() or None
        )
        lag_baseline = (
            active_model_raw_shadow_entry_count
            if active_model_raw_shadow_entry_count > 0
            else active_model_shadow_rows
        )
        shadow_row_lag = max(0, current_shadow_entry_count - lag_baseline)
        if active_model_raw_shadow_entry_count > 0 and shadow_row_lag <= 0:
            model_freshness_state = "current"
        elif (
            latest_completed_run_status == "completed"
            and latest_completed_run_raw_shadow_entry_count >= current_shadow_entry_count
            and latest_completed_promotion_state == "gated_off"
        ):
            model_freshness_state = "current_gated_no_change"
        elif reconciliation["stale_running_state"]:
            model_freshness_state = "stuck_running_state"
        elif self.is_running():
            model_freshness_state = "running_catching_up"
        else:
            model_freshness_state = "pending_retrain"
        return {
            "active_model_trained_at": active_model.get("trained_at"),
            "active_model_shadow_rows": active_model_shadow_rows,
            "active_model_raw_shadow_entry_count": active_model_raw_shadow_entry_count,
            "current_shadow_entry_count": current_shadow_entry_count,
            "new_shadow_rows_since_trigger": new_shadow_rows_since_trigger,
            "shadow_row_lag_vs_active_model": shadow_row_lag,
            "latest_completed_run_id": latest_completed_run_id,
            "latest_completed_run_status": latest_completed_run_status,
            "latest_completed_run_raw_shadow_entry_count": latest_completed_run_raw_shadow_entry_count,
            "latest_completed_promotion_state": latest_completed_promotion_state,
            "model_freshness_state": model_freshness_state,
        }

    def _scheduler_contract_payload(self) -> dict[str, Any]:
        return {
            "scheduler_enabled": self.config.scheduler_enabled,
            "scheduler_config_source": self.config.scheduler_config_source,
            "scheduler_disabled_reason": self.config.scheduler_disabled_reason,
        }

    def health_payload(self) -> dict[str, Any]:
        self._reconcile_run_state()
        freshness = self._freshness_payload()
        return {
            "status": "ok",
            **self._scheduler_contract_payload(),
            "auto_promote": self.config.auto_promote,
            "relaunch_enabled": self.config.relaunch_enabled,
            "is_running": self.is_running(),
            "algo_repo_dir": str(self.config.algo_repo_dir),
            "algo_repo_exists": self.config.algo_repo_dir.exists(),
            "active_model_path": str(self.config.model_path),
            "active_training_path": str(self.config.training_path),
            "active_calibration_path": str(self.config.calibration_path),
            "active_next_state_ledger_path": str(self.config.next_state_ledger_path),
            "shadow_index_path": str(self.config.shadow_sqlite_path),
            "shadow_index_legacy_path": str(self.config.shadow_index_path),
            "shadow_duckdb_path": str(self.config.shadow_duckdb_path),
            "train_interval_secs": self.config.train_interval_secs,
            "scheduler_poll_secs": self.config.scheduler_poll_secs,
            "train_timeout_secs": self.config.train_timeout_secs,
            "min_new_shadow_rows_to_trigger": self.config.min_new_shadow_rows_to_trigger,
            "pending_restart": self._pending_restart,
            "scheduler_trigger": self._scheduler_trigger_payload(),
            **freshness,
        }

    def status_payload(self) -> dict[str, Any]:
        self._reconcile_run_state()
        manifest = self._read_manifest_if_exists(self._active_run_id) if self._active_run_id else None
        active_run = self._manifest_for_status(self._active_run_id, manifest) if self._active_run_id else None
        freshness = self._freshness_payload()
        return {
            "status": "running" if self.is_running() else "idle",
            **self._scheduler_contract_payload(),
            "active_run_id": self._active_run_id,
            "active_run": active_run,
            "active_run_stage": (active_run or {}).get("stage"),
            "active_run_stage_started_at": (active_run or {}).get("stage_started_at"),
            "active_run_stage_updated_at": (active_run or {}).get("stage_updated_at"),
            "active_run_stage_message": (active_run or {}).get("stage_message"),
            "pending_restart": self._pending_restart,
            "active_model": self._active_artifacts(),
            "latest_run_context": self._latest_run_context_path(),
            "shadow_index_path": str(self.config.shadow_sqlite_path),
            "shadow_index_legacy_path": str(self.config.shadow_index_path),
            "shadow_duckdb_path": str(self.config.shadow_duckdb_path),
            "scheduler_trigger": self._scheduler_trigger_payload(),
            **freshness,
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
