import asyncio
import contextlib
import errno
import json
import sqlite3
import sys
import threading
import time
from pathlib import Path

import pytest

import src.api.predictive_trainer as predictive_trainer
from src.api.predictive_trainer import PredictiveTrainerConfig, PredictiveTrainerManager


def _write_shadow_sqlite_count(sqlite_path: Path, count: int) -> None:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(sqlite_path))
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS predictive_candidate_firehose_shadow_outcomes_stats (
            singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
            entry_count INTEGER NOT NULL DEFAULT 0,
            last_seq INTEGER NOT NULL DEFAULT 0,
            last_write_at TEXT
        );
        """,
    )
    conn.execute(
        """
        INSERT OR REPLACE INTO predictive_candidate_firehose_shadow_outcomes_stats
            (singleton, entry_count, last_seq, last_write_at)
        VALUES (1, ?, ?, '2026-04-07T00:00:00Z')
        """,
        (count, count),
    )
    conn.commit()
    conn.close()


def _write_active_training_artifact(config: PredictiveTrainerConfig, *, raw_shadow_entry_count: int, shadow_rows: int | None = None, rows: int | None = None) -> None:
    config.training_path.write_text(
        json.dumps(
            {
                "training": {
                    "rows": rows if rows is not None else raw_shadow_entry_count,
                    "shadow_rows": shadow_rows if shadow_rows is not None else raw_shadow_entry_count,
                    "raw_shadow_entry_count": raw_shadow_entry_count,
                },
                "validation": {
                    "positive_rows": 10,
                    "negative_rows": 5,
                    "trained_at": "2026-04-08T00:00:00Z",
                },
            }
        ),
        encoding="utf-8",
    )


def _make_config(tmp_path: Path) -> PredictiveTrainerConfig:
    algo_repo = tmp_path / "algo"
    (algo_repo / "logs/analysis/predictive_entry").mkdir(parents=True, exist_ok=True)
    (algo_repo / "logs/analysis/monte_carlo").mkdir(parents=True, exist_ok=True)
    (algo_repo / "logs/index").mkdir(parents=True, exist_ok=True)
    (algo_repo / "logs/run_context").mkdir(parents=True, exist_ok=True)
    return PredictiveTrainerConfig(
        algo_repo_dir=algo_repo,
        data_dir=tmp_path / "sidecar_data",
        train_interval_secs=600,
        scheduler_poll_secs=30,
        scheduler_enabled=False,
        scheduler_config_source="test_disabled",
        scheduler_disabled_reason="test_fixture_disabled",
        auto_promote=True,
        relaunch_enabled=True,
        python_bin="python3",
        train_timeout_secs=1800,
        min_new_shadow_rows_to_trigger=100,
        positive_share_collapse_tolerance=0.05,
        calibration_mae_degradation_factor=1.25,
        p_positive_brier_degradation_factor=1.25,
    )


def test_from_env_accepts_numeric_truthy_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    algo_repo = tmp_path / "algo"
    algo_repo.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SIDECAR_PREDICTIVE_TRAINER_ALGO_REPO_DIR", str(algo_repo))
    monkeypatch.setenv("SIDECAR_PREDICTIVE_TRAINER_ENABLED", "1")
    monkeypatch.setenv("SIDECAR_PREDICTIVE_TRAINER_AUTO_PROMOTE", "yes")
    monkeypatch.setenv("SIDECAR_PREDICTIVE_TRAINER_RELAUNCH_ENABLED", "on")
    monkeypatch.setenv("SIDECAR_PREDICTIVE_TRAINER_POLL_SECS", "45")
    monkeypatch.setenv("SIDECAR_PREDICTIVE_TRAINER_MIN_NEW_SHADOW_ROWS", "125")

    config = PredictiveTrainerConfig.from_env(tmp_path / "data")

    assert config.scheduler_enabled is True
    assert config.scheduler_config_source == "env_enabled"
    assert config.scheduler_disabled_reason is None
    assert config.auto_promote is True
    assert config.relaunch_enabled is True
    assert config.scheduler_poll_secs == 45
    assert config.min_new_shadow_rows_to_trigger == 125


def test_from_env_defaults_scheduler_enabled_when_unset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    algo_repo = tmp_path / "algo"
    algo_repo.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SIDECAR_PREDICTIVE_TRAINER_ALGO_REPO_DIR", str(algo_repo))
    monkeypatch.delenv("SIDECAR_PREDICTIVE_TRAINER_ENABLED", raising=False)
    monkeypatch.delenv("SIDECAR_GUIDANCE", raising=False)

    config = PredictiveTrainerConfig.from_env(tmp_path / "data")

    assert config.scheduler_enabled is True
    assert config.scheduler_config_source == "default_enabled"
    assert config.scheduler_disabled_reason is None


def test_from_env_reports_disabled_reason_when_guidance_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    algo_repo = tmp_path / "algo"
    algo_repo.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SIDECAR_PREDICTIVE_TRAINER_ALGO_REPO_DIR", str(algo_repo))
    monkeypatch.setenv("SIDECAR_PREDICTIVE_TRAINER_ENABLED", "0")
    monkeypatch.setenv("SIDECAR_GUIDANCE", "1")

    config = PredictiveTrainerConfig.from_env(tmp_path / "data")

    assert config.scheduler_enabled is False
    assert config.scheduler_config_source == "env_disabled"
    assert (
        config.scheduler_disabled_reason
        == "scheduler_explicitly_disabled_while_guidance_enabled"
    )


def test_evaluate_candidate_accepts_improvement(tmp_path: Path):
    manager = PredictiveTrainerManager(_make_config(tmp_path))
    active = {
        "training_rows": 100,
        "shadow_rows": 90,
        "positive_rows": 60,
        "negative_rows": 40,
        "calibration_global_mae_sol": 0.8,
        "p_positive_after_cost_brier": 0.2,
    }
    candidate_model = {
        "calibration": {
            "global_mae_sol": 0.6,
            "tradeability_head_brier": {"p_positive_after_cost": 0.18},
        }
    }
    candidate_attestation = {
        "training": {"rows": 120, "shadow_rows": 110},
        "validation": {"positive_rows": 78, "negative_rows": 42},
    }

    result = manager._evaluate_candidate(
        active=active,
        candidate_model=candidate_model,
        candidate_attestation=candidate_attestation,
    )

    assert result["ok"] is True
    assert result["issues"] == []


def test_evaluate_candidate_rejects_positive_share_collapse(tmp_path: Path):
    manager = PredictiveTrainerManager(_make_config(tmp_path))
    active = {
        "training_rows": 100,
        "shadow_rows": 90,
        "positive_rows": 70,
        "negative_rows": 30,
    }
    candidate_model = {"calibration": {}}
    candidate_attestation = {
        "training": {"rows": 120, "shadow_rows": 110},
        "validation": {"positive_rows": 50, "negative_rows": 70},
    }

    result = manager._evaluate_candidate(
        active=active,
        candidate_model=candidate_model,
        candidate_attestation=candidate_attestation,
    )

    assert result["ok"] is False
    assert "positive_share_collapsed" in result["issues"]


def test_build_attestation_carries_provenance_summary(tmp_path: Path):
    manager = PredictiveTrainerManager(_make_config(tmp_path))
    candidate_model_path = tmp_path / "model_candidate.json"
    candidate_model_path.write_text(
        json.dumps(
            {
                "version": "predictive-entry-v12.2",
                "trained_at": "2026-04-08T00:00:00Z",
                "data_quality": {
                    "row_count": 12,
                    "positive_net_sol_count": 8,
                    "negative_net_sol_count": 4,
                },
                "executed_data_quality": {"row_count": 2},
                "shadow_data_quality": {"row_count": 10},
                "training_window": {
                    "rows": 12,
                    "executed_rows": 2,
                    "shadow_rows": 10,
                    "excluded_invalid_target_rows": {"executed": 0, "shadow": 0},
                },
                "source_provenance_class_counts_shadow": {"yellowstone_authoritative": 9},
                "source_provenance_class_counts_executed": {
                    "yellowstone_authoritative": 2
                },
                "yellowstone_authoritative_shadow_rows": 9,
                "yellowstone_authoritative_executed_rows": 2,
                "mixed_event_flow_shadow_rows": 1,
                "legacy_or_unattributed_rows": 3,
                "selected_shadow_row_count": 10,
                "selected_executed_row_count": 2,
                "selected_total_training_rows": 12,
                "raw_shadow_entry_count": 222,
                "executed_prior_audit": [
                    {
                        "timestamp": "2026-04-06T18:19:35.789997Z",
                        "mint": "mint-a",
                        "source_provenance_class": "yellowstone_authoritative",
                        "strategy_family": "EVENT_FLOW_ACCELERATION",
                        "exit_policy": "fast_fee_clear_lock",
                        "net_sol": 0.003054,
                        "exit_reason": "trailing_stop_loss",
                    }
                ],
                "positive_negative_split_by_provenance": {
                    "shadow": {
                        "yellowstone_authoritative": {
                            "positives": 7,
                            "negatives": 2,
                            "total": 9,
                            "positive_rate": 0.7777,
                        }
                    }
                },
                "event_policy_provenance_executed_priors": {
                    "ignition|fast_fee_clear_lock|yellowstone_authoritative": {
                        "sample_count": 2,
                        "effective_sample_count": 5.0,
                        "realized_prior_sample_sufficient": False,
                    }
                },
                "realized_prior_sample_sufficient": {
                    "threshold_effective_sample_count": 20.0,
                    "by_event_policy_provenance": {
                        "ignition|fast_fee_clear_lock|yellowstone_authoritative": False
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    attestation = manager._build_attestation(
        candidate_model_path=candidate_model_path,
        dataset_path=tmp_path / "dataset.jsonl",
        dataset_summary={"ok": True, "rows": 12, "quality_gates": {}, "summary": "summary"},
        dataset_summary_path=tmp_path / "dataset_summary.json",
        candidate_output_hint=tmp_path / "candidate.json",
    )

    assert attestation["yellowstone_authoritative_shadow_rows"] == 9
    assert attestation["mixed_event_flow_shadow_rows"] == 1
    assert (
        attestation["event_policy_provenance_executed_priors"][
            "ignition|fast_fee_clear_lock|yellowstone_authoritative"
        ]["effective_sample_count"]
        == 5.0
    )
    assert attestation["training"]["raw_shadow_entry_count"] == 222
    assert attestation["training"]["selected_shadow_row_count"] == 10
    assert attestation["training"]["selected_executed_row_count"] == 2
    assert attestation["training"]["selected_total_training_rows"] == 12
    assert attestation["executed_prior_audit"][0]["mint"] == "mint-a"


@pytest.mark.asyncio
async def test_start_run_prevents_overlap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manager = PredictiveTrainerManager(_make_config(tmp_path))
    started = asyncio.Event()
    release = asyncio.Event()

    async def _fake_run_cycle(*, run_id: str, requested_by: str, auto_promote: bool | None) -> None:
        started.set()
        await release.wait()

    monkeypatch.setattr(manager, "_run_cycle", _fake_run_cycle)

    first = await manager.start_run(requested_by="manual", auto_promote=None)
    await asyncio.wait_for(started.wait(), timeout=1.0)
    second = await manager.start_run(requested_by="manual", auto_promote=None)

    assert first["status"] == "started"
    assert second["status"] == "already_running"
    assert second["run_id"] == first["run_id"]

    release.set()
    await asyncio.wait_for(manager._current_task, timeout=1.0)


def test_scheduler_trigger_prefers_row_threshold(tmp_path: Path):
    manager = PredictiveTrainerManager(_make_config(tmp_path))
    manager._write_scheduler_state(
        {
            "last_run_id": "run-old",
            "last_requested_by": "scheduler_interval",
            "last_run_started_at": "2026-04-07T00:00:00Z",
            "last_trigger_reason": "scheduler_interval",
            "last_trigger_shadow_entry_count": 40,
        }
    )
    _write_shadow_sqlite_count(manager.config.shadow_sqlite_path, 150)
    manager._shadow_index_stats_cache = manager._refresh_shadow_index_stats_cache_sync()

    trigger = manager._scheduler_trigger_payload()

    assert trigger["should_start"] is True
    assert trigger["requested_by"] == "scheduler_row_threshold"
    assert trigger["new_shadow_rows_since_trigger"] == 110
    assert trigger["row_threshold_reached"] is True


def test_python_cmd_forces_unbuffered_python(tmp_path: Path):
    manager = PredictiveTrainerManager(_make_config(tmp_path))

    cmd = manager._python_cmd(Path("/tmp/example.py"), "--flag", "value")

    assert cmd == ["python3", "-u", "/tmp/example.py", "--flag", "value"]


def test_read_json_retries_resource_deadlock_then_succeeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "model.json"
    path.write_text('{"ok": true}', encoding="utf-8")
    real_read_text = Path.read_text
    attempts = {"count": 0}

    def _flaky_read_text(self: Path, *args, **kwargs):
        if self == path and attempts["count"] < 2:
            attempts["count"] += 1
            raise OSError(errno.EDEADLK, "Resource deadlock avoided")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _flaky_read_text)

    assert predictive_trainer._read_json(path) == {"ok": True}
    assert attempts["count"] == 2


def test_active_artifacts_degrades_when_model_read_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manager = PredictiveTrainerManager(_make_config(tmp_path))
    manager.config.model_path.write_text('{"ok": true}', encoding="utf-8")
    real_read_json = predictive_trainer._read_json

    def _flaky_read_json(path: Path):
        if path == manager.config.model_path:
            raise OSError(errno.EDEADLK, "Resource deadlock avoided")
        return real_read_json(path)

    monkeypatch.setattr(predictive_trainer, "_read_json", _flaky_read_json)

    payload = manager._active_artifacts()

    assert payload["raw_shadow_entry_count"] == 0
    assert payload["trained_at"] is None
    assert payload["model_read_error"]["type"] == "OSError"
    assert "Resource deadlock avoided" in payload["model_read_error"]["message"]


def test_scheduler_trigger_skips_auto_retrain_when_active_model_matches_current_corpus(tmp_path: Path):
    config = _make_config(tmp_path)
    manager = PredictiveTrainerManager(config)
    _write_active_training_artifact(config, raw_shadow_entry_count=200, shadow_rows=120, rows=125)
    manager._write_scheduler_state(
        {
            "last_run_id": "run-old",
            "last_requested_by": "scheduler_interval",
            "last_run_started_at": "2026-04-07T00:00:00Z",
            "last_trigger_reason": "scheduler_interval",
            "last_trigger_shadow_entry_count": 150,
        }
    )
    _write_shadow_sqlite_count(manager.config.shadow_sqlite_path, 200)
    manager._shadow_index_stats_cache = manager._refresh_shadow_index_stats_cache_sync()

    trigger = manager._scheduler_trigger_payload()

    assert trigger["should_start"] is False
    assert trigger["reason"] == "current_corpus_already_modeled"
    assert trigger["active_model_raw_shadow_entry_count"] == 200
    assert trigger["effective_trigger_shadow_entry_count"] == 200
    assert trigger["new_shadow_rows_since_effective_baseline"] == 0
    assert trigger["corpus_advanced_beyond_active_model"] is False


def test_scheduler_trigger_uses_interval_when_active_model_lags_current_corpus(tmp_path: Path):
    config = _make_config(tmp_path)
    manager = PredictiveTrainerManager(config)
    _write_active_training_artifact(config, raw_shadow_entry_count=170, shadow_rows=110, rows=115)
    manager._write_scheduler_state(
        {
            "last_run_id": "run-old",
            "last_requested_by": "scheduler_interval",
            "last_run_started_at": "2026-04-07T00:00:00Z",
            "last_trigger_reason": "scheduler_interval",
            "last_trigger_shadow_entry_count": 195,
        }
    )
    _write_shadow_sqlite_count(manager.config.shadow_sqlite_path, 200)
    manager._shadow_index_stats_cache = manager._refresh_shadow_index_stats_cache_sync()

    trigger = manager._scheduler_trigger_payload()

    assert trigger["should_start"] is True
    assert trigger["requested_by"] == "scheduler_interval"
    assert trigger["reason"] == "interval"
    assert trigger["row_threshold_reached"] is False
    assert trigger["shadow_row_lag_vs_active_model"] == 30
    assert trigger["corpus_advanced_beyond_active_model"] is True


def test_scheduler_trigger_uses_interval_when_row_threshold_not_met(tmp_path: Path):
    manager = PredictiveTrainerManager(_make_config(tmp_path))
    manager._write_scheduler_state(
        {
            "last_run_id": "run-old",
            "last_requested_by": "scheduler_interval",
            "last_run_started_at": "2026-04-07T00:00:00Z",
            "last_trigger_reason": "scheduler_interval",
            "last_trigger_shadow_entry_count": 50,
        }
    )
    _write_shadow_sqlite_count(manager.config.shadow_sqlite_path, 80)
    manager._shadow_index_stats_cache = manager._refresh_shadow_index_stats_cache_sync()

    trigger = manager._scheduler_trigger_payload()

    assert trigger["should_start"] is True
    assert trigger["requested_by"] == "scheduler_interval"
    assert trigger["new_shadow_rows_since_trigger"] == 30
    assert trigger["row_threshold_reached"] is False


def test_scheduler_trigger_recovers_from_terminal_manifest_stale_task(tmp_path: Path):
    manager = PredictiveTrainerManager(_make_config(tmp_path))
    _write_shadow_sqlite_count(manager.config.shadow_sqlite_path, 220)
    manager._shadow_index_stats_cache = manager._refresh_shadow_index_stats_cache_sync()
    run_id = "run-stale-terminal"
    manager._active_run_id = run_id
    manager._write_manifest(
        run_id,
        {
            "run_id": run_id,
            "status": "completed",
            "raw_shadow_entry_count": 100,
        },
    )

    class _NeverDoneTask:
        def done(self) -> bool:
            return False

    manager._current_task = _NeverDoneTask()
    manager._write_scheduler_state(
        {
            "last_run_id": run_id,
            "last_requested_by": "scheduler_interval",
            "last_run_started_at": "2026-04-07T00:00:00Z",
            "last_trigger_reason": "scheduler_interval",
            "last_trigger_shadow_entry_count": 100,
        }
    )

    trigger = manager._scheduler_trigger_payload()

    assert manager._current_task is None
    assert manager._active_run_id is None
    assert trigger["should_start"] is True
    assert trigger["requested_by"] == "scheduler_row_threshold"
    assert trigger["stale_running_state"] is True
    assert trigger["stale_reason"] == "terminal_manifest_without_lock"


def test_reconcile_marks_running_manifest_without_lock_failed(tmp_path: Path):
    manager = PredictiveTrainerManager(_make_config(tmp_path))
    run_id = "run-manifest-only"
    manager._write_manifest(
        run_id,
        {
            "run_id": run_id,
            "status": "running",
            "started_at": "2026-04-07T00:00:00Z",
        },
    )

    reconciliation = manager._reconcile_run_state()
    manifest = manager._read_manifest(run_id)

    assert reconciliation["stale_running_state"] is True
    assert reconciliation["stale_reason"] == "running_manifest_without_lock"
    assert manifest["status"] == "failed"
    assert manifest["error"]["type"] == "StaleRunState"


def test_health_payload_reports_model_freshness_state(tmp_path: Path):
    manager = PredictiveTrainerManager(_make_config(tmp_path))
    manager.config.training_path.write_text(
        json.dumps(
            {
                "training": {"rows": 200, "shadow_rows": 200, "executed_rows": 1},
                "validation": {
                    "positive_rows": 120,
                    "negative_rows": 80,
                    "trained_at": "2026-04-07T03:38:24.484862Z",
                },
            }
        ),
        encoding="utf-8",
    )
    manager.config.model_path.write_text("{}", encoding="utf-8")
    manager.config.calibration_path.write_text(json.dumps({"rows": 10}), encoding="utf-8")
    _write_shadow_sqlite_count(manager.config.shadow_sqlite_path, 260)
    manager._shadow_index_stats_cache = manager._refresh_shadow_index_stats_cache_sync()
    manager._write_scheduler_state(
        {
            "last_run_id": "run-old",
            "last_requested_by": "scheduler_interval",
            "last_run_started_at": "2026-04-07T00:00:00Z",
            "last_trigger_reason": "scheduler_interval",
            "last_trigger_shadow_entry_count": 220,
        }
    )

    payload = manager.health_payload()

    assert payload["active_model_trained_at"] == "2026-04-07T03:38:24.484862Z"
    assert payload["active_model_shadow_rows"] == 200
    assert payload["current_shadow_entry_count"] == 260
    assert payload["new_shadow_rows_since_trigger"] == 40
    assert payload["model_freshness_state"] == "pending_retrain"


def test_health_payload_reports_current_when_latest_run_gated_no_change(tmp_path: Path):
    manager = PredictiveTrainerManager(_make_config(tmp_path))
    manager.config.training_path.write_text(
        json.dumps(
            {
                "training": {"rows": 200, "shadow_rows": 200, "executed_rows": 1},
                "validation": {
                    "positive_rows": 120,
                    "negative_rows": 80,
                    "trained_at": "2026-04-07T03:38:24.484862Z",
                },
            }
        ),
        encoding="utf-8",
    )
    manager.config.model_path.write_text("{}", encoding="utf-8")
    manager.config.calibration_path.write_text(json.dumps({"rows": 10}), encoding="utf-8")
    _write_shadow_sqlite_count(manager.config.shadow_sqlite_path, 260)
    manager._shadow_index_stats_cache = manager._refresh_shadow_index_stats_cache_sync()
    manager._write_scheduler_state(
        {
            "last_run_id": "run-current",
            "last_requested_by": "scheduler_interval",
            "last_run_started_at": "2026-04-07T00:00:00Z",
            "last_trigger_reason": "scheduler_interval",
            "last_trigger_shadow_entry_count": 260,
        }
    )
    manager._write_manifest(
        "run-current",
        {
            "run_id": "run-current",
            "status": "completed",
            "raw_shadow_entry_count": 260,
            "promotion": {
                "state": "gated_off",
                "gate_result": {
                    "ok": False,
                    "issues": ["training_rows_not_improved", "shadow_rows_not_improved"],
                },
            },
        },
    )
    manager._write_manifest(
        "run-later-failed",
        {
            "run_id": "run-later-failed",
            "status": "failed",
            "raw_shadow_entry_count": 260,
            "promotion": {"state": "not_attempted"},
            "error": {
                "type": "StaleRunState",
                "message": "reconciled running manifest without live task or lock",
            },
        },
    )

    payload = manager.health_payload()

    assert payload["latest_completed_run_id"] == "run-current"
    assert payload["latest_completed_run_raw_shadow_entry_count"] == 260
    assert payload["latest_completed_promotion_state"] == "gated_off"
    assert payload["model_freshness_state"] == "current_gated_no_change"
    assert payload["shadow_row_lag_vs_active_model"] == 60


def test_health_payload_uses_cached_sqlite_stats_not_json_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manager = PredictiveTrainerManager(_make_config(tmp_path))
    _write_shadow_sqlite_count(manager.config.shadow_sqlite_path, 333)
    manager._shadow_index_stats_cache = manager._refresh_shadow_index_stats_cache_sync()
    monkeypatch.setattr("src.api.predictive_trainer._count_json_array_entries", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("json hot path used")))

    payload = manager.health_payload()

    assert payload["current_shadow_entry_count"] == 333
    assert payload["scheduler_trigger"]["shadow_index_count_cached"] is True
    assert payload["scheduler_trigger"]["shadow_index_count_error"] is None


def test_status_payload_reports_shadow_store_paths(tmp_path: Path):
    manager = PredictiveTrainerManager(_make_config(tmp_path))

    payload = manager.status_payload()

    assert payload["scheduler_enabled"] is False
    assert payload["scheduler_config_source"] == "test_disabled"
    assert payload["scheduler_disabled_reason"] == "test_fixture_disabled"
    assert payload["shadow_index_path"] == str(manager.config.shadow_sqlite_path)
    assert payload["shadow_index_legacy_path"] == str(manager.config.shadow_index_path)
    assert payload["shadow_duckdb_path"] == str(manager.config.shadow_duckdb_path)


def test_status_payload_reports_active_run_stage_and_artifact_flags(tmp_path: Path):
    manager = PredictiveTrainerManager(_make_config(tmp_path))
    run_id = "run-stage"
    run_dir = manager._run_dir(run_id)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "model_candidate.json").write_text("{}", encoding="utf-8")
    manager._active_run_id = run_id

    class _NeverDoneTask:
        def done(self) -> bool:
            return False

    manager._current_task = _NeverDoneTask()
    manager._write_manifest(
        run_id,
        {
            "run_id": run_id,
            "status": "running",
            "stage": "train",
            "stage_started_at": "2026-04-07T00:00:00Z",
            "stage_updated_at": "2026-04-07T00:00:05Z",
            "stage_message": "training candidate model",
        },
    )

    payload = manager.status_payload()

    assert payload["status"] == "running"
    assert payload["active_run_stage"] == "train"
    assert payload["active_run_stage_message"] == "training candidate model"
    assert payload["active_run"]["model_candidate_exists"] is True
    assert payload["active_run"]["training_candidate_exists"] is False
    assert payload["active_run"]["eval_pack_exists"] is False
    assert payload["active_run"]["log_paths"]["train_stdout"].endswith("train.stdout.log")


def test_status_payload_tolerates_missing_manifest_for_active_run(tmp_path: Path):
    manager = PredictiveTrainerManager(_make_config(tmp_path))
    manager._active_run_id = "run-missing"

    class _NeverDoneTask:
        def done(self) -> bool:
            return False

    manager._current_task = _NeverDoneTask()

    payload = manager.status_payload()

    assert payload["status"] == "running"
    assert payload["active_run_id"] == "run-missing"
    assert payload["active_run"] is None
    assert payload["active_run_stage"] is None


def test_run_logged_subprocess_streams_output_before_completion(tmp_path: Path):
    manager = PredictiveTrainerManager(_make_config(tmp_path))
    script_path = tmp_path / "writer.py"
    script_path.write_text(
        "\n".join(
            [
                "import sys",
                "import time",
                "print('stdout-start', flush=True)",
                "print('stderr-start', file=sys.stderr, flush=True)",
                "time.sleep(0.5)",
                "print('stdout-end', flush=True)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    stdout_path = tmp_path / "stream.stdout.log"
    stderr_path = tmp_path / "stream.stderr.log"
    result: dict[str, object] = {}

    def _target() -> None:
        result["value"] = manager._run_logged_subprocess(
            cmd=[sys.executable, str(script_path)],
            cwd=tmp_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timeout=5,
        )

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()

    deadline = time.time() + 2.0
    while time.time() < deadline:
        if stdout_path.exists() and "stdout-start" in stdout_path.read_text(encoding="utf-8"):
            break
        time.sleep(0.05)
    else:
        raise AssertionError("stdout log did not update before subprocess completion")

    thread.join(timeout=3.0)
    assert thread.is_alive() is False
    assert result["value"][0] == 0
    assert "stdout-end" in stdout_path.read_text(encoding="utf-8")
    assert "stderr-start" in stderr_path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_start_run_seeds_manifest_before_background_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    manager = PredictiveTrainerManager(_make_config(tmp_path))
    manager._shadow_index_stats_cache = manager._refresh_shadow_index_stats_cache_sync()
    gate = asyncio.Event()

    async def _fake_run_cycle(*, run_id: str, requested_by: str, auto_promote):
        await gate.wait()

    monkeypatch.setattr(manager, "_run_cycle", _fake_run_cycle)

    result = await manager.start_run(requested_by="manual", auto_promote=None)
    run_id = result["run_id"]
    manifest = manager._read_manifest(run_id)

    assert result["status"] == "started"
    assert manifest["stage"] == "dataset"
    assert manifest["stage_message"] == "queued for dataset build"
    assert manager.config.lock_path.exists() is True

    manager._current_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await manager._current_task


def test_run_cycle_sync_records_stage_progress_and_promotion_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    manager = PredictiveTrainerManager(_make_config(tmp_path))
    manager.config.training_path.write_text(
        json.dumps(
            {
                "training": {"rows": 10, "shadow_rows": 10, "executed_rows": 1, "version": "old"},
                "validation": {
                    "positive_rows": 6,
                    "negative_rows": 4,
                    "trained_at": "2026-04-07T00:00:00Z",
                },
            }
        ),
        encoding="utf-8",
    )
    manager.config.model_path.write_text(
        json.dumps(
            {
                "version": "old",
                "calibration": {
                    "global_mae_sol": 1.0,
                    "tradeability_head_brier": {"p_positive_after_cost": 0.5},
                },
                "data_quality": {"row_count": 10, "positive_net_sol_count": 6, "negative_net_sol_count": 4},
            }
        ),
        encoding="utf-8",
    )
    manager.config.calibration_path.write_text(json.dumps({"rows": 1}), encoding="utf-8")
    _write_shadow_sqlite_count(manager.config.shadow_sqlite_path, 220)
    manager._shadow_index_stats_cache = manager._refresh_shadow_index_stats_cache_sync()

    stage_writes: list[str] = []
    original_write_manifest = manager._write_manifest

    def _capture_manifest(run_id: str, payload: dict[str, object]) -> None:
        stage = str(payload.get("stage") or "")
        if stage:
            stage_writes.append(stage)
        original_write_manifest(run_id, payload)

    monkeypatch.setattr(manager, "_write_manifest", _capture_manifest)

    def _fake_logged_subprocess(*, cmd, cwd, stdout_path, stderr_path, timeout=None):
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text("ok\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        script_arg = cmd[2] if len(cmd) > 2 and cmd[1] == "-u" else cmd[1]
        script_name = Path(script_arg).name
        if script_name == "build_mc_dataset.py":
            dataset_dir = stdout_path.parent.parent / "dataset"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            (dataset_dir / "dataset_latest.jsonl").write_text('{"row":1}\n', encoding="utf-8")
            (dataset_dir / "dataset_latest_summary.json").write_text(
                json.dumps(
                    {
                        "rows": 20,
                        "ok": True,
                        "quality_gates": {},
                        "summary": str(dataset_dir / "dataset_latest_summary.json"),
                        "unknown_sleeve_ratio": 0.0,
                    }
                ),
                encoding="utf-8",
            )
            return 0, '{"ok": true}', ""
        if script_name == "train_predictive_entry_model.py":
            output_path = Path(cmd[cmd.index("--output") + 1])
            output_path.write_text(
                json.dumps(
                    {
                        "version": "predictive-entry-v12.2",
                        "trained_at": "2026-04-07T00:10:00Z",
                        "training_window": {"rows": 20, "executed_rows": 5, "shadow_rows": 15},
                        "data_quality": {
                            "row_count": 20,
                            "positive_net_sol_count": 12,
                            "negative_net_sol_count": 8,
                        },
                        "executed_data_quality": {"row_count": 5},
                        "shadow_data_quality": {"row_count": 15},
                        "calibration": {
                            "global_mae_sol": 0.8,
                            "tradeability_head_brier": {"p_positive_after_cost": 0.2},
                        },
                    }
                ),
                encoding="utf-8",
            )
            return 0, '{"rows": 20}', ""
        if script_name in {"update_predictive_next_state_ledger.py", "refresh_predictive_calibration.py"}:
            Path(cmd[cmd.index("--snapshot") + 1]).write_text(json.dumps({"rows": 6}), encoding="utf-8")
            Path(cmd[cmd.index("--ledger") + 1]).write_text('{"row":1}\n', encoding="utf-8")
            return 0, '{"ok": true}', ""
        if script_name == "report_model_eval_pack.py":
            Path(cmd[cmd.index("--out-json") + 1]).write_text(
                json.dumps(
                    {
                        "window": {"n": 20},
                        "calibration_and_sample_sufficiency": {"ok": True},
                        "decision_gaps": {"mean": 0.1},
                    }
                ),
                encoding="utf-8",
            )
            Path(cmd[cmd.index("--out-md") + 1]).write_text("# ok\n", encoding="utf-8")
            return 0, '{"ok": true}', ""
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(manager, "_run_logged_subprocess", _fake_logged_subprocess)
    monkeypatch.setattr(
        manager,
        "_promote_candidate",
        lambda **_kwargs: {"state": "promoted_pending_restart", "forced": False},
    )

    manager._run_cycle_sync(run_id="run-progress", requested_by="manual", auto_promote=None)

    manifest = manager._read_manifest("run-progress")
    ordered_stages: list[str] = []
    for stage in stage_writes:
        if not ordered_stages or ordered_stages[-1] != stage:
            ordered_stages.append(stage)

    assert ordered_stages == [
        "dataset",
        "train",
        "attest",
        "calibrate",
        "eval",
        "gate",
        "promote",
        "completed",
    ]
    assert manifest["status"] == "completed"
    assert manifest["stage"] == "completed"
    assert manifest["promotion"]["state"] == "promoted_pending_restart"
    assert (manager._run_dir("run-progress") / "promotion.json").exists()
    status_manifest = manager._manifest_for_status("run-progress", manifest)
    assert status_manifest["model_candidate_exists"] is True
    assert status_manifest["training_candidate_exists"] is True
    assert status_manifest["calibration_candidate_exists"] is True
    assert status_manifest["eval_pack_exists"] is True
    assert status_manifest["promotion_record_exists"] is True


def test_run_cycle_sync_marks_failed_stage_on_train_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    manager = PredictiveTrainerManager(_make_config(tmp_path))
    manager.config.training_path.write_text(
        json.dumps(
            {
                "training": {"rows": 10, "shadow_rows": 10, "executed_rows": 1, "version": "old"},
                "validation": {
                    "positive_rows": 6,
                    "negative_rows": 4,
                    "trained_at": "2026-04-07T00:00:00Z",
                },
            }
        ),
        encoding="utf-8",
    )
    manager.config.model_path.write_text("{}", encoding="utf-8")
    manager.config.calibration_path.write_text(json.dumps({"rows": 1}), encoding="utf-8")
    _write_shadow_sqlite_count(manager.config.shadow_sqlite_path, 220)
    manager._shadow_index_stats_cache = manager._refresh_shadow_index_stats_cache_sync()

    def _fake_logged_subprocess(*, cmd, cwd, stdout_path, stderr_path, timeout=None):
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text("ok\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        script_arg = cmd[2] if len(cmd) > 2 and cmd[1] == "-u" else cmd[1]
        script_name = Path(script_arg).name
        if script_name == "build_mc_dataset.py":
            dataset_dir = stdout_path.parent.parent / "dataset"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            (dataset_dir / "dataset_latest.jsonl").write_text('{"row":1}\n', encoding="utf-8")
            (dataset_dir / "dataset_latest_summary.json").write_text(
                json.dumps(
                    {
                        "rows": 20,
                        "ok": True,
                        "quality_gates": {},
                        "summary": str(dataset_dir / "dataset_latest_summary.json"),
                        "unknown_sleeve_ratio": 0.0,
                    }
                ),
                encoding="utf-8",
            )
            return 0, '{"ok": true}', ""
        if script_name == "train_predictive_entry_model.py":
            return (
                7,
                '{"kind":"trainer_progress","stage":"shadow_snapshot_start"}\n',
                '{"error":"snapshot failed for test","reason":"shadow_sqlite_snapshot_failed"}\n',
            )
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(manager, "_run_logged_subprocess", _fake_logged_subprocess)

    manager._run_cycle_sync(run_id="run-failed", requested_by="manual", auto_promote=None)

    manifest = manager._read_manifest("run-failed")
    assert manifest["status"] == "failed"
    assert manifest["stage"] == "failed"
    assert (
        manifest["error"]["message"]
        == "train step failed (7): shadow sqlite snapshot failed: snapshot failed for test"
    )


@pytest.mark.asyncio
async def test_run_cycle_offloads_sync_work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manager = PredictiveTrainerManager(_make_config(tmp_path))
    captured: dict[str, object] = {}

    async def _fake_run_sync_on_executor(func, /, *args, **kwargs):
        captured["func_name"] = getattr(func, "__name__", "")
        captured["kwargs"] = kwargs
        return None

    monkeypatch.setattr(manager, "_run_sync_on_executor", _fake_run_sync_on_executor)

    await manager._run_cycle(run_id="run-offload", requested_by="manual", auto_promote=None)

    assert captured["func_name"] == "_run_cycle_sync"
    assert captured["kwargs"]["run_id"] == "run-offload"
    assert manager._active_run_id is None
    assert manager._current_task is None


@pytest.mark.asyncio
async def test_shutdown_marks_running_manifest_cancelled(tmp_path: Path):
    manager = PredictiveTrainerManager(_make_config(tmp_path))
    run_id = "run-shutdown"
    manager._active_run_id = run_id
    manager._write_manifest(
        run_id,
        {
            "run_id": run_id,
            "status": "running",
            "started_at": "2026-04-07T00:00:00Z",
        },
    )
    manager.config.lock_path.parent.mkdir(parents=True, exist_ok=True)
    manager.config.lock_path.write_text("{}", encoding="utf-8")
    manager._current_task = asyncio.create_task(asyncio.sleep(3600))

    await manager.shutdown()

    manifest = manager._read_manifest(run_id)
    assert manifest["status"] == "cancelled"
    assert manifest["error"]["type"] == "TrainerShutdown"


@pytest.mark.asyncio
async def test_promote_run_updates_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manager = PredictiveTrainerManager(_make_config(tmp_path))
    run_id = "run-123"
    run_dir = manager._run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / "model_candidate.json"
    training_path = run_dir / "prelaunch_training_candidate.json"
    calibration_path = run_dir / "calibration_candidate.json"
    for path in (model_path, training_path, calibration_path):
        path.write_text("{}", encoding="utf-8")

    manager._write_manifest(
        run_id,
        {
            "run_id": run_id,
            "artifacts": {
                "candidate_model": str(model_path),
                "candidate_training": str(training_path),
                "candidate_calibration": str(calibration_path),
            },
            "promotion": {"state": "not_attempted"},
        },
    )

    monkeypatch.setattr(
        manager,
        "_promote_candidate",
        lambda **kwargs: {"state": "promoted_pending_restart", "forced": True},
    )

    result = await manager.promote_run(run_id, forced=True)
    updated = manager._read_manifest(run_id)

    assert result["state"] == "promoted_pending_restart"
    assert updated["promotion"]["state"] == "promoted_pending_restart"


@pytest.mark.asyncio
async def test_promote_run_uses_executor_helper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manager = PredictiveTrainerManager(_make_config(tmp_path))
    run_id = "run-executor"
    run_dir = manager._run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / "model_candidate.json"
    training_path = run_dir / "prelaunch_training_candidate.json"
    calibration_path = run_dir / "calibration_candidate.json"
    ledger_path = run_dir / "next_state_ledger_candidate.jsonl"
    for path in (model_path, training_path, calibration_path):
        path.write_text("{}", encoding="utf-8")
    ledger_path.write_text('{"row":1}\n', encoding="utf-8")

    manager._write_manifest(
        run_id,
        {
            "run_id": run_id,
            "artifacts": {
                "candidate_model": str(model_path),
                "candidate_training": str(training_path),
                "candidate_calibration": str(calibration_path),
                "candidate_ledger": str(ledger_path),
            },
            "promotion": {"state": "not_attempted"},
        },
    )

    captured: dict[str, object] = {}

    async def _fake_run_sync_on_executor(func, /, *args, **kwargs):
        captured["func_name"] = getattr(func, "__name__", "")
        return {"state": "promoted_pending_restart", "forced": True}

    monkeypatch.setattr(manager, "_run_sync_on_executor", _fake_run_sync_on_executor)

    result = await manager.promote_run(run_id, forced=True)

    assert captured["func_name"] == "_promote_candidate"
    assert result["state"] == "promoted_pending_restart"


def test_promote_candidate_copies_next_state_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manager = PredictiveTrainerManager(_make_config(tmp_path))
    run_id = "run-ledger"
    run_dir = manager._run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / "model_candidate.json"
    training_path = run_dir / "prelaunch_training_candidate.json"
    calibration_path = run_dir / "calibration_candidate.json"
    ledger_path = run_dir / "next_state_ledger_candidate.jsonl"
    model_path.write_text("{}", encoding="utf-8")
    training_path.write_text("{}", encoding="utf-8")
    calibration_path.write_text("{}", encoding="utf-8")
    ledger_path.write_text('{"row":1}\n', encoding="utf-8")

    monkeypatch.setattr(
        manager,
        "_current_open_positions",
        lambda work_dir: {"ok": True, "open_positions_remaining": 1},
    )
    monkeypatch.setattr(
        manager,
        "_repo_launch_ready",
        lambda: {"status_clean": True, "branch": "main", "head_matches_origin_main": True},
    )

    result = manager._promote_candidate(
        run_id=run_id,
        model_candidate=model_path,
        training_candidate=training_path,
        calibration_candidate=calibration_path,
        ledger_candidate=ledger_path,
        forced=False,
    )

    assert result["state"] == "promoted_pending_restart"
    assert manager.config.next_state_ledger_path.read_text(encoding="utf-8") == '{"row":1}\n'


def test_health_payload_clears_stale_repo_not_launch_ready_pending_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    manager = PredictiveTrainerManager(_make_config(tmp_path))
    manager.set_guidance_subscriber_count_provider(lambda: 1)
    manager.config.training_path.write_text(
        json.dumps(
            {
                "training": {
                    "rows": 120,
                    "shadow_rows": 110,
                    "raw_shadow_entry_count": 220,
                    "executed_rows": 6,
                    "version": "predictive-entry-v12.2",
                },
                "validation": {
                    "positive_rows": 70,
                    "negative_rows": 50,
                    "trained_at": "2026-04-09T03:36:05Z",
                },
            }
        ),
        encoding="utf-8",
    )
    manager.config.model_path.write_text(
        json.dumps(
            {
                "version": "predictive-entry-v12.2",
                "trained_at": "2026-04-09T03:36:05Z",
                "selected_shadow_row_count": 110,
                "selected_executed_row_count": 6,
                "selected_total_training_rows": 116,
                "executed_prior_audit": [{"mint": "mint-a"}],
            }
        ),
        encoding="utf-8",
    )
    manager.config.calibration_path.write_text(json.dumps({"rows": 8}), encoding="utf-8")
    _write_shadow_sqlite_count(manager.config.shadow_sqlite_path, 220)
    manager._shadow_index_stats_cache = manager._refresh_shadow_index_stats_cache_sync()
    manager._write_manifest(
        "run-stale-pending",
        {
            "run_id": "run-stale-pending",
            "status": "completed",
            "raw_shadow_entry_count": 220,
            "promotion": {"state": "promoted_pending_restart"},
        },
    )
    manager._pending_restart = {
        "run_id": "run-stale-pending",
        "reason": "repo_not_launch_ready",
        "repo_state": {
            "branch": "main",
            "status_clean": False,
            "tracked_status_clean": False,
            "head_matches_origin_main": True,
            "untracked_count": 12,
            "untracked_examples": ["docs/DEVELOPMENT 2.md"],
        },
    }
    monkeypatch.setattr(
        manager,
        "_repo_launch_ready",
        lambda: {
            "branch": "main",
            "status_clean": True,
            "tracked_status_clean": True,
            "head_matches_origin_main": True,
            "untracked_count": 9,
            "untracked_examples": ["docs/DEVELOPMENT 2.md", "scripts/build_mc_dataset 2.py"],
        },
    )
    monkeypatch.setattr(
        manager,
        "_current_open_positions",
        lambda _work_dir: {"ok": True, "open_positions_remaining": 0},
    )

    payload = manager.health_payload()

    assert payload["latest_completed_promotion_state"] == "promoted_pending_restart"
    assert payload["effective_promotion_state"] == "promoted_current"
    assert payload["pending_restart"] is None
    assert payload["repo_state"]["status_clean"] is True
    assert payload["repo_state"]["tracked_status_clean"] is True
    assert payload["repo_state"]["untracked_count"] == 9
    assert payload["repo_state"]["untracked_examples"] == [
        "docs/DEVELOPMENT 2.md",
        "scripts/build_mc_dataset 2.py",
    ]
    assert manager._pending_restart is None


def test_history_payload_reads_jsonl(tmp_path: Path):
    manager = PredictiveTrainerManager(_make_config(tmp_path))
    manager._append_history({"timestamp": "2026-04-05T00:00:00Z", "event": "trainer_run_completed"})

    payload = manager.history_payload(limit=10)

    assert payload["history"]
    assert payload["history"][-1]["event"] == "trainer_run_completed"


def test_promote_candidate_relaunch_uses_wrapper_replace_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manager = PredictiveTrainerManager(_make_config(tmp_path))
    run_id = "run-relaunch"
    run_dir = manager._run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    model_path = run_dir / "model_candidate.json"
    training_path = run_dir / "prelaunch_training_candidate.json"
    calibration_path = run_dir / "calibration_candidate.json"
    for path in (model_path, training_path, calibration_path):
        path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        manager,
        "_current_open_positions",
        lambda work_dir: {"ok": True, "open_positions_remaining": 0},
    )
    monkeypatch.setattr(
        manager,
        "_repo_launch_ready",
        lambda: {"status_clean": True, "branch": "main", "head_matches_origin_main": True},
    )

    captured: dict[str, object] = {}

    class _Process:
        pid = 4242

    def _fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env")
        captured["stdout"] = kwargs.get("stdout")
        captured["stderr"] = kwargs.get("stderr")
        captured["start_new_session"] = kwargs.get("start_new_session")
        return _Process()

    monkeypatch.setattr("src.api.predictive_trainer.subprocess.Popen", _fake_popen)

    result = manager._promote_candidate(
        run_id=run_id,
        model_candidate=model_path,
        training_candidate=training_path,
        calibration_candidate=calibration_path,
        ledger_candidate=run_dir / "missing_ledger.jsonl",
        forced=False,
    )

    assert result["state"] == "promoted_and_relaunch_requested"
    assert result["restart_pid"] == 4242
    assert result["restart_dispatch"] == "detached_wrapper"
    assert captured["cmd"] == [str(manager.config.deploy_script_path), "--live", "--skip-build"]
    assert captured["env"]["DEPLOY_WRAPPER_REPLACE_EXISTING"] == "1"
    assert captured["env"]["ALGOTRADER_WALLET"] == str(manager.config.wallet_path)
    assert captured["env"]["SIDECAR_REQUIRE_HEALTH"] == "0"
    assert captured["env"]["SIDECAR_SKIP_POST_LAUNCH_TRAINER_TRIGGER"] == "1"
    assert "REAL_ALGOTRADER_WALLET" not in captured["env"]
    assert captured["start_new_session"] is True
