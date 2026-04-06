import asyncio
import json
from pathlib import Path

import pytest

from src.api.predictive_trainer import PredictiveTrainerConfig, PredictiveTrainerManager


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
        scheduler_enabled=False,
        auto_promote=True,
        relaunch_enabled=True,
        python_bin="python3",
        train_timeout_secs=1800,
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

    config = PredictiveTrainerConfig.from_env(tmp_path / "data")

    assert config.scheduler_enabled is True
    assert config.auto_promote is True
    assert config.relaunch_enabled is True


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

    class _Completed:
        stdout = "ok"
        stderr = ""

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env")
        return _Completed()

    monkeypatch.setattr("src.api.predictive_trainer.subprocess.run", _fake_run)

    result = manager._promote_candidate(
        run_id=run_id,
        model_candidate=model_path,
        training_candidate=training_path,
        calibration_candidate=calibration_path,
        ledger_candidate=run_dir / "missing_ledger.jsonl",
        forced=False,
    )

    assert result["state"] == "promoted_and_relaunched"
    assert captured["cmd"] == [str(manager.config.deploy_script_path), "--live", "--skip-build"]
    assert captured["env"]["DEPLOY_WRAPPER_REPLACE_EXISTING"] == "1"
