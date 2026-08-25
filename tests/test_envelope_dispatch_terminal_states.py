"""Dispatch, terminal state, and deliverable selection around a bespoke ``train(ctx)`` body.

The envelope decides the run's terminal state and its deliverable from what the body actually
did, never from the fact that a ``.pt`` exists: a cancelled body, a body that recorded its own
failure, and a body that raised all leave a checkpoint on disk and none of them may register a
model. When the run is genuinely complete, an explicit ``set_final_weights`` outranks the
filename convention and the convention itself prefers the best checkpoint over the last one.
Provenance is re-snapshotted after the body ran, so ``env.json`` carries the real outcome.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import tcip_store as ts

torch = pytest.importorskip("torch")

from tcip_mcp.audit import audit_log_key  # noqa: E402
from tcip_mcp.experiments import env_key, status_key  # noqa: E402
from tcip_mcp.pipelines.training.envelope import TrainContext, run_training_envelope  # noqa: E402
from tcip_mcp.pipelines.training.generic_trainer import create_run  # noqa: E402


def _audit_statuses(root, tool="training_run"):
    events = ts.read_log(audit_log_key(root)).records
    return [e["status"] for e in events if e.get("tool") == tool]


def _experiment_state(root, experiment_id):
    return ts.read(status_key(experiment_id, root=root))["state"]


def _start(tmp_path, experiment_id, body_name):
    from tcip_mcp.experiments import create_experiment, update_status

    config = {
        "model_source": {"builder": "x:y", "task": "detection", "in_chans": 3},
        "training_source": f"{__name__}:{body_name}",
        "device": "cpu",
    }
    create_experiment(experiment_id, config, data_source="imgs")
    update_status(experiment_id, "running")
    run = create_run(config, str(tmp_path / "out"))
    ctx = TrainContext(run=run, train_loader=None, val_loader=None, task="detection",
                       experiment_id=experiment_id)
    run_training_envelope(ctx)
    return ctx


def _train_saves_best_then_final(ctx):
    """A loop keeping a best-so-far checkpoint plus a last-epoch one, as the stock trainer does."""
    ctx.save_checkpoint({"model_state_dict": {}, "metrics": {"val_loss": 0.2, "epoch": 4}}, "model_best")
    ctx.save_checkpoint({"model_state_dict": {}, "metrics": {"val_loss": 0.8, "epoch": 9}}, "model_final")


def test_best_checkpoint_outranks_the_last_one_as_the_deliverable(tmp_path):
    from tcip_mcp.model_registry import ModelRegistry

    ctx = _start(tmp_path, "expBest", "_train_saves_best_then_final")

    assert ctx.run.status == "completed"
    assert ctx.final_weights.endswith("model_best.pt")
    entry = ModelRegistry(str(tmp_path)).get_model("expBest")
    assert entry is not None
    assert entry["checkpoint_path"].endswith("model_best.pt")
    assert entry["metrics"]["val_loss"] == 0.2
    assert entry["metrics"]["epoch"] == 4


def _train_declares_its_own_deliverable(ctx):
    """A loop whose shippable weights live under a tag outside the model_best/model_final pair."""
    ctx.save_checkpoint({"model_state_dict": {}, "metrics": {"val_loss": 0.8, "epoch": 9}}, "model_best")
    path = ctx.save_checkpoint(
        {"model_state_dict": {}, "metrics": {"val_loss": 0.2, "epoch": 4}}, "ema_weights")
    ctx.set_final_weights(path)


def test_an_explicit_deliverable_outranks_the_filename_convention(tmp_path):
    from tcip_mcp.model_registry import ModelRegistry

    ctx = _start(tmp_path, "expExplicit", "_train_declares_its_own_deliverable")

    assert ctx.run.status == "completed"
    assert ctx.final_weights.endswith("ema_weights.pt")
    entry = ModelRegistry(str(tmp_path)).get_model("expExplicit")
    assert entry is not None
    assert entry["checkpoint_path"].endswith("ema_weights.pt")
    assert entry["metrics"]["val_loss"] == 0.2


def test_registered_metrics_source_is_training_source_for_a_bespoke_loop(tmp_path):
    """``save_checkpoint``'s own contract (envelope.py): a ``metrics`` key in the saved state
    becomes the registered entry's metrics with ``metrics_source='training_source'``, since the
    platform wrote what the loop chose into the artifact and never measured it itself."""
    from tcip_mcp.model_registry import ModelRegistry

    _start(tmp_path, "expTrainingSource", "_train_saves_best_then_final")

    entry = ModelRegistry(str(tmp_path)).get_model("expTrainingSource")
    assert entry is not None
    assert entry["metrics_source"] == "training_source"
    assert entry["metrics"]["val_loss"] == 0.2


def _train_stops_on_cancel(ctx):
    """A loop that checkpoints, then honours a cancellation request and returns."""
    ctx.save_checkpoint({"model_state_dict": {}, "metrics": {"val_loss": 0.4}}, "model_final")
    ctx.run.cancel_event.set()


def test_a_cancelled_run_registers_no_model_despite_its_checkpoint(tmp_path):
    from tcip_mcp.model_registry import ModelRegistry

    ctx = _start(tmp_path, "expCancelled", "_train_stops_on_cancel")

    assert ctx.run.status == "cancelled"
    assert (tmp_path / "out" / "model_final.pt").is_file()
    assert ModelRegistry(str(tmp_path)).get_model("expCancelled") is None
    assert _experiment_state(tmp_path, "expCancelled") == "cancelled"
    assert _audit_statuses(tmp_path) == ["running", "cancelled"]


def _train_records_its_own_failure(ctx):
    """A loop that detects a bad run itself and marks it failed rather than raising."""
    ctx.save_checkpoint({"model_state_dict": {}, "metrics": {"val_loss": 0.4}}, "model_best")
    ctx.run.status = "failed"
    ctx.run.error = "loss diverged at stage 2"


def test_a_body_that_marks_itself_failed_is_not_promoted_to_completed(tmp_path):
    from tcip_mcp.model_registry import ModelRegistry

    ctx = _start(tmp_path, "expSelfFailed", "_train_records_its_own_failure")

    assert ctx.run.status == "failed"
    assert ctx.run.error == "loss diverged at stage 2"
    assert (tmp_path / "out" / "model_best.pt").is_file()
    assert ModelRegistry(str(tmp_path)).get_model("expSelfFailed") is None
    assert _experiment_state(tmp_path, "expSelfFailed") == "failed"
    assert _audit_statuses(tmp_path) == ["running", "failed"]


def _train_raises_after_checkpointing(ctx):
    """A loop that declares its best checkpoint as it improves, then dies partway through."""
    path = ctx.save_checkpoint({"model_state_dict": {}, "metrics": {"val_loss": 0.4}}, "model_best")
    ctx.set_final_weights(path)
    raise RuntimeError("device ran out of memory mid-epoch")


def test_a_raised_failure_closes_the_run_failed_and_registers_nothing(tmp_path):
    from tcip_mcp.model_registry import ModelRegistry

    ctx = _start(tmp_path, "expRaised", "_train_raises_after_checkpointing")

    assert ctx.run.status == "failed"
    assert "out of memory" in ctx.run.error
    assert (tmp_path / "out" / "model_best.pt").is_file()
    assert ModelRegistry(str(tmp_path)).get_model("expRaised") is None
    assert _experiment_state(tmp_path, "expRaised") == "failed"
    assert _audit_statuses(tmp_path) == ["running", "failed"]


def _train_restores_rng_state(ctx):
    """A loop reporting, as the stock trainer does, that it restored the checkpoint's RNG state."""
    ctx.save_checkpoint({"model_state_dict": {}, "metrics": {"val_loss": 0.4}}, "model_best")
    ctx.run.rng_state_restored = True


def test_env_provenance_carries_an_outcome_only_known_after_the_body_ran(tmp_path):
    ctx = _start(tmp_path, "expRng", "_train_restores_rng_state")

    assert ctx.run.status == "completed"
    env = ts.read(env_key("expRng"))
    assert env["rng_state_restored"] is True
    assert env["run_id"] == ctx.run.run_id
    assert env["seed"] == ctx.run.config["seed"]


def _train_races_the_wall_clock_watchdog(ctx):
    """A loop that finishes normally in-process while a separate watchdog thread has already
    marked the stored record failed, the one reachable complete_run refusal: the record was
    already terminal by the time the child reached its own finalize step."""
    from tcip_mcp.experiments import update_status

    path = ctx.save_checkpoint({"model_state_dict": {}, "metrics": {"val_loss": 0.3}}, "model_best")
    ctx.set_final_weights(path)
    update_status(ctx.experiment_id, "failed", error="killed by the wall-clock watcher")


def test_a_wall_clock_failed_record_reached_by_finalize_run_registers_nothing(tmp_path):
    """The behaviour change this row states: after the change, a run whose stored record turned
    terminal out from under it stays failed with no pointer and no registry entry, rather than
    ending completed and registered. complete_run's refusal reconciles ctx.run.status to the
    failed state the record actually holds, so the closing training_run audit event names failed
    too, rather than contradicting the record it closed over."""
    from tcip_mcp.experiments import artifacts_key
    from tcip_mcp.model_registry import ModelRegistry

    ctx = _start(tmp_path, "expWallClock", "_train_races_the_wall_clock_watchdog")

    # The body itself finished normally, but the completion write was refused (the watchdog's
    # failed already landed), so ctx.run.status is reconciled to what the record holds.
    assert ctx.run.status == "failed"
    assert ctx.final_weights.endswith("model_best.pt")
    assert Path(ctx.final_weights).is_file()
    assert ModelRegistry(str(tmp_path)).get_model("expWallClock") is None
    assert _experiment_state(tmp_path, "expWallClock") == "failed"
    assert "model_weights" not in ts.read(artifacts_key("expWallClock"))
    assert _audit_statuses(tmp_path) == ["running", "failed"]
