"""Dispatch, terminal state, and deliverable selection around a bespoke ``train(ctx)`` body.

The envelope decides the run's final status and its deliverable from what the body actually did,
never from the fact that a ``.pt`` exists: a canceled body, a body that recorded its own failure,
a body that raised and a body past its wall clock all leave a checkpoint on disk and none of them
may complete with one. When the run is genuinely complete, an explicit ``set_final_weights``
outranks the convention, and the convention itself prefers the best checkpoint the body saved
over the last one.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

import tcip_store as ts

torch = pytest.importorskip("torch")

from tcip_mcp.audit import audit_log_key  # noqa: E402
from tcip_mcp.experiments import observe  # noqa: E402
from tcip_mcp.pipelines.model_build import METRICS_KEY, STATE_DICT_KEY  # noqa: E402
from tcip_mcp.pipelines.training.envelope import TrainContext, run_training_envelope  # noqa: E402
from tests._verified_checkpoint_fixtures import completed_checkpoint as _completed  # noqa: E402


def _audit_statuses(root, tool="training_run"):
    events = ts.read_log(audit_log_key(root)).records
    return [e["status"] for e in events if e.get("tool") == tool]


def _start(tmp_path, body_name, *, deadline: float | None = None) -> tuple[TrainContext, Path]:
    """Run the body ``body_name`` of this module through the envelope, over a run directory the
    launcher's own writer opened."""
    from tcip_mcp.experiments import observe
    from tcip_mcp.pipelines.training.run_registry import observed_run
    from tests._verified_checkpoint_fixtures import detection_config, opened_run

    run_dir = opened_run(tmp_path, detection_config(
        tmp_path / "data", training_source=f"{__name__}:{body_name}", device="cpu"))
    run = observed_run(observe(run_dir))
    run.deadline = deadline
    ctx = TrainContext(run=run, train_loader=None, val_loader=None)
    run_training_envelope(ctx)
    return ctx, run_dir


def _train_saves_best_then_final(ctx):
    """A loop keeping a best-so-far checkpoint plus a last-epoch one, as the stock trainer does."""
    ctx.save_checkpoint({STATE_DICT_KEY: {}, METRICS_KEY: {"val_loss": 0.2, "epoch": 4}}, "model_best")
    ctx.save_checkpoint({STATE_DICT_KEY: {}, METRICS_KEY: {"val_loss": 0.8, "epoch": 9}}, "model_final")


def test_best_checkpoint_outranks_the_last_one_as_the_deliverable(tmp_path):
    ctx, run_dir = _start(tmp_path, "_train_saves_best_then_final")

    assert ctx.run.status == "completed"
    assert ctx.run.deliverable == ctx.run.saved["model_best"]
    checkpoint = _completed(run_dir)
    assert checkpoint is not None
    assert checkpoint["path"].endswith("model_best.pt")
    assert checkpoint["metrics"]["val_loss"] == 0.2
    assert checkpoint["metrics"]["epoch"] == 4


def _train_declares_its_own_deliverable(ctx):
    """A loop whose shippable weights live under a tag outside the model_best/model_final pair."""
    ctx.save_checkpoint({STATE_DICT_KEY: {}, METRICS_KEY: {"val_loss": 0.8, "epoch": 9}}, "model_best")
    ctx.save_checkpoint(
        {STATE_DICT_KEY: {}, METRICS_KEY: {"val_loss": 0.2, "epoch": 4}}, "ema_weights")
    ctx.set_final_weights("ema_weights")


def test_an_explicit_deliverable_outranks_the_filename_convention(tmp_path):
    ctx, run_dir = _start(tmp_path, "_train_declares_its_own_deliverable")

    assert ctx.run.status == "completed"
    assert ctx.run.deliverable == ctx.run.saved["ema_weights"]
    checkpoint = _completed(run_dir)
    assert checkpoint is not None
    assert checkpoint["path"].endswith("ema_weights.pt")
    assert checkpoint["metrics"]["val_loss"] == 0.2


def test_completed_metrics_source_is_training_source_for_a_bespoke_loop(tmp_path):
    """``save_checkpoint``'s own contract: a ``metrics`` key in the saved state becomes the
    completed checkpoint's metrics with ``metrics_source='training_source'``, since the platform
    wrote what the loop chose into the artifact and never measured it itself."""
    _, run_dir = _start(tmp_path, "_train_saves_best_then_final")

    checkpoint = _completed(run_dir)
    assert checkpoint is not None
    assert checkpoint["metrics_source"] == "training_source"
    assert checkpoint["metrics"]["val_loss"] == 0.2


class _OutsideTheContract:
    """A value ``torch.load(weights_only=True)`` refuses to unpickle."""


def _train_leaves_an_undecodable_deliverable(ctx):
    """A loop whose declared deliverable holds state the verified reader refuses to unpickle."""
    ctx.save_checkpoint({STATE_DICT_KEY: {}, "extra": _OutsideTheContract()}, "model_best")
    ctx.set_final_weights("model_best")


def test_a_deliverable_the_verified_reader_refuses_ends_the_run_failed(tmp_path):
    """Completion names only a checkpoint the verified reader admits: a declared deliverable
    whose payload it refuses ends the run failed, and its final status names no checkpoint."""
    ctx, run_dir = _start(tmp_path, "_train_leaves_an_undecodable_deliverable")

    final = observe(run_dir).final
    assert ctx.run.status == "failed"
    assert final["state"] == "failed" and final["error"]
    assert final["checkpoint"] is None
    assert _completed(run_dir) is None
    assert _audit_statuses(tmp_path) == ["running", "failed"]


def _train_stops_on_cancel(ctx):
    """A loop that checkpoints, then honors a cancellation request and returns."""
    from tcip_mcp.experiments import request_cancel

    ctx.save_checkpoint({STATE_DICT_KEY: {}, METRICS_KEY: {"val_loss": 0.4}}, "model_final")
    request_cancel(ctx.run_dir)


def test_a_canceled_run_completes_no_checkpoint_despite_its_weights(tmp_path):
    ctx, run_dir = _start(tmp_path, "_train_stops_on_cancel")

    assert ctx.run.status == "canceled"
    assert (run_dir / "model_final.pt").is_file()
    assert observe(run_dir).final["state"] == "canceled"
    assert _completed(run_dir) is None
    assert _audit_statuses(tmp_path) == ["running", "canceled"]


def _train_records_its_own_failure(ctx):
    """A loop that detects a bad run itself and marks it failed rather than raising."""
    ctx.save_checkpoint({STATE_DICT_KEY: {}, METRICS_KEY: {"val_loss": 0.4}}, "model_best")
    ctx.run.status = "failed"
    ctx.run.error = "loss diverged at stage 2"


def test_a_body_that_marks_itself_failed_is_not_promoted_to_completed(tmp_path):
    ctx, run_dir = _start(tmp_path, "_train_records_its_own_failure")

    assert ctx.run.status == "failed"
    final = observe(run_dir).final
    assert (final["state"], final["error"]) == ("failed", "loss diverged at stage 2")
    assert (run_dir / "model_best.pt").is_file()
    assert _completed(run_dir) is None
    assert _audit_statuses(tmp_path) == ["running", "failed"]


def _train_raises_after_checkpointing(ctx):
    """A loop that declares its best checkpoint as it improves, then dies partway through."""
    ctx.save_checkpoint({STATE_DICT_KEY: {}, METRICS_KEY: {"val_loss": 0.4}}, "model_best")
    ctx.set_final_weights("model_best")
    raise RuntimeError("device ran out of memory mid-epoch")


def test_a_raised_failure_closes_the_run_failed_and_completes_nothing(tmp_path):
    ctx, run_dir = _start(tmp_path, "_train_raises_after_checkpointing")

    assert ctx.run.status == "failed"
    assert "out of memory" in observe(run_dir).final["error"]
    assert (run_dir / "model_best.pt").is_file()
    assert _completed(run_dir) is None
    assert _audit_statuses(tmp_path) == ["running", "failed"]


def _train_finishes_past_the_wall_clock(ctx):
    """A loop that finishes normally and declares its weights after its deadline passed."""
    ctx.save_checkpoint({STATE_DICT_KEY: {}, METRICS_KEY: {"val_loss": 0.3}}, "model_best")
    ctx.set_final_weights("model_best")


def test_a_run_past_its_wall_clock_ends_failed_naming_it_with_no_checkpoint(tmp_path):
    """The wall clock is the run's own fact: a body that returned normally past its deadline
    ends failed naming the limit, and its weights are never the completed checkpoint."""
    ctx, run_dir = _start(tmp_path, "_train_finishes_past_the_wall_clock",
                          deadline=time.time() - 1)

    assert ctx.run.status == "failed"
    assert observe(run_dir).final["error"] == "exceeded max_wall_clock_seconds"
    assert (run_dir / "model_best.pt").is_file()
    assert _completed(run_dir) is None
    assert _audit_statuses(tmp_path) == ["running", "failed"]
