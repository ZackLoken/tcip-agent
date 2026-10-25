"""Dispatch, terminal state, and deliverable selection around a bespoke ``train(ctx)`` body.

A canceled body, a body that recorded its own failure, a body that raised and a body past its
wall clock each leave a checkpoint on disk, and none of their runs completes with one. In a
completed run an explicit ``set_final_weights`` outranks the filename convention, and the
convention picks the best checkpoint the body saved over the last one.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tcip_mcp.experiments import observe  # noqa: E402
from tcip_mcp.pipelines.model_build import METRICS_KEY, STATE_DICT_KEY  # noqa: E402
from tcip_mcp.pipelines.training.envelope import (
    TrainContext,
    run_training_envelope,
)  # noqa: E402
from tests._audit_fixtures import audit_rows  # noqa: E402
from tests._verified_checkpoint_fixtures import (
    completed_checkpoint as _completed,
)  # noqa: E402


def _start(
    tmp_path: Path, body_name: str, *, deadline: float | None = None
) -> tuple[TrainContext, Path]:
    """Run this module's body ``body_name`` through the envelope and return its context.

    The run directory is one the launcher's own writer opened, its run's deadline set to
    ``deadline``; it is returned beside the context.
    """
    from tcip_mcp.experiments import observe
    from tcip_mcp.pipelines.training.run_registry import observed_run
    from tests._verified_checkpoint_fixtures import (
        detection_config,
        detector_declaring,
        opened_run,
    )

    run_dir = opened_run(
        tmp_path,
        detection_config(
            tmp_path / "data",
            project=tmp_path,
            model_source=detector_declaring(__file__),
            training_source=f"{Path(__file__).stem}:{body_name}",
        ),
    )
    run = observed_run(observe(run_dir))
    run.deadline = deadline
    ctx = TrainContext(run=run, train_loader=None, val_loader=None)
    run_training_envelope(ctx)
    return ctx, run_dir


def _train_saves_best_then_final(ctx: TrainContext) -> None:
    """Save ``model_best`` at epoch 4, val_loss 0.2, then ``model_final`` at epoch 9, 0.8."""
    ctx.save_checkpoint(
        {STATE_DICT_KEY: {}, METRICS_KEY: {"val_loss": 0.2, "epoch": 4}},
        "model_best",
    )
    ctx.save_checkpoint(
        {STATE_DICT_KEY: {}, METRICS_KEY: {"val_loss": 0.8, "epoch": 9}},
        "model_final",
    )


def test_best_checkpoint_outranks_the_last_one_as_the_deliverable(
    tmp_path: Path,
) -> None:
    ctx, run_dir = _start(tmp_path, "_train_saves_best_then_final")

    assert ctx.run.status == "completed"
    assert ctx.run.deliverable == ctx.run.saved["model_best"]
    checkpoint = _completed(run_dir)
    assert checkpoint is not None
    assert checkpoint["path"].endswith("model_best.pt")
    assert checkpoint["metrics"]["val_loss"] == 0.2
    assert checkpoint["metrics"]["epoch"] == 4


def _train_declares_its_own_deliverable(ctx: TrainContext) -> None:
    """Declare shippable weights saved under a tag outside the model_best/model_final pair."""
    ctx.save_checkpoint(
        {STATE_DICT_KEY: {}, METRICS_KEY: {"val_loss": 0.8, "epoch": 9}},
        "model_best",
    )
    ctx.save_checkpoint(
        {STATE_DICT_KEY: {}, METRICS_KEY: {"val_loss": 0.2, "epoch": 4}},
        "ema_weights",
    )
    ctx.set_final_weights("ema_weights")


def test_an_explicit_deliverable_outranks_the_filename_convention(
    tmp_path: Path,
) -> None:
    ctx, run_dir = _start(tmp_path, "_train_declares_its_own_deliverable")

    assert ctx.run.status == "completed"
    assert ctx.run.deliverable == ctx.run.saved["ema_weights"]
    checkpoint = _completed(run_dir)
    assert checkpoint is not None
    assert checkpoint["path"].endswith("ema_weights.pt")
    assert checkpoint["metrics"]["val_loss"] == 0.2


def test_completed_metrics_source_is_training_source_for_a_bespoke_loop(
    tmp_path: Path,
) -> None:
    """A bespoke loop's completed metrics carry ``metrics_source='training_source'``.

    A ``metrics`` key in the state ``save_checkpoint`` saved becomes the completed checkpoint's
    metrics, the best checkpoint's ``val_loss`` among them.
    """
    _, run_dir = _start(tmp_path, "_train_saves_best_then_final")

    checkpoint = _completed(run_dir)
    assert checkpoint is not None
    assert checkpoint["metrics_source"] == "training_source"
    assert checkpoint["metrics"]["val_loss"] == 0.2


class _OutsideTheContract:
    """A value ``torch.load(weights_only=True)`` refuses to unpickle."""


def _train_leaves_an_undecodable_deliverable(ctx: TrainContext) -> None:
    """Save ``model_best`` with an ``_OutsideTheContract`` under ``extra``; declare it final."""
    ctx.save_checkpoint(
        {STATE_DICT_KEY: {}, "extra": _OutsideTheContract()}, "model_best"
    )
    ctx.set_final_weights("model_best")


def test_a_deliverable_the_verified_reader_refuses_ends_the_run_failed(
    tmp_path: Path,
) -> None:
    """A deliverable the verified reader refuses ends the run failed with no checkpoint."""
    ctx, run_dir = _start(tmp_path, "_train_leaves_an_undecodable_deliverable")

    final = observe(run_dir).final
    assert ctx.run.status == "failed"
    assert final.state == "failed" and final.status_error
    assert final.checkpoint is None
    assert _completed(run_dir) is None
    assert [r["status"] for r in audit_rows(tmp_path, "training_run")] == [
        "running",
        "failed",
    ]


def _train_stops_on_cancel(ctx: TrainContext) -> None:
    """Save a checkpoint, then request the run's cancellation and return."""
    from tcip_mcp.experiments import request_cancel

    ctx.save_checkpoint(
        {STATE_DICT_KEY: {}, METRICS_KEY: {"val_loss": 0.4}}, "model_final"
    )
    request_cancel(ctx.run_dir)


def test_a_canceled_run_completes_no_checkpoint_despite_its_weights(
    tmp_path: Path,
) -> None:
    ctx, run_dir = _start(tmp_path, "_train_stops_on_cancel")

    assert ctx.run.status == "canceled"
    assert (run_dir / "model_final.pt").is_file()
    assert observe(run_dir).final.state == "canceled"
    assert _completed(run_dir) is None
    assert [r["status"] for r in audit_rows(tmp_path, "training_run")] == [
        "running",
        "canceled",
    ]


def _train_records_its_own_failure(ctx: TrainContext) -> None:
    """Save a checkpoint, then set the run's status to failed with a reason and return."""
    ctx.save_checkpoint(
        {STATE_DICT_KEY: {}, METRICS_KEY: {"val_loss": 0.4}}, "model_best"
    )
    ctx.run.status = "failed"
    ctx.run.status_error = "loss diverged at stage 2"


def test_a_body_that_marks_itself_failed_is_not_promoted_to_completed(
    tmp_path: Path,
) -> None:
    ctx, run_dir = _start(tmp_path, "_train_records_its_own_failure")

    assert ctx.run.status == "failed"
    final = observe(run_dir).final
    assert (final.state, final.status_error) == (
        "failed",
        "loss diverged at stage 2",
    )
    assert (run_dir / "model_best.pt").is_file()
    assert _completed(run_dir) is None
    assert [r["status"] for r in audit_rows(tmp_path, "training_run")] == [
        "running",
        "failed",
    ]


def _train_raises_after_checkpointing(ctx: TrainContext) -> None:
    """Declare a best checkpoint, then raise partway through."""
    ctx.save_checkpoint(
        {STATE_DICT_KEY: {}, METRICS_KEY: {"val_loss": 0.4}}, "model_best"
    )
    ctx.set_final_weights("model_best")
    raise RuntimeError("device ran out of memory mid-epoch")


def test_a_raised_failure_closes_the_run_failed_and_completes_nothing(
    tmp_path: Path,
) -> None:
    ctx, run_dir = _start(tmp_path, "_train_raises_after_checkpointing")

    assert ctx.run.status == "failed"
    assert "out of memory" in observe(run_dir).final.status_error
    assert (run_dir / "model_best.pt").is_file()
    assert _completed(run_dir) is None
    assert [r["status"] for r in audit_rows(tmp_path, "training_run")] == [
        "running",
        "failed",
    ]


def _train_finishes_past_the_wall_clock(ctx: TrainContext) -> None:
    """Save and declare weights, then return normally."""
    ctx.save_checkpoint(
        {STATE_DICT_KEY: {}, METRICS_KEY: {"val_loss": 0.3}}, "model_best"
    )
    ctx.set_final_weights("model_best")


def test_a_run_past_its_wall_clock_ends_failed_naming_it_with_no_checkpoint(
    tmp_path: Path,
) -> None:
    """A run returning past its wall clock ends failed naming the limit, with no checkpoint."""
    ctx, run_dir = _start(
        tmp_path,
        "_train_finishes_past_the_wall_clock",
        deadline=time.time() - 1,
    )

    assert ctx.run.status == "failed"
    assert (
        observe(run_dir).final.status_error
        == "exceeded max_wall_clock_seconds"
    )
    assert (run_dir / "model_best.pt").is_file()
    assert _completed(run_dir) is None
    assert [r["status"] for r in audit_rows(tmp_path, "training_run")] == [
        "running",
        "failed",
    ]
