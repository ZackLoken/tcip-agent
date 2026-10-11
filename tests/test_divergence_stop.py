"""What the trainer does with a run whose loss goes non-finite: two consecutive full training
passes with no finite batch loss stop the run as failed, uniform across loader shapes and reset
at every stage boundary; anything short of that (one bad epoch, a single finite loss inside an
otherwise-bad epoch, a healthy run) must never trip the same check.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tcip_mcp.experiments import experiment_dir
from tcip_mcp.pipelines.training.generic_trainer import train
from tests.tiny_trainer_fixtures import (
    ALWAYS_DIVERGED_MODEL,
    MEAN_INTENSITY_REGRESSOR,
    frames_data,
    regressor_config,
    run_context,
)

TRANSIENT_BUILDER = "tiny_trainer_fixtures:build_transiently_diverged_model"
STEP_COUNTED_BUILDER = (
    "tiny_trainer_fixtures:build_step_counted_divergence_model"
)

TRAIN_INTENSITIES = [0.10, 0.25, 0.40, 0.55, 0.70, 0.85]
DIVERGED_PASSES_PHRASE = "2 consecutive full training passes"


def _trained(
    tmp_path,
    builder: str,
    builder_kwargs: dict,
    *,
    horizon: int,
    batch_size: int = 2,
    stages: list[dict] | None = None,
    experiment_id: str | None = None,
):
    """A regression run of ``builder`` over six training frames and no validation side, whose
    stage the schedule's ``horizon`` ends, its stop rule's patience
    (``_training_values.stop_rule``) past both that horizon and the two diverged passes that stop
    a run, opened by the launcher's own producer and trained over its own loader: its
    ``batch_size`` sets the batches-per-epoch shape a test reasons about (2 -> three
    batches/epoch, 6 -> one batch/epoch).
    """
    data = frames_data(
        tmp_path / "ds",
        TRAIN_INTENSITIES,
        [2.0 * c for c in TRAIN_INTENSITIES],
    )
    config = regressor_config(
        horizon,
        data=data,
        builder=builder,
        builder_kwargs=builder_kwargs,
        batch_size=batch_size,
    )
    if stages is not None:
        config["stages"] = stages
    ctx = run_context(tmp_path, config, experiment_id=experiment_id)
    return train(ctx.run, ctx.train_loader, val_loader=None)


@pytest.mark.parametrize(
    "batch_size",
    [2, 6],
    ids=["three_batches_per_epoch", "one_batch_per_epoch"],
)
def test_a_run_whose_loss_never_recovers_stops_after_two_diverged_epochs(
    tmp_path, batch_size
):
    """Every batch is non-finite, at two different loader shapes: the trigger is two consecutive
    full passes with no finite loss, never a batch-count-derived threshold, so both shapes stop
    at exactly the same epoch with the same wording.
    """
    run = _trained(
        tmp_path, ALWAYS_DIVERGED_MODEL, {}, horizon=30, batch_size=batch_size
    )

    assert run.status == "failed"
    assert DIVERGED_PASSES_PHRASE in run.status_error
    assert "diverged" in run.status_error
    assert run.current_epoch == 2


def test_a_healthy_run_never_trips_the_divergence_check(tmp_path):
    """A run whose loss stays finite throughout completes normally and carries no divergence
    text, proving the counter never fires on a model that never produces a bad batch.
    """
    run = _trained(
        tmp_path, MEAN_INTENSITY_REGRESSOR, {"init_weight": 0.0}, horizon=3
    )

    assert run.status == "completed", run.status_error
    assert "diverged" not in run.status_error
    assert len(run.metrics_history) == 3


def test_one_fully_diverged_epoch_followed_by_recovery_completes(tmp_path):
    """One fully diverged epoch, short of the two-pass rule, followed by a recovery to finite
    losses completes the run: the counter resets on the first epoch with a finite loss.
    """
    run = _trained(
        tmp_path,
        TRANSIENT_BUILDER,
        {"bad_batches": 3, "init_weight": 0.0},
        horizon=3,
    )

    assert run.status == "completed", run.status_error
    assert "diverged" not in run.status_error
    assert len(run.metrics_history) == 3


def test_a_single_finite_loss_among_bad_batches_does_not_count_the_epoch_as_diverged(
    tmp_path,
):
    """One finite loss inside an otherwise-nan epoch must keep that epoch off the two-pass
    count: a second, fully diverged epoch right after would only stop the run if the first
    (wrongly) counted too. Neither epoch's mean loss is finite, so the run then ends as having
    no selectable epoch, not as diverged.
    """
    # Epoch 1's middle batch only, at three batches/epoch.
    run = _trained(
        tmp_path, STEP_COUNTED_BUILDER, {"finite_at": [2]}, horizon=2
    )

    assert run.status == "failed"
    assert "no selectable epoch" in run.status_error
    assert DIVERGED_PASSES_PHRASE not in run.status_error
    assert len(run.metrics_history) == 2


def test_stage_boundary_resets_the_diverged_epoch_counter(tmp_path):
    """One fully diverged epoch at the end of stage 0 and another at the start of stage 1 must
    not stop the run: the counter resets at every stage boundary, so the two never add up.
    """
    # Three batches/epoch, twelve calls total across four epochs.
    run = _trained(
        tmp_path,
        STEP_COUNTED_BUILDER,
        {"finite_at": [1, 2, 3, 10, 11, 12]},
        horizon=2,
        stages=[{"freeze_to": 0}, {"freeze_to": 0}],
    )

    assert run.status == "completed", run.status_error
    assert run.current_epoch == 4


def test_cancel_requested_during_the_second_diverged_epoch_still_ends_failed(
    tmp_path,
):
    """Divergence is judged before cancellation: a run whose cancel is requested partway through
    the epoch that trips the two-pass rule ends failed, not canceled.
    """
    out_dir = str(experiment_dir("diverged-and-canceled", project=tmp_path))
    kwargs = {"cancel_at_call": 5, "cancel_output_dir": out_dir}
    # Three batches/epoch: epoch 2 is calls 4, 5, 6.
    run = _trained(
        tmp_path,
        ALWAYS_DIVERGED_MODEL,
        kwargs,
        horizon=30,
        experiment_id="diverged-and-canceled",
    )

    assert run.status == "failed"
    assert DIVERGED_PASSES_PHRASE in run.status_error
    assert run.current_epoch == 2


def test_launch_training_real_subprocess_reports_the_diverged_stop(
    tmp_path, monkeypatch
):
    """A model that passes the preflight contract on the synthetic (random, nonzero) batch, then
    divides by a real batch's zero pixel sum once real training starts, must be caught by a real
    launch_training subprocess: monitor_training names the diverged stop, not a silent hang
    or the launch-time placeholder.
    """
    pytest.importorskip("torchvision")
    from tcip_mcp.tools.training_tools import launch_training
    from tests._verified_checkpoint_fixtures import run_to_end
    from tests.tiny_trainer_fixtures import write_regression_dataset

    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.tensorboard_manager.launch_tensorboard",
        lambda *a, **k: {},
    )

    images_dir, csv_path = write_regression_dataset(
        tmp_path, intensities=[0.0, 0.0, 0.0, 0.0], values=[0.1, 0.2, 0.3, 0.4]
    )

    cfg = regressor_config(
        5,
        data={
            "images_dir": str(images_dir),
            "labels_dir": str(csv_path),
            "split": {"seed": 0, "val_ratio": 0.15},
        },
        builder="tiny_trainer_fixtures:build_pixel_sum_divide_model",
        batch_size=4,
    )
    res = launch_training(tmp_path, cfg, actor=None)
    assert "error" not in res, res
    experiment_id = res["experiment_id"]

    status = run_to_end(tmp_path, experiment_id, seconds=60)

    assert status.get("state") == "failed", status
    assert status["status_error"] is not None
    assert DIVERGED_PASSES_PHRASE in status["status_error"]
