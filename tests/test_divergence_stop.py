"""What ``generic_trainer.train`` does with a run whose loss goes non-finite.

The count of consecutive full training passes with no finite batch loss the stop rule states
(``nonfinite_passes``) stops the run as failed at both loader shapes, and a stage boundary resets
the count; fewer bad epochs, a single finite loss inside an epoch of nan losses, and a healthy
run do not trip the check.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tcip_mcp.experiments import experiment_dir
from tcip_mcp.pipelines.training.generic_trainer import train
from tests._training_values import drawn_split, stage, stop_rule
from tests.tiny_trainer_fixtures import (
    ALWAYS_DIVERGED_MODEL,
    MEAN_INTENSITY_REGRESSOR,
    TRAIN_INTENSITIES,
    frames_data,
    regressor_config,
    run_context,
)

TRANSIENT_BUILDER = "tiny_trainer_fixtures:build_transiently_diverged_model"
STEP_COUNTED_BUILDER = (
    "tiny_trainer_fixtures:build_step_counted_divergence_model"
)

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
    nonfinite_passes: int = 2,
):
    """Train a regression run of ``builder`` over six frames with no validation side; return it.

    The config passes ``horizon``, ``batch_size``, ``builder`` with ``builder_kwargs``, and the
    stop rule ``_training_values.stop_rule(nonfinite_passes=...)``; ``stages`` replaces the
    config's own when given. The run is opened by the launcher's own producer under
    ``experiment_id`` and trained by ``generic_trainer.train`` over its own loader.
    """
    data = frames_data(
        tmp_path / "ds",
        TRAIN_INTENSITIES,
        [2.0 * c for c in TRAIN_INTENSITIES],
    )
    config = regressor_config(
        horizon,
        data=data,
        project=tmp_path,
        builder=builder,
        builder_kwargs=builder_kwargs,
        batch_size=batch_size,
        early_stopping=stop_rule(nonfinite_passes=nonfinite_passes),
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
    """A run whose loss never recovers stops after two diverged epochs, at either loader shape.

    Every batch is non-finite, and at both loader shapes the run fails at epoch two naming two
    consecutive full passes.
    """
    run = _trained(
        tmp_path, ALWAYS_DIVERGED_MODEL, {}, horizon=30, batch_size=batch_size
    )

    assert run.status == "failed"
    assert DIVERGED_PASSES_PHRASE in run.status_error
    assert "diverged" in run.status_error
    assert run.current_epoch == 2


def test_a_run_stops_at_the_diverged_pass_count_its_stop_rule_states(tmp_path):
    """A run stops at the diverged-pass count its stop rule states.

    A stop rule stating three diverged passes stops an always-diverged run at its third epoch
    naming the count, and lets a run diverged for two passes, one short of it, recover and
    complete.
    """
    stopped = _trained(
        tmp_path / "stopped",
        ALWAYS_DIVERGED_MODEL,
        {},
        horizon=30,
        nonfinite_passes=3,
    )
    assert stopped.status == "failed"
    assert "3 consecutive full training passes" in stopped.status_error
    assert stopped.current_epoch == 3

    # Three batches/epoch: six bad batches are two fully diverged epochs.
    recovered = _trained(
        tmp_path / "recovered",
        TRANSIENT_BUILDER,
        {"bad_batches": 6, "init_weight": 0.0},
        horizon=4,
        nonfinite_passes=3,
    )
    assert recovered.status == "completed", recovered.status_error
    assert len(recovered.metrics_history) == 4


def test_a_healthy_run_never_trips_the_divergence_check(tmp_path):
    """A healthy run does not trip the divergence check.

    A run whose loss stays finite throughout completes its three epochs and carries no
    divergence text.
    """
    run = _trained(
        tmp_path, MEAN_INTENSITY_REGRESSOR, {"init_weight": 0.0}, horizon=3
    )

    assert run.status == "completed", run.status_error
    assert "diverged" not in run.status_error
    assert len(run.metrics_history) == 3


def test_one_fully_diverged_epoch_followed_by_recovery_completes(tmp_path):
    """One fully diverged epoch followed by recovery completes the run.

    One fully diverged epoch, short of the two-pass rule, followed by a recovery to finite
    losses completes all three epochs.
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
    """A single finite loss among bad batches keeps the epoch off the diverged count.

    With one finite loss among the first epoch's nan losses and a fully diverged second epoch,
    the run is not stopped as diverged; it fails after two epochs as having no selectable epoch.
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
    """A stage boundary resets the diverged-epoch counter.

    One fully diverged epoch at the end of stage 0 and another at the start of stage 1 do not
    stop the run, which completes four epochs.
    """
    # Three batches/epoch, twelve calls total across four epochs.
    run = _trained(
        tmp_path,
        STEP_COUNTED_BUILDER,
        {"finite_at": [1, 2, 3, 10, 11, 12]},
        horizon=2,
        stages=[stage(0), stage(0)],
    )

    assert run.status == "completed", run.status_error
    assert run.current_epoch == 4


def test_cancel_requested_during_the_second_diverged_epoch_still_ends_failed(
    tmp_path,
):
    """A cancel requested during the second diverged epoch ends the run failed, not canceled.

    A run whose cancel is requested partway through the epoch that trips the two-pass rule ends
    failed at epoch two, naming the diverged stop.
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
    """A real ``launch_training`` subprocess reports the diverged stop.

    A model dividing by its batch's spread across bands, over gray frames of different
    intensities, launched through a real subprocess ends failed, naming the diverged stop.
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
        tmp_path, intensities=[0.1, 0.3, 0.5, 0.7], values=[0.1, 0.2, 0.3, 0.4]
    )

    cfg = regressor_config(
        5,
        data={
            "images_dir": str(images_dir),
            "labels_dir": str(csv_path),
            "split": drawn_split(val_ratio=0.15),
        },
        project=tmp_path,
        builder="tiny_trainer_fixtures:build_pixel_spread_divide_model",
        batch_size=4,
    )
    res = launch_training(tmp_path, cfg, actor=None)
    assert "error" not in res, res
    experiment_id = res["experiment_id"]

    status = run_to_end(tmp_path, experiment_id, seconds=60)

    assert status.get("state") == "failed", status
    assert status["status_error"] is not None
    assert DIVERGED_PASSES_PHRASE in status["status_error"]
