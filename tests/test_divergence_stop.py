"""What the trainer does with a run whose loss goes non-finite: two consecutive full training
passes with no finite batch loss stop the run as failed, uniform across loader shapes and reset
at every stage boundary; anything short of that (one bad epoch, a single finite loss inside an
otherwise-bad epoch, a healthy run) must never trip the same check.
"""

from __future__ import annotations

import time

import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader

from tcip_mcp.pipelines.training.generic_trainer import train
from tcip_mcp.pipelines.training.collation import task_collate
from tcip_mcp.pipelines.training.run_registry import create_run
from tests.tiny_trainer_fixtures import ConstantImageDataset

DIVERGED_BUILDER = "tests.tiny_trainer_fixtures:build_always_diverged_model"
HEALTHY_BUILDER = "tests.tiny_trainer_fixtures:build_mean_intensity_regressor"
TRANSIENT_BUILDER = "tests.tiny_trainer_fixtures:build_transiently_diverged_model"
STEP_COUNTED_BUILDER = "tests.tiny_trainer_fixtures:build_step_counted_divergence_model"

TRAIN_INTENSITIES = [0.10, 0.25, 0.40, 0.55, 0.70, 0.85]
DIVERGED_PASSES_PHRASE = "2 consecutive full training passes"


def _train_loader(batch_size: int = 2) -> DataLoader:
    """Six samples; ``batch_size`` sets the batches-per-epoch shape a test reasons about
    (2 -> three batches/epoch, 6 -> one batch/epoch)."""
    train_ds = ConstantImageDataset(TRAIN_INTENSITIES, [2.0 * c for c in TRAIN_INTENSITIES])
    return DataLoader(train_ds, batch_size=batch_size, collate_fn=task_collate("regression"))


def _config(builder: str, builder_kwargs: dict, *, epochs: int) -> dict:
    return {
        "model_source": {"builder": builder, "builder_kwargs": builder_kwargs,
                         "task": "regression", "in_chans": 1},
        "device": "cpu",
        "mixed_precision": False,
        "stages": [{"freeze_to": 0, "epochs": epochs}],
        "optimizer": {"name": "adamw", "backbone_lr": 0.05, "head_lr": 0.05, "weight_decay": 0.0},
        "checkpoint_every_n_epochs": 0,
        "early_stopping": {"enabled": False},
    }


@pytest.mark.parametrize("batch_size", [2, 6], ids=["three_batches_per_epoch", "one_batch_per_epoch"])
def test_a_run_whose_loss_never_recovers_stops_after_two_diverged_epochs(tmp_path, batch_size):
    """Every batch is non-finite, at two different loader shapes: the trigger is two consecutive
    full passes with no finite loss, never a batch-count-derived threshold, so both shapes stop
    at exactly the same epoch with the same wording."""
    train_loader = _train_loader(batch_size)
    run = create_run(_config(DIVERGED_BUILDER, {}, epochs=30), str(tmp_path / "out"))
    run = train(run, train_loader, val_loader=None, task="regression")

    assert run.status == "failed"
    assert DIVERGED_PASSES_PHRASE in run.error
    assert "diverged" in run.error
    assert run.current_epoch == 2


def test_a_healthy_run_never_trips_the_divergence_check(tmp_path):
    """A run whose loss stays finite throughout completes normally and carries no divergence
    text, proving the counter never fires on a model that never produces a bad batch."""
    train_loader = _train_loader()
    run = create_run(_config(HEALTHY_BUILDER, {"init_weight": 0.0}, epochs=3), str(tmp_path / "out"))
    run = train(run, train_loader, val_loader=None, task="regression")

    assert run.status == "completed", run.error
    assert "diverged" not in run.error
    assert len(run.metrics_history) == 3


def test_one_fully_diverged_epoch_followed_by_recovery_completes(tmp_path):
    """One fully diverged epoch, short of the two-pass rule, followed by a recovery to finite
    losses completes the run: the counter resets on the first epoch with a finite loss."""
    train_loader = _train_loader()
    run = create_run(
        _config(TRANSIENT_BUILDER, {"bad_batches": 3, "init_weight": 0.0}, epochs=3),
        str(tmp_path / "out"))
    run = train(run, train_loader, val_loader=None, task="regression")

    assert run.status == "completed", run.error
    assert "diverged" not in run.error
    assert len(run.metrics_history) == 3


def test_a_single_finite_loss_among_bad_batches_does_not_count_the_epoch_as_diverged(tmp_path):
    """One finite loss inside an otherwise-nan epoch must keep that epoch off the two-pass
    count: a second, fully diverged epoch right after would only stop the run if the first
    (wrongly) counted too."""
    train_loader = _train_loader()  # three batches/epoch
    run = create_run(
        _config(STEP_COUNTED_BUILDER, {"finite_at": [2]}, epochs=2),  # epoch 1's middle batch only
        str(tmp_path / "out"))
    run = train(run, train_loader, val_loader=None, task="regression")

    assert run.status == "completed", run.error
    assert len(run.metrics_history) == 2


def test_stage_boundary_resets_the_diverged_epoch_counter(tmp_path):
    """One fully diverged epoch at the end of stage 0 and another at the start of stage 1 must
    not stop the run: the counter resets at every stage boundary, so the two never add up."""
    train_loader = _train_loader()  # three batches/epoch, twelve calls total across four epochs
    config = _config(STEP_COUNTED_BUILDER, {"finite_at": [1, 2, 3, 10, 11, 12]}, epochs=2)
    config["stages"] = [{"freeze_to": 0, "epochs": 2}, {"freeze_to": 0, "epochs": 2}]
    run = create_run(config, str(tmp_path / "out"))
    run = train(run, train_loader, val_loader=None, task="regression")

    assert run.status == "completed", run.error
    assert run.current_epoch == 4


def test_cancel_requested_during_the_second_diverged_epoch_still_ends_failed(tmp_path):
    """The landed diverged-before-cancel ordering: a run whose cancel is requested partway
    through the epoch that trips the two-pass rule ends failed, not cancelled."""
    from tests.tiny_trainer_fixtures import CancelSentinelAtCall

    train_loader = _train_loader()  # three batches/epoch: epoch 2 is calls 4, 5, 6
    out_dir = str(tmp_path / "out")
    on_forward = CancelSentinelAtCall(out_dir, at_call=5)
    run = create_run(_config(DIVERGED_BUILDER, {"on_forward": on_forward}, epochs=30), out_dir)
    run = train(run, train_loader, val_loader=None, task="regression")

    assert run.status == "failed"
    assert DIVERGED_PASSES_PHRASE in run.error
    assert run.current_epoch == 2


def test_launch_training_real_subprocess_reports_the_diverged_stop(tmp_path, monkeypatch):
    """A model that passes the preflight contract on the synthetic (random, nonzero) batch, then
    divides by a real batch's zero pixel sum once real training starts, must be caught by a real
    launch_training subprocess: check_training_status names the diverged stop, not a silent hang
    or the launch-time placeholder."""
    pytest.importorskip("torchvision")
    from tcip_mcp.tools.training_tools import check_training_status, launch_training
    from tests.tiny_trainer_fixtures import write_regression_dataset

    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.tensorboard_manager.launch_tensorboard", lambda *a, **k: {})

    images_dir, csv_path = write_regression_dataset(
        tmp_path, intensities=[0.0, 0.0, 0.0, 0.0], values=[0.1, 0.2, 0.3, 0.4])
    labels_dir = tmp_path / "unused_labels"  # required by the unconditional directory check, unread
    labels_dir.mkdir()

    cfg = {
        "model_source": {"builder": "tests.tiny_trainer_fixtures:build_pixel_sum_divide_model",
                         "task": "regression", "in_chans": 3},
        "data": {"images_dir": str(images_dir), "csv_path": str(csv_path), "labels_dir": str(labels_dir)},
        "training": {"batch_size": 4, "stages": [{"freeze_to": 0, "epochs": 5}],
                     "mixed_precision": False, "device": "cpu",
                     "checkpoint_every_n_epochs": 0, "early_stopping": {"enabled": False}},
    }
    res = launch_training(cfg, str(tmp_path / "out"))
    assert "error" not in res, res
    run_id = res["run_id"]

    deadline = time.monotonic() + 60
    status: dict = {}
    while time.monotonic() < deadline:
        status = check_training_status(run_id)
        if status.get("status") in ("failed", "completed", "cancelled"):
            break
        time.sleep(0.5)

    assert status.get("status") == "failed", status
    assert status.get("error") is not None
    assert DIVERGED_PASSES_PHRASE in status["error"]
