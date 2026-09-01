"""What the trainer does with a run whose loss goes non-finite: a full epoch's worth of
consecutive non-finite batches stops the run as failed, but neither a single bad batch nor a
healthy run ever trips the same check.
"""

from __future__ import annotations


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

TRAIN_INTENSITIES = [0.10, 0.25, 0.40, 0.55, 0.70, 0.85]


def _train_loader() -> DataLoader:
    """Six samples at batch_size=2: three batches per epoch, the divergence threshold this
    module's tests all reason about."""
    train_ds = ConstantImageDataset(TRAIN_INTENSITIES, [2.0 * c for c in TRAIN_INTENSITIES])
    return DataLoader(train_ds, batch_size=2, collate_fn=task_collate("regression"))


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


def test_a_run_whose_loss_never_recovers_is_stopped_as_diverged(tmp_path):
    """Every batch is non-finite, so the streak reaches the run's own three-batch threshold at
    the end of the first epoch; the run must stop there, not run out its configured 30 epochs."""
    train_loader = _train_loader()
    run = create_run(_config(DIVERGED_BUILDER, {}, epochs=30), str(tmp_path / "out"))
    run = train(run, train_loader, val_loader=None, task="regression")

    assert run.status == "failed"
    assert "non-finite" in run.error
    assert "3" in run.error
    assert "diverged" in run.error
    assert run.current_epoch <= 2


def test_a_healthy_run_never_trips_the_divergence_check(tmp_path):
    """A run whose loss stays finite throughout completes normally and carries no divergence
    text, proving the counter never fires on a model that never produces a bad batch."""
    train_loader = _train_loader()
    run = create_run(_config(HEALTHY_BUILDER, {"init_weight": 0.0}, epochs=3), str(tmp_path / "out"))
    run = train(run, train_loader, val_loader=None, task="regression")

    assert run.status == "completed", run.error
    assert "diverged" not in run.error
    assert len(run.metrics_history) == 3


def test_a_transient_bad_streak_short_of_the_threshold_does_not_kill_the_run(tmp_path):
    """Two non-finite batches (short of the three-batch threshold this loader's epoch carries)
    followed by a recovery to finite losses must complete the run: one bad streak, or a
    transient spike, is not divergence."""
    train_loader = _train_loader()
    run = create_run(
        _config(TRANSIENT_BUILDER, {"bad_batches": 2, "init_weight": 0.0}, epochs=3),
        str(tmp_path / "out"))
    run = train(run, train_loader, val_loader=None, task="regression")

    assert run.status == "completed", run.error
    assert "diverged" not in run.error
    assert len(run.metrics_history) == 3
