"""Coverage: TensorBoard event files carry train and validation loss (and accuracy where the
task defines one) at every epoch step, through the run's one writer, on both the default
trainer's path and the sweep-trial body an HPO run executes each trial through.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tcip_mcp.experiments import TENSORBOARD_DIR
from tcip_mcp.pipelines.training.generic_trainer import train
from tests.tiny_trainer_fixtures import (
    classifier_config,
    separable_classifier_loaders,
    trainer_run,
)


def _scalar_steps(log_dir: Path, tag: str) -> list[int]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    acc.Reload()
    return [e.step for e in acc.Scalars(tag)]


def test_classification_training_writes_train_and_val_scalars_every_epoch(tmp_path):
    train_loader, val_loader = separable_classifier_loaders()
    config = classifier_config(3)
    from tcip_mcp.experiments import METRICS_FILE
    from tcip_mcp.pipelines.training.envelope import TrainContext

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / METRICS_FILE).touch()
    run = trainer_run(config, out_dir, project=tmp_path, has_val_loader=True, id="auto-run-42")
    ctx = TrainContext(run=run, train_loader=train_loader, val_loader=val_loader)
    run = ctx.default_train()
    ctx.tb.close()
    assert run.status == "completed", run.status_error

    tb_dir = out_dir / TENSORBOARD_DIR
    assert _scalar_steps(tb_dir, "train_loss") == [1, 2, 3]
    assert _scalar_steps(tb_dir, "val_loss") == [1, 2, 3]
    assert _scalar_steps(tb_dir, "val_accuracy") == [1, 2, 3]


def test_hpo_trial_body_writes_train_and_val_loss_every_epoch(tmp_path):
    from tcip_mcp.experiments import board_of, run_dirs
    from tcip_mcp.tools.training_tools import _run_hpo_trial

    from tcip_annotation.state import Annotation, BBox

    from tests._chain_fixtures import training_config
    from tests._producer_fixtures import seed_labeled_images
    from tests._verified_checkpoint_fixtures import BUILT_DETECTOR, opened_sweep

    images_dir = seed_labeled_images(
        tmp_path / "ds" / "images" / "train",
        [Annotation(subject="leaf", geometry=BBox(10, 10, 30, 30))], n=2, width=128, height=128)
    base_config = training_config(
        BUILT_DETECTOR, {"images_dir": str(images_dir), "scope": {"subject": "leaf"},
                         "split": {"seed": 0, "val_ratio": 0.15}},
        stages=[{"freeze_to": -1, "epochs": 2}])
    reported: list[float] = []
    _run_hpo_trial({}, reported.append, opened_sweep(tmp_path, base_config), "x")
    (trial_dir,) = run_dirs(tmp_path)

    # One report per epoch's metrics row, which is the trial's whole result.
    assert len(reported) == 2

    tb_dir = board_of(trial_dir)
    assert _scalar_steps(tb_dir, "train_loss") == [1, 2]
    assert _scalar_steps(tb_dir, "val_loss") == [1, 2]


def test_the_epoch_console_line_carries_validation_metrics_beyond_loss(tmp_path, caplog):
    """The one-line-per-epoch summary a breeder or an operator tails must name accuracy and F1,
    not only the loss a plain reader would take for the whole story."""
    import logging

    train_loader, val_loader = separable_classifier_loaders()
    run = trainer_run(classifier_config(2), tmp_path / "out", project=tmp_path, has_val_loader=True,
                      id="auto-run-43")
    with caplog.at_level(logging.INFO, logger="tcip_mcp.pipelines.training.generic_trainer"):
        run = train(run, train_loader, val_loader=val_loader)
    assert run.status == "completed", run.status_error

    epoch_lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Epoch")]
    assert len(epoch_lines) == 2
    for line in epoch_lines:
        assert "val_accuracy=" in line
        assert "val_f1=" in line
