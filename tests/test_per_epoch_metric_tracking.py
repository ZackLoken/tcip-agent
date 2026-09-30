"""Coverage: TensorBoard event files carry train and validation loss (and accuracy where the
task defines one) at every epoch step, on both the direct training path and the sweep-trial
body an HPO run executes each trial through.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader

from tcip_mcp.pipelines.training.collation import task_collate
from tcip_mcp.pipelines.training.generic_trainer import train
from tests.tiny_trainer_fixtures import ConstantImageClassDataset, trainer_run

CLASSIFIER_BUILDER = "tests.tiny_trainer_fixtures:build_mean_intensity_classifier"


def _scalar_steps(log_dir: Path, tag: str) -> list[int]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    acc.Reload()
    return [e.step for e in acc.Scalars(tag)]


def _seed_leaf_detection_dataset(root: Path) -> tuple[Path, Path, Path, Path]:
    from PIL import Image
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    images_dir, labels_dir = root / "images", root / "labels"
    val_images, val_labels = root / "val_images", root / "val_labels"
    for d in (images_dir, labels_dir, val_images, val_labels):
        d.mkdir(parents=True)
    box = BBox(10, 10, 30, 30)
    for i in range(2):
        Image.new("RGB", (128, 128)).save(images_dir / f"t{i}.png")
        json_io.write_annotations(str(labels_dir / f"t{i}.json"),
                                  [Annotation(subject="leaf", geometry=box)], 128, 128)
    Image.new("RGB", (128, 128)).save(val_images / "v0.png")
    json_io.write_annotations(str(val_labels / "v0.json"),
                              [Annotation(subject="leaf", geometry=box)], 128, 128)
    return images_dir, labels_dir, val_images, val_labels


def test_classification_training_writes_train_and_val_scalars_every_epoch(tmp_path):
    train_ds = ConstantImageClassDataset(
        [-2.0, -1.5, -1.0, 1.0, 1.5, 2.0], [0, 0, 0, 1, 1, 1])
    val_ds = ConstantImageClassDataset([-1.8, -0.4, 0.4, 1.8], [0, 0, 1, 1])
    collate = task_collate("classification")
    train_loader = DataLoader(train_ds, batch_size=3, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=4, collate_fn=collate)

    config = {
        "model_source": {"builder": CLASSIFIER_BUILDER, "builder_kwargs": {"init_weight": -1.0},
                         "task": "classification"},
        "data": {"num_channels": 1, "num_classes": 2, "scope": {}},
        "device": "cpu",
        "mixed_precision": False,
        "stages": [{"freeze_to": 0, "epochs": 3}],
        "optimizer": {"name": "adamw", "backbone_lr": 0.2, "head_lr": 0.2, "weight_decay": 0.0},
        "checkpoint_every_n_epochs": 0,
        "early_stopping": {"enabled": False},
    }
    out_dir = tmp_path / "out"
    run = trainer_run(config, out_dir, project=tmp_path, has_val_loader=True, id="auto-run-42")
    run = train(run, train_loader, val_loader=val_loader)
    assert run.status == "completed", run.error

    tb_dir = out_dir / "tensorboard"
    assert _scalar_steps(tb_dir, "train/loss") == [1, 2, 3]
    assert _scalar_steps(tb_dir, "val/val_loss") == [1, 2, 3]
    assert _scalar_steps(tb_dir, "val/val_accuracy") == [1, 2, 3]


def test_hpo_trial_body_writes_train_and_val_loss_every_epoch(tmp_path):
    from tcip_mcp.pipelines.training.generic_trainer import resolve_objective
    from tcip_mcp.tools.training_tools import _run_hpo_trial, sweep_dir

    images_dir, labels_dir, val_images, val_labels = _seed_leaf_detection_dataset(tmp_path / "ds")
    base_config = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"min_size": 64, "max_size": 128},
                         "task": "detection"},
        "data": {"images_dir": str(images_dir), "labels_dir": str(labels_dir),
                 "scope": {"subject": "leaf"}},
        "batch_size": 1, "stages": [{"freeze_to": -1, "epochs": 2}],
                     "mixed_precision": False, "device": "cpu",
    }
    trial_dir = sweep_dir("hpo_study", project=tmp_path) / "trial_x"
    trial_dir.parent.mkdir(parents=True)
    reported: list[float] = []
    _run_hpo_trial({}, reported.append, base_config, trial_dir, project=tmp_path,
                   objective=resolve_objective(base_config, has_val_loader=True, project=tmp_path),
                   launched_by={"launcher": "process"})

    # One report per epoch's metrics row, which is the trial's whole result.
    assert len(reported) == 2

    tb_dir = trial_dir / "tensorboard"
    assert _scalar_steps(tb_dir, "train/loss") == [1, 2]
    assert _scalar_steps(tb_dir, "val/val_loss") == [1, 2]


def test_the_epoch_console_line_carries_validation_metrics_beyond_loss(tmp_path, caplog):
    """The one-line-per-epoch summary a breeder or an operator tails must name accuracy and F1,
    not only the loss a plain reader would take for the whole story."""
    import logging

    train_ds = ConstantImageClassDataset(
        [-2.0, -1.5, -1.0, 1.0, 1.5, 2.0], [0, 0, 0, 1, 1, 1])
    val_ds = ConstantImageClassDataset([-1.8, -0.4, 0.4, 1.8], [0, 0, 1, 1])
    collate = task_collate("classification")
    train_loader = DataLoader(train_ds, batch_size=3, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=4, collate_fn=collate)

    config = {
        "model_source": {"builder": CLASSIFIER_BUILDER, "builder_kwargs": {"init_weight": -1.0},
                         "task": "classification"},
        "data": {"num_channels": 1, "num_classes": 2, "scope": {}},
        "device": "cpu",
        "mixed_precision": False,
        "stages": [{"freeze_to": 0, "epochs": 2}],
        "optimizer": {"name": "adamw", "backbone_lr": 0.2, "head_lr": 0.2, "weight_decay": 0.0},
        "checkpoint_every_n_epochs": 0,
        "early_stopping": {"enabled": False},
    }
    run = trainer_run(config, tmp_path / "out", project=tmp_path, has_val_loader=True, id="auto-run-43")
    with caplog.at_level(logging.INFO, logger="tcip_mcp.pipelines.training.generic_trainer"):
        run = train(run, train_loader, val_loader=val_loader)
    assert run.status == "completed", run.error

    epoch_lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Epoch")]
    assert len(epoch_lines) == 2
    for line in epoch_lines:
        assert "val_accuracy=" in line
        assert "val_f1=" in line
