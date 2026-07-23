"""Minimal end-to-end smoke tests for each task type via bespoke ``model_source`` builders.

The existing ``test_full_pipeline.py`` covers classification (synthetic) and
detection (gated on real sample data). This file locks the
build_model -> build_dataset -> train contract for the task types that were
otherwise untested, on tiny synthetic data, CPU, one epoch. The assertion is
deliberately weak: the pipeline runs to completion and produces finite losses
and a checkpoint. That is enough to catch the regressions this repo is prone to
(e.g. a dataset/head/collate shape mismatch).

Each smoke drives one of the sibling bespoke builders in ``tests.bespoke_models``.

Kept tiny (a handful of 64x64 images, 1 epoch) to stay well under the CI
per-test timeout.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from torch.utils.data import DataLoader

from tcip_mcp.pipelines.data.datasets import build_dataset  # noqa: E402
from tcip_mcp.pipelines.training.generic_trainer import (  # noqa: E402
    create_run,
    task_collate,
    train,
)
from tcip_annotation import json_io  # noqa: E402
from tcip_annotation.state import Annotation, BBox, Polygon  # noqa: E402

IMG = 64


# --------------------------------------------------------------------------
# Synthetic-data helpers
# --------------------------------------------------------------------------

def _save_png(path: Path, bright: bool = False) -> None:
    from torchvision.utils import save_image

    base = 0.7 if bright else 0.0
    img = torch.rand(3, IMG, IMG) * 0.3 + base
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(img, str(path))


def _model_source(builder: str, **kwargs) -> dict:
    return {"builder": f"tests.bespoke_models:{builder}", "builder_kwargs": kwargs}


def _train_config(model_source: dict) -> dict:
    return {
        "model_source": model_source,
        "device": "cpu",
        "stages": [{"freeze_to": -1, "epochs": 1}],
        "mixed_precision": False,
        "optimizer": {"name": "adamw", "backbone_lr": 1e-4, "head_lr": 1e-3, "weight_decay": 0},
        "scheduler": {"type": "cosine"},
        "early_stopping": {"enabled": False},
        "gradient_accumulation_steps": 1,
        "checkpoint_every_n_epochs": 1,
    }


def _assert_trained(run, output_dir: Path) -> None:
    assert run.status == "completed", getattr(run, "error", run.status)
    assert run.current_epoch == 1
    assert run.metrics_history
    train_loss = run.metrics_history[-1]["train_loss"]
    assert math.isfinite(train_loss), f"non-finite train_loss: {train_loss}"
    assert (output_dir / "model_best.pt").is_file()


# --------------------------------------------------------------------------
# Box-style tasks (detection, instance_seg) — torchvision detector path
# --------------------------------------------------------------------------

def test_detection_e2e(tmp_path: Path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    for i in range(4):
        _save_png(images_dir / f"img{i}.png")
        # one centered box covering the middle of the image
        json_io.write_annotations(
            str(labels_dir / f"img{i}.json"),
            [Annotation(subject="catkin", geometry=BBox(19.2, 19.2, 44.8, 44.8))],
            IMG,
            IMG,
            keep_empty=True,
        )

    dataset = build_dataset(
        "detection", images_dir=str(images_dir), labels_dir=str(labels_dir), subject="catkin"
    )
    loader = DataLoader(dataset, batch_size=2, collate_fn=task_collate("detection"))

    model_source = _model_source("build_bespoke_detection", num_classes=1,
                                 min_size=IMG, max_size=IMG * 2)
    run = create_run(_train_config(model_source), str(tmp_path / "out"))
    run = train(run, loader, val_loader=None, task="detection")
    _assert_trained(run, tmp_path / "out")


def test_instance_seg_e2e(tmp_path: Path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    for i in range(4):
        _save_png(images_dir / f"img{i}.png")
        # a square polygon (>= 3 vertices)
        json_io.write_annotations(
            str(labels_dir / f"img{i}.json"),
            [Annotation(subject="catkin",
                        geometry=Polygon([(19.2, 19.2), (44.8, 19.2), (44.8, 44.8), (19.2, 44.8)]))],
            IMG,
            IMG,
            keep_empty=True,
        )

    dataset = build_dataset(
        "instance_seg", images_dir=str(images_dir), labels_dir=str(labels_dir), subject="catkin"
    )
    # Guard the polygon -> mask rasterization path (datasets.py). With the mask_rcnn
    # detector these masks now reach the Mask R-CNN mask loss during training.
    assert dataset[0][1]["masks"].shape[0] > 0
    loader = DataLoader(dataset, batch_size=2, collate_fn=task_collate("instance_seg"))

    model_source = _model_source("build_bespoke_instance_seg", num_classes=1,
                                 min_size=IMG, max_size=IMG * 2)
    run = create_run(_train_config(model_source), str(tmp_path / "out"))
    run = train(run, loader, val_loader=None, task="instance_seg")
    _assert_trained(run, tmp_path / "out")


# --------------------------------------------------------------------------
# Dense / scalar tasks (semantic_seg, ordinal, regression) — stack-collate path
# --------------------------------------------------------------------------

def test_semantic_seg_e2e(tmp_path: Path):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    import numpy as np

    for i in range(4):
        _save_png(images_dir / f"img{i}.png")
        m = np.zeros((IMG, IMG), dtype=np.uint8)
        m[IMG // 4 : IMG // 2, IMG // 4 : IMG // 2] = 1  # a foreground block
        Image.fromarray(m, mode="L").save(masks_dir / f"img{i}.png")

    dataset = build_dataset(
        "semantic_seg", images_dir=str(images_dir), masks_dir=str(masks_dir), num_classes=2
    )
    loader = DataLoader(dataset, batch_size=2, collate_fn=task_collate("semantic_seg"))

    model_source = _model_source("build_bespoke_semantic_seg", num_classes=2)
    run = create_run(_train_config(model_source), str(tmp_path / "out"))
    run = train(run, loader, val_loader=None, task="semantic_seg")
    _assert_trained(run, tmp_path / "out")


def _write_csv(path: Path, rows: list[tuple[str, object]], header: tuple[str, str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def test_ordinal_e2e(tmp_path: Path):
    images_dir = tmp_path / "images"
    rows = []
    for i in range(6):
        _save_png(images_dir / f"img{i}.png", bright=(i % 3 == 0))
        rows.append((f"img{i}", i % 3))  # ranks 0..2
    csv_path = tmp_path / "ranks.csv"
    _write_csv(csv_path, rows, ("stem", "rank"))

    dataset = build_dataset(
        "ordinal", images_dir=str(images_dir), csv_path=str(csv_path), num_ranks=3
    )
    loader = DataLoader(dataset, batch_size=3, collate_fn=task_collate("ordinal"))

    model_source = _model_source("build_bespoke_ordinal", num_ranks=3)
    run = create_run(_train_config(model_source), str(tmp_path / "out"))
    run = train(run, loader, val_loader=None, task="ordinal")
    _assert_trained(run, tmp_path / "out")


def test_regression_e2e(tmp_path: Path):
    images_dir = tmp_path / "images"
    rows = []
    for i in range(6):
        _save_png(images_dir / f"img{i}.png", bright=(i % 2 == 0))
        rows.append((f"img{i}", float(i) / 6.0))
    csv_path = tmp_path / "values.csv"
    _write_csv(csv_path, rows, ("stem", "value"))

    dataset = build_dataset(
        "regression", images_dir=str(images_dir), csv_path=str(csv_path)
    )
    loader = DataLoader(dataset, batch_size=3, collate_fn=task_collate("regression"))

    model_source = _model_source("build_bespoke_regressor")
    run = create_run(_train_config(model_source), str(tmp_path / "out"))
    run = train(run, loader, val_loader=None, task="regression")
    _assert_trained(run, tmp_path / "out")
