"""Minimal end-to-end smoke tests for each composable task type.

The existing ``test_full_pipeline.py`` covers classification (synthetic) and
detection (gated on real sample data). This file locks the
compose -> build_dataset -> train contract for the task types that were
otherwise untested, on tiny synthetic data, CPU, one epoch. The assertion is
deliberately weak: the pipeline runs to completion and produces finite losses
and a checkpoint. That is enough to catch the regressions this repo is prone to
(e.g. a dataset/head/collate shape mismatch).

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

# Trigger component registration (side-effect imports).
import tcip_mcp.pipelines.components.backbones  # noqa: F401,E402
import tcip_mcp.pipelines.components.necks  # noqa: F401,E402
import tcip_mcp.pipelines.components.heads  # noqa: F401,E402
import tcip_mcp.pipelines.components.losses  # noqa: F401,E402

from tcip_mcp.pipelines.composer import compose_model  # noqa: E402
from tcip_mcp.pipelines.data.datasets import build_dataset  # noqa: E402
from tcip_mcp.pipelines.training.generic_trainer import (  # noqa: E402
    create_run,
    task_collate,
    train,
)

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


def _train_config(spec: dict) -> dict:
    return {
        "model_spec": spec,
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


def _detection_backbone_spec(head_name: str, num_classes: int, detector: str | None = None) -> dict:
    # Small input sizes keep the torchvision detector fast on CPU.
    head = {
        "name": head_name,
        "num_classes": num_classes,
        "min_size": IMG,
        "max_size": IMG * 2,
    }
    if detector:
        head["detector"] = detector
    return {
        "backbone": {"name": "resnet18", "pretrained": False},
        "neck": {"name": "fpn", "out_channels": 256},
        "heads": [head],
    }


def _gap_spec(head: dict) -> dict:
    return {
        "backbone": {"name": "resnet18", "pretrained": False},
        "neck": {"name": "gap"},
        "heads": [head],
    }


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
        (labels_dir / f"img{i}.txt").write_text("0 0.5 0.5 0.4 0.4\n")

    dataset = build_dataset(
        "detection", images_dir=str(images_dir), labels_dir=str(labels_dir), num_classes=1
    )
    loader = DataLoader(dataset, batch_size=2, collate_fn=task_collate("detection"))

    spec = _detection_backbone_spec("anchor_detection", num_classes=1)
    compose_model(spec)  # must compose without error
    run = create_run(_train_config(spec), str(tmp_path / "out"))
    run = train(run, loader, val_loader=None, task="detection")
    _assert_trained(run, tmp_path / "out")


def test_instance_seg_e2e(tmp_path: Path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    for i in range(4):
        _save_png(images_dir / f"img{i}.png")
        # a square polygon (>= 3 vertices -> >= 7 fields)
        (labels_dir / f"img{i}.txt").write_text(
            "0 0.3 0.3 0.7 0.3 0.7 0.7 0.3 0.7\n"
        )

    dataset = build_dataset(
        "instance_seg", images_dir=str(images_dir), labels_dir=str(labels_dir), num_classes=1
    )
    # Guard the polygon -> mask rasterization path (datasets.py). With the mask_rcnn
    # detector these masks now reach the Mask R-CNN mask loss during training.
    assert dataset[0][1]["masks"].shape[0] > 0
    loader = DataLoader(dataset, batch_size=2, collate_fn=task_collate("instance_seg"))

    spec = _detection_backbone_spec("anchor_detection", num_classes=1, detector="mask_rcnn")
    compose_model(spec)
    run = create_run(_train_config(spec), str(tmp_path / "out"))
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

    spec = {
        "backbone": {"name": "resnet18", "pretrained": False},
        "neck": {"name": "fpn", "out_channels": 256},
        "heads": [{"name": "semantic_seg", "num_classes": 2}],
    }
    compose_model(spec)
    run = create_run(_train_config(spec), str(tmp_path / "out"))
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

    spec = _gap_spec({"name": "ordinal", "num_ranks": 3})
    compose_model(spec)
    run = create_run(_train_config(spec), str(tmp_path / "out"))
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

    spec = _gap_spec({"name": "regression"})
    compose_model(spec)
    run = create_run(_train_config(spec), str(tmp_path / "out"))
    run = train(run, loader, val_loader=None, task="regression")
    _assert_trained(run, tmp_path / "out")
