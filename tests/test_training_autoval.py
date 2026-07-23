"""Auto train/val wiring (W4): _auto_train_val + detection val-loss pass.

These exercise the helper that derives a group-aware val split and the
generic_trainer ``_validate`` detection path (which W4 makes correct so a real
val loader can be wired without crashing the run).
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from torch.utils.data import DataLoader  # noqa: E402

from tcip_mcp.tools.training_tools import _auto_train_val  # noqa: E402
from tcip_mcp.pipelines.training.generic_trainer import (  # noqa: E402
    create_run,
    task_collate,
    train,
)
from tcip_annotation import json_io  # noqa: E402
from tcip_annotation.state import Annotation, BBox  # noqa: E402

IMG = 64


def _save_png(path: Path, bright: bool = False) -> None:
    from torchvision.utils import save_image

    base = 0.7 if bright else 0.0
    img = torch.rand(3, IMG, IMG) * 0.3 + base
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(img, str(path))


def _detection_dataset(root: Path, prefixes=("srcA", "srcB", "srcC", "srcD"), tiles=2):
    images_dir = root / "images"
    labels_dir = root / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    all_stems = []
    for pref in prefixes:
        for t in range(tiles):
            stem = f"{pref}_{t}_0"
            _save_png(images_dir / f"{stem}.png")
            json_io.write_annotations(
                str(labels_dir / f"{stem}.json"),
                [Annotation(subject="catkin", geometry=BBox(19.2, 19.2, 44.8, 44.8))],
                IMG,
                IMG,
                keep_empty=True,
            )
            all_stems.append(stem)
    return images_dir, labels_dir, all_stems


def test_auto_train_val_detection_splits(tmp_path: Path):
    images_dir, labels_dir, all_stems = _detection_dataset(tmp_path / "ds")
    data_cfg = {
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "subject": "catkin",
        "auto_val": True,
        "split": {"val_ratio": 0.4, "seed": 1},
    }
    train_ds, val_ds = _auto_train_val("detection", data_cfg, None)
    assert val_ds is not None
    assert set(train_ds.stems).isdisjoint(set(val_ds.stems))
    assert sorted(train_ds.stems + val_ds.stems) == sorted(all_stems)
    assert val_ds.transforms is None


def test_auto_train_val_ordinal_returns_none(tmp_path: Path):
    images_dir = tmp_path / "images"
    rows = []
    for i in range(4):
        _save_png(images_dir / f"img{i}.png")
        rows.append((f"img{i}", i % 2))
    csv_path = tmp_path / "ranks.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("stem", "rank"))
        w.writerows(rows)

    data_cfg = {"images_dir": str(images_dir), "csv_path": str(csv_path), "auto_val": True}
    _ds, val_ds = _auto_train_val("ordinal", data_cfg, None)
    assert val_ds is None


def test_auto_train_val_tiny_dataset_guard(tmp_path: Path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    _save_png(images_dir / "src_0_0.png")
    json_io.write_annotations(
        str(labels_dir / "src_0_0.json"),
        [Annotation(subject="catkin", geometry=BBox(19.2, 19.2, 44.8, 44.8))],
        IMG,
        IMG,
        keep_empty=True,
    )

    data_cfg = {"images_dir": str(images_dir), "labels_dir": str(labels_dir),
                "subject": "catkin", "auto_val": True}
    _train_ds, val_ds = _auto_train_val("detection", data_cfg, None)
    assert val_ds is None  # single group -> no leakage-free val possible


def test_train_emits_val_loss_with_autoval(tmp_path: Path):
    images_dir, labels_dir, _ = _detection_dataset(tmp_path / "ds")
    data_cfg = {
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "subject": "catkin",
        "auto_val": True,
        "split": {"val_ratio": 0.4, "seed": 1},
    }
    train_ds, val_ds = _auto_train_val("detection", data_cfg, None)
    assert val_ds is not None
    train_loader = DataLoader(train_ds, batch_size=2, collate_fn=task_collate("detection"))
    val_loader = DataLoader(val_ds, batch_size=2, collate_fn=task_collate("detection"))

    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1, "min_size": IMG, "max_size": IMG * 2},
                         "task": "detection"},
        "device": "cpu",
        "stages": [{"freeze_to": -1, "epochs": 1}],
        "mixed_precision": False,
        "optimizer": {"name": "adamw", "backbone_lr": 1e-4, "head_lr": 1e-3, "weight_decay": 0},
        "early_stopping": {"enabled": False},
    }
    run = create_run(cfg, str(tmp_path / "out"))
    run = train(run, train_loader, val_loader=val_loader, task="detection")

    assert run.status == "completed", getattr(run, "error", run.status)
    assert "val_loss" in run.metrics_history[-1]
    assert run.metrics_history[-1]["val_loss"] >= 0.0
