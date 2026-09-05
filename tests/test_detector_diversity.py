"""Detector & architecture diversity: add_p2 necks, FCOS end-to-end.

Uses ``resnet18`` (FPN normalizes the channel difference); CPU, tiny inputs, 1 epoch.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from tcip_mcp.pipelines.components.necks import FPN, PAN  # noqa: E402


def _features():
    return {
        "s0": torch.randn(1, 64, 32, 32),
        "s1": torch.randn(1, 128, 16, 16),
        "s2": torch.randn(1, 256, 8, 8),
        "s3": torch.randn(1, 512, 4, 4),
    }


# --------------------------------------------------------------------------
# Necks: add_p2
# --------------------------------------------------------------------------

def test_fpn_add_p2_adds_finer_level():
    out = FPN([64, 128, 256, 512], 64, add_p2=True)(_features())
    assert set(out.keys()) == {"p0", "p1", "p2", "p3", "p4"}
    assert out["p0"].shape[-1] == 2 * out["p1"].shape[-1]  # extra level is 2x finer


def test_pan_add_p2_levels_and_convs():
    pan = PAN([64, 128, 256, 512], 64, add_p2=True)
    assert len(pan(_features())) == 5
    assert len(pan.bottom_up_convs) == 4
    pan2 = PAN([64, 128, 256, 512], 64)
    assert len(pan2(_features())) == 4
    assert len(pan2.bottom_up_convs) == 3


def test_necks_default_off_unchanged():
    fpn_out = FPN([64, 128, 256, 512], 64)(_features())
    assert set(fpn_out.keys()) == {"p0", "p1", "p2", "p3"}
    assert fpn_out["p0"].shape[1] == 64
    assert set(PAN([64, 128, 256, 512], 64)(_features()).keys()) == {"p0", "p1", "p2", "p3"}


# --------------------------------------------------------------------------
# Detectors
# --------------------------------------------------------------------------

def test_detection_anchor_count_with_p2():
    import timm

    from tcip_mcp.pipelines.components.backbones import BackboneWrapper
    from tcip_mcp.pipelines.components.detectors import BackboneNeckAdapter, _build_faster_rcnn

    _m = timm.create_model("resnet18", pretrained=False, features_only=True,
                           out_indices=(1, 2, 3, 4))
    bb = BackboneWrapper(_m, _m.feature_info.channels())
    neck = FPN(bb.out_channels, 256, add_p2=True)  # 5 pyramid levels
    adapter = BackboneNeckAdapter(bb, neck)
    with torch.no_grad():
        names = list(adapter(torch.zeros(1, 3, 64, 64)).keys())
    assert len(names) == 5
    det = _build_faster_rcnn(  # must not raise (guards the [..][:5] truncation crash)
        adapter, 1, featmap_names=names, num_levels=len(names), min_size=64, max_size=128)
    assert len(det.rpn.anchor_generator.sizes) == 5


# --------------------------------------------------------------------------
# End-to-end (anchor-free)
# --------------------------------------------------------------------------

def test_detection_anchor_free_e2e(tmp_path: Path):
    from torchvision.utils import save_image
    from torch.utils.data import DataLoader
    from tcip_mcp.pipelines.data.datasets import build_dataset
    from tcip_mcp.pipelines.training.generic_trainer import train
    from tcip_mcp.pipelines.training.collation import task_collate
    from tcip_mcp.pipelines.training.run_registry import create_run
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    for i in range(4):
        save_image(torch.rand(3, 64, 64), str(images_dir / f"img{i}.png"))
        json_io.write_annotations(
            str(labels_dir / f"img{i}.json"),
            [Annotation(subject="bud", geometry=BBox(19.2, 19.2, 44.8, 44.8))],
            64,
            64,
            keep_empty=True,
        )

    ds = build_dataset("detection", images_dir=str(images_dir), labels_dir=str(labels_dir), subject="bud")
    loader = DataLoader(ds, batch_size=2, collate_fn=task_collate("detection"))

    model_source = {
        "builder": "tests.bespoke_models:build_bespoke_detection",
        "builder_kwargs": {"num_classes": 1, "detector": "fcos", "min_size": 64, "max_size": 128},
        "task": "detection",
    }
    cfg = {
        "model_source": model_source, "device": "cpu", "stages": [{"freeze_to": -1, "epochs": 1}],
        "mixed_precision": False,
        "optimizer": {"name": "adamw", "backbone_lr": 1e-4, "head_lr": 1e-3, "weight_decay": 0},
        "early_stopping": {"enabled": False},
        # No val_loader below: loss is the only metric coherent to select on without one.
        "evaluation": {"selection_metric": "loss"},
    }
    run = create_run(cfg, str(tmp_path / "out"))
    run = train(run, loader, val_loader=None, task="detection")
    assert run.status == "completed", getattr(run, "error", run.status)
    assert math.isfinite(run.metrics_history[-1]["train_loss"])
    assert (tmp_path / "out" / "model_best.pt").is_file()
