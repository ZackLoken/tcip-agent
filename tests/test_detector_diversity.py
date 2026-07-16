"""W6 — detector & architecture diversity: add_p2 necks, FCOS/RetinaNet, recommender.

Build specs use ``resnet18`` (FPN normalizes the channel difference); CPU, tiny inputs, 1 epoch.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

import tcip_mcp.pipelines.components.backbones  # noqa: F401,E402
import tcip_mcp.pipelines.components.necks  # noqa: F401,E402
import tcip_mcp.pipelines.components.heads  # noqa: F401,E402
import tcip_mcp.pipelines.components.losses  # noqa: F401,E402
from tcip_mcp.pipelines.components.necks import FPN, PAN  # noqa: E402
from tcip_mcp.pipelines.composer import compose_model, recommend_model_spec, validate_model_spec  # noqa: E402
from tcip_mcp.pipelines.registry import HEADS  # noqa: E402


def _features():
    return {
        "s0": torch.randn(1, 64, 32, 32),
        "s1": torch.randn(1, 128, 16, 16),
        "s2": torch.randn(1, 256, 8, 8),
        "s3": torch.randn(1, 512, 4, 4),
    }


def _det_spec(head_name, **head_extra):
    head = {"name": head_name, "num_classes": 1, "min_size": 64, "max_size": 128}
    head.update(head_extra)
    return {
        "backbone": {"name": "resnet18", "pretrained": False},
        "neck": {"name": "fpn", "out_channels": 256},
        "heads": [head],
    }


# --------------------------------------------------------------------------
# Necks — add_p2
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
    spec = _det_spec("anchor_detection")
    spec["neck"]["add_p2"] = True  # 5 pyramid levels
    model = compose_model(spec)  # must not raise (guards the [..][:5] truncation crash)
    assert len(model.detector.rpn.anchor_generator.sizes) == 5


def test_anchor_free_head_registered_and_valid():
    assert "anchor_free_detection" in HEADS
    spec = {
        "backbone": {"name": "resnet18"},
        "neck": {"name": "fpn"},
        "heads": [{"name": "anchor_free_detection", "num_classes": 1}],
    }
    assert validate_model_spec(spec) == []


# --------------------------------------------------------------------------
# Recommender
# --------------------------------------------------------------------------

def test_recommend_detection_tiny_objects():
    spec = recommend_model_spec("detection", 300, num_classes=2, object_size="tiny")
    assert spec["heads"][0]["name"] == "anchor_free_detection"
    assert spec["heads"][0]["detector"] == "fcos"
    assert spec["neck"].get("add_p2") is True
    assert spec["heads"][0]["anchor_base_size"] <= 8


def test_recommend_detection_default_unchanged():
    spec = recommend_model_spec("detection", 1000, num_classes=3)
    assert spec["neck"]["name"] == "fpn"
    assert "add_p2" not in spec["neck"]
    assert spec["heads"][0]["name"] == "anchor_detection"


# --------------------------------------------------------------------------
# End-to-end (anchor-free)
# --------------------------------------------------------------------------

def test_detection_anchor_free_e2e(tmp_path: Path):
    from torchvision.utils import save_image
    from torch.utils.data import DataLoader
    from tcip_mcp.pipelines.data.datasets import build_dataset
    from tcip_mcp.pipelines.training.generic_trainer import create_run, task_collate, train
    from tcip_annotation import json_io
    from tcip_annotation.state import BBox

    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    for i in range(4):
        save_image(torch.rand(3, 64, 64), str(images_dir / f"img{i}.png"))
        json_io.write_detect(
            str(labels_dir / f"img{i}.json"),
            [BBox(19.2, 19.2, 44.8, 44.8, 0)],
            64,
            64,
            keep_empty=True,
        )

    ds = build_dataset("detection", images_dir=str(images_dir), labels_dir=str(labels_dir), num_classes=1)
    loader = DataLoader(ds, batch_size=2, collate_fn=task_collate("detection"))

    spec = _det_spec("anchor_free_detection", detector="fcos")
    cfg = {
        "model_spec": spec, "device": "cpu", "stages": [{"freeze_to": -1, "epochs": 1}],
        "mixed_precision": False,
        "optimizer": {"name": "adamw", "backbone_lr": 1e-4, "head_lr": 1e-3, "weight_decay": 0},
        "early_stopping": {"enabled": False},
    }
    run = create_run(cfg, str(tmp_path / "out"))
    run = train(run, loader, val_loader=None, task="detection")
    assert run.status == "completed", getattr(run, "error", run.status)
    assert math.isfinite(run.metrics_history[-1]["train_loss"])
    assert (tmp_path / "out" / "model_best.pt").is_file()
