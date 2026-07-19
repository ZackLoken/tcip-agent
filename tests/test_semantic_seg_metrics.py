"""Semantic-segmentation mIoU / Dice scorer.

Unit tests pin the metric to hand-computed values on tiny synthetic label maps
(perfect overlap -> 1.0, disjoint -> 0.0, a known partial-overlap case, ignore-index,
absent-class handling), plus one end-to-end check that a ``semantic_seg`` ``evaluate()``
run surfaces the metric alongside ``loss``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")  # evaluation.py imports torch at module load

from tcip_mcp.pipelines.training.evaluation import (  # noqa: E402
    evaluate,
    semantic_seg_metrics,
)


# --------------------------------------------------------------------------
# semantic_seg_metrics — exact values on tiny synthetic maps
# --------------------------------------------------------------------------

def test_perfect_overlap_is_one():
    gt = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])
    m = semantic_seg_metrics(gt.clone(), gt, num_classes=2)
    assert m["mIoU"] == pytest.approx(1.0)
    assert m["dice"] == pytest.approx(1.0)
    assert m["pixel_acc"] == pytest.approx(1.0)
    assert m["per_class_iou"] == {0: pytest.approx(1.0), 1: pytest.approx(1.0)}


def test_disjoint_is_zero():
    gt = torch.zeros(4, 4, dtype=torch.long)
    pred = torch.ones(4, 4, dtype=torch.long)  # every pixel wrong, class swapped
    m = semantic_seg_metrics(pred, gt, num_classes=2)
    assert m["mIoU"] == pytest.approx(0.0)
    assert m["dice"] == pytest.approx(0.0)
    assert m["pixel_acc"] == pytest.approx(0.0)
    assert m["per_class_iou"] == {0: pytest.approx(0.0), 1: pytest.approx(0.0)}


def test_partial_overlap_hand_computed():
    # gt   = [0,0,0,0, 1,1,1,1]; pred = [0,0,1,1, 1,1,1,1]
    # class 0: inter=2 union=4 -> IoU 0.5,     dice 2*2/(2+4)=0.6667
    # class 1: inter=4 union=6 -> IoU 0.6667,  dice 2*4/(6+4)=0.8
    gt = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    pred = torch.tensor([0, 0, 1, 1, 1, 1, 1, 1])
    m = semantic_seg_metrics(pred, gt, num_classes=2)
    assert m["per_class_iou"][0] == pytest.approx(0.5)
    assert m["per_class_iou"][1] == pytest.approx(4 / 6)
    assert m["per_class_dice"][0] == pytest.approx(4 / 6)
    assert m["per_class_dice"][1] == pytest.approx(0.8)
    assert m["mIoU"] == pytest.approx((0.5 + 4 / 6) / 2)
    assert m["dice"] == pytest.approx((4 / 6 + 0.8) / 2)
    assert m["pixel_acc"] == pytest.approx(6 / 8)  # idx 2,3 wrong


def test_absent_class_excluded_from_mean():
    # num_classes=3 but class 2 appears in neither map -> reported None, out of the mean.
    gt = torch.tensor([0, 0, 1, 1])
    pred = torch.tensor([0, 0, 1, 1])
    m = semantic_seg_metrics(pred, gt, num_classes=3)
    assert m["per_class_iou"][2] is None
    assert m["per_class_dice"][2] is None
    assert m["mIoU"] == pytest.approx(1.0)  # only classes 0,1 counted


def test_ignore_index_drops_pixels():
    # Pixels 2,3 are ignore (255); the rest match perfectly.
    gt = torch.tensor([0, 0, 255, 255, 1, 1, 1, 1])
    pred = torch.tensor([0, 0, 1, 1, 1, 1, 1, 1])
    m = semantic_seg_metrics(pred, gt, num_classes=2, ignore_index=255)
    assert m["mIoU"] == pytest.approx(1.0)
    assert m["dice"] == pytest.approx(1.0)
    assert m["pixel_acc"] == pytest.approx(1.0)


def test_batched_shapes_flatten():
    # [N,H,W] input flattens the same way for pred and gt.
    gt = torch.zeros(2, 4, 4, dtype=torch.long)
    gt[:, :2] = 1
    m = semantic_seg_metrics(gt.clone(), gt, num_classes=2)
    assert m["mIoU"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# evaluate() surfaces the metric for a semantic_seg run
# --------------------------------------------------------------------------

def test_evaluate_semantic_seg_surfaces_miou(tmp_path: Path):
    pytest.importorskip("torchvision")
    from PIL import Image
    import numpy as np
    from torch.utils.data import DataLoader

    import tcip_mcp.pipelines.components.backbones  # noqa: F401
    import tcip_mcp.pipelines.components.necks  # noqa: F401
    import tcip_mcp.pipelines.components.heads  # noqa: F401
    from tcip_mcp.pipelines.composer import compose_model
    from tcip_mcp.pipelines.data.datasets import build_dataset
    from tcip_mcp.pipelines.training.generic_trainer import task_collate

    IMG = 64
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    for i in range(4):
        arr = (np.random.rand(IMG, IMG, 3) * 255).astype(np.uint8)
        Image.fromarray(arr).save(images_dir / f"img{i}.png")
        m = np.zeros((IMG, IMG), dtype=np.uint8)
        m[IMG // 4:IMG // 2, IMG // 4:IMG // 2] = 1  # a foreground block
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
    model = compose_model(spec)

    result = evaluate(model, loader, torch.device("cpu"), "semantic_seg")

    assert "loss" in result
    for key in ("mIoU", "dice", "pixel_acc", "per_class_iou", "per_class_dice"):
        assert key in result, f"missing {key}"
    assert 0.0 <= result["mIoU"] <= 1.0
    assert 0.0 <= result["dice"] <= 1.0
    assert 0.0 <= result["pixel_acc"] <= 1.0
    assert set(result["per_class_iou"]) == {0, 1}
