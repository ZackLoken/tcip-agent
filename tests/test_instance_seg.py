"""Functional instance_seg via Mask R-CNN: masks reach the loss, predictions
carry masks, and metrics use segmentation IoU (segm AP)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("pycocotools")


def test_mask_rcnn_uses_masks_in_loss_and_predicts_masks():
    from tests import bespoke_models

    # resnet18: loss_mask / predicted-masks are mask-head assertions, independent of backbone
    # depth: the FPN out_channels normalizes the channel difference.
    model = bespoke_models.build_bespoke_instance_seg(num_classes=1, min_size=64, max_size=128)
    assert isinstance(model, bespoke_models.BespokeDetection)

    img = torch.rand(3, 64, 64)
    masks = torch.zeros((1, 64, 64), dtype=torch.uint8)
    masks[0, 10:40, 10:40] = 1
    target = {"boxes": torch.tensor([[10.0, 10.0, 40.0, 40.0]]),
              "labels": torch.tensor([1]), "masks": masks}

    model.train()
    losses = model([img], [target])
    assert "loss_mask" in losses                 # the mask GT actually drives a loss now

    model.eval()
    out = model([img])
    assert "masks" in out[0]                      # predictions carry instance masks


def test_segm_metrics_score_mask_overlap():
    from tcip_mcp.pipelines.training.evaluation import (
        coco_detection_metrics,
        records_from_detector,
    )

    mask = torch.zeros((1, 32, 32), dtype=torch.uint8)
    mask[0, 8:24, 8:24] = 1
    target = {"boxes": torch.tensor([[8.0, 8.0, 24.0, 24.0]]),
              "labels": torch.tensor([1]), "masks": mask}
    output = {"boxes": torch.tensor([[8.0, 8.0, 24.0, 24.0]]),
              "labels": torch.tensor([1]), "scores": torch.tensor([0.99]),
              "masks": mask.unsqueeze(0).float()}  # [N, 1, H, W] soft masks

    rec = records_from_detector(target, output, width=32, height=32, include_masks=True)
    m = coco_detection_metrics([rec], iou_type="segm")
    assert m["iou_type"] == "segm"
    assert m["map50"] > 0.5                       # a perfect mask match scores high
