"""Phase 2.2 — functional instance_seg via Mask R-CNN: masks reach the loss, predictions
carry masks, and metrics use segmentation IoU (segm AP)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("pycocotools")


def test_mask_rcnn_registered_and_recommended_for_instance_seg():
    import tcip_mcp.pipelines.components.detectors  # noqa: F401
    from tcip_mcp.pipelines.composer import recommend_model_spec
    from tcip_mcp.pipelines.registry import DETECTORS

    assert "mask_rcnn" in DETECTORS
    assert DETECTORS.describe("mask_rcnn")["valid_tasks"] == ["instance_seg"]
    spec = recommend_model_spec("instance_seg", dataset_size=300, num_classes=2)
    assert spec["heads"][0]["detector"] == "mask_rcnn"


def test_mask_rcnn_uses_masks_in_loss_and_predicts_masks():
    from tcip_mcp.pipelines.composer import DetectionModel, compose_model

    model = compose_model({
        "backbone": {"name": "tv_resnet50"},
        "neck": {"name": "fpn", "out_channels": 64},
        "heads": [{"name": "anchor_detection", "num_classes": 1, "detector": "mask_rcnn"}],
    })
    assert isinstance(model, DetectionModel)

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
