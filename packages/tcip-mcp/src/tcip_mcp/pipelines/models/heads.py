"""Detection and segmentation head builders — wraps torchvision models."""

from __future__ import annotations

import torch.nn as nn
import torchvision.models.detection as det
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator


_HEAD_REGISTRY = {
    "faster_rcnn": "FasterRCNN with ResNet50-FPN backbone",
    "fcos": "FCOS anchor-free detector",
    "retinanet": "RetinaNet single-stage detector",
    "mask_rcnn": "Mask R-CNN instance segmentation",
}


def list_heads() -> list[str]:
    """Return names of available detection/segmentation heads."""
    return sorted(_HEAD_REGISTRY.keys())


def build_detection_model(
    head: str,
    num_classes: int,
    pretrained_backbone: bool = True,
    min_size: int = 800,
    max_size: int = 1333,
    anchor_sizes: list[tuple[int, ...]] | None = None,
) -> nn.Module:
    """Build a complete detection model (backbone + head).

    Args:
        head: Detection head name.
        num_classes: Number of object classes (excluding background).
                     The model will use num_classes + 1 internally.
        pretrained_backbone: Use ImageNet-pretrained backbone weights.
        min_size: Minimum image size for the transform.
        max_size: Maximum image size for the transform.
        anchor_sizes: Custom anchor sizes for anchor-based detectors.

    Returns:
        A torchvision detection model (nn.Module).
    """
    if head not in _HEAD_REGISTRY:
        raise ValueError(f"Unknown head: {head}. Available: {list_heads()}")

    weights_backbone = "DEFAULT" if pretrained_backbone else None

    if head == "faster_rcnn":
        model = det.fasterrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=weights_backbone,
            min_size=min_size,
            max_size=max_size,
        )
        # Replace the classifier head for our number of classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)

        if anchor_sizes is not None:
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            model.rpn.anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

        return model

    elif head == "fcos":
        model = det.fcos_resnet50_fpn(
            weights=None,
            weights_backbone=weights_backbone,
            num_classes=num_classes + 1,
            min_size=min_size,
            max_size=max_size,
        )
        return model

    elif head == "retinanet":
        model = det.retinanet_resnet50_fpn(
            weights=None,
            weights_backbone=weights_backbone,
            num_classes=num_classes + 1,
            min_size=min_size,
            max_size=max_size,
        )
        return model

    elif head == "mask_rcnn":
        model = det.maskrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=weights_backbone,
            min_size=min_size,
            max_size=max_size,
        )
        # Replace box predictor
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
        # Replace mask predictor
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes + 1
        )

        if anchor_sizes is not None:
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            model.rpn.anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

        return model

    raise ValueError(f"No factory for head: {head}")
