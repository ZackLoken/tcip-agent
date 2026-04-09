"""Model builder — high-level API to construct models from config dicts."""

from __future__ import annotations

import torch.nn as nn

from tcip_mcp.pipelines.models.heads import build_detection_model, list_heads
from tcip_mcp.pipelines.models.backbones import list_backbones


def build_model(config: dict) -> nn.Module:
    """Build a model from a configuration dictionary.

    Config keys:
        task: 'detection' or 'segmentation' (required)
        head: Detection head name (default: 'faster_rcnn' for detection, 'mask_rcnn' for segmentation)
        num_classes: Number of classes excluding background (required)
        pretrained_backbone: bool (default: True)
        min_size: int (default: 800)
        max_size: int (default: 1333)
        anchor_sizes: optional list of tuples

    Returns:
        Constructed nn.Module.
    """
    task = config.get("task", "detection")
    supported_tasks = ("detection", "segmentation")
    if task not in supported_tasks:
        raise ValueError(f"Unsupported task: {task}. Supported: {supported_tasks}")

    # Default head depends on task
    if task == "segmentation":
        default_head = "mask_rcnn"
    else:
        default_head = "faster_rcnn"

    return build_detection_model(
        head=config.get("head", default_head),
        num_classes=config["num_classes"],
        pretrained_backbone=config.get("pretrained_backbone", True),
        min_size=config.get("min_size", 800),
        max_size=config.get("max_size", 1333),
        anchor_sizes=config.get("anchor_sizes"),
    )


def validate_model_config(config: dict) -> list[str]:
    """Validate a model config dict, returning a list of issues (empty = valid)."""
    issues: list[str] = []

    if "num_classes" not in config:
        issues.append("Missing required key: 'num_classes'")
    elif not isinstance(config["num_classes"], int) or config["num_classes"] < 1:
        issues.append("'num_classes' must be a positive integer")

    task = config.get("task", "detection")
    if task not in ("detection", "segmentation"):
        issues.append(f"Unsupported task: {task}")

    head = config.get("head", "faster_rcnn")
    if head not in list_heads():
        issues.append(f"Unknown head: {head}. Available: {list_heads()}")

    # Warn if task/head mismatch
    if task == "segmentation" and head != "mask_rcnn":
        issues.append(f"Segmentation task requires 'mask_rcnn' head, got '{head}'")

    return issues


def get_model_info() -> dict:
    """Return available model configurations."""
    return {
        "tasks": ["detection", "segmentation"],
        "heads": list_heads(),
        "backbones": list_backbones(),
        "default_head": "faster_rcnn",
        "default_backbone": "resnet50",
    }
