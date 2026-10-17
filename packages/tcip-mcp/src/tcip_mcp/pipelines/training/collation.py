"""Collate functions that batch a task's per-sample ``(image, target)`` pairs."""

from __future__ import annotations

from typing import Any

import torch


def _detection_collate(batch):
    """Batch detection pairs as ``(list of images, list of targets)``, unstacked."""
    images, targets = zip(*batch)
    return list(images), list(targets)


def _stack_collate(batch):
    """Stack a batch's images into one tensor and merge its target dicts key by key.

    A number per sample becomes a tensor, a tensor per sample a stacked tensor, and anything
    else a list.
    """
    raw_images, targets = zip(*batch)
    images = torch.stack(raw_images)
    # Merge target dicts, stack numeric values
    merged: dict[str, Any] = {}
    for key in targets[0]:
        vals = [t[key] for t in targets]
        if isinstance(vals[0], (int, float)):
            merged[key] = torch.tensor(vals)
        elif isinstance(vals[0], torch.Tensor):
            merged[key] = torch.stack(vals)
        else:
            merged[key] = vals
    return images, merged


def task_collate(task: str):
    """Return the collate function for ``task``: unstacked for a detector task, else stacked."""
    from tcip_mcp.pipelines.model_contract import DETECTOR_TASKS

    if task in DETECTOR_TASKS:
        return _detection_collate
    return _stack_collate
