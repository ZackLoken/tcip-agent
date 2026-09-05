"""Collate functions for a task's ``DataLoader``: batches a list of per-sample
``(image, target)`` pairs into the shape ``train()`` and ``evaluate()`` both expect.
"""

from __future__ import annotations

from typing import Any

import torch


def _detection_collate(batch):
    """Detection/instance_seg: list of (img, target) → (list[img], list[target])."""
    images, targets = zip(*batch)
    return list(images), list(targets)


def _stack_collate(batch):
    """Classification/ordinal/regression/semantic_seg: stack into tensors."""
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
    """Return the right collate_fn for a task type."""
    if task in ("detection", "instance_seg"):
        return _detection_collate
    return _stack_collate
