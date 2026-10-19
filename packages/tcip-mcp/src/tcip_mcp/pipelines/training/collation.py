"""Collate functions that batch a task's per-sample ``(image, target)`` pairs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

type Example = tuple[torch.Tensor, dict]
type StackedTargets = dict[str, torch.Tensor | list]
type Batch = (
    tuple[list[torch.Tensor], list[dict]] | tuple[torch.Tensor, StackedTargets]
)
type Collate = Callable[[list[Example]], Batch]


def _detection_collate(batch: list[Example]) -> Batch:
    """Batch detection pairs as ``(list of images, list of targets)``, unstacked."""
    images, targets = zip(*batch)
    return list(images), list(targets)


def _stack_collate(batch: list[Example]) -> Batch:
    """Stack a batch's images into one tensor and merge its target dicts key by key.

    A number per sample becomes a tensor, a tensor per sample a stacked tensor, and anything
    else a list.
    """
    raw_images, targets = zip(*batch)
    images = torch.stack(raw_images)
    # Merge target dicts, stack numeric values
    merged: StackedTargets = {}
    for key in targets[0]:
        vals = [t[key] for t in targets]
        if isinstance(vals[0], (int, float)):
            merged[key] = torch.tensor(vals)
        elif isinstance(vals[0], torch.Tensor):
            merged[key] = torch.stack(vals)
        else:
            merged[key] = vals
    return images, merged


def task_collate(task: str) -> Collate:
    """Return the collate function for ``task``: unstacked for a detector task, else stacked."""
    from tcip_mcp.pipelines.model_contract import DETECTOR_TASKS

    if task in DETECTOR_TASKS:
        return _detection_collate
    return _stack_collate
