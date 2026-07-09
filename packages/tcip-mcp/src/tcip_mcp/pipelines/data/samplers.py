"""Task-aware data samplers for handling class imbalance.

Each sampler wraps torch samplers but auto-computes weights from
dataset.class_distribution, so the agent just picks a strategy name.
"""

from __future__ import annotations

import math

import torch
from torch.utils.data import Sampler, WeightedRandomSampler as _TorchWeightedRandom

from tcip_mcp.pipelines.data.datasets import BaseDataset


def _target_class_id(target: dict, class_key: str | None = None) -> int | None:
    """Class id for a sample's target, honoring an explicit ``class_key`` or task defaults.

    Replaces the previously-hardcoded ``label``/``ranks``/``labels`` detection: a dataset
    that names its class field differently can pass ``class_key`` instead of silently
    bucketing every sample as class 0. Returns None when no class is found.

    Detection/instance-seg tensor ``labels`` are 1-indexed (cid + 1, background = 0)
    while ``class_distribution`` keys are 0-indexed cids, so that fallback branch shifts
    back by 1. An explicit ``class_key`` returns the raw value unshifted — don't pass
    ``class_key="labels"`` for detection targets; rely on the fallback instead.
    """
    if class_key is not None:
        val = target.get(class_key)
    elif "label" in target:
        val = target["label"]
    elif "ranks" in target:
        val = target["ranks"]
    elif torch.is_tensor(target.get("labels")) and len(target["labels"]) > 0:
        # 1-indexed detection label -> 0-indexed class_distribution key.
        return int(target["labels"].reshape(-1)[0].item()) - 1
    else:
        return None
    if val is None:
        return None
    if torch.is_tensor(val):
        return int(val.reshape(-1)[0].item()) if val.numel() else None
    return int(val)


class ClassBalancedSampler(Sampler):
    """Over-/under-samples so each class appears equally often per epoch.

    Good for disease scoring and ordinal traits where some ranks are rare.
    """

    def __init__(self, dataset: BaseDataset, class_key: str | None = None) -> None:
        self.class_key = class_key
        dist = dataset.class_distribution
        if not dist:
            self._indices = list(range(len(dataset)))
            self._length = len(dataset)
            return

        # Build per-sample weight: inverse class frequency
        self._weights = self._compute_weights(dataset, dist, class_key)
        self._length = len(dataset)

    @staticmethod
    def _compute_weights(
        dataset: BaseDataset, dist: dict[int, int], class_key: str | None = None,
    ) -> torch.Tensor:
        total = sum(dist.values())
        n_classes = len(dist)
        class_weight = {cid: total / (n_classes * cnt) for cid, cnt in dist.items()}

        weights = []
        for i in range(len(dataset)):
            _, target = dataset[i]
            cid = _target_class_id(target, class_key)
            weights.append(class_weight.get(cid, 1.0) if cid is not None else 1.0)
        return torch.tensor(weights, dtype=torch.double)

    def __iter__(self):
        if hasattr(self, "_weights"):
            idx = torch.multinomial(self._weights, self._length, replacement=True)
            return iter(idx.tolist())
        return iter(self._indices)

    def __len__(self) -> int:
        return self._length


class OverSampler(Sampler):
    """Duplicate minority-class samples so all classes have >= min_count.

    Use when some classes have <10 samples (common for rare phenotypes).
    """

    def __init__(self, dataset: BaseDataset, min_count: int = 50, class_key: str | None = None) -> None:
        dist = dataset.class_distribution
        self._indices: list[int] = list(range(len(dataset)))
        if not dist:
            return

        # Gather per-class indices
        class_indices: dict[int, list[int]] = {cid: [] for cid in dist}
        for i in range(len(dataset)):
            _, target = dataset[i]
            cid = _target_class_id(target, class_key)
            if cid is not None and cid in class_indices:
                class_indices[cid].append(i)

        # Duplicate minority classes
        extras: list[int] = []
        for cid, indices in class_indices.items():
            if len(indices) < min_count and len(indices) > 0:
                reps = math.ceil(min_count / len(indices))
                extras.extend(indices * reps)
        self._indices.extend(extras)

    def __iter__(self):
        # Global RNG (like ClassBalancedSampler's multinomial): order varies per
        # epoch and is controlled by set_seed(). A default-constructed Generator
        # has a fixed seed, which froze the order identically every epoch.
        perm = torch.randperm(len(self._indices))
        return iter([self._indices[i] for i in perm])

    def __len__(self) -> int:
        return len(self._indices)


class WeightedRandomSampler(Sampler):
    """Thin wrapper around torch's WeightedRandomSampler with auto-weights."""

    def __init__(self, dataset: BaseDataset, class_key: str | None = None) -> None:
        dist = dataset.class_distribution
        n = len(dataset)
        if not dist:
            self._sampler = _TorchWeightedRandom(
                torch.ones(n), num_samples=n, replacement=True
            )
            return

        weights = ClassBalancedSampler._compute_weights(dataset, dist, class_key)
        self._sampler = _TorchWeightedRandom(weights, num_samples=n, replacement=True)

    def __iter__(self):
        return iter(self._sampler)

    def __len__(self) -> int:
        return len(self._sampler)


# ====================================================================
# Factory
# ====================================================================

_SAMPLER_MAP = {
    "random": None,  # use default DataLoader shuffle
    "class_balanced": ClassBalancedSampler,
    "oversample": OverSampler,
    "weighted_random": WeightedRandomSampler,
}


def build_sampler(name: str, dataset: BaseDataset, **kwargs) -> Sampler | None:
    """Build a sampler by name. Returns None for 'random' (use shuffle=True)."""
    if name not in _SAMPLER_MAP:
        raise ValueError(f"Unknown sampler '{name}'. Available: {list(_SAMPLER_MAP.keys())}")
    cls = _SAMPLER_MAP[name]
    if cls is None:
        return None
    return cls(dataset, **kwargs)
