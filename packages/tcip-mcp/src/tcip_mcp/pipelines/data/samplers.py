"""Task-aware data samplers for handling class imbalance.

Each sampler wraps torch samplers but auto-computes weights from
dataset.class_distribution, so the agent just picks a strategy name.
"""

from __future__ import annotations

import math
from collections import Counter

import torch
from torch.utils.data import Sampler, WeightedRandomSampler as _TorchWeightedRandom

from tcip_mcp.pipelines.data.datasets import BaseDataset


class ClassBalancedSampler(Sampler):
    """Over-/under-samples so each class appears equally often per epoch.

    Good for disease scoring and ordinal traits where some ranks are rare.
    """

    def __init__(self, dataset: BaseDataset) -> None:
        dist = dataset.class_distribution
        if not dist:
            self._indices = list(range(len(dataset)))
            self._length = len(dataset)
            return

        max_count = max(dist.values())
        # Build per-sample weight: inverse class frequency
        # We need per-sample classes — iterate dataset target dicts
        self._weights = self._compute_weights(dataset, dist)
        self._length = len(dataset)

    @staticmethod
    def _compute_weights(dataset: BaseDataset, dist: dict[int, int]) -> torch.Tensor:
        total = sum(dist.values())
        n_classes = len(dist)
        class_weight = {cid: total / (n_classes * cnt) for cid, cnt in dist.items()}

        weights = []
        for i in range(len(dataset)):
            _, target = dataset[i]
            # Detect which key holds the class info
            if "label" in target:
                cid = target["label"]
            elif "rank" in target:
                cid = target["rank"]
            elif "labels" in target and len(target["labels"]) > 0:
                cid = target["labels"][0].item()
            else:
                cid = 0
            weights.append(class_weight.get(cid, 1.0))
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

    def __init__(self, dataset: BaseDataset, min_count: int = 50) -> None:
        dist = dataset.class_distribution
        self._indices: list[int] = list(range(len(dataset)))
        if not dist:
            return

        # Gather per-class indices
        class_indices: dict[int, list[int]] = {cid: [] for cid in dist}
        for i in range(len(dataset)):
            _, target = dataset[i]
            if "label" in target:
                cid = target["label"]
            elif "rank" in target:
                cid = target["rank"]
            elif "labels" in target and len(target["labels"]) > 0:
                cid = target["labels"][0].item()
            else:
                continue
            if cid in class_indices:
                class_indices[cid].append(i)

        # Duplicate minority classes
        extras: list[int] = []
        for cid, indices in class_indices.items():
            if len(indices) < min_count and len(indices) > 0:
                reps = math.ceil(min_count / len(indices))
                extras.extend(indices * reps)
        self._indices.extend(extras)

    def __iter__(self):
        g = torch.Generator()
        perm = torch.randperm(len(self._indices), generator=g)
        return iter([self._indices[i] for i in perm])

    def __len__(self) -> int:
        return len(self._indices)


class WeightedRandomSampler(Sampler):
    """Thin wrapper around torch's WeightedRandomSampler with auto-weights."""

    def __init__(self, dataset: BaseDataset) -> None:
        dist = dataset.class_distribution
        n = len(dataset)
        if not dist:
            self._sampler = _TorchWeightedRandom(
                torch.ones(n), num_samples=n, replacement=True
            )
            return

        weights = ClassBalancedSampler._compute_weights(dataset, dist)
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
