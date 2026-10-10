"""Task-aware data samplers: class-imbalance handling plus read-locality ordering.

The imbalance samplers wrap torch samplers but compute weights from
dataset.class_distribution; a config's ``sampler`` block names one beside its own values. The
locality sampler orders a tiled dataset's reads to stay inside GDAL's block cache.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import cast

import torch
from torch.utils.data import Sampler, WeightedRandomSampler as _TorchWeightedRandom

from tcip_mcp.pipelines.data.datasets import BaseDataset

logger = logging.getLogger(__name__)


def _target_class_id(target: dict, class_key: str | None = None) -> int | None:
    """A sample's class id: its target's ``class_key`` value, raw, when one is named, else its
    ``label``, its ``ranks``, or its first detection ``labels`` row shifted from 1-indexed to the
    0-indexed id ``class_distribution`` keys; ``None`` when none is found."""
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
    """Over-/under-samples so each class appears equally often per epoch."""

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
        """Each sample's draw weight: its class's balanced weight
        (:func:`~tcip_mcp.pipelines.components.losses.compute_class_weights`), 1.0 for a sample
        with no class or one outside ``dist``."""
        from tcip_mcp.pipelines.components.losses import compute_class_weights

        class_weight = compute_class_weights(dist, normalize=False).tolist()
        weights = []
        for i in range(len(dataset)):
            _, target = dataset[i]
            cid = _target_class_id(target, class_key)
            weights.append(class_weight[cid] if cid in dist else 1.0)
        return torch.tensor(weights, dtype=torch.double)

    def __iter__(self):
        if hasattr(self, "_weights"):
            idx = torch.multinomial(self._weights, self._length, replacement=True)
            return iter(idx.tolist())
        return iter(self._indices)

    def __len__(self) -> int:
        return self._length


class OverSampler(Sampler):
    """Duplicate minority-class samples so all classes have >= min_count."""

    def __init__(self, dataset: BaseDataset, min_count: int, class_key: str | None = None) -> None:
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
        # has a fixed seed, which would freeze the order identically every epoch.
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
            # torch's own stub types weights as Sequence[float]; a Tensor is what the
            # constructor actually accepts and converts internally.
            self._sampler = _TorchWeightedRandom(
                cast(Sequence[float], torch.ones(n)), num_samples=n, replacement=True
            )
            return

        weights = ClassBalancedSampler._compute_weights(dataset, dist, class_key)
        self._sampler = _TorchWeightedRandom(
            cast(Sequence[float], weights), num_samples=n, replacement=True)

    def __iter__(self):
        return iter(self._sampler)

    def __len__(self) -> int:
        return len(self._sampler)


def _interleave_lanes(lanes: list[list[int]], batch_size: int) -> list[int]:
    """Interleave per-lane index sequences at ``batch_size`` granularity: one chunk from each
    lane in rotation, dropping a lane from the rotation once it is exhausted."""
    order: list[int] = []
    positions = [0] * len(lanes)
    active = [i for i, lane in enumerate(lanes) if lane]
    while active:
        still_active = []
        for lane_i in active:
            pos = positions[lane_i]
            order.extend(lanes[lane_i][pos:pos + batch_size])
            positions[lane_i] = pos + batch_size
            if positions[lane_i] < len(lanes[lane_i]):
                still_active.append(lane_i)
        active = still_active
    return order


class TileLocalitySampler(Sampler):
    """A tiled dataset's tiles in bands of contiguous tile rows, sources, bands and tiles within
    a band each shuffled per epoch; with ``num_workers > 1`` bands are dealt round-robin onto one
    lane per worker and the lanes interleaved ``batch_size`` tiles at a time. The band height is
    derived at construction from half the per-reader GDAL cache share over the costliest windowed
    source's decoded row bytes, and logged.

    Requires a dataset exposing ``tile_entries`` (index-ordered ``(stem, (x0, y0, x1, y1))``)
    and ``source_frames`` (per-stem frame facts including ``windowed``), at least one
    windowed source, and the loader context: ``num_workers`` always, ``batch_size`` when
    ``num_workers > 1``; anything else refuses naming why.
    """

    def __init__(self, dataset, num_workers: int | None = None,
                 batch_size: int | None = None) -> None:
        tile_entries = getattr(dataset, "tile_entries", None)
        source_frames = getattr(dataset, "source_frames", None)
        if tile_entries is None or source_frames is None:
            raise ValueError(
                "tile_locality requires a tiled dataset exposing tile_entries and "
                "source_frames; build the dataset with tiling enabled, or state another "
                "sampler or none."
            )
        if num_workers is None:
            raise ValueError(
                "tile_locality requires the loader's num_workers: pass it to build_sampler "
                "(the band derivation depends on the per-process GDAL cache regime)."
            )
        self._lane_count = max(1, int(num_workers))
        if self._lane_count > 1 and (batch_size is None or batch_size < 1):
            raise ValueError(
                "tile_locality with num_workers > 1 requires the loader's batch_size: pass "
                "it to build_sampler (per-worker read order is interleaved in batches)."
            )
        self._batch_size = int(batch_size) if batch_size else None
        windowed = {stem: frame for stem, frame in source_frames.items()
                    if frame.get("windowed")}
        if not windowed:
            raise ValueError(
                "tile_locality orders reads for windowed raster sources; every source in "
                "this dataset decodes whole, so read order cannot reduce decodes. State no "
                "sampler."
            )
        row_costs = {}
        for stem, frame in windowed.items():
            width = frame.get("width")
            channels = frame.get("channels")
            itemsize = frame.get("dtype_itemsize")
            if not width or not channels or not itemsize:
                raise ValueError(
                    f"tile_locality cannot derive a band height: windowed source '{stem}' "
                    "reports no width/channels/dtype_itemsize in source_frames."
                )
            row_costs[stem] = int(width) * int(channels) * int(itemsize)

        from tcip_mcp.pipelines import raster_source

        cache_bytes = raster_source.gdal_cache_bytes()
        cache_share = cache_bytes // self._lane_count
        max_row_bytes = max(row_costs.values())

        rows_by_stem: dict[str, dict[int, list[int]]] = {}
        for idx, (stem, box) in enumerate(tile_entries):
            rows_by_stem.setdefault(stem, {}).setdefault(int(box[1]), []).append(idx)
        pitches: list[int] = []
        for rows in rows_by_stem.values():
            ys = sorted(rows)
            pitches.extend(b - a for a, b in zip(ys, ys[1:]))
        row_pitch = min(pitches) if pitches else None

        if row_pitch is None:
            band_tile_rows = 1
        else:
            band_tile_rows = max(1, (cache_share // 2) // max_row_bytes // row_pitch)
        self.cache_bytes = cache_bytes
        self.cache_share_bytes = cache_share
        self.max_row_bytes = max_row_bytes
        self.row_pitch = row_pitch
        self.band_tile_rows = int(band_tile_rows)
        logger.info(
            "tile_locality: band height %d tile rows (gdal cache %d bytes, %d reader "
            "lane(s) at %d bytes each, max windowed source row %d bytes, tile row pitch "
            "%s px)",
            self.band_tile_rows, cache_bytes, self._lane_count, cache_share,
            max_row_bytes, row_pitch)

        self._bands_by_stem: dict[str, list[list[int]]] = {}
        for stem, rows in rows_by_stem.items():
            ys = sorted(rows)
            bands = []
            for i in range(0, len(ys), self.band_tile_rows):
                bands.append([idx for y in ys[i:i + self.band_tile_rows] for idx in rows[y]])
            self._bands_by_stem[stem] = bands
        self._length = len(tile_entries)

    def _draw_bands(self) -> list[list[int]]:
        """One epoch's globally shuffled band sequence, drawn from the global RNG (like
        OverSampler): order varies per epoch and is controlled by set_seed()."""
        out: list[list[int]] = []
        stems = list(self._bands_by_stem)
        for stem_i in torch.randperm(len(stems)).tolist():
            bands = self._bands_by_stem[stems[stem_i]]
            for band_i in torch.randperm(len(bands)).tolist():
                band = bands[band_i]
                out.append([band[i] for i in torch.randperm(len(band)).tolist()])
        return out

    def __iter__(self):
        bands = self._draw_bands()
        if self._lane_count <= 1:
            return iter([idx for band in bands for idx in band])
        lanes: list[list[int]] = [[] for _ in range(self._lane_count)]
        for i, band in enumerate(bands):
            lanes[i % self._lane_count].extend(band)
        # __init__ already refuses a lane_count > 1 with no batch_size; this branch is only
        # reached when lane_count > 1 (the <= 1 case returns above).
        assert self._batch_size is not None
        return iter(_interleave_lanes(lanes, self._batch_size))

    def __len__(self) -> int:
        return self._length


# ====================================================================
# Factory
# ====================================================================

_SAMPLER_MAP: dict[str, type] = {
    "class_balanced": ClassBalancedSampler,
    "oversample": OverSampler,
    "weighted_random": WeightedRandomSampler,
    "tile_locality": TileLocalitySampler,
}


LOADER_CONTEXT = ("num_workers", "batch_size")
"""The settings the train loader supplies to a sampler whose constructor takes them, never a
sampler block."""


def sampler_settings(block: dict) -> tuple[type, dict]:
    """The sampler class ``block["name"]`` names and the rest of ``block`` as its own values.
    Refuses (``ValueError``) an unknown name, and by name a value its constructor requires that
    ``block`` leaves unstated, one it states that the constructor does not take, or one of
    :data:`LOADER_CONTEXT`, which the loader supplies."""
    import inspect

    from tcip_mcp.pipelines.model_build import resolve_named

    cls = resolve_named(block.get("name", ""), _SAMPLER_MAP, kind="sampler")
    settings = {k: v for k, v in block.items() if k != "name"}
    supplied = sorted(set(settings) & set(LOADER_CONTEXT))
    if supplied:
        raise ValueError(f"sampler {block['name']!r} states {supplied}, which the train loader "
                         "supplies; drop them from the block")
    try:
        inspect.signature(cls).bind(None, **settings)
    except TypeError as exc:
        raise ValueError(f"sampler {block['name']!r}: {exc}") from exc
    return cls, settings


def build_sampler(block: dict, dataset: BaseDataset, *, num_workers: int | None = None,
                  batch_size: int | None = None) -> Sampler:
    """The sampler ``block`` states (:func:`sampler_settings`) over ``dataset``.

    ``num_workers``/``batch_size`` are the loader context (:data:`LOADER_CONTEXT`), handed to a
    sampler whose constructor takes them; a sampler that needs one and was built without it
    refuses, naming what to pass. A sampler whose constructor takes neither is built without
    them.
    """
    from tcip_mcp.pipelines.model_build import keyword_parameters

    cls, kwargs = sampler_settings(block)
    params, _open = keyword_parameters(cls)
    for name, value in zip(LOADER_CONTEXT, (num_workers, batch_size), strict=True):
        if name in params:
            kwargs[name] = value
    return cls(dataset, **kwargs)
