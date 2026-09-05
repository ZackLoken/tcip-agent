"""Task-aware data samplers: class-imbalance handling plus read-locality ordering.

The imbalance samplers wrap torch samplers but auto-compute weights from
dataset.class_distribution, so the agent just picks a strategy name. The locality
sampler orders a tiled dataset's reads to stay inside GDAL's block cache.
"""

from __future__ import annotations

import inspect
import logging
import math
from collections.abc import Sequence
from typing import cast

import torch
from torch.utils.data import Sampler, WeightedRandomSampler as _TorchWeightedRandom

from tcip_mcp.pipelines.data.datasets import BaseDataset

logger = logging.getLogger(__name__)


def _target_class_id(target: dict, class_key: str | None = None) -> int | None:
    """Class id for a sample's target, honoring an explicit ``class_key`` or task defaults.

    A dataset that names its class field differently than the ``label``/``ranks``/``labels``
    defaults below can pass ``class_key`` instead of silently bucketing every sample as class 0.
    Returns None when no class is found.

    Detection/instance-seg tensor ``labels`` are 1-indexed (cid + 1, background = 0)
    while ``class_distribution`` keys are 0-indexed cids, so that fallback branch shifts
    back by 1. An explicit ``class_key`` returns the raw value unshifted: don't pass
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
    """Spatially local read order over a tiled dataset's tile lattice.

    Fully shuffled tile access on a windowed raster whose compression blocks span full-width
    strips re-decodes each strip for nearly every tile that touches it, because the block
    cache evicts a strip long before the shuffle returns to its row. This sampler keeps each
    reading process inside a contiguous band of tile rows so a band's strips stay resident,
    while training still sees randomness at every level per epoch: source order is shuffled,
    band order within a source is shuffled, and tile order within a band is shuffled.

    The band height is derived at construction, never pinned: half the per-reader GDAL cache
    share (the other half absorbs overlap reads crossing into the neighbor band) divided by
    the costliest windowed source's decoded row bytes, converted to tile rows through the
    lattice's row pitch. With ``num_workers > 1`` each worker process holds its own GDAL
    cache, so the share is the process budget divided by the worker count. The derived band
    and its inputs are logged at construction.

    Multi-worker loading keeps per-worker locality through lanes: torch's multiprocessing
    DataLoader dispatches batch ``i`` to worker ``i mod num_workers``, in order (measured to
    hold across batch sizes, prefetch factors, and persistent_workers), so bands are dealt
    round-robin onto ``num_workers`` lanes and the emitted order interleaves the lanes at
    ``batch_size`` granularity, meaning worker ``w`` reads lane ``w``'s bands back to back.
    Once a lane exhausts it leaves the rotation, so alignment loosens over the epoch's tail
    chunks; every earlier batch lands on its lane's worker.

    Requires a dataset exposing ``tile_entries`` (index-ordered ``(stem, tile_x, tile_y)``)
    and ``source_frames`` (per-stem frame facts including ``windowed``), at least one
    windowed source, and the loader context: ``num_workers`` always, ``batch_size`` when
    ``num_workers > 1`` (the lane interleaving is defined in batches).
    """

    def __init__(self, dataset, num_workers: int | None = None,
                 batch_size: int | None = None) -> None:
        tile_entries = getattr(dataset, "tile_entries", None)
        source_frames = getattr(dataset, "source_frames", None)
        if tile_entries is None or source_frames is None:
            raise ValueError(
                "tile_locality requires a tiled dataset exposing tile_entries and "
                "source_frames; build the dataset with tiling enabled, or pick another "
                "sampler (e.g. 'random')."
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
                "this dataset decodes whole, so read order cannot reduce decodes. Use the "
                "'random' sampler."
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
        for idx, (stem, _tile_x, tile_y) in enumerate(tile_entries):
            rows_by_stem.setdefault(stem, {}).setdefault(int(tile_y), []).append(idx)
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

_SAMPLER_MAP = {
    "random": None,  # use default DataLoader shuffle
    "class_balanced": ClassBalancedSampler,
    "oversample": OverSampler,
    "weighted_random": WeightedRandomSampler,
    "tile_locality": TileLocalitySampler,
}


def build_sampler(name: str, dataset: BaseDataset, *, num_workers: int | None = None,
                  batch_size: int | None = None, **kwargs) -> Sampler | None:
    """Build a sampler by name. Returns None for 'random' (use shuffle=True).

    ``num_workers``/``batch_size`` are the loader context, forwarded only to samplers whose
    constructor accepts them; a sampler that needs one and was built without it refuses,
    naming what to pass. Samplers that ignore the context keep building unchanged.
    """
    if name not in _SAMPLER_MAP:
        raise ValueError(f"Unknown sampler '{name}'. Available: {list(_SAMPLER_MAP.keys())}")
    cls = _SAMPLER_MAP[name]
    if cls is None:
        return None
    params = inspect.signature(cls).parameters
    if "num_workers" in params:
        kwargs.setdefault("num_workers", num_workers)
    if "batch_size" in params:
        kwargs.setdefault("batch_size", batch_size)
    return cls(dataset, **kwargs)
