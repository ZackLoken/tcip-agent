"""Multi-task datasets with standardized interfaces.

Every loader here is built from admitted samples, each reading its own source and the ground
truth that answers for it, and returns (image_tensor, target_dict) where the target format is
task-specific but always dict-based. `build_dataset` dispatches to the class for a task, or to a
bespoke ``dataset_source`` builder for a task no loader here covers.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


from tcip_mcp.pipelines import raster_source
from tcip_mcp.pipelines.data.band_groups import BandGroupRef
from tcip_annotation.json_io import LabelDocument
from tcip_annotation.state import box_derivable, object_rows, polygonal

from tcip_mcp.pipelines.data.label_queries import acquired, json_det_targets, resolved
from tcip_mcp.pipelines.derivations import num_classes_from_distribution
from tcip_mcp.pipelines.data.selection import (
    DOCUMENT, MASK, SHAPE_DESCRIPTIONS, TABLE, ClassScope, Sample, refuse_unreadable_samples,
)
from tcip_mcp.pipelines.image_utils import (
    frame_size, image_dimensions, load_image, pil_to_tensor, pixel_array,
    to_pil_if_faithful,
)
from tcip_mcp.pipelines.execution import DEFAULT_OVERLAP

logger = logging.getLogger(__name__)


class BaseDataset(Dataset, ABC):
    """Abstract base for all task-specific datasets."""

    task_type: str = ""
    expected_channels: int  # input channels the dataset yields, stamped by build_dataset

    @property
    @abstractmethod
    def num_samples(self) -> int: ...

    @property
    def class_distribution(self) -> dict[int, int]:
        """Class ID → count. Subclasses should override for efficiency."""
        return {}

    @abstractmethod
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]: ...

    def __len__(self) -> int:
        return self.num_samples


class BaseImageDataset(BaseDataset):
    """Base for image datasets, centralizes channel-aware loading + finalization.

    Subclasses set ``self.transforms``, then build only the task-specific target. Each sample is
    indexed by its location (:meth:`sample_of`), so one dataset spans capture dates and keeps two
    dates' same-named images apart.

    ``ground_truth_shape`` is the one shape this loader reads
    (:data:`~tcip_mcp.pipelines.data.selection.GROUND_TRUTH_SHAPES`), declared by each subclass and
    refused in :meth:`refuse_other_shapes`. ``ground_truth_count`` names the count that ground
    truth derives for a loader whose ground truth carries its own classes (:func:`resolve_sizes`).
    ``reads_geometry`` declares which geometries answer for this loader's measurement, for the
    loaders whose ground truth is a document.
    """

    ground_truth_shape: str = DOCUMENT
    ground_truth_count: str | None = None
    reads_geometry: "Callable[[Any], bool] | None" = None
    reads_description: str = ""
    scope: ClassScope = ClassScope()
    transforms: Any = None
    _samples: dict[str, Sample]
    _keys: list[str]

    @property
    def stems(self) -> list[str]:
        """Each indexed sample's key, its location, in index order."""
        return self._keys

    @property
    def num_samples(self) -> int:
        return len(self.stems)

    def sample_of(self, key: str) -> Sample:
        """The recorded sample one index key names."""
        return self._samples[key]

    @classmethod
    def refuse_other_shapes(cls, samples: Sequence[Sample]) -> None:
        """Refuse a sample whose own ground truth is not the shape this loader reads, naming it."""
        wrong = [s.location for s in samples if s.shape != cls.ground_truth_shape]
        if wrong:
            raise ValueError(
                f"{len(wrong)} sample(s) name ground truth a {cls.task_type} loader does not "
                f"read ({wrong[:5]}): it reads "
                f"{SHAPE_DESCRIPTIONS[cls.ground_truth_shape]}, and reading what these name "
                f"instead would train on something other than the ground truth recorded for "
                f"them. Draw a selection over the ground truth {cls.task_type} reads."
            )

    @staticmethod
    def read_mask(sample: Sample) -> np.ndarray:
        """One mask sample's raster as the integer class ids it carries, decoded from the bytes its
        one read answered (``stored``,
        :func:`~tcip_mcp.pipelines.data.label_queries.acquired`)."""
        return np.array(raster_source.PhotographicSource(sample.stored.value, 1).image)

    def _init_from_samples(self, samples: Sequence[Sample]) -> None:
        """Index a recorded sample list: each sample's own source and ground truth.

        Each sample is keyed by its own location, distinct across capture dates, with its member
        name beside the key. Refuses, before indexing any of it, a sample no loader here can
        read (:func:`~tcip_mcp.pipelines.data.selection.refuse_unreadable_samples`) and, where this
        loader declares which geometries it reads, one whose document carries the subject only in
        geometries it does not (:meth:`_refuse_unreadable_geometry`). Each sample's ground truth and
        logical image are the ones it carries
        (:func:`~tcip_mcp.pipelines.data.label_queries.acquired`,
        :func:`~tcip_mcp.pipelines.data.label_queries.resolved`), never read again by this loader.
        """
        refuse_unreadable_samples(samples)
        samples = resolved(acquired(samples))
        self._refuse_unreadable_geometry(samples)
        self._samples = {s.location: s for s in samples}
        self._keys = [s.location for s in samples]

    def _refuse_unreadable_geometry(self, samples: Sequence[Sample]) -> None:
        """Refuse a sample whose document carries this run's subject only in geometries this loader
        does not read (:attr:`reads_geometry`), naming it.

        A document carrying the subject as a point, or as an image-level record, has real ground
        truth this loader cannot turn into a target. A loader that declares no geometry (a mask
        raster, a table row) refuses nothing here.
        """
        if self.reads_geometry is None:
            return
        reads = self.reads_geometry
        wrong = []
        for sample in samples:
            mine = [a for a in sample.read.annotations if a.subject == self.scope.subject]
            if mine and not any(reads(a.geometry) for a in mine):
                wrong.append(sample.location)
        if wrong:
            raise ValueError(
                f"{len(wrong)} sample(s) carry {self.scope.subject!r} only in geometries a "
                f"{self.task_type} loader does not read ({wrong[:5]}): it reads "
                f"{self.reads_description}, and training an image whose real objects it cannot "
                f"read would teach them as background. Run a task whose loader reads what these "
                f"documents carry, or supply a builder that reads them."
            )

    def image_of(self, stem: str) -> Path | BandGroupRef:
        """The logical image one sample key names, as its sample carries it."""
        return self._samples[stem].image

    def _open_image(self, stem: str):
        """Open an image honoring ``expected_channels``: PIL where the pixels have a faithful
        PIL mode (1/3 channels always; 4 only when the source declares its 4th band alpha), else
        an ``[H, W, C]`` ndarray. See :func:`image_utils.to_pil_if_faithful`."""
        return load_image(self.image_of(stem), self.expected_channels)

    _warned_ndarray_transforms = False

    def _finalize(self, img, target: dict) -> tuple[torch.Tensor, dict]:
        """Apply PIL transforms (when applicable) or convert straight to a tensor."""
        if self.transforms is not None and isinstance(img, Image.Image):
            return self.transforms(img, target)
        if self.transforms is not None and not BaseImageDataset._warned_ndarray_transforms:
            # Warned once: config claiming augmentation the model never saw is a provenance break.
            BaseImageDataset._warned_ndarray_transforms = True
            logger.warning(
                "augmentation is configured but skipped for images whose dtype or band count "
                "PIL cannot represent faithfully (e.g. uint16 or 5-band pixels, or a 4-band "
                "uint8 raster whose 4th band the source doesn't declare alpha): the transform "
                "pipeline is PIL-only. This run trains those images unaugmented."
            )
        return pil_to_tensor(img), target


# ====================================================================
# Detection
# ====================================================================

def indexed_sample_keys(dataset: Any) -> set[str]:
    """The sample keys a built dataset actually indexes, however many examples each one yields.

    A per-image dataset indexes one example per sample, so every sample it was handed is here. A
    tiled dataset indexes one example per kept tile and names no example at all for a source whose
    tiles all fall outside its keep regions or carry no ground truth, so such a source is absent
    here.
    """
    return set(getattr(dataset, "stems", None) or [])


def target_tensors(target: Any) -> dict[str, torch.Tensor]:
    """A ``{"boxes", "labels", "iscrowd"}`` target of parallel per-box values, with its
    ``attributes`` rows when it carries them, as the tensors every detection loader emits,
    ``iscrowd`` under torchvision's reference key."""
    return {
        "boxes": torch.tensor(target["boxes"], dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(target["labels"], dtype=torch.int64),
        "iscrowd": torch.tensor(crowd_of(target), dtype=torch.int64),
        **({"attributes": torch.as_tensor(target["attributes"], dtype=torch.int64)}
           if "attributes" in target else {}),
    }


PER_BOX_KEYS = ("boxes", "scores", "labels", "masks", "iscrowd", "attributes")
"""The keys of a detection target or a detection record holding one row per box, which every
row filter keeps in step."""


def crowd_of(target: Mapping[str, Any]) -> Any:
    """A detection target's per-box crowd flags, which every detection producer states; a target
    stating none refuses (``ValueError``) naming the key."""
    if target.get("iscrowd") is None:
        raise ValueError("this detection target states no 'iscrowd': whether each row is one "
                         "object or a crowd region is its producer's to state, so a bespoke "
                         "dataset emits it beside its boxes and labels.")
    return target["iscrowd"]


def instance_targets(targets: list[dict]) -> list[dict]:
    """``targets`` as the built-in torchvision heads take them: every crowd row
    (:func:`object_rows`) dropped from each per-box key, every other key as it was; the input
    targets are unchanged."""
    out = []
    for t in targets:
        keep = object_rows(crowd_of(t))
        out.append({k: v[keep] if k in PER_BOX_KEYS else v
                    for k, v in t.items()})
    return out


class DocumentDataset(BaseImageDataset):
    """A loader over samples whose ground truth is a per-image label document, read under the
    run's own admitted class space (``scope``)."""

    ground_truth_shape = DOCUMENT

    def __init__(
        self, samples: Sequence[Sample], transforms: Any = None, *, scope: ClassScope,
    ) -> None:
        self.transforms = transforms
        self.scope = scope
        self._init_from_samples(samples)

    def document(self, stem: str) -> LabelDocument:
        """The label document one sample key names, as its one read answered it."""
        return self._samples[stem].read

    def det_targets(self, document: LabelDocument) -> dict[str, Any]:
        """One sample's label ``document`` (:meth:`document`) as the target
        :func:`json_det_targets` reads under this run's own class space and this task's own
        geometry (``reads_geometry``)."""
        return json_det_targets(document.annotations, self.scope,
                                reads=cast("Callable[[Any], bool]", self.reads_geometry))


class DetectionDataset(DocumentDataset):
    """Object detection over a recorded sample list: each sample reads its own source and the label
    document that answers for it.
    """

    task_type = "detection"
    reads_geometry = staticmethod(box_derivable)
    reads_description = "a box or a polygon of its subject"

    @property
    def class_distribution(self) -> dict[int, int]:
        counts: Counter[int] = Counter()
        for stem in self.stems:
            target = self.det_targets(self.document(stem))
            for lab in np.asarray(target["labels"])[object_rows(crowd_of(target))].tolist():
                counts[lab - 1] += 1  # back to 0-indexed cid
        return dict(counts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        stem = self.stems[idx]
        img = self._open_image(stem)
        return self._finalize(
            img, {**target_tensors(self.det_targets(self.document(stem))), "image_id": idx})


# ====================================================================
# Tiled Detection (SAHI's slice lattice)
# ====================================================================

TILE_SIZE = 224
"""Tile edge in pixels a tiled detection run uses when its config states none."""

_EMPTY_BOXES = np.zeros((0, 4), dtype=np.float32)
_EMPTY_LABELS = np.zeros((0,), dtype=np.int64)


def clip_boxes_to_tile(
    boxes: np.ndarray, labels: np.ndarray, tile_x: int, tile_y: int,
    tile_w: int, tile_h: int, min_box_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Intersect full-image-px boxes with a ``tile_w`` x ``tile_h`` tile; drop seam slivers; emit
    tile-local xyxy.

    A box clipped by the tile edge is dropped only when the visible (clipped) part is a sliver: its
    characteristic size ``sqrt(iw*ih) < min_box_size``. Boxes fully inside the tile are always
    kept. ``tile_w``/``tile_h`` need not be equal.
    """
    from tcip_annotation.matching import box_sizes

    boxes = np.asarray(boxes)
    labels = np.asarray(labels)
    if len(boxes) == 0:
        return _EMPTY_BOXES.copy(), _EMPTY_LABELS.copy()
    tx2, ty2 = tile_x + tile_w, tile_y + tile_h
    ix1 = np.maximum(boxes[:, 0], tile_x)
    iy1 = np.maximum(boxes[:, 1], tile_y)
    ix2 = np.minimum(boxes[:, 2], tx2)
    iy2 = np.minimum(boxes[:, 3], ty2)
    iw, ih = ix2 - ix1, iy2 - iy1
    visible = (iw > 0) & (ih > 0)
    was_clipped = ((boxes[:, 0] < tile_x) | (boxes[:, 1] < tile_y)
                   | (boxes[:, 2] > tx2) | (boxes[:, 3] > ty2))
    char_size = box_sizes(np.stack([ix1, iy1, ix2, iy2], axis=1))
    keep = visible & ~(was_clipped & (char_size < min_box_size))
    if not keep.any():
        return _EMPTY_BOXES.copy(), _EMPTY_LABELS.copy()
    out = np.stack([ix1[keep] - tile_x, iy1[keep] - tile_y,
                    ix2[keep] - tile_x, iy2[keep] - tile_y], axis=1)
    return out.astype(np.float32), labels[keep].astype(np.int64)


def clipped_boxes_per_slice(
    boxes: np.ndarray, labels: np.ndarray, slices: list[tuple[int, int, int, int]],
    min_box_size: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """:func:`clip_boxes_to_tile` for every slice of a lattice at once, each clipped to its own
    ``(x0, y0, x1, y1)`` extent: result ``i`` equals ``clip_boxes_to_tile(boxes, labels, x0, y0,
    x1 - x0, y1 - y0, ...)`` for ``slices[i]``. Slice origins are distinct, as a lattice's are.
    """
    n = len(slices)
    boxes = np.asarray(boxes)
    labels = np.asarray(labels)
    if n == 0 or len(boxes) == 0:
        return [(_EMPTY_BOXES.copy(), _EMPTY_LABELS.copy()) for _ in range(n)]
    xs = np.unique(np.asarray([s[0] for s in slices], dtype=np.int64))
    ys = np.unique(np.asarray([s[1] for s in slices], dtype=np.int64))
    # float64 holds every float32/float64 coordinate and every origin exactly, so these strict
    # bounds agree with the clip's own visibility arithmetic instead of re-rounding it.
    bx1 = boxes[:, 0].astype(np.float64)
    by1 = boxes[:, 1].astype(np.float64)
    bx2 = boxes[:, 2].astype(np.float64)
    by2 = boxes[:, 3].astype(np.float64)
    span_x = max(x1 - x0 for x0, _y0, x1, _y1 in slices)
    span_y = max(y1 - y0 for _x0, y0, _x1, y1 in slices)
    col_lo = np.searchsorted(xs, bx1 - span_x, side="right")
    col_hi = np.searchsorted(xs, bx2, side="left")
    row_lo = np.searchsorted(ys, by1 - span_y, side="right")
    row_hi = np.searchsorted(ys, by2, side="left")
    candidates: dict[tuple[int, int], list[int]] = {}
    for i in range(len(boxes)):
        for k in range(row_lo[i], row_hi[i]):
            ty = int(ys[k])
            for j in range(col_lo[i], col_hi[i]):
                candidates.setdefault((int(xs[j]), ty), []).append(i)
    results: list[tuple[np.ndarray, np.ndarray] | None] = [None] * n
    index_of = {(x0, y0): m for m, (x0, y0, _x1, _y1) in enumerate(slices)}
    for (x0, y0), idxs in candidates.items():
        m = index_of.get((x0, y0))
        if m is None:
            continue  # an origin pair the lattice never laid down
        _x0, _y0, x1, y1 = slices[m]
        results[m] = clip_boxes_to_tile(
            boxes[idxs], labels[idxs], x0, y0, x1 - x0, y1 - y0, min_box_size)
    return [(_EMPTY_BOXES.copy(), _EMPTY_LABELS.copy()) if r is None else r for r in results]


def dedup_boxes(boxes: np.ndarray, labels: np.ndarray, iou_thresh: float) -> list[int]:
    """Greedy largest-first, class-aware dedup: the sorted indices of the boxes kept, dropping a
    box that overlaps a kept box of its own class by >= iou_thresh. Indices, so a caller keeps
    every per-box array (a crowd flag beside the labels) in step by indexing each one the same
    way."""
    from tcip_annotation.matching import box_areas, iou_matrix

    n = len(boxes)
    if not iou_thresh or iou_thresh >= 1.0 or n < 2:
        return list(range(n))
    areas = box_areas(boxes)
    ious = iou_matrix(boxes, boxes)
    kept: list[int] = []
    for i in sorted(range(n), key=lambda i: -areas[i]):  # largest first
        if not any(labels[i] == labels[k] and 0 < ious[i, k] and ious[i, k] >= iou_thresh
                   for k in kept):
            kept.append(i)
    return sorted(kept)


def _validated_keep_regions(
    keep_regions: "Sequence[tuple[int, int, int, int]] | None",
) -> list[tuple[int, int, int, int]] | None:
    """``keep_regions`` as int 4-tuples, or ``None`` when no filter was asked for.

    A malformed rect refuses by name rather than silently keeping or dropping tiles it never
    described. An empty sequence is a real filter that keeps nothing, distinct from ``None``.
    """
    if keep_regions is None:
        return None
    regions: list[tuple[int, int, int, int]] = []
    for region in keep_regions:
        vals = tuple(int(v) for v in region)
        if len(vals) != 4:
            raise ValueError(
                f"keep region {region!r} must be a half-open pixel rect (x0, y0, x1, y1)")
        x0, y0, x1, y1 = vals
        if x1 <= x0 or y1 <= y0:
            raise ValueError(
                f"keep region {vals!r} has no extent: half-open needs x0 < x1 and y0 < y1")
        regions.append(vals)
    return regions


class TiledDetectionDataset(BaseImageDataset):
    """Wrap a ``DetectionDataset`` and expand each source image into native-resolution
    slices of SAHI's lattice (:func:`~tcip_mcp.pipelines.slicing.slice_lattice`) with labels
    clipped/remapped to slice space.

    Slice membership is computed at ``__init__`` without decoding pixels. Sources whose backend
    opens without a decode (``raster_source.opens_windowed``: a GDAL-served raster, a
    memory-mapped ``.npy``) are opened through the process source pool, so their dims come from
    the open source and layout refusals surface here; every other container keeps a header-only
    dimension probe, and its refusals surface at first read. ``__getitem__`` reads a windowed
    stem one slice window at a time through the pool, and a whole-decode stem by decoding once
    and indexing the slice; both emit the same target dict shape as ``DetectionDataset``. The
    dataset itself never holds an open source object, so it pickles into spawned DataLoader
    workers.

    ``keep_regions``, when given, is a sequence of half-open pixel rects ``(x0, y0, x1, y1)``
    in each image's own full-resolution frame: only slices lying fully inside one of them are
    indexed (an empty sequence keeps none). A slice shorter than ``tile_size`` (a frame shorter
    than the tile on that axis) is dropped first and counted in ``tiles_dropped_past_extent``;
    slices no rect contains count in ``tiles_dropped_outside_regions``. Without ``keep_regions``
    both counts stay 0 and every slice is kept.

    A clipped box whose visible part falls under ``sliver_frac`` of the class's average size is a
    tile-seam sliver and dropped; unstated, the fraction derives from the ground truth's own
    size spread (``derivations.derive_sliver_frac``), and ground truth too sparse to derive it
    refuses (``ValueError``) naming ``tiling.sliver_frac``.
    """

    task_type = "detection"

    def __init__(
        self,
        base: "DetectionDataset",
        tile_size: int = TILE_SIZE,
        overlap: float = DEFAULT_OVERLAP,
        sliver_frac: float | None = None,
        dedup_iou: float = 0.8,
        skip_empty: bool = False,
        transforms: Any = None,
        keep_regions: Sequence[tuple[int, int, int, int]] | None = None,
    ) -> None:
        from tcip_annotation.matching import box_sizes

        from tcip_mcp.pipelines.derivations import derive_sliver_frac
        from tcip_mcp.pipelines.raster_source import rect_contains_rect
        from tcip_mcp.pipelines.slicing import is_full_slice, slice_lattice

        self._samples = base._samples
        self.expected_channels = base.expected_channels
        self.tile_size = tile_size
        self.overlap = overlap
        self.transforms = transforms
        self._index: list[dict] = []
        # Plain values only, so the dataset pickles into spawned workers; asserted at decode time.
        self._source_frames: dict[str, dict[str, Any]] = {}
        regions = _validated_keep_regions(keep_regions)
        self.tiles_dropped_past_extent = 0
        self.tiles_dropped_outside_regions = 0

        # Pass 1: every image's dims and boxes, and the GT box sizes the sliver cutoff derives from.
        stems_data: list[tuple[str, np.ndarray, dict[str, np.ndarray], int, int]] = []
        object_sizes: list[float] = []
        for stem in base.stems:
            img_source = base.image_of(stem)
            windowed = raster_source.opens_windowed(img_source, self.expected_channels)
            if windowed:
                # Opened now, so an unreadable layout refuses here rather than mid-epoch.
                src = raster_source.pooled_source(img_source, self.expected_channels)
                w, h = int(src.width), int(src.height)
                channels = int(src.num_channels)
                itemsize: int | None = int(np.dtype(src.dtype).itemsize)
            else:
                # A header probe: opening a whole-decode source would hold its pixels resident.
                w, h = image_dimensions(img_source, self.expected_channels)
                channels = int(self.expected_channels)
                itemsize = None
            self._source_frames[stem] = {
                "width": int(w), "height": int(h), "channels": channels,
                "dtype_itemsize": itemsize, "windowed": windowed,
            }
            # The frame the boxes were drawn in, which a multi-band raster can decode differently.
            from tcip_mcp.pipelines.data.splits import label_document_extent

            document = base.document(stem)
            authored = label_document_extent(document, f"{stem}'s label document")
            if authored != (w, h):
                raise ValueError(
                    f"tiled dataset frame mismatch for stem {stem!r}: the labels record a "
                    f"{authored[0]}x{authored[1]} image but it decodes as {w}x{h} at "
                    f"{self.expected_channels} channels. Tiles would be cut from a different frame "
                    f"than the boxes were drawn in, displacing every box. Re-author the labels "
                    f"against the multi-band frame, or ingest this raster as {authored[0]}x"
                    f"{authored[1]}."
                )
            full = base.det_targets(document)
            fb = np.asarray(full["boxes"], dtype=np.float32).reshape(-1, 4)
            rows_of: dict[str, np.ndarray] = {k: np.asarray(full[k], dtype=np.int64)
                                              for k in PER_BOX_KEYS if k != "boxes" and k in full}
            # A crowd region is not one object, so its extent says nothing about object size.
            object_sizes.extend(box_sizes(fb[object_rows(crowd_of(rows_of))]).tolist())
            stems_data.append((stem, fb, rows_of, w, h))

        char_sizes = [s for s in object_sizes if s > 0]
        class_avg_size = float(np.mean(char_sizes)) if char_sizes else 0.0
        if sliver_frac is None:
            sliver_frac = derive_sliver_frac(char_sizes)
        if sliver_frac is None:
            raise ValueError(
                f"the tile-seam sliver cutoff derives from the ground truth's own box-size spread, "
                f"and this dataset holds {len(char_sizes)} box(es), too few to measure one; state "
                "tiling.sliver_frac for this run.")
        min_box_size = sliver_frac * class_avg_size

        for stem, fb, rows_of, w, h in stems_data:
            slices = slice_lattice(h, w, tile_size, overlap)
            if regions is not None:
                kept: list[tuple[int, int, int, int]] = []
                for s in slices:
                    if not is_full_slice(s, tile_size):
                        self.tiles_dropped_past_extent += 1
                    elif any(rect_contains_rect(r, s) for r in regions):
                        kept.append(s)
                    else:
                        self.tiles_dropped_outside_regions += 1
                slices = kept
            per_slice = clipped_boxes_per_slice(fb, np.arange(len(fb)), slices, min_box_size)
            for s, (tb, rows) in zip(slices, per_slice):
                if len(tb) > 1:
                    keep = dedup_boxes(tb, rows_of["labels"][rows], dedup_iou)
                    tb, rows = tb[keep], rows[keep]
                if skip_empty and len(tb) == 0:
                    continue
                self._index.append({"stem": stem, "slice": s, "boxes": tb,
                                    **{k: v[rows] for k, v in rows_of.items()}})

    @property
    def stems(self) -> list[str]:
        return [e["stem"] for e in self._index]

    @property
    def tile_entries(self) -> list[tuple[str, tuple[int, int, int, int]]]:
        """``(stem, slice)`` per sample, in index order, each slice its half-open ``(x0, y0, x1,
        y1)`` lattice box."""
        return [(e["stem"], e["slice"]) for e in self._index]

    @property
    def source_frames(self) -> dict[str, dict[str, Any]]:
        """Per-stem frame facts recorded when the index was built: ``width``, ``height``,
        ``channels``, ``dtype_itemsize`` (``None`` where only a header probe ran, so no dtype was
        read), and ``windowed`` (whether this source reads through a windowed backend)."""
        return {stem: dict(info) for stem, info in self._source_frames.items()}

    @property
    def class_distribution(self) -> dict[int, int]:
        counts: Counter[int] = Counter()
        for e in self._index:
            for lab in e["labels"][object_rows(crowd_of(e))].tolist():
                counts[int(lab) - 1] += 1  # 0-indexed cid, matching DetectionDataset
        return dict(counts)

    def _read_windowed_tile(self, stem: str, info: dict, s: tuple[int, int, int, int]):
        """One slice through the pooled windowed source: its ``[H, W, C]`` array and the source's
        band interpretations.

        Refuses when the recorded frame disagrees with the pooled source's own dims, or the
        returned window's shape disagrees with the requested rect.
        """
        src = raster_source.pooled_source(self.image_of(stem), self.expected_channels)
        if (src.width, src.height) != (info["width"], info["height"]):
            raise ValueError(
                f"tiled dataset frame changed for stem {stem!r}: indexed at "
                f"{info['width']}x{info['height']} but the source now opens as "
                f"{src.width}x{src.height} at {self.expected_channels} channels. Cropping here "
                f"would displace every box."
            )
        x0, y0, x1, y1 = s
        region, _spec = src.read_region(raster_source.Rect(x0, y0, x1, y1))
        if region.shape[:2] != (y1 - y0, x1 - x0):
            raise ValueError(
                f"windowed read for stem {stem!r} returned {region.shape[0]}x{region.shape[1]} "
                f"pixels for the {y1 - y0}x{x1 - x0} window at ({x0}, {y0}): the decoder "
                f"disagrees with its own header, refusing to serve displaced pixels."
            )
        return region, getattr(src, "band_interpretations", None)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        """One slice and its clipped targets. The slice is PIL where its dtype has a faithful mode
        (so augmentation applies), else ``[H, W, C]``; a 4-channel slice converts only when the
        source names its 4th band alpha (:func:`to_pil_if_faithful`)."""
        e = self._index[idx]
        stem = e["stem"]
        info = self._source_frames[stem]
        interpretations = None
        if info["windowed"]:
            region, interpretations = self._read_windowed_tile(stem, info, e["slice"])
        else:
            img = self._open_image(stem)
            w, h = frame_size(img)
            if (w, h) != (info["width"], info["height"]):
                raise ValueError(
                    f"tiled dataset frame changed for stem {stem!r}: indexed at "
                    f"{info['width']}x{info['height']} but now decodes as {w}x{h} at "
                    f"{self.expected_channels} channels. Cropping here would displace every box."
                )
            arr, interpretations = pixel_array(img)
            x0, y0, x1, y1 = e["slice"]
            region = arr[y0:y1, x0:x1]
        tile = to_pil_if_faithful(region, band_interpretations=interpretations)
        return self._finalize(tile, {**target_tensors(e), "image_id": idx})


# ====================================================================
# Instance Segmentation
# ====================================================================

class InstanceSegDataset(DocumentDataset):
    """Instance masks from per-image polygons: the producer's own samples, each naming its own
    source and the label document that answers for it."""

    task_type = "instance_seg"
    reads_geometry = staticmethod(polygonal)
    reads_description = "a polygon of its subject"

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        stem = self.stems[idx]
        img = self._open_image(stem)
        w, h = frame_size(img)

        target = self.det_targets(self.document(stem))
        from PIL import ImageDraw

        masks = []
        for polygon in target["geometry"]:
            # Every ring fills one instance mask: a multi-ring instance is one occlusion-split object.
            poly_img = Image.new("L", (w, h), 0)
            draw = ImageDraw.Draw(poly_img)
            for ring in polygon.rings:
                draw.polygon([(p[0], p[1]) for p in ring], fill=1)
            masks.append(np.array(poly_img))

        return self._finalize(img, {
            **target_tensors(target),
            "masks": torch.tensor(np.stack(masks) if masks else np.zeros((0, h, w)), dtype=torch.uint8),
            "image_id": idx,
        })


# ====================================================================
# Semantic Segmentation
# ====================================================================

class SemanticSegDataset(BaseImageDataset):
    """PNG mask images where pixel values are class IDs, over a recorded sample list.

    Membership is exactly what the producer recorded and each sample reads the mask it names, so
    the dataset spans whatever capture dates the draw did.
    """

    task_type = "semantic_seg"
    ground_truth_shape = MASK
    ground_truth_count = "num_classes"

    def __init__(self, samples: Sequence[Sample], transforms: Any = None) -> None:
        self.transforms = transforms
        self._init_from_samples(samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        stem = self.stems[idx]
        img = self._open_image(stem)
        mask = self.read_mask(self._samples[stem])
        target = {"masks": torch.tensor(mask, dtype=torch.int64)}
        return self._finalize(img, target)


# ====================================================================
# Classification
# ====================================================================

def table_values(samples: Sequence[Sample]) -> list[str]:
    """Each sample's own table-row value, as its one read answered it
    (:func:`~tcip_mcp.pipelines.data.label_queries.acquired`)."""
    return [s.read for s in acquired(samples)]


class TableDataset(BaseImageDataset):
    """A loader over a recorded sample list whose ground truth is one row of a table: each sample
    reads the value its ``row_key`` names in the table it names (:func:`table_values`), read by
    ``convert`` and handed to the head under ``target_key``, its loss contract's own key. A loader
    whose ground truth derives a count also counts its values per class."""

    ground_truth_shape = TABLE
    target_key = ""
    convert: "staticmethod[[str], Any]" = staticmethod(int)

    def __init__(self, samples: Sequence[Sample], transforms: Any = None) -> None:
        self.transforms = transforms
        self._init_from_samples(samples)
        self._values = [self.convert(self._samples[k].read) for k in self._keys]

    @property
    def class_distribution(self) -> dict[int, int]:
        return dict(Counter(self._values)) if self.ground_truth_count else {}

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        return self._finalize(self._open_image(self.stems[idx]),
                              {self.target_key: self._values[idx]})


class ClassificationDataset(TableDataset):
    """Image classification: each sample's class id."""

    task_type = "classification"
    ground_truth_count = "num_classes"
    target_key = "labels"


class OrdinalDataset(TableDataset):
    """Ordinal regression: each sample's rank."""

    task_type = "ordinal"
    ground_truth_count = "num_ranks"
    target_key = "ranks"


class RegressionDataset(TableDataset):
    """Continuous-value regression: each sample's value."""

    task_type = "regression"
    target_key = "values"
    convert = staticmethod(float)


# ====================================================================
# Factory
# ====================================================================

_DATASET_MAP: dict[str, type[BaseImageDataset]] = {
    "detection": DetectionDataset,
    "instance_seg": InstanceSegDataset,
    "semantic_seg": SemanticSegDataset,
    "classification": ClassificationDataset,
    "ordinal": OrdinalDataset,
    "regression": RegressionDataset,
}
"""Which built-in loader reads a run's samples, by task. A task outside this map reaches a dataset
only through a bespoke ``dataset_source`` builder."""


def builtin_loader(task: str, dataset_source: dict | None = None) -> type[BaseImageDataset] | None:
    """The built-in loader a run of ``task`` is read by: ``None`` for a run naming a bespoke
    ``dataset_source`` and for a task no built-in loader reads."""
    return None if dataset_source else _DATASET_MAP.get(task)


def build_from_dataset_source(
    dataset_source: dict, *, task: str, samples: Sequence[Sample],
    scope: ClassScope, transforms: Any,
) -> Dataset:
    """Import the agent's dataset builder and call it.

    The builder is called with a context of ``samples`` (the sample list for the side being built,
    each carrying its logical image, :func:`~tcip_mcp.pipelines.data.label_queries.resolved`),
    ``scope`` (the :class:`~tcip_mcp.pipelines.data.selection.ClassScope` those samples were
    admitted under, every field ``None`` where the ground truth carries its own classes), ``task``
    and ``transforms``.

    ``builder_kwargs`` configure the builder; a key the context already states refuses by name.
    Declare ``**kwargs`` on the builder to ignore context keys it doesn't use.

    ``dataset_source`` schema (parallels ``model_source``)::

        {"builder": "my_module:build_ds",  # required, 'module:function' (or 'module.function')
         "builder_kwargs": {...},          # optional, the builder's own configuration
         "source_files": [...]}            # optional, provenance
    """
    if not isinstance(dataset_source, dict):
        raise ValueError("dataset_source must be a dict")
    from tcip_mcp.pipelines.model_build import import_source_builder

    fn = import_source_builder(dataset_source)
    builder_kwargs = dataset_source.get("builder_kwargs") or {}
    if not isinstance(builder_kwargs, dict):
        raise ValueError("dataset_source.builder_kwargs must be a dict")
    context = {"task": task, "samples": resolved(samples), "scope": scope,
               "transforms": transforms}
    restated = sorted(set(context) & set(builder_kwargs))
    if restated:
        raise ValueError(
            f"dataset_source.builder_kwargs restates {restated}: the samples this run trains on, "
            f"the class space they were admitted under, the task and the augmentation are the "
            f"platform's to state, and a builder given a second value for one of them would build "
            f"over something the run's own record does not describe. Drop {restated} from "
            "builder_kwargs."
        )
    return fn(**context, **builder_kwargs)


def _band_count(samples: Sequence[Sample]) -> int:
    """The band count the sources of ``samples`` carry, one count for all of them.

    Every source is probed, off a header where its container carries one and by decoding where it
    does not; sources that disagree, or a source that will not probe, refuse by name.
    """
    from tcip_mcp.pipelines.derivations import probe_channels

    counts: dict[int, str] = {}
    for source in (s.image for s in resolved(samples)):
        try:
            counts.setdefault(int(probe_channels(source)), str(source))
        except Exception as exc:
            raise ValueError(
                f"the band count of {source} could not be read ({exc}): a run's input channels "
                f"are derived from its own sources, and training at a guessed count would size "
                f"the model wrong for every image it reads. Fix the source, or state "
                "data.num_channels."
            ) from exc
    if not counts:
        raise ValueError(
            "a loader is built over the samples the platform's own producer named, and none were "
            "handed here, so there is no source to read a band count from.")
    if len(counts) > 1:
        named = ", ".join(f"{count} in {counts[count]}" for count in sorted(counts))
        raise ValueError(
            f"the sources this run was handed carry different band counts ({named}): one model "
            f"reads one band count, so training over both would read every image of one of them "
            f"as something it is not. Run over sources of one band count, or state "
            "data.num_channels to read them all at that count."
        )
    return next(iter(counts))


GROUND_TRUTH_COUNTS = ("num_classes", "num_ranks")
"""The counts ground truth carrying its own classes derives, as a loader declares its own in
``ground_truth_count``."""

SIZE_NAMES = GROUND_TRUTH_COUNTS + ("num_channels",)
"""Every size a loader is built at, in the spelling a config and a caller state them by."""


def stated_sizes(stated: "Mapping[str, Any]") -> dict[str, int]:
    """The sizes a mapping (a config's data section) states, absent where it states none."""
    return {name: int(stated[name]) for name in SIZE_NAMES if stated.get(name) is not None}


def resolve_sizes(
    task: str, stated: "Mapping[str, Any]", samples: Sequence[Sample],
    dataset_source: dict | None = None,
) -> dict[str, int]:
    """The sizes a run's loaders are built at: what its caller states, and, for each size stated
    nowhere, what the samples themselves carry.

    Read over every sample the loaders are built from: a class reaching only one side sizes both,
    and two sources disagreeing about their band count refuse. A stated band count is taken as
    given and nothing is probed. A built-in loader's ground truth derives at most one count, its
    ``ground_truth_count``, and that derived count is the one resolved: any stated count refuses
    (a scoped run's class count is its map's length).

    For a task no built-in loader reads and for a bespoke ``dataset_source``, the band count the
    sources carry and only the counts the caller states.
    """
    resolved = stated_sizes(stated)
    cls = builtin_loader(task, dataset_source)
    if cls is not None:
        # Before any ground truth is read, so no size is read off a sample this loader cannot read.
        cls.refuse_other_shapes(samples)
    if "num_channels" not in resolved:
        resolved["num_channels"] = _band_count(samples)
    if cls is None:
        return resolved
    stated_counts = [count for count in GROUND_TRUTH_COUNTS if count in resolved]
    if stated_counts:
        raise ValueError(
            f"this {task} run states {stated_counts}: a built-in loader's count is the one its "
            f"ground truth or class map derives, never an input, and a second value beside it "
            f"would size a head the record does not describe. Drop {stated_counts} from data."
        )
    name = cls.ground_truth_count
    if name is None:
        return resolved
    ids = (Counter(int(v) for s in acquired(samples) for v in np.unique(cls.read_mask(s)))
           if cls.ground_truth_shape == MASK
           else Counter(int(value) for value in table_values(samples)))
    resolved[name] = num_classes_from_distribution(ids)
    return resolved


def stated_tiling(tiling: "Mapping[str, Any] | None") -> dict | None:
    """The ``TiledDetectionDataset`` constructor kwargs a ``tiling`` section states, keys omitted
    so the class's own constructor defaults apply, when it is stated and not ``enabled: false``;
    ``None`` otherwise."""
    if not tiling or not tiling.get("enabled", True):
        return None
    return {k: tiling[k] for k in
            ("tile_size", "overlap", "sliver_frac", "dedup_iou", "skip_empty", "keep_regions")
            if k in tiling}


def run_tiling(task: str, tiling: "Mapping[str, Any] | None") -> dict | None:
    """:func:`stated_tiling` for a run of ``task`` that tiles its training loader: a detection
    run; ``None`` for a run that does not."""
    return stated_tiling(tiling) if task == "detection" else None


def build_dataset(
    task: str, dataset_source: dict | None = None, *,
    samples: Sequence[Sample], sizes: "Mapping[str, int]", scope: ClassScope,
    transforms: Any = None, tiling: dict | None = None, **unowned: Any,
) -> Dataset:
    """Factory: build a dataset by task type, or via a bespoke ``dataset_source`` builder.

    ``samples`` is the producer's own sample list, required on every route: each sample reads its
    own source and the ground truth that answers for it, its own label document, its own mask
    raster or the row its ``row_key`` names. ``scope`` is the class space those samples were
    admitted under (:class:`~tcip_mcp.pipelines.data.selection.ClassScope`), handed to a loader
    over label documents and to a bespoke builder.

    ``sizes`` is what the caller resolved for this run (:func:`resolve_sizes`); a loader reads its
    sources at its band count. Anything given that no recipient could take refuses by name.

    An optional ``tiling`` dict (``{enabled, tile_size, overlap, sliver_frac, dedup_iou,
    skip_empty, keep_regions}``) wraps the detection dataset in a :class:`TiledDetectionDataset`; a
    bespoke builder composes its own tiling, and a ``tiling`` beside it refuses. An unknown task
    with no builder raises ``Unknown task``.
    """
    if unowned:
        raise ValueError(
            f"build_dataset was given {sorted(unowned)}: a loader is built from the samples the "
            f"platform's own producer named, the class space they were admitted under and the "
            f"augmentation this run resolved, and nothing else, so anything further could answer "
            f"for a membership, a class space or a format this run's own record does not state. "
            f"Drop {sorted(unowned)}; a bespoke builder's own configuration goes in "
            "dataset_source.builder_kwargs."
        )
    if dataset_source is not None:
        if tiling is not None:
            raise ValueError(
                "build_dataset was given ['tiling'] beside a dataset_source: a bespoke builder "
                "composes its own tiling over the samples it was handed, so the platform states "
                "none for a dataset it does not build. Drop ['tiling'], or put it in "
                "dataset_source.builder_kwargs."
            )
        return build_from_dataset_source(
            dataset_source, task=task, samples=samples, scope=scope, transforms=transforms)

    from tcip_mcp.pipelines.model_build import resolve_named

    cls = resolve_named(task, _DATASET_MAP, kind="task")
    declared = {"scope": scope} if cls.ground_truth_shape == DOCUMENT else {}
    construct: Any = cls

    tiler = run_tiling(task, tiling)
    if tiler is not None:
        base = construct(samples=samples, **declared)
        assert isinstance(base, DetectionDataset), "_DATASET_MAP's detection entry is this class"
        # Stamped before the tiler indexes every image at the base's band count.
        base.expected_channels = sizes["num_channels"]
        ds: BaseDataset = TiledDetectionDataset(base, transforms=transforms, **tiler)
    else:
        if stated_tiling(tiling) is not None:
            logger.warning("tiling is only supported for task='detection'; ignoring for task=%r", task)
        ds = construct(samples=samples, transforms=transforms, **declared)

    ds.expected_channels = sizes["num_channels"]
    return ds
