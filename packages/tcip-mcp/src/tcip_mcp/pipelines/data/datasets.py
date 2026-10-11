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
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


from tcip_mcp.pipelines import raster_source
from tcip_mcp.pipelines.data.band_groups import BandGroupRef
from tcip_annotation.json_io import LabelDocument
from tcip_annotation.state import (
    Polygon,
    box_derivable,
    object_rows,
    polygonal,
)

from tcip_mcp.pipelines.data.label_queries import (
    acquired,
    json_det_targets,
    resolved,
)
from tcip_mcp.pipelines.derivations import (
    Region,
    num_classes_from_distribution,
)
from tcip_mcp.pipelines.data.selection import (
    DOCUMENT,
    MASK,
    SHAPE_DESCRIPTIONS,
    TABLE,
    ClassScope,
    Sample,
    refuse_unreadable_samples,
)
from tcip_mcp.pipelines.image_utils import (
    frame_size,
    load_image,
    pil_to_tensor,
    pixel_array,
    to_pil_if_faithful,
)
from tcip_mcp.pipelines.schemas import (
    DatasetSourceSchema,
    DataSpec,
    TilingSpec,
    run_tiling,
)

if TYPE_CHECKING:
    from tcip_mcp.pipelines.model_build import SourceLayout

logger = logging.getLogger(__name__)


class BaseDataset(Dataset, ABC):
    """Abstract base for all task-specific datasets."""

    task_type: str = ""
    expected_channels: (
        int  # input channels the dataset yields, stamped by build_dataset
    )

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
        wrong = [
            s.location for s in samples if s.shape != cls.ground_truth_shape
        ]
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
        import io

        opened = Image.open(io.BytesIO(sample.stored.value))
        return np.array(raster_source.PhotographicSource(opened, 1).image)

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
            mine = [
                a
                for a in sample.read.annotations
                if a.subject == self.scope.subject
            ]
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
        if (
            self.transforms is not None
            and not BaseImageDataset._warned_ndarray_transforms
        ):
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
        "boxes": torch.tensor(target["boxes"], dtype=torch.float32).reshape(
            -1, 4
        ),
        "labels": torch.tensor(target["labels"], dtype=torch.int64),
        "iscrowd": torch.tensor(crowd_of(target), dtype=torch.int64),
        **(
            {
                "attributes": torch.as_tensor(
                    target["attributes"], dtype=torch.int64
                )
            }
            if "attributes" in target
            else {}
        ),
    }


PER_BOX_KEYS = ("boxes", "scores", "labels", "masks", "iscrowd", "attributes")
"""The keys of a detection target or a detection record holding one row per box, which every
row filter keeps in step."""


def crowd_of(target: Mapping[str, Any]) -> Any:
    """A detection target's per-box crowd flags, which every detection producer states; a target
    stating none refuses (``ValueError``) naming the key."""
    if target.get("iscrowd") is None:
        raise ValueError(
            "this detection target states no 'iscrowd': whether each row is one "
            "object or a crowd region is its producer's to state, so a bespoke "
            "dataset emits it beside its boxes and labels."
        )
    return target["iscrowd"]


def object_region(target: Mapping[str, Any], area: float) -> Region:
    """The objects of a detection ``target`` (its rows but the crowd regions, :func:`object_rows`),
    their xyxy boxes and labels, over a region of ``area`` pixels."""
    keep = object_rows(crowd_of(target))
    return Region(
        boxes=np.asarray(target["boxes"], dtype=np.float32).reshape(-1, 4)[
            keep
        ],
        labels=np.asarray(target["labels"], dtype=np.int64).reshape(-1)[keep],
        area=float(area),
    )


def class_counts(regions: Iterable[Region]) -> dict[int, int]:
    """Objects per 0-indexed class id over ``regions``."""
    return dict(
        Counter(int(label) - 1 for r in regions for label in r.labels.tolist())
    )


def instance_targets(targets: list[dict]) -> list[dict]:
    """``targets`` as the built-in torchvision heads take them: every crowd row
    (:func:`object_rows`) dropped from each per-box key, every other key as it was; the input
    targets are unchanged."""
    out = []
    for t in targets:
        keep = object_rows(crowd_of(t))
        out.append(
            {k: v[keep] if k in PER_BOX_KEYS else v for k, v in t.items()}
        )
    return out


class DocumentDataset(BaseImageDataset):
    """A loader over samples whose ground truth is a per-image label document, read under the
    run's own admitted class space (``scope``)."""

    ground_truth_shape = DOCUMENT

    def __init__(
        self,
        samples: Sequence[Sample],
        transforms: Any = None,
        *,
        scope: ClassScope,
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
        return json_det_targets(
            document.annotations,
            self.scope,
            reads=cast("Callable[[Any], bool]", self.reads_geometry),
        )

    def region(self, stem: str) -> Region:
        """One sample's objects over the frame its label document records
        (:func:`object_region`)."""
        from tcip_mcp.pipelines.data.splits import label_document_extent

        document = self.document(stem)
        width, height = label_document_extent(
            document, f"{stem}'s label document"
        )
        return object_region(self.det_targets(document), width * height)

    @property
    def regions(self) -> list[Region]:
        """Each sample's :meth:`region`, in index order."""
        return [self.region(stem) for stem in self.stems]

    @staticmethod
    def tensors(target: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        """The tensors this task's loader emits for ``target``'s rows, each in the coordinates of
        the image it is served with: :func:`target_tensors`."""
        return target_tensors(target)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        stem = self.stems[idx]
        img = self._open_image(stem)
        target = self.tensors(
            {**self.det_targets(self.document(stem)), "size": frame_size(img)}
        )
        return self._finalize(img, {**target, "image_id": idx})


class DetectionDataset(DocumentDataset):
    """Object detection over a recorded sample list: each sample reads its own source and the label
    document that answers for it.
    """

    task_type = "detection"
    reads_geometry = staticmethod(box_derivable)
    reads_description = "a box or a polygon of its subject"

    @property
    def class_distribution(self) -> dict[int, int]:
        return class_counts(self.regions)


# ====================================================================
# Tiled Detection (SAHI's slice lattice)
# ====================================================================


def slice_candidates(
    boxes: np.ndarray, slices: list[tuple[int, int, int, int]]
) -> list[np.ndarray]:
    """Each slice's candidate rows of a frame whose full-image xyxy ``boxes`` are given: the
    ascending indices of every row whose box can meet the slice's half-open ``(x0, y0, x1, y1)``
    extent, a superset :func:`clip_rows_to_tile` decides on. Found from the slice origins each
    box's extent spans, so the cost follows the box-slice incidences; slice origins are distinct,
    as a lattice's are. Each slice gets its own fresh array."""
    found: list[list[int]] = [[] for _ in slices]
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if slices and len(boxes):
        xs = np.unique(np.asarray([s[0] for s in slices], dtype=np.int64))
        ys = np.unique(np.asarray([s[1] for s in slices], dtype=np.int64))
        span_x = max(x1 - x0 for x0, _y0, x1, _y1 in slices)
        span_y = max(y1 - y0 for _x0, y0, _x1, y1 in slices)
        col_lo = np.searchsorted(xs, boxes[:, 0] - span_x, side="right")
        col_hi = np.searchsorted(xs, boxes[:, 2], side="left")
        row_lo = np.searchsorted(ys, boxes[:, 1] - span_y, side="right")
        row_hi = np.searchsorted(ys, boxes[:, 3], side="left")
        index_of = {(x0, y0): m for m, (x0, y0, _x1, _y1) in enumerate(slices)}
        for i in range(len(boxes)):
            for k in range(row_lo[i], row_hi[i]):
                for j in range(col_lo[i], col_hi[i]):
                    m = index_of.get((int(xs[j]), int(ys[k])))
                    if m is not None:
                        found[m].append(i)
    return [np.asarray(rows, dtype=np.int64) for rows in found]


def clip_rows_to_tile(
    shapes: Sequence[Any],
    boxes: np.ndarray,
    rows: np.ndarray,
    tile: tuple[int, int, int, int],
    min_box_size: float,
) -> tuple[np.ndarray, np.ndarray, list[Polygon]]:
    """The ``rows`` of a frame clipped once to the half-open pixel ``tile`` ``(x0, y0, x1, y1)``:
    each row's region (``shapes``, its Shapely geometry,
    :func:`~tcip_annotation.matching.shapely_geometry`) intersected with the tile.

    A row whose clipped region has no area leaves the tile. A row whose full-image xyxy box
    (``boxes``) crosses the tile edge also leaves it when its clipped region is a seam sliver: the
    characteristic size of the region's bounds under ``min_box_size``. Returns the surviving
    rows' tile-local boxes (the bounds of their clipped regions, float32 xyxy), their indices, and
    each one's clipped region as a tile-local ``Polygon``, one ring per piece the tile cut.
    """
    import shapely
    from tcip_annotation.matching import box_sizes

    x0, y0, x1, y1 = tile
    rows = np.asarray(rows, dtype=np.int64)
    clipped = shapely.intersection(
        np.asarray(shapes, dtype=object)[rows], shapely.box(x0, y0, x1, y1)
    )
    bounds = np.nan_to_num(shapely.bounds(clipped)).reshape(-1, 4)
    frame = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)[rows]
    crosses = (
        (frame[:, 0] < x0)
        | (frame[:, 1] < y0)
        | (frame[:, 2] > x1)
        | (frame[:, 3] > y1)
    )
    keep = (shapely.area(clipped) > 0) & ~(
        crosses & (box_sizes(bounds) < min_box_size)
    )
    pieces = [
        [
            p
            for p in shapely.get_parts(region)
            if isinstance(p, shapely.Polygon) and p.area
        ]
        for region in clipped[keep]
    ]
    regions = [
        Polygon(
            rings=[
                [(x - x0, y - y0) for x, y in p.exterior.coords[:-1]]
                for p in parts
            ]
        )
        for parts in pieces
    ]
    local = (bounds[keep] - [x0, y0, x0, y0]).astype(np.float32)
    return local, rows[keep], regions


def _validated_keep_regions(
    keep_regions: "Sequence[tuple[int, int, int, int]] | None",
) -> list[tuple[int, int, int, int]] | None:
    """``keep_regions`` as int 4-tuples, or ``None`` when no filter was asked for.

    A malformed rect refuses by name. An empty sequence is a real filter that keeps nothing,
    distinct from ``None``.
    """
    if keep_regions is None:
        return None
    regions: list[tuple[int, int, int, int]] = []
    for region in keep_regions:
        vals = tuple(int(v) for v in region)
        if len(vals) != 4:
            raise ValueError(
                f"keep region {region!r} must be a half-open pixel rect (x0, y0, x1, y1)"
            )
        x0, y0, x1, y1 = vals
        if x1 <= x0 or y1 <= y0:
            raise ValueError(
                f"keep region {vals!r} has no extent: half-open needs x0 < x1 and y0 < y1"
            )
        regions.append(vals)
    return regions


class TiledDetectionDataset(BaseImageDataset):
    """Wrap a :class:`DocumentDataset` (a detection or an instance-segmentation one) and expand
    each source image into native-resolution slices of SAHI's lattice
    (:func:`~tcip_mcp.pipelines.slicing.slice_lattice`), each row's region clipped once to the
    slice (:func:`clip_rows_to_tile`): a row whose clipped region has no area or is a seam sliver
    leaves the slice, and one that stays carries the bounds of its clipped region as its box and
    that region, tile-local, as its geometry. Each slice emits the base task's target
    (``DocumentDataset.tensors``), an instance mask the clipped region rasterized.

    Slice membership is computed at ``__init__``. Each source's frame comes from its
    :class:`~tcip_mcp.pipelines.raster_source.SourceHeader` (a TIFF's or a photograph's header,
    a numpy container's array, which for a ``.npz`` is its pixels loaded), and the same header
    is handed to the process source pool (``raster_source.pooled_source``), which holds open a
    source whose backend reads windows without a decode (a GDAL-served raster, a memory-mapped
    ``.npy``); layout refusals surface here, and a refusal of the pixels themselves surfaces at
    first read. ``__getitem__`` reads a windowed stem one slice window at a time through the
    pool, and a whole-decode stem by decoding the whole source for each slice and indexing the
    slice out of it; both emit the base's target dict shape. The dataset itself never holds an
    open source object, so it pickles into spawned DataLoader workers.

    ``keep_regions``, when given, is a sequence of half-open pixel rects ``(x0, y0, x1, y1)``
    in each image's own full-resolution frame: only slices lying fully inside one of them are
    indexed (an empty sequence keeps none). A slice shorter than ``tile_size`` (a frame shorter
    than the tile on that axis) is dropped first and counted in ``tiles_dropped_past_extent``;
    slices no rect contains count in ``tiles_dropped_outside_regions``. Without ``keep_regions``
    both counts stay 0 and every slice is kept.

    ``tile_size`` and ``overlap`` are the lattice's, both required. A row crossing the slice edge
    whose clipped region's bounds fall under ``sliver_frac`` of the class's average size is a
    tile-seam sliver and dropped; unstated, the fraction derives from the ground truth's own
    size spread (``derivations.derive_sliver_frac``), and ground truth too sparse to derive it
    refuses (``ValueError``) naming ``tiling.sliver_frac``.
    """

    def __init__(
        self,
        base: DocumentDataset,
        tile_size: int,
        overlap: float,
        sliver_frac: float | None = None,
        skip_empty: bool = False,
        transforms: Any = None,
        keep_regions: Sequence[tuple[int, int, int, int]] | None = None,
    ) -> None:
        from tcip_mcp.pipelines.derivations import (
            char_sizes,
            derive_sliver_frac,
        )
        from tcip_mcp.pipelines.raster_source import rect_contains_rect
        from tcip_mcp.pipelines.slicing import is_full_slice, slice_lattice

        from tcip_annotation.matching import geometry_rings, shapely_geometry

        self._samples = base._samples
        self.expected_channels = base.expected_channels
        self._tensors = type(base).tensors
        self.transforms = transforms
        self._index: list[dict] = []
        # Plain values only, so the dataset pickles into spawned workers; asserted at decode time.
        self._source_frames: dict[str, dict[str, Any]] = {}
        regions = _validated_keep_regions(keep_regions)
        self.tiles_dropped_past_extent = 0
        self.tiles_dropped_outside_regions = 0

        # Pass 1: every image's dims and rows, and the objects of each whole frame.
        stems_data: list[
            tuple[str, np.ndarray, dict[str, np.ndarray], list, int, int]
        ] = []
        frames: list[Region] = []
        for stem in base.stems:
            # Acquired now, so an unreadable layout refuses here rather than mid-epoch.
            header = raster_source.SourceHeader(base.image_of(stem))
            w, h = header.frame_at(self.expected_channels)
            src = raster_source.pooled_source(header, self.expected_channels)
            windowed = src is not None
            channels = None if src is None else int(src.num_channels)
            itemsize = (
                None if src is None else int(np.dtype(src.dtype).itemsize)
            )
            self._source_frames[stem] = {
                "width": int(w),
                "height": int(h),
                "channels": channels,
                "dtype_itemsize": itemsize,
                "windowed": windowed,
            }
            # The frame the boxes were drawn in, which a multi-band raster can decode differently.
            from tcip_mcp.pipelines.data.splits import label_document_extent

            document = base.document(stem)
            authored = label_document_extent(
                document, f"{stem}'s label document"
            )
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
            rows_of: dict[str, np.ndarray] = {
                k: np.asarray(full[k], dtype=np.int64)
                for k in PER_BOX_KEYS
                if k != "boxes" and k in full
            }
            stems_data.append((stem, fb, rows_of, full["geometry"], w, h))
            frames.append(object_region(full, w * h))

        self.tile_size, self.overlap = tile_size, overlap
        sizes = char_sizes(r.boxes for r in frames)
        class_avg_size = float(np.mean(sizes)) if sizes else 0.0
        if sliver_frac is None:
            sliver_frac = derive_sliver_frac(sizes)
        if sliver_frac is None:
            raise ValueError(
                f"the tile-seam sliver cutoff derives from the ground truth's own box-size spread, "
                f"and this dataset holds {len(sizes)} box(es), too few to measure one; state "
                "tiling.sliver_frac for this run."
            )
        min_box_size = sliver_frac * class_avg_size

        for stem, fb, rows_of, geometries, w, h in stems_data:
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
            shapes = [shapely_geometry(geometry_rings(g)) for g in geometries]
            for s, candidates in zip(slices, slice_candidates(fb, slices)):
                tb, rows, clipped = clip_rows_to_tile(
                    shapes, fb, candidates, s, min_box_size
                )
                if skip_empty and len(tb) == 0:
                    continue
                self._index.append(
                    {
                        "stem": stem,
                        "slice": s,
                        "boxes": tb,
                        "geometry": clipped,
                        **{k: v[rows] for k, v in rows_of.items()},
                    }
                )

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
        """Per-stem frame facts recorded when the index was built: ``width`` and ``height`` (the
        source header's frame), ``channels`` and ``dtype_itemsize`` (both ``None`` for a source
        no pooled reader was opened for), and ``windowed`` (whether this source reads through a
        windowed backend)."""
        return {stem: dict(info) for stem, info in self._source_frames.items()}

    @property
    def regions(self) -> list[Region]:
        """Each indexed tile's objects over its own pixel area (:func:`object_region`), in index
        order."""
        return [
            object_region(e, (x1 - x0) * (y1 - y0))
            for e in self._index
            for x0, y0, x1, y1 in [e["slice"]]
        ]

    @property
    def class_distribution(self) -> dict[int, int]:
        return class_counts(self.regions)

    def _read_windowed_tile(
        self, stem: str, info: dict, s: tuple[int, int, int, int]
    ):
        """One slice through the pooled windowed source: its ``[H, W, C]`` array and the source's
        band interpretations.

        Refuses when the recorded frame disagrees with the pooled source's own dims, or the
        returned window's shape disagrees with the requested rect.
        """
        src = raster_source.pooled_source(
            raster_source.SourceHeader(self.image_of(stem)),
            self.expected_channels,
        )
        if src is None or (src.width, src.height) != (
            info["width"],
            info["height"],
        ):
            opened = (
                "a whole decode"
                if src is None
                else f"{src.width}x{src.height}"
            )
            raise ValueError(
                f"tiled dataset frame changed for stem {stem!r}: indexed at "
                f"{info['width']}x{info['height']} through a windowed reader but the source now "
                f"opens as {opened} at {self.expected_channels} channels. Cropping here would "
                f"displace every box."
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
        x0, y0, x1, y1 = e["slice"]
        info = self._source_frames[stem]
        interpretations = None
        if info["windowed"]:
            region, interpretations = self._read_windowed_tile(
                stem, info, e["slice"]
            )
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
            region = arr[y0:y1, x0:x1]
        tile = to_pil_if_faithful(region, band_interpretations=interpretations)
        return self._finalize(
            tile,
            {
                **self._tensors({**e, "size": (x1 - x0, y1 - y0)}),
                "image_id": idx,
            },
        )


# ====================================================================
# Instance Segmentation
# ====================================================================


class InstanceSegDataset(DocumentDataset):
    """Instance masks from per-image polygons: the producer's own samples, each naming its own
    source and the label document that answers for it."""

    task_type = "instance_seg"
    reads_geometry = staticmethod(polygonal)
    reads_description = "a polygon of its subject"

    @staticmethod
    def tensors(target: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        """:func:`target_tensors` beside ``masks``, each row's polygon rasterized on the
        ``size`` ``(width, height)`` of the image it is served with (:func:`instance_masks`)."""
        return {
            **target_tensors(target),
            "masks": instance_masks(target["geometry"], target["size"]),
        }


def instance_masks(
    polygons: Sequence[Any], size: tuple[int, int]
) -> torch.Tensor:
    """Each polygon of ``polygons`` filled on its own ``size`` ``(width, height)`` canvas, as a
    ``[N, height, width]`` uint8 tensor. Every ring of a polygon fills its one mask: a multi-ring
    instance is one occlusion-split object."""
    from PIL import ImageDraw

    w, h = size
    masks = np.zeros((len(polygons), h, w), dtype=np.uint8)
    for mask, polygon in zip(masks, polygons):
        canvas = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(canvas)
        for ring in polygon.rings:
            draw.polygon([(x, y) for x, y in ring], fill=1)
        mask[:] = np.asarray(canvas)
    return torch.from_numpy(masks)


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

    def __init__(
        self, samples: Sequence[Sample], transforms: Any = None
    ) -> None:
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

    def __init__(
        self, samples: Sequence[Sample], transforms: Any = None
    ) -> None:
        self.transforms = transforms
        self._init_from_samples(samples)
        self._values = [
            self.convert(self._samples[k].read) for k in self._keys
        ]

    @property
    def class_distribution(self) -> dict[int, int]:
        return dict(Counter(self._values)) if self.ground_truth_count else {}

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        return self._finalize(
            self._open_image(self.stems[idx]),
            {self.target_key: self._values[idx]},
        )


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


def builtin_loader(
    task: str, dataset_source: DatasetSourceSchema | None = None
) -> type[BaseImageDataset] | None:
    """The built-in loader a run of ``task`` is read by: ``None`` for a run naming a bespoke
    ``dataset_source`` and for a task no built-in loader reads."""
    return None if dataset_source else _DATASET_MAP.get(task)


BespokeSource = tuple[DatasetSourceSchema, "SourceLayout"]
"""A run's ``data.dataset_source`` and the layout of its run's declared files it imports from
(``model_build.SourceLayout``), arriving together."""


def build_from_dataset_source(
    bespoke: BespokeSource,
    *,
    task: str,
    samples: Sequence[Sample],
    scope: ClassScope,
    transforms: Any,
) -> Dataset:
    """Import the agent's dataset builder a ``dataset_source`` names from its run's layout
    (``bespoke``, :data:`BespokeSource`; ``model_build.import_source_builder``) and call it.

    The builder is called with a context of ``samples`` (the sample list for the side being built,
    each carrying its logical image, :func:`~tcip_mcp.pipelines.data.label_queries.resolved`),
    ``scope`` (the :class:`~tcip_mcp.pipelines.data.selection.ClassScope` those samples were
    admitted under, every field ``None`` where the ground truth carries its own classes), ``task``
    and ``transforms``.

    ``builder_kwargs`` configure the builder; a key the context already states refuses by name.
    Declare ``**kwargs`` on the builder to ignore context keys it doesn't use.
    """
    from tcip_mcp.pipelines.model_build import import_source_builder

    dataset_source, layout = bespoke
    fn = import_source_builder(dataset_source.builder, layout)
    builder_kwargs = dataset_source.builder_kwargs or {}
    context = {
        "task": task,
        "samples": resolved(samples),
        "scope": scope,
        "transforms": transforms,
    }
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
            "handed here, so there is no source to read a band count from."
        )
    if len(counts) > 1:
        named = ", ".join(
            f"{count} in {counts[count]}" for count in sorted(counts)
        )
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


def stated_sizes(data: DataSpec) -> dict[str, int]:
    """The sizes a data block states or records, absent where it states none."""
    sizes = {name: getattr(data, name) for name in SIZE_NAMES}
    return {
        name: int(size) for name, size in sizes.items() if size is not None
    }


def resolve_sizes(
    task: str,
    stated: "Mapping[str, int]",
    samples: Sequence[Sample],
    dataset_source: DatasetSourceSchema | None = None,
) -> dict[str, int]:
    """The sizes a run's loaders are built at: what its caller states (``stated``, by
    :data:`SIZE_NAMES`), and, for each size stated nowhere, what the samples themselves carry.

    Read over every sample the loaders are built from: a class reaching only one side sizes both,
    and two sources disagreeing about their band count refuse. A stated band count is taken as
    given and nothing is probed. A built-in loader's ground truth derives at most one count, its
    ``ground_truth_count``, and that derived count is the one resolved: any stated count refuses
    (a scoped run's class count is its map's length).

    For a task no built-in loader reads and for a bespoke ``dataset_source``, the band count the
    sources carry and only the counts the caller states.
    """
    resolved = dict(stated)
    cls = builtin_loader(task, dataset_source)
    if cls is not None:
        # Before any ground truth is read, so no size is read off a sample this loader cannot read.
        cls.refuse_other_shapes(samples)
    if "num_channels" not in resolved:
        resolved["num_channels"] = _band_count(samples)
    if cls is None:
        return resolved
    stated_counts = [
        count for count in GROUND_TRUTH_COUNTS if count in resolved
    ]
    if stated_counts:
        raise ValueError(
            f"this {task} run states {stated_counts}: a built-in loader's count is the one its "
            f"ground truth or class map derives, never an input, and a second value beside it "
            f"would size a head the record does not describe. Drop {stated_counts} from data."
        )
    name = cls.ground_truth_count
    if name is None:
        return resolved
    ids = (
        Counter(
            int(v)
            for s in acquired(samples)
            for v in np.unique(cls.read_mask(s))
        )
        if cls.ground_truth_shape == MASK
        else Counter(int(value) for value in table_values(samples))
    )
    resolved[name] = num_classes_from_distribution(ids)
    return resolved


def build_dataset(
    task: str,
    *,
    samples: Sequence[Sample],
    sizes: "Mapping[str, int]",
    scope: ClassScope,
    transforms: Any = None,
    tiling: TilingSpec | None = None,
    **unowned: Any,
) -> Dataset:
    """Factory: build the platform's dataset for ``task``.

    ``samples`` is the producer's own sample list: each sample reads its own source and the
    ground truth that answers for it, its own label document, its own mask raster or the row its
    ``row_key`` names. ``scope`` is the class space those samples were admitted under
    (:class:`~tcip_mcp.pipelines.data.selection.ClassScope`), handed to a loader over label
    documents.

    ``sizes`` is what the caller resolved for this run (:func:`resolve_sizes`); a loader reads its
    sources at its band count. Anything else given refuses (``ValueError``) by name.

    The tiling ``task`` builds at (``schemas.run_tiling``, whose refusal propagates) wraps the
    task's dataset in a :class:`TiledDetectionDataset` at the options it states
    (``TilingSpec.tiler_options``), and one missing its ``tile_size`` or ``overlap`` refuses
    (``ValueError``). An unknown task raises ``Unknown task``.
    """
    if unowned:
        raise ValueError(
            f"build_dataset was given {sorted(unowned)}: a loader is built from the samples the "
            f"platform's own producer named, the class space they were admitted under and the "
            f"augmentation this run resolved, and nothing else, so anything further could answer "
            f"for a membership, a class space or a format this run's own record does not state. "
            f"Drop {sorted(unowned)}; a dataset_source builds through build_from_dataset_source "
            "(ctx.build_dataset from a training body), configured by its builder_kwargs."
        )

    from tcip_mcp.pipelines.model_build import resolve_named

    cls = resolve_named(task, _DATASET_MAP, kind="task")
    declared = {"scope": scope} if cls.ground_truth_shape == DOCUMENT else {}
    construct: Any = cls

    tiler = run_tiling(task, tiling)
    if tiler is not None:
        if tiler.tile_size is None or tiler.overlap is None:
            raise ValueError(
                f"build_dataset was given an enabled tiling with tile_size {tiler.tile_size!r} "
                f"and overlap {tiler.overlap!r}: a tiler runs on one whole lattice; state both, "
                "or resolve them with split_construction.resolved_tiling."
            )
        base = construct(samples=samples, **declared)
        assert isinstance(base, DocumentDataset), (
            "a task that tiles reads label documents"
        )
        # Stamped before the tiler indexes every image at the base's band count.
        base.expected_channels = sizes["num_channels"]
        ds: BaseDataset = TiledDetectionDataset(
            base, transforms=transforms, **tiler.tiler_options()
        )
    else:
        ds = construct(samples=samples, transforms=transforms, **declared)

    ds.expected_channels = sizes["num_channels"]
    return ds
