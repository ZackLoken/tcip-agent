"""Multi-task datasets with standardized interfaces.

Every loader here is built from the samples the producer named
(``label_queries.admit``): each sample reads its own source and the ground truth that answers for
it, and nothing here enumerates a directory or a table. Each dataset type returns
(image_tensor, target_dict) where the target format is task-specific but always dict-based. A
factory function `build_dataset` dispatches to the correct class by task type, or, for a task the
known loaders don't cover, to a bespoke ``dataset_source`` builder the agent supplies (mirrors
``model_source``; see `build_from_dataset_source`).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


from tcip_mcp.pipelines import raster_source
from tcip_mcp.pipelines.data.band_groups import BandGroupRef
from tcip_annotation.state import box_derivable, polygonal

from tcip_mcp.pipelines.data.label_queries import (
    authored_frame, ground_truth_table, json_det_targets,
)
from tcip_mcp.pipelines.derivations import num_classes_from_distribution
from tcip_mcp.pipelines.data.selection import (
    DOCUMENT, MASK, SHAPE_DESCRIPTIONS, TABLE, ClassScope, Sample, refuse_unreadable_samples,
)
from tcip_mcp.pipelines.image_utils import (
    crop_pad_tile, image_dimensions, load_image,
    pad_tile, pil_to_tensor, resolve_source_path, to_pil_if_faithful,
)

logger = logging.getLogger(__name__)


class BaseDataset(Dataset, ABC):
    """Abstract base for all task-specific datasets."""

    task_type: str = ""
    expected_channels: int  # input channels the dataset yields, stamped by build_dataset

    @property
    @abstractmethod
    def num_classes(self) -> int: ...

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

    Subclasses set ``self.transforms`` (and inherit ``expected_channels`` from build_dataset),
    then build only the task-specific target.

    Every loader here is built from a recorded sample list and sets ``sample_sources`` and
    ``sample_ground_truth``: each sample's own source and ground-truth path, keyed by the sample
    key this dataset indexes by. Every read then goes to the path the sample recorded rather than
    to a directory listing, which is what lets one dataset span capture dates and keeps two dates'
    same-named images apart. ``sample_member_stems`` holds the bare stem a membership record names
    each sample by.

    ``ground_truth_shape`` is the one shape this loader reads
    (:data:`~tcip_mcp.pipelines.data.selection.GROUND_TRUTH_SHAPES`), declared by each subclass
    and refused once in :meth:`refuse_other_shapes`. ``takes`` names the class-space facts this
    loader is built with beyond its samples and transforms, which the factory reads to hand it
    exactly those and nothing else. ``reads_geometry`` declares which geometries answer for this
    loader's measurement, for the loaders whose ground truth is a document.
    """

    ground_truth_shape: str = DOCUMENT
    takes: tuple[str, ...] = ()
    reads_geometry: "Callable[[Any], bool] | None" = None
    reads_description: str = ""
    subject: str | None = None
    transforms: Any = None
    sample_sources: dict[str, str]
    sample_ground_truth: dict[str, str]
    sample_member_stems: dict[str, str]
    sample_counts: dict[str, int]
    id_map: dict[str, int] | None
    _num_classes: int

    @property
    def record_stems(self) -> list[str]:
        """The bare ground-truth stem naming each indexed sample, in index order.

        The key a membership record, a cal/holdout lock and a leakage join all name a member by.
        A loader indexes by each sample's source identity, which keeps two dates' same-named
        images apart but is not what those records spell, so the sample's own member stem is read
        back here rather than each measurement door converting on its own.
        """
        keys: list[str] = list(getattr(self, "stems", None) or getattr(self, "_stems", []))
        return [self.member_stem_of(key) for key in keys]

    def member_stem_of(self, key: str) -> str:
        """The bare stem a membership record names one indexed sample by, as the sample itself
        stated it."""
        return self.sample_member_stems[key]

    @classmethod
    def refuse_other_shapes(cls, samples: Sequence[Sample]) -> None:
        """Refuse a sample whose own ground truth is not the shape this loader reads, naming it.

        The one statement of that refusal, asked where a run's sizes are resolved
        (:func:`resolve_sizes`) before anything reads a sample's ground truth: the sizes a run is
        built at are read off that ground truth, so a sample of another shape is named there
        rather than by whichever reader opens it first.
        """
        wrong = [s.identity for s in samples if s.shape != cls.ground_truth_shape]
        if wrong:
            raise ValueError(
                f"{len(wrong)} sample(s) name ground truth a {cls.task_type} loader does not "
                f"read ({wrong[:5]}): it reads "
                f"{SHAPE_DESCRIPTIONS[cls.ground_truth_shape]}, and reading what these name "
                f"instead would train on something other than the ground truth recorded for "
                f"them. Draw a selection over the ground truth {cls.task_type} reads."
            )

    @staticmethod
    def read_mask(path: "str | Path") -> np.ndarray:
        """One mask raster as the integer class ids it carries.

        The one read of a mask ground truth: what a mask loader serves a sample from, and what the
        run's class count is resolved over (:func:`resolve_sizes`), so the two can never read one
        file differently. A sample is admitted on the strength of this exact file, so a mask gone
        since raises from here rather than reading as background.
        """
        return np.array(load_image(path, 1))

    def _init_from_samples(self, samples: Sequence[Sample]) -> list[str]:
        """Index a recorded sample list and answer the keys this dataset indexes: each sample's
        own source and ground truth, and nothing rediscovered from a directory.

        Each sample is keyed by its own source identity, distinct across capture dates by
        construction, so two dates holding a same-named image index as two samples rather than
        collapsing into one, and the bare stem a membership record names it by rides beside the
        key. Refuses, before indexing any of it, a sample no loader here can read
        (:func:`~tcip_mcp.pipelines.data.selection.refuse_unreadable_samples`) and, where this
        loader declares which geometries it reads, one whose document carries the subject only in
        geometries it does not (:meth:`_refuse_unreadable_geometry`), which reads this instance's
        own subject. Whether a sample's ground truth is the shape this loader reads is
        :meth:`refuse_other_shapes`, asked where the run's sizes are resolved, before this runs.
        """
        refuse_unreadable_samples(samples)
        self._refuse_unreadable_geometry(samples)
        self.sample_sources = {s.identity: s.source for s in samples}
        self.sample_ground_truth = {s.identity: s.ground_truth for s in samples}
        self.sample_member_stems = {s.identity: s.member_stem for s in samples}
        self.sample_counts = {}
        return [s.identity for s in samples]

    def _refuse_unreadable_geometry(self, samples: Sequence[Sample]) -> None:
        """Refuse a sample whose document carries this run's subject only in geometries this
        loader does not read, naming it.

        Admission asks whether a document carries the subject at all; which geometries answer for
        a measurement is this loader's own fact, declared in :attr:`reads_geometry`. A document
        carrying the subject as a point, or as an image-level record, has real ground truth this
        loader cannot turn into a target, so training it would put a real object's pixels in the
        background class with no human having said the image is empty. A loader that declares no
        geometry (a mask raster, a table row) reads the whole of what it was handed and refuses
        nothing here.
        """
        if self.reads_geometry is None:
            return
        from tcip_annotation import json_io

        reads = self.reads_geometry
        wrong = []
        for sample in samples:
            mine = [a for a in json_io.read_annotations(sample.ground_truth)
                    if a.subject == self.subject]
            if mine and not any(reads(a.geometry) for a in mine):
                wrong.append(sample.identity)
        if wrong:
            raise ValueError(
                f"{len(wrong)} sample(s) carry {self.subject!r} only in geometries a "
                f"{self.task_type} loader does not read ({wrong[:5]}): it reads "
                f"{self.reads_description}, and training an image whose real objects it cannot "
                f"read would teach them as background. Run a task whose loader reads what these "
                f"documents carry, or supply a builder that reads them."
            )

    def _init_class_ids_from_draw(self, id_map: dict[str, int] | None) -> None:
        """Set the class ids a sample-built geometry loader reads targets under.

        ``id_map`` is required and is the draw's own: recorded samples can span label trees, so
        there is no single registry beside them to resolve class ids from.
        """
        if not id_map:
            raise ValueError(
                f"a {self.task_type} dataset built from recorded samples needs the draw's own "
                "id_map: its samples can span label trees, so there is no single registry beside "
                "them to resolve class ids from."
            )
        self.id_map = dict(id_map)
        self._num_classes = len(id_map)

    def _resolve_path(self, stem: str) -> Path | BandGroupRef:
        """The logical image one sample key names: the sample's own recorded source (a
        ``BandGroupRef`` when a ``.bandgroup`` manifest groups it)."""
        return resolve_source_path(self.sample_sources[stem])

    def _label_path(self, stem: str) -> Path:
        """The ground truth one sample key names: the path the sample itself recorded."""
        return Path(self.sample_ground_truth[stem])

    def _open_image(self, stem: str):
        """Open an image honoring ``expected_channels``: PIL where the pixels have a faithful
        PIL mode (1/3 channels always; 4 only when the source declares its 4th band alpha), else
        an ``[H, W, C]`` ndarray. See :func:`image_utils.to_pil_if_faithful`."""
        return load_image(self._resolve_path(stem), self.expected_channels)

    @staticmethod
    def _image_size(img) -> tuple[int, int]:
        """Return ``(width, height)`` for a PIL image or an ``[H, W, C]`` array."""
        if isinstance(img, Image.Image):
            return img.size
        return int(img.shape[1]), int(img.shape[0])

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

def record_stems_of(dataset: Any) -> list[str] | None:
    """The bare ground-truth stem naming each of ``dataset``'s samples, in index order, or
    ``None`` when the dataset names its samples nothing.

    The one place a measurement asks a loader what to call its samples. A platform loader answers
    from :attr:`BaseImageDataset.record_stems`, which reads a sample-built dataset's own recorded
    ground truth back. A dataset the ``dataset_source`` seam admitted answers from the ``stems``
    list that interface has always exposed, in the vocabulary its own builder chose: the seam
    accepts any Torch dataset, so nothing may require it to have grown a second attribute. A
    dataset with neither leaves each record's ``image_id`` the integer index it was generated
    with, which is what a per-image join downstream then reports it cannot resolve.
    """
    for attribute in ("record_stems", "stems"):
        named = getattr(dataset, attribute, None)
        if named is not None:
            return list(named)
    return None


def indexed_sample_keys(dataset: Any) -> set[str]:
    """The sample keys a built dataset actually indexes, however many examples each one yields.

    A per-image dataset indexes one example per sample, so every sample it was handed is here. A
    tiled dataset indexes one example per kept tile and names no example at all for a source whose
    tiles all fall outside its keep regions or carry no ground truth, so such a source is absent
    here. A caller that handed a dataset an explicit sample set asks here which of them the loader
    will still read, rather than reading a per-example count that cannot answer it.
    """
    return set(getattr(dataset, "stems", None) or getattr(dataset, "_stems", None) or [])


class DetectionDataset(BaseImageDataset):
    """Object detection over a recorded sample list.

    Membership is exactly what the producer recorded: each sample reads its own source and the
    label document that answers for it, no directory is scanned and no admission is re-derived,
    so the dataset spans whatever capture dates the draw did. Targets come from each sample's own
    per-image document of the json_io schema, read through the run's own ``id_map``.
    """

    task_type = "detection"
    ground_truth_shape = DOCUMENT
    takes = ("subject", "attribute", "id_map")
    reads_geometry = staticmethod(box_derivable)
    reads_description = "a box or a polygon of its subject"

    def __init__(
        self,
        samples: Sequence[Sample],
        transforms: Any = None,
        subject: str | None = None,
        attribute: str | None = None,
        id_map: dict[str, int] | None = None,
    ) -> None:
        self.transforms = transforms
        self.subject = subject
        self.attribute = attribute
        self.stems = self._init_from_samples(samples)
        self._init_class_ids_from_draw(id_map)

    def det_targets(self, stem: str) -> tuple[list, list]:
        """Pixel-xyxy boxes + 1-indexed labels for one sample's own label document.

        Public, because a delivery-grade measurement scores against the ground truth this run
        trains on and reads it here rather than opening the document itself: one statement of
        what this run's targets are, under this run's own class map.

        The samples were already admitted with any image carrying an instance unlabeled for
        ``attribute`` held out (the producer's ``skipped_incomplete_attribute`` rail, a
        fixed-length dataset can't act on this per-``__getitem__`` call, only once, up front), so
        ``n_unlabeled`` is always 0 here by construction; the 3-tuple is unpacked for the shared
        ``json_det_targets`` signature, not because a nonzero count is expected at this point.
        """
        boxes, labels, _n_unlabeled = json_det_targets(
            str(self._label_path(stem)), self.subject, self.attribute, self.id_map)
        return boxes, labels

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def num_samples(self) -> int:
        return len(self.stems)

    @property
    def class_distribution(self) -> dict[int, int]:
        counts: Counter[int] = Counter()
        for stem in self.stems:
            _, labels = self.det_targets(stem)
            for lab in labels:
                counts[lab - 1] += 1  # back to 0-indexed cid
        return dict(counts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        stem = self.stems[idx]
        img = self._open_image(stem)
        w, h = self._image_size(img)
        boxes, labels = self.det_targets(stem)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": idx,
        }
        return self._finalize(img, target)


# ====================================================================
# Tiled Detection (SAHI-style sliding window)
# ====================================================================

TILE_SIZE = 224
"""Tile edge in pixels a tiled detection run uses when its config states none. Stated here, where
the tiler reads it, so a split deriving its block geometry before the dataset exists and the
dataset itself resolve one lattice."""

TILE_OVERLAP = 0.2
"""Fraction of a tile shared with its neighbour when a tiled run's config states none, beside
:data:`TILE_SIZE` and read by the same two callers."""


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
    tiles with labels clipped/remapped to tile space.

    Tile membership is computed at ``__init__`` without decoding pixels. Sources whose backend
    opens without a decode (``raster_source.opens_windowed``: a GDAL-served raster, a
    memory-mapped ``.npy``) are opened through the process source pool, so their dims come from
    the open source and layout refusals surface here; every other container keeps a header-only
    dimension probe, and its refusals surface at first read. ``__getitem__`` reads a windowed
    stem one tile window at a time through the pool, and a whole-decode stem by decoding once
    and cropping; both zero-pad border tiles to ``tile_size`` and emit the same target dict
    shape as ``DetectionDataset``. The dataset itself never holds an open source object, so it
    pickles into spawned DataLoader workers.

    ``keep_regions``, when given, is a sequence of half-open pixel rects ``(x0, y0, x1, y1)``
    in each image's own full-resolution frame: only tiles whose rect lies fully inside one of
    them are indexed (an empty sequence keeps none). Tiles overhanging the image extent are
    dropped first and counted in ``tiles_dropped_past_extent``, since they can never lie inside
    a rect clipped to the image; tiles no rect contains count in
    ``tiles_dropped_outside_regions``. Without ``keep_regions`` both counts stay 0 and
    overhanging tiles are kept and zero-padded.
    """

    task_type = "detection"

    def __init__(
        self,
        base: "DetectionDataset",
        tile_size: int = TILE_SIZE,
        overlap: float = TILE_OVERLAP,
        sliver_frac: float | None = None,
        dedup_iou: float = 0.8,
        skip_empty: bool = False,
        transforms: Any = None,
        keep_regions: Sequence[tuple[int, int, int, int]] | None = None,
    ) -> None:
        from tcip_mcp.pipelines.data.tiling import (
            compute_stride, tile_positions, clipped_boxes_per_tile, dedup_boxes,
            tile_within_extent, rect_contains_tile,
        )
        from tcip_mcp.pipelines.derivations import char_sizes_from_boxes, derive_sliver_frac

        self.base = base
        # This wrapper does its own channel-aware reads and its own tile index over the base's
        # sample maps, so it takes the band count and the paths off the base it was handed.
        self.sample_sources = base.sample_sources
        self.sample_ground_truth = base.sample_ground_truth
        self.sample_member_stems = base.sample_member_stems
        self.expected_channels = base.expected_channels
        self.tile_size = tile_size
        self.overlap = overlap
        self.transforms = transforms
        self.stride = compute_stride(tile_size, overlap)
        self._index: list[dict] = []
        # Per-stem frame facts this index was built against (plain values only; a RasterSource
        # attribute would break pickling into spawned workers), asserted again at decode time.
        self._source_frames: dict[str, dict[str, Any]] = {}
        regions = _validated_keep_regions(keep_regions)
        self.tiles_dropped_past_extent = 0
        self.tiles_dropped_outside_regions = 0

        # Pass 1: read every image's upright dims + full-image-px boxes, and accumulate GT box sizes
        # so the seam-sliver cutoff is derived from this dataset's class-average object size, not a
        # fixed fraction (derive-don't-pin). skip_empty defaults False: empty tiles are valid
        # negatives.
        stems_data: list[tuple[str, np.ndarray, np.ndarray, int, int]] = []
        # xywh per image (char_sizes_from_boxes's own expected shape), converted from the xyxy boxes
        # this loop otherwise deals in, so the class-average size uses the same computation
        # derive_localization_kind/derive_iou_match_threshold already share, never a second formula.
        gt_boxes_per_image: list[list[tuple[float, float, float, float]]] = []
        for stem in base.stems:
            # Through the base's own resolver, the one the read path uses, so the frame this index
            # is built against and the pixels __getitem__ later crops come from one source.
            img_source = base._resolve_path(stem)
            windowed = raster_source.opens_windowed(img_source, self.expected_channels)
            if windowed:
                # Header-only open, so an unreadable layout refuses now rather than at step N of
                # an epoch, and the dims are the served source's own.
                src = raster_source.pooled_source(img_source, self.expected_channels)
                w, h = int(src.width), int(src.height)
                channels = int(src.num_channels)
                itemsize: int | None = int(np.dtype(src.dtype).itemsize)
            else:
                # A whole-decode backend keeps the header probe (measured the way __getitem__
                # decodes it): opening it here would hold every source's pixels resident.
                w, h = image_dimensions(img_source, self.expected_channels)
                channels = int(self.expected_channels)
                itemsize = None
            self._source_frames[stem] = {
                "width": int(w), "height": int(h), "channels": channels,
                "dtype_itemsize": itemsize, "windowed": windowed,
            }
            # The frame the boxes were actually drawn in, recorded in the label file itself. The
            # annotation stack measures with PIL, which reports a 40x24x5 GeoTIFF as 5x40, so on a
            # multi-band raster the authored frame and the decoded frame genuinely disagree, and
            # every box would be cropped from somewhere it was never drawn. Comparing the two
            # decoders instead would prove nothing: they share a branch and agree by construction.
            authored = authored_frame(base._label_path(stem))
            if authored is not None and authored != (w, h):
                raise ValueError(
                    f"tiled dataset frame mismatch for stem {stem!r}: the labels record a "
                    f"{authored[0]}x{authored[1]} image but it decodes as {w}x{h} at "
                    f"{self.expected_channels} channels. Tiles would be cut from a different frame "
                    f"than the boxes were drawn in, displacing every box. Re-author the labels "
                    f"against the multi-band frame, or ingest this raster as {authored[0]}x"
                    f"{authored[1]}."
                )
            # Through the base dataset's own targeting, over this sample's own document.
            full_boxes, full_labels = base.det_targets(stem)
            fb = np.asarray(full_boxes, dtype=np.float32).reshape(-1, 4)
            fl = np.asarray(full_labels, dtype=np.int64)
            if len(fb):
                gt_boxes_per_image.append(
                    [(x1, y1, x2 - x1, y2 - y1) for x1, y1, x2, y2 in fb.tolist()])
            stems_data.append((stem, fb, fl, w, h))

        char_sizes = char_sizes_from_boxes(gt_boxes_per_image)
        self.class_avg_size = float(np.mean(char_sizes)) if char_sizes else 0.0
        # A caller-supplied fraction wins; otherwise derive it from this dataset's own size spread
        # (a class with wide natural size variation needs a lower cutoff than a tightly-sized one,
        # a fixed fraction can't tell a genuinely small-but-complete object from a real tile-seam
        # fragment). Falls back to 0.5 only when the spread itself is underivable (too few boxes to
        # measure a spread from, or none at all).
        if sliver_frac is None:
            sliver_frac = derive_sliver_frac(char_sizes)
            self.sliver_frac_source = (
                "GT characteristic-size spread (p10 / mean)" if sliver_frac is not None
                else "documented default (underivable: too few GT boxes to measure a spread)")
            if sliver_frac is None:
                sliver_frac = 0.5
        else:
            self.sliver_frac_source = "explicit"
        self.sliver_frac = sliver_frac
        self.min_box_size = sliver_frac * self.class_avg_size

        # Pass 2: tile using the derived sliver cutoff, boxes clipped in bulk per stem.
        for stem, fb, fl, w, h in stems_data:
            positions = tile_positions(h, w, tile_size, self.stride)
            if regions is not None:
                kept: list[tuple[int, int]] = []
                for tile_x, tile_y in positions:
                    if not tile_within_extent(tile_x, tile_y, tile_size, w, h):
                        self.tiles_dropped_past_extent += 1
                    elif any(rect_contains_tile(r, tile_x, tile_y, tile_size) for r in regions):
                        kept.append((tile_x, tile_y))
                    else:
                        self.tiles_dropped_outside_regions += 1
                positions = kept
            per_tile = clipped_boxes_per_tile(fb, fl, positions, tile_size, self.min_box_size)
            for (tile_x, tile_y), (tb, tl) in zip(positions, per_tile):
                if len(tb) > 1:
                    tb, tl = dedup_boxes(tb, tl, dedup_iou)
                if skip_empty and len(tb) == 0:
                    continue
                self._index.append({"stem": stem, "tile_x": tile_x, "tile_y": tile_y, "boxes": tb, "labels": tl})

    @property
    def num_classes(self) -> int:
        return self.base.num_classes

    @property
    def num_samples(self) -> int:
        return len(self._index)

    @property
    def stems(self) -> list[str]:
        return [e["stem"] for e in self._index]

    @property
    def tile_entries(self) -> list[tuple[str, int, int]]:
        """``(stem, tile_x, tile_y)`` per sample, in index order: the tile geometry a sampler
        needs to order reads for locality without touching a pixel."""
        return [(e["stem"], e["tile_x"], e["tile_y"]) for e in self._index]

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
            for lab in e["labels"].tolist():
                counts[int(lab) - 1] += 1  # 0-indexed cid, matching DetectionDataset
        return dict(counts)

    def _read_windowed_tile(self, stem: str, info: dict, tile_x: int, tile_y: int):
        """One tile through the pooled windowed source, clipped to bounds and zero-padded; PIL
        where the dtype has a faithful mode (so augmentation applies), else ndarray. A 4-channel
        tile only converts when the source's own ``band_interpretations`` names the 4th band
        alpha (see :func:`to_pil_if_faithful`); an untagged or genuinely spectral 4th band stays
        ndarray, same as any other mode PIL can't represent faithfully.

        The recorded frame is checked against the pooled source's own dims: the pool keys on the
        file's mtime and size, so a file replaced since the index was built opens fresh here, and
        a dims disagreement means tiles cut from a frame the boxes were never clipped to. The
        returned window's shape is also checked against the requested rect, the one witness left
        against a decoder disagreeing with its own header. Refuse either way, don't reconcile.
        """
        src = raster_source.pooled_source(self._resolve_path(stem), self.expected_channels)
        if (src.width, src.height) != (info["width"], info["height"]):
            raise ValueError(
                f"tiled dataset frame changed for stem {stem!r}: indexed at "
                f"{info['width']}x{info['height']} but the source now opens as "
                f"{src.width}x{src.height} at {self.expected_channels} channels. Cropping here "
                f"would displace every box."
            )
        y0, y1 = tile_y, min(tile_y + self.tile_size, src.height)
        x0, x1 = tile_x, min(tile_x + self.tile_size, src.width)
        region, _spec = src.read_region(raster_source.Rect(x0, y0, x1, y1))
        if region.shape[:2] != (y1 - y0, x1 - x0):
            raise ValueError(
                f"windowed read for stem {stem!r} returned {region.shape[0]}x{region.shape[1]} "
                f"pixels for the {y1 - y0}x{x1 - x0} window at ({x0}, {y0}): the decoder "
                f"disagrees with its own header, refusing to serve displaced pixels."
            )
        return to_pil_if_faithful(
            pad_tile(region, self.tile_size),
            band_interpretations=getattr(src, "band_interpretations", None))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        e = self._index[idx]
        stem = e["stem"]
        info = self._source_frames[stem]
        if info["windowed"]:
            tile = self._read_windowed_tile(stem, info, e["tile_x"], e["tile_y"])
        else:
            # Channel-aware and EXIF-oriented (via load_image) so cropped pixels align with the
            # tile geometry and the labels clipped in __init__.
            img = self._open_image(stem)
            w, h = self._image_size(img)
            if (w, h) != (info["width"], info["height"]):
                # If the file now decodes differently than the frame the index was built against,
                # the tile would be cut where the boxes were never clipped. Refuse, don't reconcile.
                raise ValueError(
                    f"tiled dataset frame changed for stem {stem!r}: indexed at "
                    f"{info['width']}x{info['height']} but now decodes as {w}x{h} at "
                    f"{self.expected_channels} channels. Cropping here would displace every box."
                )
            tile = crop_pad_tile(img, e["tile_x"], e["tile_y"], self.tile_size, w, h)
        target = {
            "boxes": torch.tensor(e["boxes"], dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(e["labels"], dtype=torch.int64),
            "image_id": idx,
        }
        return self._finalize(tile, target)


# ====================================================================
# Instance Segmentation
# ====================================================================

class InstanceSegDataset(BaseImageDataset):
    """Instance masks from per-image polygons: the producer's own samples, each naming its own
    source and the label document that answers for it."""

    task_type = "instance_seg"
    ground_truth_shape = DOCUMENT
    takes = ("subject", "attribute", "id_map")
    reads_geometry = staticmethod(polygonal)
    reads_description = "a polygon of its subject"

    def __init__(
        self,
        samples: Sequence[Sample],
        transforms: Any = None,
        subject: str | None = None,
        attribute: str | None = None,
        id_map: dict[str, int] | None = None,
    ) -> None:
        self.transforms = transforms
        self.subject = subject
        self.attribute = attribute
        self.stems = self._init_from_samples(samples)
        self._init_class_ids_from_draw(id_map)

    def _read_polys(self, stem: str, w: int, h: int) -> list[tuple[list[list[tuple[float, float]]], int]]:
        """(pixel polygon rings, 1-indexed label) per instance, from this sample's own
        ``<stem>.json``, filtered to ``subject`` and the geometry this loader declares it reads
        (:attr:`reads_geometry`). Already pixel-space; the +1
        background offset is the loader's, nothing on disk carries it.
        An instance's rings is a list, an occlusion-split instance (a leaf crossed by a stem) is
        genuinely more than one ring; ``__getitem__`` rasterizes every ring of an instance into
        that instance's one mask."""
        out: list[tuple[list[list[tuple[float, float]]], int]] = []
        from tcip_annotation import json_io
        for ann in json_io.read_annotations(str(self._label_path(stem))):
            if ann.subject != self.subject or not polygonal(ann.geometry):
                continue
            key = ann.attributes.get(self.attribute) if self.attribute else self.subject
            if key is None or self.id_map is None or key not in self.id_map:
                raise ValueError(
                    f"annotation of subject {self.subject!r} has class key {key!r} not in the run's "
                    f"id map, the registry cannot decode its own labels")
            out.append(([list(ring) for ring in ann.geometry.rings], self.id_map[key] + 1))
        return out

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def num_samples(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        stem = self.stems[idx]
        img = self._open_image(stem)
        w, h = self._image_size(img)

        boxes, labels, masks = [], [], []
        for rings, lab in self._read_polys(stem, w, h):
            all_pts = [p for ring in rings for p in ring]
            if not all_pts:
                continue
            xs = [p[0] for p in all_pts]
            ys = [p[1] for p in all_pts]
            boxes.append([min(xs), min(ys), max(xs), max(ys)])
            labels.append(lab)

            # Rasterize every ring into the same instance mask, a multi-ring instance is one
            # occlusion-split object, not several separate ones; ImageDraw fills union naturally
            # since a pixel already painted 1 stays 1.
            mask = np.zeros((h, w), dtype=np.uint8)
            try:
                from PIL import ImageDraw
                poly_img = Image.new("L", (w, h), 0)
                draw = ImageDraw.Draw(poly_img)
                for ring in rings:
                    if len(ring) >= 3:
                        draw.polygon([(p[0], p[1]) for p in ring], fill=1)
                mask = np.array(poly_img)
            except Exception:
                pass
            masks.append(mask)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "masks": torch.tensor(np.stack(masks) if masks else np.zeros((0, h, w)), dtype=torch.uint8),
            "image_id": idx,
        }
        return self._finalize(img, target)


# ====================================================================
# Semantic Segmentation
# ====================================================================

class SemanticSegDataset(BaseImageDataset):
    """PNG mask images where pixel values are class IDs, over a recorded sample list.

    Membership is exactly what the producer recorded and each sample reads the mask it names, so
    the dataset spans whatever capture dates the draw did. ``num_classes`` is the run's own count,
    resolved once for the run (:func:`resolve_sizes`), never read off this half's own masks.
    """

    task_type = "semantic_seg"
    ground_truth_shape = MASK
    takes = ("num_classes",)

    def __init__(
        self,
        samples: Sequence[Sample],
        transforms: Any = None,
        *,
        num_classes: int,
    ) -> None:
        self.transforms = transforms
        self.stems = self._init_from_samples(samples)
        self.sample_counts = {"annotated": len(self.stems), "skipped_unannotated": 0}
        self._num_classes = num_classes

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def num_samples(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        stem = self.stems[idx]
        img = self._open_image(stem)
        mask = self.read_mask(self._label_path(stem))
        # Key matches the SemanticSegHead loss contract.
        target = {"masks": torch.tensor(mask, dtype=torch.int64)}
        return self._finalize(img, target)


# ====================================================================
# Classification
# ====================================================================

def _values_by_sample(samples: Sequence[Sample]) -> list[str]:
    """Each sample's own value, read out of the table its ``row_key`` names a row of.

    Each table is read once however many samples it answers for. Whether a recorded row is still
    there is the one re-admission's question
    (:func:`~tcip_mcp.pipelines.data.label_queries.refuse_inadmissible_samples`), which every
    bound route runs before a loader is built and which a drawn route cannot fail by
    construction, so this reads the row the sample names rather than restating that check.
    """
    tables: dict[str, dict[str, str]] = {}
    values: list[str] = []
    for sample in samples:
        assert sample.row_key is not None, "refuse_unreadable_samples requires a row key here"
        if sample.ground_truth not in tables:
            tables[sample.ground_truth] = ground_truth_table(sample.ground_truth)
        values.append(tables[sample.ground_truth][sample.row_key])
    return values


class ClassificationDataset(BaseImageDataset):
    """Image classification over a recorded sample list.

    Each sample reads the row its ``row_key`` names in the table it names, so the dataset spans
    whatever tables the producer admitted. ``num_classes`` is the run's own count, resolved once
    for the run (:func:`resolve_sizes`).
    """

    task_type = "classification"
    ground_truth_shape = TABLE
    takes = ("num_classes",)

    def __init__(
        self,
        samples: Sequence[Sample],
        transforms: Any = None,
        *,
        num_classes: int,
    ) -> None:
        self.transforms = transforms
        self._stems = self._init_from_samples(samples)
        self._labels = [int(v) for v in _values_by_sample(samples)]
        self._num_classes = num_classes

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def num_samples(self) -> int:
        return len(self._stems)

    @property
    def class_distribution(self) -> dict[int, int]:
        return dict(Counter(self._labels))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        stem = self._stems[idx]
        img = self._open_image(stem)
        target = {"labels": self._labels[idx]}
        return self._finalize(img, target)


# ====================================================================
# Ordinal
# ====================================================================

class OrdinalDataset(BaseImageDataset):
    """Ordinal regression over a recorded sample list: each sample reads the rank its ``row_key``
    names in the table it names. ``num_ranks`` is the run's own count, resolved once for the run
    (:func:`resolve_sizes`)."""

    task_type = "ordinal"
    ground_truth_shape = TABLE
    takes = ("num_ranks",)

    def __init__(
        self,
        samples: Sequence[Sample],
        transforms: Any = None,
        *,
        num_ranks: int,
    ) -> None:
        self.transforms = transforms
        self._stems = self._init_from_samples(samples)
        self._ranks = [int(v) for v in _values_by_sample(samples)]
        self._num_ranks = num_ranks

    @property
    def num_classes(self) -> int:
        return self._num_ranks

    @property
    def num_samples(self) -> int:
        return len(self._stems)

    @property
    def class_distribution(self) -> dict[int, int]:
        return dict(Counter(self._ranks))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        stem = self._stems[idx]
        img = self._open_image(stem)
        # Key matches the OrdinalHead loss contract (plural, like "labels"/"masks"). The rank
        # count is the head's own, never restated per item.
        target = {"ranks": self._ranks[idx]}
        return self._finalize(img, target)


# ====================================================================
# Regression
# ====================================================================

class RegressionDataset(BaseImageDataset):
    """Continuous-value regression over a recorded sample list: each sample reads the value its
    ``row_key`` names in the table it names."""

    task_type = "regression"
    ground_truth_shape = TABLE

    def __init__(
        self,
        samples: Sequence[Sample],
        transforms: Any = None,
    ) -> None:
        self.transforms = transforms
        self._stems = self._init_from_samples(samples)
        self._values = [float(v) for v in _values_by_sample(samples)]

    @property
    def num_classes(self) -> int:
        return 1

    @property
    def num_samples(self) -> int:
        return len(self._stems)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        stem = self._stems[idx]
        img = self._open_image(stem)
        # Key matches the RegressionHead loss contract.
        target = {"values": self._values[idx]}
        return self._finalize(img, target)


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
"""The one place a task name selects code: which built-in loader reads a run's samples. A task
outside this map reaches a dataset only through a bespoke ``dataset_source`` builder, which is
handed the same samples the producer named for any other task."""

def build_from_dataset_source(
    dataset_source: dict, *, task: str, samples: Sequence[Sample],
    id_map: dict[str, int] | None, transforms: Any,
) -> Dataset:
    """Import the agent's dataset builder and call it, the bespoke-task escape (mirrors
    ``build_from_model_source``). Registry-free, no ``exec``: the builder is imported like any
    module.

    The lowest boundary, and the only one: the context below is the whole call, every field
    stated, so there is no shape of call that hands a builder anything else. ``samples`` is the
    sample list for the side being built, ``id_map``
    the class map those samples were admitted under (``None`` where the ground-truth shape carries
    its own classes and no admitted map exists, which is a mask raster or a table row: derive the
    class space from the ground truth you were handed, the way :func:`resolve_sizes` reads it
    for the platform's own loaders), plus ``task`` and ``transforms``. Never a directory, a
    document path or a format flag: the platform names the samples and the builder builds over
    them, so nothing here asks what a bespoke dataset looks like.

    ``builder_kwargs`` configure the builder and never restate what the producer named: a key the
    context already states refuses by name, since a builder that overrode ``samples`` or
    ``id_map`` would train on membership and a class space the run's own record does not describe.
    Declare ``**kwargs`` on the builder to ignore context keys it doesn't use.

    ``dataset_source`` schema (parallels ``model_source``)::

        {"builder": "my_module:build_ds",  # required, 'module:function' (or 'module.function')
         "builder_kwargs": {...},          # optional, the builder's own configuration
         "source_files": [...],            # optional, provenance (snapshot_model_source copies these)
         "task": "..."}                    # optional, measurement/eval routing
    """
    if not isinstance(dataset_source, dict):
        raise ValueError("dataset_source must be a dict")
    from tcip_mcp.pipelines.model_build import import_source_builder

    fn = import_source_builder(dataset_source)
    builder_kwargs = dataset_source.get("builder_kwargs") or {}
    if not isinstance(builder_kwargs, dict):
        raise ValueError("dataset_source.builder_kwargs must be a dict")
    # What a bespoke builder is handed, and all it is handed: the membership the platform's own
    # producer named, the class map it admitted it under, the task and the augmentation.
    context = {"task": task, "samples": samples, "id_map": id_map, "transforms": transforms}
    restated = sorted(set(context) & set(builder_kwargs))
    if restated:
        raise ValueError(
            f"dataset_source.builder_kwargs restates {restated}: the samples this run trains on, "
            f"the class map they were admitted under, the task and the augmentation are the "
            f"platform's to state, and a builder given a second value for one of them would build "
            f"over something the run's own record does not describe. Drop {restated} from "
            "builder_kwargs."
        )
    return fn(**context, **builder_kwargs)


def _band_count(samples: Sequence[Sample]) -> int:
    """The band count the sources of ``samples`` carry, one count for all of them.

    Every source is probed, not one of them, so a run whose sources disagree refuses by name
    rather than sizing the model for whichever one a probe happened to open first and reading
    every other image at the wrong band count. A source that will not probe refuses the same way:
    a confidently-wrong count sizes the model wrong for every image the run reads. Each source is
    read once for the count, off a header where its container carries one and by decoding where it
    does not, paid where a run's sizes are resolved and only where no width is stated.
    """
    from tcip_mcp.pipelines.derivations import probe_channels

    counts: dict[int, str] = {}
    for sample in samples:
        source = resolve_source_path(sample.source)
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
"""The sizes ground truth carrying its own classes states, as a loader declares them in ``takes``.
A stated one below what that ground truth reaches refuses; the band count beside them is a
property of the source, so a caller narrowing it (reading an RGB source as one channel) states how
to read rather than a vocabulary smaller than the data holds."""

SIZE_NAMES = GROUND_TRUTH_COUNTS + ("num_channels",)
"""Every size a loader is built at, in the spelling a config and a caller state them by."""


def stated_sizes(stated: "Mapping[str, Any]") -> dict[str, int]:
    """The sizes a mapping states, absent where it states none: the one read of a config, for what
    it states before a run and for what that run recorded on it after."""
    return {name: int(stated[name]) for name in SIZE_NAMES if stated.get(name) is not None}


def resolve_sizes(
    task: str, stated: "Mapping[str, Any]", samples: Sequence[Sample],
    dataset_source: dict | None = None,
) -> dict[str, int]:
    """The sizes a run's loaders are built at: what its caller states, and, for each size stated
    nowhere, what the samples themselves carry.

    The one resolution, for the run's own routes and for a single-loader measurement door alike,
    read over every sample the loaders are built from: a class reaching only one side sizes both,
    and two sources disagreeing about their band count refuse rather than one of them governing. A
    stated band count is how the caller reads its sources, so it is taken as given and nothing is
    probed; a stated class or rank count below what the ground truth reaches refuses.

    Only what the caller states, for a task no built-in loader reads and for a bespoke
    ``dataset_source``: a builder sizes the dataset it builds, so nothing is derived for one the
    platform does not build.
    """
    resolved = stated_sizes(stated)
    cls = _DATASET_MAP.get(task)
    if cls is None or dataset_source is not None:
        return resolved
    # Asked before any ground truth is read: reading a size off a sample this loader cannot read
    # is what the refusal exists to stop.
    cls.refuse_other_shapes(samples)
    if "num_channels" not in resolved:
        resolved["num_channels"] = _band_count(samples)
    names = [name for name in cls.takes if name in GROUND_TRUTH_COUNTS]
    if not names:
        return resolved
    ids = (Counter(int(v) for s in samples
                   for v in np.unique(cls.read_mask(Path(s.ground_truth))))
           if cls.ground_truth_shape == MASK
           else Counter(int(value) for value in _values_by_sample(samples)))
    held = num_classes_from_distribution(ids)
    for name in names:
        if name not in resolved:
            resolved[name] = held
        elif resolved[name] < held:
            raise ValueError(
                f"the ground truth this run was handed reaches {held - 1}, which needs "
                f"{name} >= {held}, but {name}={resolved[name]} was configured: a head sized "
                f"under what the ground truth carries would index past its own outputs, or train "
                f"every value beyond its last as that last one. Fix {name} or the ground truth."
            )
    return resolved


def tile_kwargs_from_tiling(tiling: dict) -> dict:
    """The ``TiledDetectionDataset`` constructor kwargs a ``tiling`` config dict carries, keys
    omitted so the class's own constructor defaults apply. Shared by ``build_dataset`` and any
    caller that must resolve tiling geometry before construction (a spatial split derives its
    block geometry at the same ``tile_size``/``overlap`` the dataset will actually use)."""
    return {k: tiling[k] for k in
            ("tile_size", "overlap", "sliver_frac", "dedup_iou", "skip_empty", "keep_regions")
            if k in tiling}


def build_dataset(
    task: str, dataset_source: dict | None = None, *,
    samples: Sequence[Sample], sizes: "Mapping[str, int]", transforms: Any = None,
    scope: ClassScope | None = None, tiling: dict | None = None, **unowned: Any,
) -> Dataset:
    """Factory: build a dataset by task type, or via a bespoke ``dataset_source`` builder.

    ``samples`` is the producer's own sample list and the one membership any loader here is built
    from, required on every route: each sample reads its own source and the ground truth that
    answers for it, its own label document, its own mask raster or the row its ``row_key`` names,
    so the dataset spans whatever capture dates the draw did and nothing here scans a directory or
    a table. ``scope`` is the class space those samples were admitted under
    (:class:`~tcip_mcp.pipelines.data.selection.ClassScope`), ``None`` for ground truth that
    carries its own classes.

    ``sizes`` is what the caller resolved for this run (:func:`resolve_sizes`), the band count its
    sources are read at and the class or rank count its ground truth carries. Nothing is derived
    here: one run's loaders are built at one set of sizes because one resolution answered for all
    of them, and a sample whose ground truth is not the shape the selected loader reads has
    already refused by name there. Each recipient is handed exactly what it declares and nothing
    else: a built-in loader its own :attr:`BaseImageDataset.takes`, a bespoke builder the
    producer-owned context :func:`build_from_dataset_source` states. Anything this factory was
    given that no recipient could take refuses by name, before anything is read off it, here and
    through a bespoke ``train(ctx)`` body's own ``ctx.build_dataset`` call, which is this same
    factory.

    An optional ``tiling`` dict (``{enabled, tile_size, overlap, sliver_frac,
    dedup_iou, skip_empty, keep_regions}``) wraps the detection dataset in a
    :class:`TiledDetectionDataset`; a bespoke builder composes its own tiling over its own
    samples, so it never reaches one, and it states no size either: a builder's own dataset sizes
    itself, and this factory neither reads nor writes anything on an object it did not build.

    The known loaders stay the default; the ``Unknown task`` error below is still raised for a bad
    known-task name with no builder (an honest typo signal), the seam is the escape for a
    genuinely new task.
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
    scope = scope or ClassScope()
    if dataset_source is not None:
        platform_owned = sorted(name for name, value in {"tiling": tiling, **dict(sizes)}.items()
                                if value is not None)
        if platform_owned:
            raise ValueError(
                f"build_dataset was given {platform_owned} beside a dataset_source: a bespoke "
                f"builder composes its own tiling, band count and class count over the samples it "
                f"was handed, so the platform states none of them for a dataset it does not "
                f"build. Drop {platform_owned}, or put them in dataset_source.builder_kwargs."
            )
        return build_from_dataset_source(
            dataset_source, task=task, samples=samples, id_map=scope.id_map,
            transforms=transforms)

    cls = _DATASET_MAP.get(task)
    if cls is None:
        raise ValueError(f"Unknown task '{task}'. Available: {list(_DATASET_MAP.keys())}")
    available: dict[str, Any] = {"subject": scope.subject, "attribute": scope.attribute,
                                 "id_map": scope.id_map, **sizes}
    declared = {name: available[name] for name in cls.takes if available.get(name) is not None}
    # Each loader declares its own constructor keywords, so the call is made through the class
    # object rather than a signature this factory restates.
    construct: Any = cls

    if tiling and tiling.get("enabled", True) and task == "detection":
        base = construct(samples=samples, **declared)
        assert isinstance(base, DetectionDataset), "_DATASET_MAP's detection entry is this class"
        # The tiler's __init__ indexes every image at this band count, reading it off the base it
        # wraps, so the base is stamped before the wrapper is built.
        base.expected_channels = sizes["num_channels"]
        ds: BaseDataset = TiledDetectionDataset(
            base, transforms=transforms, **tile_kwargs_from_tiling(tiling))
    else:
        if tiling and tiling.get("enabled", True) and task != "detection":
            logger.warning("tiling is only supported for task='detection'; ignoring for task=%r", task)
        ds = construct(samples=samples, transforms=transforms, **declared)

    ds.expected_channels = sizes["num_channels"]
    return ds
