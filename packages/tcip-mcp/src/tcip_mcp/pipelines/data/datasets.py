"""Multi-task datasets with standardized interfaces.

Each dataset type returns (image_tensor, target_dict) where the target
format is task-specific but always dict-based. A factory function
`build_dataset` dispatches to the correct class by task type, or, for a
task the known loaders don't cover, to a bespoke ``dataset_source`` builder
the agent supplies (mirrors ``model_source``; see `build_from_dataset_source`).
"""

from __future__ import annotations

import csv
import logging
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


from tcip_mcp.pipelines import raster_source
from tcip_mcp.pipelines.data.band_groups import BandGroupIncomplete, BandGroupRef
from tcip_mcp.pipelines.data.label_queries import (
    assemble_coco, authored_frame, coco_det_targets, dir_label_format, first_labels_json,
    image_name_map, json_det_targets, require_samples, resolve_registry_id_map, trainable_stems,
)
from tcip_mcp.pipelines.image_utils import (
    IMAGE_EXTS, crop_pad_tile, image_dimensions, list_logical_images, load_image,
    pad_tile, pil_to_tensor, resolve_image_source, to_pil_if_faithful,
)

logger = logging.getLogger(__name__)


class BaseDataset(Dataset, ABC):
    """Abstract base for all task-specific datasets."""

    task_type: str = ""
    expected_channels: int = 3  # input channels the dataset yields (3=RGB; set by build_dataset)

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

    Subclasses set ``self.images_dir`` and ``self.transforms`` (and inherit
    ``expected_channels`` from build_dataset), then build only the task-specific target.
    """

    images_dir: Path
    transforms: Any = None

    def _resolve_path(self, stem: str) -> Path | BandGroupRef:
        """A ``stem`` may be a literal path (classification folder mode), a stem in images_dir, or
        (when a ``.bandgroup`` manifest groups sibling band files under it) a ``BandGroupRef``."""
        p = Path(stem)
        if p.is_absolute() or p.exists():
            return p
        return resolve_image_source(self.images_dir, stem)

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

class DetectionDataset(BaseImageDataset):
    """Object detection. ``label_format`` selects the on-disk label format:

    - ``json`` (default): canonical per-image ``<labels_dir>/<stem>.json`` (json_io schema)
    - ``coco``: a single COCO JSON at ``coco_json``, the assembled dataset view of the per-image
      JSON, used for training (annotations matched by file name)
    """

    task_type = "detection"

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        stems: list[str] | None = None,
        transforms: Any = None,
        num_classes: int = 1,
        label_format: str = "json",
        coco_json: str | None = None,
        coco_data: dict | None = None,
        subject: str | None = None,
        attribute: str | None = None,
        id_map: dict[str, int] | None = None,
        date=None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transforms = transforms
        self._num_classes = num_classes
        self.subject = subject
        self.attribute = attribute
        self.label_format = (label_format or "json").lower()
        self._coco = None
        if coco_data is not None:  # in-memory COCO assembled from per-image JSON (train/eval)
            self._coco = coco_data
            self.label_format = "coco"
        elif self.label_format == "coco":
            if not coco_json:
                raise ValueError("label_format='coco' requires coco_json (path to the COCO JSON).")
            from tcip_annotation import json_io

            # An assembled dataset-level COCO document is interop, never checked against this
            # platform's own per-image schema_version ceiling (see tcip_annotation.format_io).
            self._coco = json_io.load_json_document(coco_json)
        # The single name→id map: resolved here for a direct-json build, else supplied by
        # build_dataset (which resolved it once for the COCO assembly). One derivation either way.
        if id_map is None and self.label_format == "json":
            _reg, id_map = resolve_registry_id_map(self.labels_dir, subject, attribute)
            self._num_classes = len(id_map)
        self.id_map = id_map
        # The attribute-completeness rail (an image with any instance never assessed for `attribute`
        # is held out whole, never trained on its labelled subset) lives inside trainable_stems, the
        # one partition that already decides admission and records why, see its docstring. A second
        # filter over trainable_stems' *output* would both corrupt those counts and never run at all
        # on the real build_dataset path (which assembles COCO and so takes the
        # `label_format == "coco"` branch). Applying it at the partition covers both label paths
        # from one implementation and reports a truthful reason for each drop.
        self.stems, self.sample_counts = trainable_stems(
            self.labels_dir, self.images_dir, stems,
            subject=subject, date=date, coco=self._coco,
            attribute=attribute, id_map=self.id_map,
        )
        require_samples(self.stems, self.sample_counts, self.labels_dir)
        # Real on-disk filenames, for matching a stem to the COCO's ``file_name`` (which carries the
        # true name), image_name_map reads the actual directory listing, never a constructed guess.
        self._image_names = image_name_map(self.images_dir)

    def _det_targets(self, stem: str, file_name: str) -> tuple[list, list]:
        """Pixel-xyxy boxes + 1-indexed labels for one image (coco or name-based json).

        ``self.stems`` already excludes any image with an instance unlabeled for ``attribute``
        (``trainable_stems``' ``skipped_incomplete_attribute`` rail, a fixed-length dataset can't
        act on this per-``__getitem__`` call, only once, up front), so ``n_unlabeled`` is always 0
        here by construction; the 3-tuple is unpacked for the shared ``json_det_targets``
        signature, not because a nonzero count is expected at this point.
        """
        if self.label_format == "coco":
            return coco_det_targets(self._coco, file_name)
        boxes, labels, _n_unlabeled = json_det_targets(
            str(self.labels_dir / f"{stem}.json"), self.subject, self.attribute, self.id_map)
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
        if self.label_format == "coco" and self._coco:
            # self._coco may be shared across a full/train/val split trio (assembled once in
            # training_tools.py, threaded into all three builds rather than re-assembled per
            # split), its annotations cover the whole dataset, not just this dataset's own
            # self.stems, so every consumer must filter to its own image set or a split's
            # class_distribution reports the identical, unsplit whole for train and val alike.
            own_names = {self._image_names.get(s, "") for s in self.stems}
            image_names_by_id = {e.get("id"): str(e.get("file_name", ""))
                                 for e in self._coco.get("images", [])}
            for ann in self._coco.get("annotations", []):
                if image_names_by_id.get(ann.get("image_id")) in own_names:
                    counts[ann.get("category_id", 0)] += 1
        else:  # json: parse each image's annotation
            for stem in self.stems:
                _, labels = self._det_targets(stem, "")
                for lab in labels:
                    counts[lab - 1] += 1  # back to 0-indexed cid
        return dict(counts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        stem = self.stems[idx]
        img = self._open_image(stem)
        w, h = self._image_size(img)
        file_name = self._image_names.get(stem, "") if self.label_format == "coco" else ""
        boxes, labels = self._det_targets(stem, file_name)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": idx,
        }
        return self._finalize(img, target)


# ====================================================================
# Tiled Detection (SAHI-style sliding window)
# ====================================================================

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
    overhanging tiles are kept, zero-padded, as always.
    """

    task_type = "detection"

    def __init__(
        self,
        base: "DetectionDataset",
        tile_size: int = 224,
        overlap: float = 0.2,
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
        # This wrapper does its own channel-aware reads rather than delegating to base. Inherit the
        # band count from the dataset being wrapped so every construction path carries it,
        # build_dataset stamps it afterwards, but ctx.tiled_dataset constructs this directly and
        # would otherwise fall back to the 3-channel class default.
        self.images_dir = base.images_dir
        self.expected_channels = getattr(base, "expected_channels", 3)
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
        # fixed fraction (Q5 / derive-don't-pin). skip_empty defaults False: empty tiles are valid
        # negatives.
        stems_data: list[tuple[str, np.ndarray, np.ndarray, int, int]] = []
        # xywh per image (char_sizes_from_boxes's own expected shape), converted from the xyxy boxes
        # this loop otherwise deals in, so the class-average size uses the same computation
        # derive_localization_kind/derive_iou_match_threshold already share, never a second formula.
        gt_boxes_per_image: list[list[tuple[float, float, float, float]]] = []
        for stem in base.stems:
            img_source = resolve_image_source(base.images_dir, stem)
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
            authored = authored_frame(stem, base.labels_dir, base.label_format,
                                       base._coco, base._image_names.get(stem, ""))
            if authored is not None and authored != (w, h):
                raise ValueError(
                    f"tiled dataset frame mismatch for stem {stem!r}: the labels record a "
                    f"{authored[0]}x{authored[1]} image but it decodes as {w}x{h} at "
                    f"{self.expected_channels} channels. Tiles would be cut from a different frame "
                    f"than the boxes were drawn in, displacing every box. Re-author the labels "
                    f"against the multi-band frame, or ingest this raster as {authored[0]}x"
                    f"{authored[1]}."
                )
            # Format-aware read via the base dataset's own targeting (json/coco share one path);
            # only coco needs the image file name to match its annotations. Use the real on-disk name
            # (img_path.name can be miscased on Windows), or the coco match silently finds nothing.
            file_name = base._image_names.get(stem, "") if base.label_format == "coco" else ""
            full_boxes, full_labels = base._det_targets(stem, file_name)
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
    """Instance masks from per-image polygons: canonical per-image JSON ``<stem>.json`` (default),
    or a COCO dict / assembled per-image JSON (``label_format='coco'`` / ``coco_data``)."""

    task_type = "instance_seg"

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        stems: list[str] | None = None,
        transforms: Any = None,
        num_classes: int = 1,
        label_format: str = "json",
        coco_json: str | None = None,
        coco_data: dict | None = None,
        subject: str | None = None,
        attribute: str | None = None,
        id_map: dict[str, int] | None = None,
        date=None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transforms = transforms
        self._num_classes = num_classes
        self.subject = subject
        self.attribute = attribute
        self.label_format = (label_format or "json").lower()
        self._coco = None
        if coco_data is not None:
            self._coco = coco_data
            self.label_format = "coco"
        elif self.label_format == "coco":
            if not coco_json:
                raise ValueError("label_format='coco' requires coco_json (path to the COCO JSON).")
            from tcip_annotation import json_io

            # An assembled dataset-level COCO document is interop, never checked against this
            # platform's own per-image schema_version ceiling (see tcip_annotation.format_io).
            self._coco = json_io.load_json_document(coco_json)
        if id_map is None and self.label_format == "json":
            _reg, id_map = resolve_registry_id_map(self.labels_dir, subject, attribute)
            self._num_classes = len(id_map)
        self.id_map = id_map
        # attribute/id_map must be threaded through: without them the direct-JSON instance_seg path
        # has no attribute-completeness rail at all, an image with any instance never assessed for
        # `attribute` trains on its labeled subset instead of being held out whole, the same gap
        # DetectionDataset's own call already closes.
        self.stems, self.sample_counts = trainable_stems(
            self.labels_dir, self.images_dir, stems,
            subject=subject, date=date, coco=self._coco,
            attribute=attribute, id_map=self.id_map,
        )
        require_samples(self.stems, self.sample_counts, self.labels_dir)
        # Real on-disk filenames for the COCO ``file_name`` match (see DetectionDataset / image_name_map).
        self._image_names = image_name_map(self.images_dir)

    def _read_polys(self, stem: str, w: int, h: int) -> list[tuple[list[list[tuple[float, float]]], int]]:
        """(pixel polygon rings, 1-indexed label) per instance, from the assembled COCO or the
        name-based per-image ``<stem>.json`` (filtered to ``subject`` + polygon geometry). Both are
        already pixel-space; the +1 background offset is the loader's, nothing on disk carries it.
        An instance's rings is a list, an occlusion-split instance (a leaf crossed by a stem) is
        genuinely more than one ring; ``__getitem__`` rasterizes every ring of an instance into
        that instance's one mask."""
        out: list[tuple[list[list[tuple[float, float]]], int]] = []
        if self.label_format == "coco":
            from tcip_annotation import format_io
            assert self._coco is not None, "__init__ always sets _coco when label_format is 'coco'"
            file_name = self._image_names.get(stem, "")
            anns, _, _ = format_io._coco_image_annotations(self._coco, file_name=file_name)
            for a in anns:
                seg = a.get("segmentation")
                if not (isinstance(seg, list) and seg):
                    continue
                rings = []
                for coords in seg:
                    if not (isinstance(coords, list) and len(coords) >= 6):
                        continue
                    rings.append([(float(coords[i]), float(coords[i + 1]))
                                 for i in range(0, len(coords) - 1, 2)])
                if not rings:
                    continue
                out.append((rings, int(a.get("category_id", 0)) + 1))
            return out
        from tcip_annotation import json_io
        from tcip_annotation.state import Polygon
        for ann in json_io.read_annotations(str(self.labels_dir / f"{stem}.json")):
            if ann.subject != self.subject or not isinstance(ann.geometry, Polygon):
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
    """PNG mask images where pixel values are class IDs."""

    task_type = "semantic_seg"

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        stems: list[str] | None = None,
        transforms: Any = None,
        num_classes: int = 2,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transforms = transforms
        self._num_classes = num_classes
        # A sample needs a mask. Unlike detection there is no unconfirmed-empty case: an
        # all-background mask is an explicit annotation, so existence is the whole rail here.
        # Serving an image with no mask would train it as entirely background, a fabricated
        # negative by another route.
        mask_stems = {p.stem for p in self.masks_dir.iterdir()} if self.masks_dir.is_dir() else set()
        candidates = stems or sorted(image_name_map(self.images_dir))
        self.stems = [s for s in candidates if s in mask_stems]
        self.sample_counts = {"annotated": len(self.stems),
                              "skipped_unannotated": len(candidates) - len(self.stems)}
        if not self.stems:
            raise ValueError(
                f"no trainable samples: none of the {len(candidates)} image(s) in "
                f"{self.images_dir} have a mask in {self.masks_dir}. An image with no mask would "
                f"train as entirely background."
            )

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
        mask_path = self.masks_dir / f"{stem}.png"
        # load_image EXIF-orients so the mask shares the image's upright frame (no-op for a
        # plain PNG mask; matters only if a mask ever carries EXIF orientation).
        mask = np.array(load_image(mask_path, 1)) if mask_path.exists() else np.zeros((h, w), dtype=np.int64)
        # Key matches the SemanticSegHead loss contract.
        target = {"masks": torch.tensor(mask, dtype=torch.int64)}
        return self._finalize(img, target)


# ====================================================================
# Classification
# ====================================================================

class ClassificationDataset(BaseImageDataset):
    """Image classification from CSV (image_stem, label) or folder structure."""

    task_type = "classification"

    def __init__(
        self,
        images_dir: str,
        csv_path: str | None = None,
        stems: list[str] | None = None,
        labels: list[int] | None = None,
        transforms: Any = None,
        num_classes: int | None = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.transforms = transforms
        if csv_path is not None:
            self._stems, self._labels = self._load_csv(csv_path)
        elif stems is not None and labels is not None:
            self._stems = stems
            self._labels = labels
        else:
            # Folder-based: images_dir/<class_name>/<image>
            self._stems, self._labels = self._load_folder_structure()
        # Derived from the labels actually loaded, the same way build_dataset derives
        # detection's num_classes from the label registry: a class id the loaded labels reach
        # but the configured num_classes doesn't cover would index past the head's logits.
        derived_classes = (max(self._labels) + 1) if self._labels else 0
        if num_classes is not None and derived_classes > num_classes:
            raise ValueError(
                f"the loaded labels reach class {derived_classes - 1}, which needs "
                f"num_classes >= {derived_classes}, but num_classes={num_classes} was "
                f"configured; fix num_classes or the data."
            )
        self._num_classes = num_classes if num_classes is not None else max(derived_classes, 1)

    def _load_csv(self, path: str) -> tuple[list[str], list[int]]:
        stems, labels = [], []
        with open(path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if len(row) >= 2:
                    stems.append(row[0].strip())
                    labels.append(int(row[1].strip()))
        return stems, labels

    def _load_folder_structure(self) -> tuple[list[str], list[int]]:
        stems, labels = [], []
        class_dirs = sorted(d for d in self.images_dir.iterdir() if d.is_dir())
        for cid, cdir in enumerate(class_dirs):
            for f in cdir.iterdir():
                if f.suffix.lower() in IMAGE_EXTS:
                    stems.append(str(f))
                    labels.append(cid)
        return stems, labels

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
    """Ordinal regression from CSV (image_stem, rank). E.g., disease severity 0-4."""

    task_type = "ordinal"

    def __init__(
        self,
        images_dir: str,
        csv_path: str,
        transforms: Any = None,
        num_ranks: int | None = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.transforms = transforms
        self._stems: list[str] = []
        self._ranks: list[int] = []
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    self._stems.append(row[0].strip())
                    self._ranks.append(int(row[1].strip()))
        # Derived from the ranks actually loaded, the same way build_dataset derives
        # detection's num_classes from the label registry: the OrdinalHead's CORN loss and
        # decode() both loop over range(num_ranks - 1) using this count with no check against
        # the data, so a rank the head was never told about silently folds into the top rank
        # instead of raising.
        derived_ranks = (max(self._ranks) + 1) if self._ranks else 0
        if num_ranks is not None and derived_ranks > num_ranks:
            raise ValueError(
                f"{csv_path} carries ranks up to {derived_ranks - 1}, which needs "
                f"num_ranks >= {derived_ranks}, but num_ranks={num_ranks} was configured. "
                f"The CORN head/loss would silently train every rank >= num_ranks - 1 as the "
                f"same top rank rather than raising; fix num_ranks or the data."
            )
        self._num_ranks = num_ranks if num_ranks is not None else max(derived_ranks, 1)

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
        # Key matches the OrdinalHead loss contract (plural, like "labels"/"masks").
        target = {"ranks": self._ranks[idx], "num_ranks": self._num_ranks}
        return self._finalize(img, target)


# ====================================================================
# Regression
# ====================================================================

class RegressionDataset(BaseImageDataset):
    """Continuous-value regression from CSV (image_stem, value)."""

    task_type = "regression"

    def __init__(
        self,
        images_dir: str,
        csv_path: str,
        transforms: Any = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.transforms = transforms
        self._stems: list[str] = []
        self._values: list[float] = []
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    self._stems.append(row[0].strip())
                    self._values.append(float(row[1].strip()))

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

_DATASET_MAP = {
    "detection": DetectionDataset,
    "instance_seg": InstanceSegDataset,
    "semantic_seg": SemanticSegDataset,
    "classification": ClassificationDataset,
    "ordinal": OrdinalDataset,
    "regression": RegressionDataset,
}

def build_from_dataset_source(dataset_source: dict, **kwargs: Any) -> Dataset:
    """Import the agent's dataset builder and call it, the bespoke-task escape (mirrors
    ``build_from_model_source``). Registry-free, no ``exec``: the builder is imported like any
    module. It receives the run's data context (``images_dir`` / ``labels_dir`` / ``stems`` /
    ``transforms`` / ``task``, whatever ``build_dataset`` was given) merged with its own
    ``builder_kwargs`` (which win on conflict), and must return a torch ``Dataset``. Declare
    ``**kwargs`` on the builder to ignore context keys it doesn't use.

    ``dataset_source`` schema (parallels ``model_source``)::

        {"builder": "my_module:build_ds",  # required, 'module:function' (or 'module.function')
         "builder_kwargs": {...},          # optional, passed to the builder (win on conflict)
         "source_files": [...],            # optional, provenance (snapshot_model_source copies these)
         "task": "..."}                    # optional, measurement/eval routing
    """
    if not isinstance(dataset_source, dict):
        raise ValueError("dataset_source must be a dict")
    from tcip_mcp.pipelines.model_build import _import_dotted

    fn = _import_dotted(dataset_source.get("builder"))
    builder_kwargs = dataset_source.get("builder_kwargs") or {}
    if not isinstance(builder_kwargs, dict):
        raise ValueError("dataset_source.builder_kwargs must be a dict")
    return fn(**{**kwargs, **builder_kwargs})


def _autoresolve_json_labels(kwargs: dict, *, subject: str, attribute: str | None,
                             id_map: dict[str, int], date) -> None:
    """Route a name-based per-image-JSON label dir onto the assembled-COCO path for training/eval.

    No-op when the caller pinned a format, already supplied COCO data, or has no ``images_dir`` to
    assemble against (both branches below share that one precondition). The single ``id_map`` is
    threaded into ``assemble_coco`` (and thus the one ``to_coco_dataset`` call), so the assembled
    categories rest on the same name→id derivation as the loader targets and the contract dims.
    A dataset-level COCO export sitting alongside real per-image label files is refused by name,
    with both remedies stated (move the export out, or point ``data.coco_json`` at it), since
    only the breeder knows which of the two is this dataset's real label source.

    ``date`` is the caller's own, the same one the dataset class hands ``trainable_stems``, so the
    assembled COCO and the partition that consumes it read one confirmation bucket rather than two
    keys that can disagree.
    """
    if kwargs.get("coco_data") is not None or kwargs.get("coco_json") or kwargs.get("label_format"):
        return
    labels_dir = kwargs.get("labels_dir", "")
    images_dir = kwargs.get("images_dir", "")
    if not labels_dir or not images_dir:
        return
    detected = dir_label_format(labels_dir)
    if detected == "coco":
        offending = first_labels_json(labels_dir)
        raise ValueError(
            f"labels_dir={labels_dir!r} holds a dataset-level COCO file ({offending}): if the "
            "per-image label files in this directory are the ones that should train, move it out "
            "of labels_dir; if this COCO export is the intended label source, pass data.coco_json "
            "(or label_format='coco') to train on it directly."
        )
    if detected == "json":
        kwargs["coco_data"] = assemble_coco(
            labels_dir, images_dir, stems=kwargs.get("stems"),
            subject=subject, attribute=attribute, id_map=id_map, date=date)
        kwargs["label_format"] = "coco"


def _probe_num_channels(images_dir: str | Path | None, stems: list[str] | None,
                        default: int = 3) -> int:
    """Band count of one sample raster from ``images_dir`` (derive-don't-pin, not a pinned 3).

    Probes a single image (guard: one sample, not every image) so a multi-band raster threads its
    real channel count through ``in_chans`` instead of silently defaulting to RGB. Falls back to
    ``default`` only when no readable raster is found at all, or a genuinely unexpected decode error
    hits it, never for a stale ``.bandgroup`` manifest (``BandGroupIncomplete`` propagates loudly
    instead), since a confidently-wrong channel count silently sizes the model wrong for every
    dataset that hits it.
    """
    if not images_dir:
        return default
    images_dir = Path(images_dir)
    sample: Path | BandGroupRef | None = None
    for stem in (stems or []):
        try:
            sample = resolve_image_source(images_dir, stem)
            break
        except FileNotFoundError:
            # Per-stem skip-and-try-the-next-one (BandGroupIncomplete included): with multiple
            # candidate stems, one stale/missing entry doesn't preclude probing a different, intact
            # one, only the single-sample fallback below has no "next stem" to fall back to.
            continue
    if sample is None:
        logical = list_logical_images(images_dir)
        if logical:
            # Through resolve_image_source (not a bare dict pick): its completeness check is what
            # turns a stale manifest into a named BandGroupIncomplete here, rather than a bare
            # decode error surfacing later inside probe_channels.
            sample = resolve_image_source(images_dir, sorted(logical)[0])
    if sample is None:
        return default
    from tcip_mcp.pipelines.derivations import probe_channels

    try:
        return int(probe_channels(sample))
    except BandGroupIncomplete:
        raise
    except Exception:
        return default


def tile_kwargs_from_tiling(tiling: dict) -> dict:
    """The ``TiledDetectionDataset`` constructor kwargs a ``tiling`` config dict carries, keys
    omitted so the class's own constructor defaults apply. Shared by ``build_dataset`` and any
    caller that must resolve tiling geometry before construction (a spatial split derives its
    block geometry at the same ``tile_size``/``overlap`` the dataset will actually use)."""
    return {k: tiling[k] for k in
            ("tile_size", "overlap", "sliver_frac", "dedup_iou", "skip_empty", "keep_regions")
            if k in tiling}


def build_dataset(task: str, dataset_source: dict | None = None, **kwargs) -> Dataset:
    """Factory: build a dataset by task type, or via a bespoke ``dataset_source`` builder.

    An optional ``tiling`` dict (``{enabled, tile_size, overlap, sliver_frac,
    dedup_iou, skip_empty, keep_regions}``) wraps the detection dataset in a
    :class:`TiledDetectionDataset`. Ignored for non-detection tasks.

    ``num_channels`` is derived by probing one sample raster when the caller does not pin it, so a
    multi-band input threads its real band count through ``in_chans`` instead of defaulting to RGB.

    ``date`` names the capture date whose confirmed negatives this build may admit, the same key
    the GUI recorded them under, and reaches both the assembled COCO and the partition unchanged.
    A run over ``annotations/<date>/`` states that date; ``None`` (the default) is the key a tree
    that carries no date was written under, and no date is ever recovered from the labels path.

    ``dataset_source`` is the bespoke seam (mirrors ``model_source``): when given, an agent-supplied
    importable builder produces the dataset for a task the known loaders don't cover. The known
    loaders stay the default; the ``Unknown task`` error below is still raised for a bad known-task
    name (an honest typo signal), the seam is the escape for a genuinely new task.
    """
    tiling = kwargs.pop("tiling", None)
    num_channels = kwargs.pop("num_channels", None)
    if num_channels is None:
        num_channels = _probe_num_channels(kwargs.get("images_dir"), kwargs.get("stems"))

    if dataset_source is not None:
        ds = build_from_dataset_source(dataset_source, task=task, **kwargs)
        if getattr(ds, "expected_channels", None) is None:
            # ds's real type is whatever the agent's bespoke builder returned; setattr (like the
            # getattr above) reaches it without narrowing this seam to BaseDataset's own shape.
            setattr(ds, "expected_channels", num_channels)
        return ds

    cls = _DATASET_MAP.get(task)
    if cls is None:
        raise ValueError(f"Unknown task '{task}'. Available: {list(_DATASET_MAP.keys())}")

    if task in ("detection", "instance_seg"):
        subject = kwargs.get("subject")
        attribute = kwargs.get("attribute")
        has_coco = (kwargs.get("coco_data") is not None or kwargs.get("coco_json")
                    or (kwargs.get("label_format") or "").lower() == "coco")
        if not has_coco and kwargs.get("labels_dir"):
            # Name-based json: resolve the single id map once, set num_classes and assemble the COCO
            # from it, the loader, the categories, and resolve_contract_dims all read this one map.
            _registry, id_map = resolve_registry_id_map(kwargs["labels_dir"], subject, attribute)
            # resolve_registry_id_map above already raises when subject is missing or empty.
            assert subject, "resolve_registry_id_map raises above when subject is missing or empty"
            kwargs["id_map"] = id_map
            kwargs["num_classes"] = len(id_map)
            _autoresolve_json_labels(kwargs, subject=subject, attribute=attribute, id_map=id_map,
                                     date=kwargs.get("date"))
        elif has_coco and kwargs.get("num_classes") is None:
            coco = kwargs.get("coco_data")
            if isinstance(coco, dict):
                kwargs["num_classes"] = len(coco.get("categories") or []) or 1

    if tiling and tiling.get("enabled", True) and task == "detection":
        transforms = kwargs.pop("transforms", None)
        base = cls(**kwargs)
        # Before constructing the tiler: its __init__ indexes every image, and that pass must
        # measure frames at the band count the tiles will be decoded at. Stamping only the wrapper
        # afterwards left the index built at 3 channels and the tiles read at N.
        base.expected_channels = num_channels
        ds = TiledDetectionDataset(base, transforms=transforms, **tile_kwargs_from_tiling(tiling))
    else:
        if tiling and tiling.get("enabled", True) and task != "detection":
            logger.warning("tiling is only supported for task='detection'; ignoring for task=%r", task)
        ds = cls(**kwargs)

    ds.expected_channels = num_channels
    return ds
