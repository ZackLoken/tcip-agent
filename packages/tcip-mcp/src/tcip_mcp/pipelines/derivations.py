"""Tier-A derivations: compute a parameter (channels, num_classes, anchor ratios) from the artifact
in hand.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from tcip_mcp.pipelines.data.band_groups import BandGroupRef
    from tcip_mcp.pipelines.raster_source import WindowSampling


def probe_channels(image_path: "str | Path | BandGroupRef") -> int:
    """Band count of a raster read from disk, as its own header states it
    (:attr:`~tcip_mcp.pipelines.raster_source.SourceHeader.channels`): a grouped capture's
    one band per member."""
    from tcip_mcp.pipelines.raster_source import SourceHeader

    return SourceHeader(image_path).channels


def num_classes_from_distribution(class_distribution: dict[int, int]) -> int:
    """Number of classes implied by the label set = max class id + 1 (ids are 0-indexed)."""
    if not class_distribution:
        return 0
    return int(max(class_distribution)) + 1


def gt_aspect_ratios(class_distribution_boxes: list[tuple[float, float]],
                     quantiles: tuple[float, float] = (0.1, 0.9)) -> list[float] | None:
    """Aspect ratios (h/w) spanning the GT box-shape distribution, for anchor coverage.

    Returns a small ratio set covering the p10..p90 of GT box aspect ratios (plus 1.0).
    ``class_distribution_boxes`` is a list of ``(w, h)`` in pixels. Returns ``None`` when no valid
    box gives a ratio.

    Wiring (derive, then pass to the builder)::

        from tcip_mcp.pipelines.derivations import gt_aspect_ratios
        from tcip_mcp.pipelines.components.detectors import build_detector
        ratios = gt_aspect_ratios([(b.w, b.h) for b in gt_boxes])   # this dataset's shapes
        if ratios is None:
            ratios = [0.5, 1.0, 2.0]                                 # underivable, yours to stamp
        names = list(adapter(torch.zeros(1, in_chans, h, w)).keys())  # this adapter's actual keys
        model = build_detector("faster_rcnn", adapter, num_classes=n,
                               featmap_names=names, num_levels=len(names),
                               aspect_ratios=tuple(ratios))
    """
    import numpy as np
    ratios = [h / w for (w, h) in class_distribution_boxes if w > 0 and h > 0]
    if not ratios:
        # Underivable: a pinned (0.5, 1, 2) returned as if derived would be indistinguishable from
        # a real result, so the caller states its own.
        return None
    lo, hi = np.quantile(ratios, quantiles[0]), np.quantile(ratios, quantiles[1])
    out = sorted({round(float(lo), 2), 1.0, round(float(hi), 2)})
    return [r for r in out if r > 0]


def _validate_gt_boxes_per_image(
    gt_boxes_per_image: "Sequence[Sequence[Sequence[float]]]", *, fn_name: str,
) -> list[list[tuple[float, float, float, float]]]:
    """Validate and normalize ``gt_boxes_per_image`` into concrete ``(x, y, w, h)`` float tuples,
    or raise ``ValueError`` naming exactly what was wrong.
    """
    if not isinstance(gt_boxes_per_image, Sequence) or isinstance(gt_boxes_per_image, (str, bytes)):
        raise ValueError(
            f"{fn_name}: gt_boxes_per_image must be a sequence of per-image box lists, got "
            f"{type(gt_boxes_per_image).__name__}"
        )
    validated: list[list[tuple[float, float, float, float]]] = []
    for i, boxes in enumerate(gt_boxes_per_image):
        if not isinstance(boxes, Sequence) or isinstance(boxes, (str, bytes)):
            raise ValueError(
                f"{fn_name}: gt_boxes_per_image[{i}] must be a sequence of boxes, got "
                f"{type(boxes).__name__}"
            )
        image_boxes: list[tuple[float, float, float, float]] = []
        for j, box in enumerate(boxes):
            if not isinstance(box, Sequence) or isinstance(box, (str, bytes)) or len(box) != 4:
                raise ValueError(
                    f"{fn_name}: gt_boxes_per_image[{i}][{j}] must be a 4-element (x, y, w, h) "
                    f"box, got {box!r}"
                )
            try:
                x, y, w, h = (float(v) for v in box)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"{fn_name}: gt_boxes_per_image[{i}][{j}] has a non-numeric coordinate "
                    f"({box!r}): {e}"
                ) from e
            image_boxes.append((x, y, w, h))
        validated.append(image_boxes)
    return validated


def _validate_char_sizes(char_sizes: Sequence[float], *, fn_name: str) -> list[float]:
    """Validate and normalize ``char_sizes`` into concrete floats, or raise ``ValueError`` naming
    exactly what was wrong, the same contract as ``_validate_gt_boxes_per_image``."""
    if not isinstance(char_sizes, Sequence) or isinstance(char_sizes, (str, bytes)):
        raise ValueError(
            f"{fn_name}: char_sizes must be a sequence of numbers, got {type(char_sizes).__name__}"
        )
    validated: list[float] = []
    for i, s in enumerate(char_sizes):
        try:
            validated.append(float(s))
        except (TypeError, ValueError) as e:
            raise ValueError(f"{fn_name}: char_sizes[{i}] is not numeric ({s!r}): {e}") from e
    return validated


def _neighbor_max_ious(boxes: np.ndarray) -> list[float]:
    """Each box's max IoU with any other box in the same region (xyxy px); fewer than 2 boxes ->
    []."""
    import numpy as np
    from tcip_annotation.matching import iou_matrix

    if len(boxes) < 2:
        return []
    overlap = iou_matrix(boxes, boxes, over="union")
    np.fill_diagonal(overlap, 0.0)  # exclude a box's overlap with itself (1.0)
    return overlap.max(axis=1).tolist()


def _neighbor_min_center_distances(boxes: np.ndarray) -> list[float]:
    """Each box's distance to its nearest same-image neighbor's center (xyxy px); fewer than 2
    boxes -> []."""
    import numpy as np
    from tcip_annotation.matching import box_centers

    if len(boxes) < 2:
        return []
    centers = box_centers(boxes)
    dist = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
    np.fill_diagonal(dist, np.inf)
    return dist.min(axis=1).tolist()


def derive_localization_tolerance_frac(
    gt_boxes_per_image: Sequence[Sequence[Sequence[float]]], *,
    percentile: float = 10.0, margin_frac: float = 0.5,
) -> float | None:
    """Center-match tolerance, as a fraction of the class's characteristic size, from the GT's own
    nearest-neighbor spacing, or ``None`` if underivable.

    Takes each GT box's distance to its nearest same-image neighbor, pools that across images,
    takes a low percentile with a safety margin, and normalizes by the characteristic size
    ``gt_class_avg_size`` measures. No image anywhere has two or more of this class -> ``None``.
    Refuses (``ValueError``) when the selected spacing percentile is zero, as many boxes as that
    percentile reaches sharing a same-class center, where the model states no tolerance a match
    could fall within; a coincident pair below that share leaves the percentile positive.

    ``gt_boxes_per_image`` is one list of ``[x, y, w, h]`` boxes (COCO xywh, px) per image, already
    filtered to the trait's own class.
    """
    import numpy as np
    from tcip_annotation.matching import box_sizes, xywh_corners

    gt_boxes_per_image = _validate_gt_boxes_per_image(
        gt_boxes_per_image, fn_name="derive_localization_tolerance_frac")
    dists: list[float] = []
    sizes: list[float] = []
    for boxes in gt_boxes_per_image:
        corners = xywh_corners(boxes)
        dists.extend(_neighbor_min_center_distances(corners))
        sizes.extend(box_sizes(corners).tolist())
    if not dists or not sizes:
        return None
    avg_size = float(np.mean(sizes))
    if avg_size <= 0:
        return None
    spacing = float(np.percentile(dists, percentile))
    if spacing <= 0:
        raise ValueError(
            f"the reference's p{percentile:g} nearest same-class center spacing is {spacing:g} px: "
            "that percentile of its boxes share a center, so the spacing model derives no "
            "center-match tolerance")
    return spacing * margin_frac / avg_size


@dataclass(frozen=True)
class Region:
    """One counted region of ground truth: the objects in it (crowd regions excluded), ``boxes``
    in xyxy pixels beside their 1-indexed ``labels``, and its pixel ``area``."""

    boxes: np.ndarray
    labels: np.ndarray
    area: float


def char_sizes(boxes: Iterable[np.ndarray]) -> list[float]:
    """The positive characteristic sizes (:func:`~tcip_annotation.matching.box_sizes`) of every
    xyxy box of every array of ``boxes``."""
    from tcip_annotation.matching import box_sizes

    return [s for arr in boxes for s in box_sizes(arr).tolist() if s > 0]


OBJECT_DENSITY_DERIVATION = ("p99 of the counted regions' objects per pixel; a frame keeps "
                             "ceil(density x its pixels) detections")
"""The label an execution record names a density :func:`derive_object_density` derived by."""


def derive_object_density(regions: Sequence[Region]) -> float:
    """The object density (objects per pixel) a frame's detection cap scales by
    (:func:`detection_cap`): the 0.99 quantile of the ``regions``' densities, each its object
    count over its area. The quantile is this method's own setting, not measured; a frame denser
    than it can still exceed its cap. Refuses (``ValueError``) no regions, and a quantile of zero,
    which no frame's cap could rest on."""
    import numpy as np

    if not regions:
        raise ValueError("an object density is derived from counted regions, and none were "
                         "counted.")
    density = float(np.quantile([len(r.boxes) / r.area for r in regions], 0.99))
    if density <= 0:
        raise ValueError(f"the 0.99 quantile of the densities of the {len(regions)} counted "
                         "regions is zero, so no frame's detection cap has a basis.")
    return density


def detection_cap(density: float, pixels: float) -> int:
    """The most detections a frame of ``pixels`` keeps at ``density``: ``ceil(density x
    pixels)``."""
    import math

    return math.ceil(density * pixels)


TILE_EDGE_DERIVATION = "5 x the p99 of the objects' larger box side, rounded up to a pixel"
"""The label an execution record names a tile edge :func:`derive_tile_geometry` derived by."""
TILE_OVERLAP_DERIVATION = ("the p99 of the objects' larger box side, rounded up to a pixel, as "
                           "the least overlap the lattice lays between neighboring tiles")
"""The label an execution record names a tile overlap :func:`derive_tile_geometry` derived
by."""


def derive_tile_geometry(regions: Sequence[Region], *, tile_size: int | None,
                         overlap: float | None) -> tuple[int, float]:
    """A tile lattice's ``(edge, overlap)``: a stated value as given; else, with the extent the
    99th percentile of the larger side of the ``regions``' boxes rounded up to a pixel, the edge
    five times the extent (:data:`TILE_EDGE_DERIVATION`) and the overlap the ratio under which
    :func:`~tcip_mcp.pipelines.slicing.slice_lattice` lays neighboring tiles overlapping by at
    least the extent (:data:`TILE_OVERLAP_DERIVATION`, ``slicing.overlap_ratio``), so an object no
    longer than the extent lies whole in some tile. The percentile and the five are this method's
    own settings, not measured. Refuses (``ValueError``) a value to derive from regions holding no
    object, and an extent at or past the edge, which no overlap fits."""
    import math

    import numpy as np

    from tcip_mcp.pipelines.slicing import overlap_ratio

    if tile_size is not None and overlap is not None:
        return int(tile_size), float(overlap)
    sides = [s for r in regions for s in np.maximum(r.boxes[:, 2] - r.boxes[:, 0],
                                                    r.boxes[:, 3] - r.boxes[:, 1]).tolist()
             if s > 0]
    if not sides:
        raise ValueError("the tile edge and overlap derive from the ground truth's object "
                         "extents, and it holds no object; state tile_size and overlap.")
    extent = math.ceil(float(np.percentile(sides, 99)))
    edge = int(tile_size) if tile_size is not None else 5 * extent
    if overlap is None:
        if extent >= edge:
            raise ValueError(f"the ground truth's p99 object side of {extent} px reaches the "
                             f"{edge} px tile edge, so no overlap holds it whole in a tile; "
                             "state a larger tile_size.")
        overlap = overlap_ratio(edge, extent)
    return edge, float(overlap)


IOU_MATCH_DERIVATION = (
    "IoU of two equal boxes of the GT's mean characteristic size sqrt(w*h) displaced along one "
    "axis by the trait's jitter, minus its absolute margin"
)
"""The label a criterion names a threshold :func:`derive_iou_match_threshold` derived by."""


def derive_iou_match_threshold(
    gt_boxes_per_image: Sequence[Sequence[Sequence[float]]], *, jitter_px: float, margin: float,
) -> float | None:
    """The IoU threshold for an ``iou_match`` trait, from the GT's own characteristic box size, or
    ``None`` if underivable.

    Models two equal boxes of the GT's mean characteristic size ``s`` (``sqrt(w*h)``) displaced
    by ``jitter_px`` along one axis, whose IoU is ``(s - jitter_px) / (s + jitter_px)``; the
    threshold is that less the absolute IoU ``margin``. Refuses (``ValueError``) a ``jitter_px``
    at or past ``s``, where the displaced boxes no longer overlap and the model states nothing,
    and a ``margin`` that leaves the threshold at or below zero.

    ``jitter_px``, the repeat-annotation displacement on the reference grid in pixels, and
    ``margin`` are the trait's own authored values (``TraitEntry.iou_jitter_px``,
    ``iou_margin``); nothing here stands in for them.

    ``gt_boxes_per_image`` is one list of ``[x, y, w, h]`` boxes (COCO xywh, px) per image, already
    filtered to the trait's own class.
    """
    gt_boxes_per_image = _validate_gt_boxes_per_image(
        gt_boxes_per_image, fn_name="derive_iou_match_threshold")
    from tcip_annotation.matching import xywh_corners

    sizes = char_sizes(xywh_corners(boxes) for boxes in gt_boxes_per_image)
    if not sizes:
        return None
    import numpy as np

    avg_size = float(np.mean(sizes))
    if jitter_px >= avg_size:
        raise ValueError(
            f"iou_jitter_px {jitter_px} is at or past the reference's mean characteristic box size "
            f"{avg_size:.4g} px: two boxes displaced that far do not overlap, so the IoU model "
            "derives no match threshold")
    modeled = (avg_size - jitter_px) / (avg_size + jitter_px)
    if modeled - margin <= 0:
        raise ValueError(
            f"iou_margin {margin} swallows the modeled IoU {modeled:.4g} of two boxes "
            f"{jitter_px} px apart: a threshold at or below zero matches every detection")
    return modeled - margin


def derive_sliver_frac(
    char_sizes: Sequence[float], *, percentile: float = 10.0, min_samples: int = 5,
) -> float | None:
    """Tile-seam sliver cutoff, as a fraction of the class's characteristic size, from the GT's own
    size spread, or ``None`` if underivable.

    Takes a low percentile of this dataset's own characteristic-size distribution relative to its
    mean, so a class with wide natural size variation gets a lower cutoff. The ratio is positive
    for any positive sizes and can exceed 1 when a few small boxes drag the mean below the low
    percentile; the consumer multiplies it back by that mean, so it stays a size cutoff either way.
    Fewer than ``min_samples`` positive sizes -> ``None``.

    ``char_sizes`` is ``sqrt(w*h)`` per GT box (px), already filtered to the trait's own class; see
    :func:`char_sizes`.
    """
    import numpy as np
    char_sizes = _validate_char_sizes(char_sizes, fn_name="derive_sliver_frac")
    sizes = [s for s in char_sizes if s > 0]
    if len(sizes) < min_samples:
        return None
    mean = float(np.mean(sizes))
    if mean <= 0:
        return None
    return float(np.percentile(sizes, percentile)) / mean


def derive_block_scale_px(
    *, tile_size: int, objects: Region, plants: "list | None" = None,
    raster_path: "str | Path | None" = None,
) -> tuple[int, str]:
    """The pixel buffer/block scale for block-aware calibration's recursive sub-banding, floored at
    ``tile_size`` (:func:`~tcip_mcp.pipelines.data.splits.spatial_strip_split`'s own floor for a
    boundary buffer).

    Two derivation paths:

    - Plant-spacing-derived, only attempted when ``plants`` is supplied: this dataset's own
      planting-grid pitch (``plant_mapping.grid_pitch_m``) converted to pixels through the raster's
      own pixel size in meters, resolved by
      :func:`~tcip_mcp.pipelines.pixel_size.resolve_pixel_size` (the CRS unit read from the EPSG
      code, so a foot-unit raster converts correctly). A raster whose georeferencing falls short
      (rotated, incomplete tags, unprojected, a user-defined or compound CRS, a unit other than
      meter or foot, a zero, negative or anisotropic scale) falls back to the GT-object-spacing
      path below. A ``raster_path`` that is not a raster file, or one this derivation cannot open
      (truncated, corrupt, or otherwise unreadable), is refused by name. A ``plants`` list with
      fewer than two georeferenced plants (``grid_pitch_m`` returns ``0.0``) is refused by name.
    - GT-object-spacing-derived (``plants`` omitted, or the raster's pixel size unresolvable): the
      median nearest-neighbor spacing of the centers of ``objects``' boxes.

    Raises ``ValueError`` naming exactly why when neither path can derive a scale (no ``plants``
    and fewer than two objects to measure a spacing from).
    """
    import statistics

    if plants:
        from tcip_mcp.pipelines.postprocessing.plant_mapping import grid_pitch_m

        pitch_m = grid_pitch_m(plants)
        if pitch_m <= 0.0:
            raise ValueError(
                "derive_block_scale_px: plant grid pitch is underivable (fewer than two plants "
                "in the supplied registry carry resolvable coordinates); pass plants=None to use "
                "the GT-object-spacing-derived block scale instead of an insufficient registry"
            )
        if raster_path is not None:
            from tcip_mcp.pipelines import pixel_size as pixel_size_module
            from tcip_mcp.pipelines.raster_source import SourceHeader

            raster = Path(raster_path)
            header = SourceHeader(raster)
            if not raster.is_file() or header.kind == "photo":
                raise ValueError(
                    "derive_block_scale_px: raster_path is not a raster file this derivation can "
                    f"read a pixel size from ({raster}); pass the training raster, or "
                    "raster_path=None to use the GT-object-spacing-derived block scale")
            if header.kind != "tif":
                raise ValueError(
                    f"derive_block_scale_px: raster_path ({raster}) is an array container, which "
                    "carries no georeferencing tags, so there is no pixel size to read; pass "
                    "raster_path=None to use the GT-object-spacing-derived block scale instead")
            try:
                header.georeference  # the one header read; an unreadable raster refuses here
            except ValueError as exc:
                raise ValueError(
                    f"derive_block_scale_px: raster_path could not be opened as a raster "
                    f"({raster}): {exc}"
                ) from exc
            resolved, _reason = pixel_size_module.resolve_pixel_size(header)
            if resolved is not None:
                pitch_px = pitch_m / resolved.meters_per_px
                return (
                    max(tile_size, round(pitch_px)),
                    f"plant grid pitch ({pitch_m:.2f}m) via {resolved.source_clause}, "
                    "floored at tile_size",
                )

    dists = _neighbor_min_center_distances(objects.boxes)
    if not dists:
        raise ValueError(
            "derive_block_scale_px: no block scale is derivable (no plant registry supplied, or "
            "its raster's georeferencing falls short, and the reserved region's own GT holds "
            "fewer than two objects to measure a spacing from)"
        )
    spacing_px = statistics.median(dists)
    return max(tile_size, round(spacing_px)), (
        "GT object-spacing (median nearest-neighbor), floored at tile_size"
    )


def band_normalization_stats(
    image_paths: "Sequence[str | Path | BandGroupRef]", num_channels: int, *, max_images: int = 50,
) -> tuple[list[float], list[float], list[str]] | None:
    """Per-band ``(mean, std, paths_read)`` in [0, 1] over at most ``max_images`` of
    ``image_paths`` read at ``num_channels`` bands, or ``None`` when no raster could be read.
    ``paths_read`` names the rasters decoded and accepted, a raster whose band count disagreed
    dropped; :func:`image_stats_provenance` renders it into ``model_source.image_stats_sampling``.
    """
    import numpy as np

    from tcip_mcp.pipelines.image_utils import load_image, pil_to_tensor, source_path_of

    moments = _BandMoments(num_channels)
    paths_read: list[str] = []
    for path in list(image_paths)[:max_images]:
        try:
            arr = pil_to_tensor(load_image(path, num_channels)).numpy().astype(np.float64)
        except Exception:  # noqa: BLE001, an unreadable raster is skipped, not fatal
            continue
        if moments.add(arr):
            paths_read.append(source_path_of(path))
    result = moments.result()
    if result is None:
        return None
    mean, std = result
    return mean, std, paths_read


class _BandMoments:
    """Pixel-weighted per-band first and second moments over [0, 1] tensor pixels."""

    def __init__(self, num_channels: int):
        import numpy as np

        self.num_channels = int(num_channels)
        self.sums = np.zeros(self.num_channels, dtype=np.float64)
        self.sqs = np.zeros(self.num_channels, dtype=np.float64)
        self.pixels = 0

    def add(self, tensor_pixels) -> bool:
        """Accumulate one ``[C, H, W]`` float64 block; returns whether it was accepted (its band
        count disagreeing with this accumulator's means the pixels are not the same bands)."""
        if tensor_pixels.shape[0] != self.num_channels:
            return False
        flat = tensor_pixels.reshape(self.num_channels, -1)
        self.sums += flat.sum(axis=1)
        self.sqs += (flat ** 2).sum(axis=1)
        self.pixels += flat.shape[1]
        return True

    def result(self) -> tuple[list[float], list[float]] | None:
        """Per-band ``(mean, std)``, or ``None`` when no pixels were accumulated at all."""
        import numpy as np

        if not self.pixels:
            return None
        mean = self.sums / self.pixels
        var = np.maximum(self.sqs / self.pixels - mean ** 2, 0.0)
        return [float(m) for m in mean], [float(s) for s in np.sqrt(var)]


@dataclass(frozen=True)
class SampledNormalizationStats:
    """Per-band ``(mean, std)`` in [0, 1] tensor units over a sample of a dataset's pixels.
    ``sampling`` names the windows read, the seed that chose them and the pixel fraction they
    cover.
    """

    mean: list[float]
    std: list[float]
    sampling: WindowSampling


def band_normalization_stats_sampled(
    image_paths: "Sequence[str | Path | BandGroupRef]", num_channels: int, *, seed: int,
    window_size: int, max_windows_per_image: int, max_images: int = 50,
) -> SampledNormalizationStats | None:
    """Per-band ``(mean, std)`` in [0, 1] over seeded pixel windows of this dataset's rasters.

    The windowed sibling of :func:`band_normalization_stats`: the same statistic in the same unit
    system, for a source whose full decode is unaffordable (an orthomosaic, say). Each raster is
    opened through ``raster_source`` and only ``max_windows_per_image`` windows of ``window_size``
    pixels are read from it, chosen by ``raster_source.sample_windows`` from ``seed``; a
    ``max_windows_per_image`` covering every cell of a raster's grid reads all of it and gives the
    exact sibling's own answer.

    Pixels are scaled by ``image_utils.pil_to_tensor``. ``seed`` is required. Returns ``None`` when
    no raster could be read.
    """
    import numpy as np

    from tcip_mcp.pipelines import raster_source
    from tcip_mcp.pipelines.image_utils import pil_to_tensor, source_path_of

    moments = _BandMoments(num_channels)
    read: list[tuple[str, raster_source.Rect]] = []
    covered = 0
    total = 0
    for path in list(image_paths)[:max_images]:
        try:
            with raster_source.open_raster(path, num_channels) as src:
                total += src.width * src.height
                label = source_path_of(path)
                for rect in raster_source.sample_windows(
                        src.width, src.height, seed=seed, window_size=window_size,
                        max_windows=max_windows_per_image):
                    moments.add(pil_to_tensor(src.read_region(rect)[0]).numpy().astype(np.float64))
                    read.append((label, rect))
                    covered += rect.width * rect.height
        except Exception:  # noqa: BLE001, an unreadable raster is skipped, not fatal
            continue
    stats = moments.result()
    if stats is None:
        return None
    mean, std = stats
    sampling = raster_source.WindowSampling(
        tuple(read), int(seed), covered / total if total else 0.0)
    return SampledNormalizationStats(mean, std, sampling)


def image_stats_provenance(
    result: "tuple[list[float], list[float], list[str]] | SampledNormalizationStats",
    *, window_size: int | None = None, max_windows_per_image: int | None = None,
) -> dict:
    """Render either band-normalization result into ``model_source.image_stats_sampling``.

    Accepts :func:`band_normalization_stats`'s ``(mean, std, paths_read)`` tuple or
    :func:`band_normalization_stats_sampled`'s :class:`SampledNormalizationStats`, and returns the
    mapping that rides beside ``builder_kwargs``. ``window_size``/``max_windows_per_image`` are the
    sampled call's own parameters, not carried on ``WindowSampling`` itself; they are ignored for
    the exact result.
    """
    if isinstance(result, SampledNormalizationStats):
        sampling = result.sampling
        return {
            "windows": [[label, {"x0": r.x0, "y0": r.y0, "x1": r.x1, "y1": r.y1}]
                        for label, r in sampling.windows],
            "seed": sampling.seed,
            "pixel_fraction": sampling.pixel_fraction,
            "window_size": window_size,
            "max_windows_per_image": max_windows_per_image,
        }
    _, _, paths_read = result
    return {
        "windows": [[path, None] for path in paths_read],
        "seed": None,
        "pixel_fraction": 1.0,
        "window_size": None,
        "max_windows_per_image": None,
    }


CROSS_TILE_NMS_DERIVATION = "GT neighbor-IoU distribution (p99 + margin)"
"""The ``derived_from`` label :func:`derive_cross_tile_nms`'s value is stamped under. A merge
comparing detections by IoS has no derivation: its threshold is the project's to state."""


def derive_cross_tile_nms(regions: Sequence[Region], *,
                          percentile: float = 99.0, margin: float = 0.05) -> float | None:
    """Cross-tile merge threshold for a merge comparing detections by IoU, from the GT
    neighbor-IoU distribution, or None if underivable.

    The merge joins two detections whose overlap exceeds this threshold, so it sits just above how
    much real neighboring GT objects overlap: per region each object's max overlap with any other
    object, the nonzero tail pooled across ``regions``, its ``percentile`` plus ``margin``. No two
    objects of one region overlapping returns None. Refuses (``ValueError``) a result at or above
    1: the ``margin`` added to the neighbors' own tail exhausts the IoU interval, so this method
    states no threshold for this reference, though a smaller margin or a stated value could.

    ``percentile`` (99) takes the overlap nearly every real neighboring pair stays under, and
    ``margin`` (0.05) the step above it a seam duplicate must clear; both are this method's own
    chosen settings, not measured.
    """
    import numpy as np

    tail: list[float] = []
    for region in regions:
        tail.extend(v for v in _neighbor_max_ious(region.boxes) if v > 0.0)
    if not tail:
        return None
    threshold = float(np.percentile(tail, percentile)) + margin
    if threshold >= 1.0:
        raise ValueError(
            f"the reference's neighboring boxes overlap up to IoU {threshold - margin:.3g} at "
            f"p{percentile:g}, and the margin {margin:g} carries the threshold to "
            f"{threshold:.3g}, past the IoU interval; state cross_tile_nms")
    return threshold


_STATIC_DERIVATION_IMPLEMENTATIONS: dict[str, object] = {
    "probed bands of": "tcip_mcp.pipelines.derivations.probe_channels",  # f-string prefix
    "max class id + 1 in the label set": (
        "tcip_mcp.pipelines.derivations.num_classes_from_distribution"
    ),
    CROSS_TILE_NMS_DERIVATION: "tcip_mcp.pipelines.derivations.derive_cross_tile_nms",
    "GT nearest-neighbor spacing (p10 + margin)": (
        "tcip_mcp.pipelines.derivations.derive_localization_tolerance_frac"
    ),
    "GT characteristic-size spread (p10 / mean)": (
        "tcip_mcp.pipelines.derivations.derive_sliver_frac"
    ),
    IOU_MATCH_DERIVATION: "tcip_mcp.pipelines.derivations.derive_iou_match_threshold",
    OBJECT_DENSITY_DERIVATION: "tcip_mcp.pipelines.derivations.derive_object_density",
    TILE_EDGE_DERIVATION: "tcip_mcp.pipelines.derivations.derive_tile_geometry",
    TILE_OVERLAP_DERIVATION: "tcip_mcp.pipelines.derivations.derive_tile_geometry",
}
"""Each derivation label authored here, every one but the count-objective ones, mapped to the
callable that computes it."""

_CURVE_IMPLEMENTATION = "tcip_mcp.pipelines.training.evaluation.derive_operating_point_curve"


def _derivation_implementations() -> dict[str, object]:
    """Every registered derivation label mapped to its implementation, the count-objective ones
    read from the picker registry (``operating_point.COUNT_OBJECTIVE_PICKERS``) when called."""
    from tcip_mcp.pipelines.operating_point import COUNT_OBJECTIVE_PICKERS

    return {
        **{label: _CURVE_IMPLEMENTATION for _picker, label in COUNT_OBJECTIVE_PICKERS.values()},
        **_STATIC_DERIVATION_IMPLEMENTATIONS,
    }


def __getattr__(name: str) -> dict[str, object]:
    """Serve ``DERIVATION_IMPLEMENTATIONS`` on access, so its picker half is read live."""
    if name == "DERIVATION_IMPLEMENTATIONS":
        return _derivation_implementations()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
