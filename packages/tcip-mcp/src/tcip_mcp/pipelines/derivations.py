"""Tier-A derivations: compute a parameter (channels, num_classes, anchor ratios) from the artifact
in hand.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tcip_mcp.pipelines.data.band_groups import BandGroupRef
    from tcip_mcp.pipelines.raster_source import WindowSampling


def probe_channels(image_path: "str | Path | BandGroupRef") -> int:
    """Band count of a raster read from disk.

    A :class:`~tcip_mcp.pipelines.data.band_groups.BandGroupRef` (sibling single-band files grouped
    as one logical image) probes each sibling on its own and sums them.
    """
    from tcip_mcp.pipelines.data.band_groups import BandGroupRef
    from tcip_mcp.pipelines.image_utils import _channels_from_shape
    from tcip_mcp.pipelines.raster_source import tiff_series_shape

    if isinstance(image_path, BandGroupRef):
        return sum(probe_channels(p) for p in image_path.bands.values())
    path = Path(image_path)
    ext = path.suffix.lower()
    if ext == ".npy":
        import numpy as np
        arr = np.load(str(path))
        return int(arr.shape[-1]) if arr.ndim == 3 else 1
    if ext == ".npz":
        import numpy as np
        with np.load(str(path)) as z:
            arr = z[list(z.files)[0]]
        return int(arr.shape[-1]) if arr.ndim == 3 else 1
    if ext in (".tif", ".tiff"):
        shape = tiff_series_shape(path)  # header-only; no pixel decode when this succeeds
        if shape is not None:
            return _channels_from_shape(shape)
        import numpy as np
        import tifffile
        arr = np.asarray(tifffile.imread(str(path)))
        return _channels_from_shape(arr.shape)
    from PIL import Image
    with Image.open(path) as im:
        return len(im.getbands())  # RGB->3, L->1, RGBA->4, already header-only (PIL is lazy)


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
        # Underivable: no valid boxes. Return None rather than the fixed (0.5, 1, 2) this function
        # exists to replace, a pinned default returned as if derived is indistinguishable from a
        # real result. Same contract as derive_cross_tile_nms: the caller stamps an honest default.
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


def _neighbor_max_ious(boxes: Sequence[Sequence[float]]) -> list[float]:
    """Each box's max IoU with any other box in the same image (xywh px); fewer than 2 boxes -> []."""
    import numpy as np
    if len(boxes) < 2:
        return []
    b = np.asarray(boxes, dtype=float)
    x1, y1, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    x2, y2 = x1 + w, y1 + h
    area = w * h
    ix1 = np.maximum(x1[:, None], x1[None, :])
    iy1 = np.maximum(y1[:, None], y1[None, :])
    ix2 = np.minimum(x2[:, None], x2[None, :])
    iy2 = np.minimum(y2[:, None], y2[None, :])
    inter = np.clip(ix2 - ix1, 0.0, None) * np.clip(iy2 - iy1, 0.0, None)
    union = area[:, None] + area[None, :] - inter
    iou = np.where(union > 0, inter / union, 0.0)
    np.fill_diagonal(iou, 0.0)  # exclude a box's self-IoU (1.0)
    return iou.max(axis=1).tolist()


def _neighbor_min_center_distances(boxes: Sequence[Sequence[float]]) -> list[float]:
    """Each box's distance to its nearest same-image neighbor's center (xywh px); fewer than 2 boxes -> []."""
    import numpy as np
    if len(boxes) < 2:
        return []
    b = np.asarray(boxes, dtype=float)
    cx, cy = b[:, 0] + b[:, 2] / 2.0, b[:, 1] + b[:, 3] / 2.0
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dist = np.sqrt(dx ** 2 + dy ** 2)
    np.fill_diagonal(dist, np.inf)
    return dist.min(axis=1).tolist()


def derive_localization_tolerance_frac(
    gt_boxes_per_image: Sequence[Sequence[Sequence[float]]], *,
    percentile: float = 10.0, margin_frac: float = 0.5,
    clamp: tuple[float, float] = (0.1, 0.75),
) -> float | None:
    """Center-match tolerance, as a fraction of the class's characteristic size, from the GT's own
    nearest-neighbor spacing, or ``None`` if underivable.

    Takes each GT box's distance to its nearest same-image neighbor, pools that across images,
    takes a low percentile with a safety margin, and normalizes by the characteristic size
    ``gt_class_avg_size`` measures. No image anywhere has two or more of this class -> ``None``.

    ``gt_boxes_per_image`` is one list of ``[x, y, w, h]`` boxes (COCO xywh, px) per image, already
    filtered to the trait's own class.
    """
    import numpy as np
    gt_boxes_per_image = _validate_gt_boxes_per_image(
        gt_boxes_per_image, fn_name="derive_localization_tolerance_frac")
    dists: list[float] = []
    sizes: list[float] = []
    for boxes in gt_boxes_per_image:
        dists.extend(_neighbor_min_center_distances(boxes))
        sizes.extend((max(w, 0.0) * max(h, 0.0)) ** 0.5 for _, _, w, h in boxes)
    if not dists or not sizes:
        return None
    avg_size = float(np.mean(sizes))
    if avg_size <= 0:
        return None
    lo, hi = clamp
    raw = float(np.percentile(dists, percentile)) * margin_frac / avg_size
    return float(min(max(raw, lo), hi))


def char_sizes_from_boxes(gt_boxes_per_image: Sequence[Sequence[Sequence[float]]]) -> list[float]:
    """``sqrt(w*h)`` per positive GT box across every image; boxes are ``(x, y, w, h)``."""
    sizes = [
        (max(w, 0.0) * max(h, 0.0)) ** 0.5
        for boxes in gt_boxes_per_image for _, _, w, h in boxes
    ]
    return [s for s in sizes if s > 0]


def _achievable_iou(avg_size: float, jitter_px: float) -> float:
    """Best-case IoU between two same-size boxes of characteristic size ``avg_size``, offset by
    ``jitter_px`` along one axis.
    """
    return max(0.0, avg_size - jitter_px) / (avg_size + jitter_px)


def derive_localization_kind(
    gt_boxes_per_image: Sequence[Sequence[Sequence[float]]], *,
    jitter_px: float = 15.0, iou_floor: float = 0.5,
) -> str | None:
    """Whether IoU-matching or center-matching should govern this trait's "found the object" call,
    from the GT's own characteristic box size, or ``None`` if underivable.

    Models two same-size boxes of characteristic size ``s`` (``sqrt(w*h)``) offset by ``jitter_px``
    along one axis: their achievable IoU is ``(s - jitter_px) / (s + jitter_px)``. When that falls
    below ``iou_floor`` (0.5), center-match governs. No valid boxes anywhere -> ``None``.

    ``jitter_px``'s default (15.0 px) is provisional, not validated against detector precision; it
    puts the crossover at ``s = 3 * jitter_px``. A derived kind is recorded with
    ``data_derived_at_runtime`` provenance and re-checked for divergence on later calls
    (``resolve_match_criterion``).

    ``gt_boxes_per_image`` is one list of ``[x, y, w, h]`` boxes (COCO xywh, px) per image, already
    filtered to the trait's own class.
    """
    gt_boxes_per_image = _validate_gt_boxes_per_image(
        gt_boxes_per_image, fn_name="derive_localization_kind")
    sizes = char_sizes_from_boxes(gt_boxes_per_image)
    if not sizes:
        return None
    import numpy as np

    from tcip_mcp.traits import CENTER_MATCH, IOU_MATCH

    avg_size = float(np.mean(sizes))
    return CENTER_MATCH if _achievable_iou(avg_size, jitter_px) < iou_floor else IOU_MATCH


def derive_iou_match_threshold(
    gt_boxes_per_image: Sequence[Sequence[Sequence[float]]], *,
    jitter_px: float = 15.0, margin: float = 0.1, clamp: tuple[float, float] = (0.3, 0.7),
) -> float | None:
    """The IoU threshold for an ``iou_match`` trait, from the GT's own characteristic box size, or
    ``None`` if underivable.

    Uses the achievable-IoU-under-jitter basis :func:`derive_localization_kind` uses, less
    ``margin``, clamped to a sane range around IoU@0.5. A recorded ``iou_match`` trait revisited
    with small current-call GT can see an achievable IoU below the floor; the clamp bounds the
    result then.

    ``jitter_px`` (15.0 px) and ``margin`` (0.1) are provisional defaults, not validated against
    detector precision.

    ``gt_boxes_per_image`` is one list of ``[x, y, w, h]`` boxes (COCO xywh, px) per image, already
    filtered to the trait's own class.
    """
    gt_boxes_per_image = _validate_gt_boxes_per_image(
        gt_boxes_per_image, fn_name="derive_iou_match_threshold")
    sizes = char_sizes_from_boxes(gt_boxes_per_image)
    if not sizes:
        return None
    import numpy as np

    avg_size = float(np.mean(sizes))
    achievable = _achievable_iou(avg_size, jitter_px)
    lo, hi = clamp
    return float(min(max(achievable - margin, lo), hi))


def derive_sliver_frac(
    char_sizes: Sequence[float], *, percentile: float = 10.0,
    clamp: tuple[float, float] = (0.25, 0.9), min_samples: int = 5,
) -> float | None:
    """Tile-seam sliver cutoff, as a fraction of the class's characteristic size, from the GT's own
    size spread, or ``None`` if underivable.

    Takes a low percentile of this dataset's own characteristic-size distribution relative to its
    mean, clamped to a sane range, so a class with wide natural size variation gets a lower cutoff.
    Fewer than ``min_samples`` boxes -> ``None``.

    ``char_sizes`` is ``sqrt(w*h)`` per GT box (px), already filtered to the trait's own class; see
    :func:`char_sizes_from_boxes`.
    """
    import numpy as np
    char_sizes = _validate_char_sizes(char_sizes, fn_name="derive_sliver_frac")
    sizes = [s for s in char_sizes if s > 0]
    if len(sizes) < min_samples:
        return None
    mean = float(np.mean(sizes))
    if mean <= 0:
        return None
    lo, hi = clamp
    raw = float(np.percentile(sizes, percentile)) / mean
    return float(min(max(raw, lo), hi))


def derive_block_scale_px(
    *, tile_size: int, gt_boxes_per_image: Sequence[Sequence[Sequence[float]]],
    plants: "list | None" = None, raster_path: "str | Path | None" = None,
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
      median nearest-neighbor spacing of ``gt_boxes_per_image``'s own box centers.

    Raises ``ValueError`` naming exactly why when neither path can derive a scale (no ``plants``
    and no image with two or more GT boxes to measure a spacing from).
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
            from tcip_mcp.pipelines.image_utils import capture_kind
            from tcip_mcp.pipelines.postprocessing.orthomosaic_mapping import (
                GeoreferencingError,
                RotatedRasterError,
                read_geotransform,
            )
            from tcip_mcp.pipelines.raster_source import UNGEOREFERENCED_ARRAY_EXTS

            raster = Path(raster_path)
            if not raster.is_file() or capture_kind(raster) != "raster":
                raise ValueError(
                    "derive_block_scale_px: raster_path is not a raster file this derivation can "
                    f"read a pixel size from ({raster}); pass the training raster, or "
                    "raster_path=None to use the GT-object-spacing-derived block scale")
            if raster.suffix.casefold() in UNGEOREFERENCED_ARRAY_EXTS:
                raise ValueError(
                    f"derive_block_scale_px: raster_path ({raster}) is an array container, which "
                    "carries no georeferencing tags, so there is no pixel size to read; pass "
                    "raster_path=None to use the GT-object-spacing-derived block scale instead")
            try:
                read_geotransform(raster)
            except (GeoreferencingError, RotatedRasterError):
                pass
            except (ValueError, OSError) as exc:
                raise ValueError(
                    f"derive_block_scale_px: raster_path could not be opened as a raster "
                    f"({raster}): {exc}"
                ) from exc
            else:
                resolved, _reason = pixel_size_module.resolve_pixel_size(raster)
                if resolved is not None:
                    pitch_px = pitch_m / resolved.meters_per_px
                    return (
                        max(tile_size, round(pitch_px)),
                        f"plant grid pitch ({pitch_m:.2f}m) via {resolved.source_clause}, "
                        "floored at tile_size",
                    )

    gt_boxes_per_image = _validate_gt_boxes_per_image(
        gt_boxes_per_image, fn_name="derive_block_scale_px")
    dists: list[float] = []
    for boxes in gt_boxes_per_image:
        dists.extend(_neighbor_min_center_distances(boxes))
    if not dists:
        raise ValueError(
            "derive_block_scale_px: no block scale is derivable (no plant registry supplied, or "
            "its raster's georeferencing falls short, and the reserved region's own GT has no "
            "image with two or more objects to measure a spacing from)"
        )
    spacing_px = statistics.median(dists)
    return max(tile_size, round(spacing_px)), "GT object-spacing (median nearest-neighbor), floored at tile_size"


def _image_stats_label(path) -> str:
    """The path string both normalization-statistics derivations record for one raster: its own
    path, or a band group's manifest path when the source is a :class:`BandGroupRef`.
    """
    return str(getattr(path, "manifest_path", path))


def band_normalization_stats(
    image_paths: "Sequence[str | Path | BandGroupRef]", num_channels: int, *, max_images: int = 50,
) -> tuple[list[float], list[float], list[str]] | None:
    """Per-band ``(mean, std, paths_read)`` in [0, 1] over this dataset's rasters, or ``None`` when
    no raster could be read.

    The statistics a detector normalizes with. torchvision defaults to 3-element ImageNet values,
    which are wrong on any band set that is not RGB photography: at 1 channel they silently
    broadcast the image to 3, and at any count other than 3 they raise inside the transform. Sample
    the training split and pass ``mean``/``std`` to ``build_detector`` as
    ``image_mean``/``image_std`` through ``model_source.builder_kwargs``: torchvision keeps them as
    plain lists on the transform, not as buffers, so they are absent from the checkpoint.
    ``paths_read`` (the paths this call decoded and accepted, after the ``max_images`` cap and
    after dropping any raster whose band count disagreed) is what :func:`image_stats_provenance`
    renders into ``model_source.image_stats_sampling``.

    ``max_images`` caps the sample.
    """
    import numpy as np

    from tcip_mcp.pipelines.image_utils import load_image, pil_to_tensor

    moments = _BandMoments(num_channels)
    paths_read: list[str] = []
    for path in list(image_paths)[:max_images]:
        try:
            arr = pil_to_tensor(load_image(path, num_channels)).numpy().astype(np.float64)
        except Exception:  # noqa: BLE001, an unreadable raster is skipped, not fatal
            continue
        if moments.add(arr):
            paths_read.append(_image_stats_label(path))
    result = moments.result()
    if result is None:
        return None
    mean, std = result
    return mean, std, paths_read


class _BandMoments:
    """Pixel-weighted per-band first and second moments over [0, 1] tensor pixels, for
    :func:`band_normalization_stats` and :func:`band_normalization_stats_sampled`.
    """

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
    from tcip_mcp.pipelines.image_utils import pil_to_tensor

    moments = _BandMoments(num_channels)
    read: list[tuple[str, raster_source.Rect]] = []
    covered = 0
    total = 0
    for path in list(image_paths)[:max_images]:
        try:
            with raster_source.open_raster(path, num_channels) as src:
                total += src.width * src.height
                label = _image_stats_label(path)
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


def derive_cross_tile_nms(gt_boxes_per_image: Sequence[Sequence[Sequence[float]]], *,
                          percentile: float = 99.0, margin: float = 0.05,
                          clamp: tuple[float, float] = (0.2, 0.8)) -> float | None:
    """Cross-tile NMS IoU threshold from the GT neighbor-overlap distribution, or None if underivable.

    Cross-tile NMS drops one of two boxes when their IoU exceeds this threshold; its job is to suppress
    duplicate detections of the same object split across a tile seam without merging two genuinely
    distinct objects that happen to overlap. So the threshold is set just above how much *real*
    neighboring GT objects overlap: per image take each GT box's max IoU with any other box, pool the
    nonzero tail across images, and use a high percentile (dense clusters overlap more, pushing the
    threshold up) plus a small margin, clamped to a sane range. No genuine overlaps anywhere -> return
    None (underivable; the caller stamps an honest default, never a derivation label on that number).

    ``gt_boxes_per_image`` is one list of ``[x, y, w, h]`` boxes (COCO xywh, px) per image.
    """
    import numpy as np
    gt_boxes_per_image = _validate_gt_boxes_per_image(
        gt_boxes_per_image, fn_name="derive_cross_tile_nms")
    tail: list[float] = []
    for boxes in gt_boxes_per_image:
        tail.extend(v for v in _neighbor_max_ious(boxes) if v > 0.0)
    if not tail:
        return None
    lo, hi = clamp
    return float(min(max(float(np.percentile(tail, percentile)) + margin, lo), hi))


_STATIC_DERIVATION_IMPLEMENTATIONS: dict[str, object] = {
    "probed bands of": "tcip_mcp.pipelines.derivations.probe_channels",  # f-string prefix
    "max class id + 1 in the label set": "tcip_mcp.pipelines.derivations.num_classes_from_distribution",
    "GT neighbor-IoU distribution (p99 + margin)": "tcip_mcp.pipelines.derivations.derive_cross_tile_nms",
    "GT nearest-neighbor spacing (p10 + margin)": "tcip_mcp.pipelines.derivations.derive_localization_tolerance_frac",
    "GT characteristic-size spread (p10 / mean)": "tcip_mcp.pipelines.derivations.derive_sliver_frac",
    "achievable IoU under annotation jitter (GT characteristic size)":
        "tcip_mcp.pipelines.derivations.derive_localization_kind",
    "achievable IoU under annotation jitter, minus margin (GT characteristic size)":
        "tcip_mcp.pipelines.derivations.derive_iou_match_threshold",
    "~1.5x p99 GT objects/image": "tcip_mcp.pipelines.operating_point._max_dets_from_density",
    "model imgsz / persisted training geometry": "tcip_mcp.pipelines.resolution.raw_operating_point",
    "persisted training tile geometry": "tcip_mcp.pipelines.resolution.raw_operating_point",
    "the checkpoint's own uniform untiled training frame":
        "tcip_mcp.pipelines.inference.predictor._native_ratio_tile_size",
    "read back from the bucket's own stamp with no accepted geometry reference behind it":
        "placeholder",
    "caller override": "caller-input",
    "no GT for this dataset; unvalidated placeholder": "placeholder",
}
"""The derivation labels that are authored here, everything except the count-objective ones.

Every ``derived_from`` label ``resolution.derived()`` can stamp resolves through
:data:`DERIVATION_IMPLEMENTATIONS` to the callable that computes it, or to an explicit
non-derivation marker ("caller-input" / "placeholder") when the constructor carries a value nothing
derived. tests/test_provenance_honesty.py enforces that, so a data-sounding label cannot be stamped
without an implementation behind it.
"""

_CURVE_IMPLEMENTATION = "tcip_mcp.pipelines.operating_point.derive_operating_point_curve"


def _derivation_implementations() -> dict[str, object]:
    """Every registered derivation label, the count-objective ones read from the picker registry
    (``operating_point.COUNT_OBJECTIVE_PICKERS`` plus
    ``operating_point.REVIEW_VERDICT_LABEL_SUFFIX``).

    Resolved on access rather than at import: ``operating_point`` imports this module and pulls
    torch in with it.
    """
    from tcip_mcp.pipelines.operating_point import (
        COUNT_OBJECTIVE_PICKERS,
        REVIEW_VERDICT_LABEL_SUFFIX,
    )

    return {
        **{
            label + suffix: _CURVE_IMPLEMENTATION
            for _picker, label in COUNT_OBJECTIVE_PICKERS.values()
            for suffix in ("", REVIEW_VERDICT_LABEL_SUFFIX)
        },
        **_STATIC_DERIVATION_IMPLEMENTATIONS,
    }


def __getattr__(name: str) -> dict[str, object]:
    """Serve ``DERIVATION_IMPLEMENTATIONS`` on access, so its picker half is read live."""
    if name == "DERIVATION_IMPLEMENTATIONS":
        return _derivation_implementations()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
