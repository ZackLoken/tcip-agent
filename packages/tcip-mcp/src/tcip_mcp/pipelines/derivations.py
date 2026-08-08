"""Tier-A data/model derivations, read the artifact in hand, compute the value.

These are the "deterministic" and "distribution" derivations (CLAUDE.md "Parameters: derive, don't
pin"): channels from *this* raster, num_classes from *this* label set, anchor aspect ratios from
*this* dataset's GT box shapes. The agent's model builder / train(ctx) calls these to size a bespoke
model to the data in hand, never a value frozen from a different dataset.

Heavy deps (PIL/numpy/tifffile) are imported lazily so this stays cheap to import.
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
    """Band count of a raster read from disk, the artifact in hand, not a sensor-name guess.

    A :class:`~tcip_mcp.pipelines.data.band_groups.BandGroupRef` (sibling single-band files
    grouped as one logical image) probes each sibling on its own and sums them, usually 1 each,
    never assumed, since a group's members are independent files with no shared header to trust.
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

    Returns a small ratio set covering the p10..p90 of GT box aspect ratios (plus 1.0), so anchors
    match the actual object shapes in this dataset rather than a fixed (0.5, 1, 2).
    ``class_distribution_boxes`` is a list of ``(w, h)`` in pixels. Returns ``None`` when no valid
    box gives a ratio, underivable, so the caller stamps an honest default rather than receiving
    this function's own pinned fallback dressed as a derivation.

    Wiring (derive -> pass; the agent's builder path, never auto-injected into build_detector so a
    method isn't re-pinned)::

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
    """Validate and normalize ``gt_boxes_per_image`` into concrete ``(x, y, w, h)`` float tuples, or
    raise ``ValueError`` naming exactly what was wrong.

    Every ``derive_*`` function that consumes this shape calls this once, so a malformed call fails
    the same way everywhere in this module, one exception type with a real message, rather than
    surfacing whatever bare exception the first downstream operation (an unpack, a comparison, a
    numpy cast) happened to raise first.
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

    A tolerance that reaches into a neighboring object's territory starts double-matching two
    distinct nearby objects to the same detection, so it has to stay well inside how close real
    same-class neighbors actually sit: take each GT box's distance to its nearest same-image
    neighbor, pool that across images, take a low percentile (the closest real pairs set the
    ceiling a safe tolerance cannot cross) with a safety margin, and normalize by the same
    characteristic size ``gt_class_avg_size`` measures, so the fraction is comparable across
    datasets. No image anywhere has two or more of this class -> ``None`` (underivable; the caller
    stamps an honest default, never a derivation label on that number).

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
    """``sqrt(w*h)`` per GT box across every image, filtered to positive sizes, the shared size
    measure ``derive_localization_kind`` and ``derive_iou_match_threshold`` both derive from, so
    the two agree on what "this trait's characteristic object size" means by construction. Boxes
    are ``(x, y, w, h)`` tuples, not ``xyxy``: a caller holding ``xyxy`` boxes (``datasets.py``'s
    ``TiledDetectionDataset``) converts before calling, rather than this function guessing a shape."""
    sizes = [
        (max(w, 0.0) * max(h, 0.0)) ** 0.5
        for boxes in gt_boxes_per_image for _, _, w, h in boxes
    ]
    return [s for s in sizes if s > 0]


def _achievable_iou(avg_size: float, jitter_px: float) -> float:
    """Best-case IoU between two same-size boxes of characteristic size ``avg_size``, offset by
    ``jitter_px`` along one axis, the shared geometric basis ``derive_localization_kind`` and
    ``derive_iou_match_threshold`` both compute from (never two independent formulas for the same
    fact)."""
    return max(0.0, avg_size - jitter_px) / (avg_size + jitter_px)


def derive_localization_kind(
    gt_boxes_per_image: Sequence[Sequence[Sequence[float]]], *,
    jitter_px: float = 15.0, iou_floor: float = 0.5,
) -> str | None:
    """Whether IoU-matching or center-matching should govern this trait's "found the object" call,
    from the GT's own characteristic box size, or ``None`` if underivable.

    IoU is unreliable on small objects: a realistic detector-vs-GT localization disagreement (not
    just human annotator imprecision, a genuinely correct detection's box rarely lands pixel-
    identical to the GT box either) changes IoU by a large relative amount when the box itself is
    small. Model two same-size boxes of characteristic size ``s`` (``sqrt(w*h)``, matching
    ``derive_localization_tolerance_frac``'s own size measure) offset by ``jitter_px`` along one
    axis: their achievable IoU is ``(s - jitter_px) / (s + jitter_px)``. When that achievable IoU
    falls below ``iou_floor`` (0.5, the standard "hit" convention this platform already uses
    elsewhere), even a correctly-localized detection could not clear an IoU-match criterion under
    realistic jitter, so center-match must govern instead, this is a genuine geometric fact about
    the object's scale, not a per-trait preference. No valid boxes anywhere -> ``None``
    (underivable; the caller stamps an honest default, never a derivation label on that number),
    same contract as every other function in this module.

    **``jitter_px``'s default (15.0px) is a provisional platform-chosen constant, not a value
    validated against this platform's real detector/annotation precision**, same shape and same
    caveat as ``operating_point._PROVISIONAL_KAPPA_FLOOR``. It sets where the center-match/IoU-match
    crossover falls (currently ``s = 3 * jitter_px`` = ~45px characteristic size); a future pass
    with real per-trait localization-agreement data (repeated-annotation studies, or measured
    prediction-vs-GT offset on a validated model) could derive it properly instead of assuming it.
    This is exactly why a derived kind is recorded with ``data_derived_at_runtime`` provenance and
    re-checked for divergence on later calls (see ``resolve_match_criterion``) rather than trusted
    as a one-shot final answer, the safety net is the revisit check, not a perfectly-tuned formula
    up front.

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
    """The IoU threshold for an ``iou_match`` trait, from the GT's own characteristic box size,
    or ``None`` if underivable.

    Uses the same achievable-IoU-under-jitter basis ``derive_localization_kind`` uses to decide
    whether ``iou_match`` should govern at all: an ``iou_match`` trait must derive its own IoU
    threshold here rather than fall back to a pinned ``0.5`` literal with no real basis behind it.

    When ``resolve_match_criterion`` derives the kind fresh from this same call's GT
    (``kind_source == "data_derived_at_runtime"``), the characteristic size already cleared
    ``iou_floor``'s achievable-IoU bar by construction (see ``derive_localization_kind``), so the
    achievable IoU here is at or above that floor for that call. That is not guaranteed for a
    recorded ``iou_match`` trait revisited with a different call's GT, the kind isn't re-validated
    against the current box sizes, only compared for a divergence warning, so a recorded trait
    with unusually small current-call GT can still see an achievable IoU below the floor; the
    ``clamp`` below is what keeps the result sane in that case, not an assumption that it can't
    occur. ``margin`` subtracts a safety buffer below the achievable IoU so a typically-jittered
    correct detection clears the threshold, not only the mathematical best case, the same
    "percentile plus margin" shape every other derivation in this module uses (e.g.
    ``derive_cross_tile_nms``), rather than gating on the exact boundary value. Clamped to a sane
    range around the conventional IoU@0.5 comparability convention.

    **``jitter_px``'s default (15.0px) and ``margin``'s default (0.1) are provisional
    platform-chosen constants**, same caveat as ``derive_localization_kind``'s own ``jitter_px``,
    not validated against this platform's real detector precision yet; the same
    ``data_derived_at_runtime`` recording and revisit-on-divergence discipline applies once this
    is wired into ``resolve_match_criterion``.

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

    The cutoff has to tell a genuinely small-but-complete object (natural size variation, e.g. an
    earlier growth/bloom stage) from a real object a tile boundary clipped down to a fragment, a
    fixed fraction can't: a class with wide natural size variation needs a lower cutoff or it
    discards real small instances as slivers, while a tightly-sized class can use a higher one. So
    take a low percentile of this dataset's own characteristic-size distribution (the small end of
    genuinely complete objects) relative to its mean, clamped to a sane range. Fewer than
    ``min_samples`` boxes -> ``None``: a percentile from a handful of points is not a spread, it is
    noise (with 1-2 boxes the ratio is trivially ~1.0 regardless of the class's real variation), the
    caller stamps an honest default, never a derivation label on that number.

    ``char_sizes`` is ``sqrt(w*h)`` per GT box (px), already filtered to the trait's own class; see
    :func:`char_sizes_from_boxes` for the shared computation callers derive it from.
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


def band_normalization_stats(
    image_paths: Sequence[str | Path], num_channels: int, *, max_images: int = 50,
) -> tuple[list[float], list[float]] | None:
    """Per-band ``(mean, std)`` in [0, 1] over *this* dataset's rasters, or ``None``.

    The statistics a detector normalizes with. torchvision defaults to 3-element ImageNet values,
    which are wrong on any band set that is not RGB photography: at 1 channel they silently
    broadcast the image to 3, and at any count other than 3 they raise inside the transform.
    Sample the training split and pass the result to ``build_detector`` as ``image_mean``/
    ``image_std``.

    Derive, then pass; never auto-inject (the ``gt_aspect_ratios`` rule): the factory never reads
    the dataset. Pass the derived values through ``model_source.builder_kwargs`` rather than calling
    this from inside your builder body, torchvision keeps ``image_mean``/``image_std`` as plain
    lists on the transform, not as buffers, so they are absent from the checkpoint and a builder
    that re-derives them at load time will normalize differently at inference than at training.
    Returns ``None`` when no raster could be read, an honest underivable, not a stand-in constant.

    ``max_images`` caps the sample; band statistics converge long before a full orchard is read.
    """
    import numpy as np

    from tcip_mcp.pipelines.image_utils import load_image, pil_to_tensor

    moments = _BandMoments(num_channels)
    for path in list(image_paths)[:max_images]:
        try:
            arr = pil_to_tensor(load_image(path, num_channels)).numpy().astype(np.float64)
        except Exception:  # noqa: BLE001, an unreadable raster is skipped, not fatal
            continue
        moments.add(arr)
    return moments.result()


class _BandMoments:
    """Pixel-weighted per-band first and second moments over [0, 1] tensor pixels.

    The arithmetic :func:`band_normalization_stats` and :func:`band_normalization_stats_sampled`
    share, so the exact statistic and the sampled one can only ever differ in which pixels they
    read, never in how those pixels are composed into a mean and a standard deviation.
    """

    def __init__(self, num_channels: int):
        import numpy as np

        self.num_channels = int(num_channels)
        self.sums = np.zeros(self.num_channels, dtype=np.float64)
        self.sqs = np.zeros(self.num_channels, dtype=np.float64)
        self.pixels = 0

    def add(self, tensor_pixels) -> None:
        """Accumulate one ``[C, H, W]`` float64 block, skipping a block whose band count disagrees
        with this accumulator's (its pixels are not the same bands)."""
        if tensor_pixels.shape[0] != self.num_channels:
            return
        flat = tensor_pixels.reshape(self.num_channels, -1)
        self.sums += flat.sum(axis=1)
        self.sqs += (flat ** 2).sum(axis=1)
        self.pixels += flat.shape[1]

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
    cover. A caller passes ``mean``/``std`` to ``build_detector`` through
    ``model_source.builder_kwargs`` and carries ``sampling`` into the same provenance record, since
    these numbers describe a sample of the rasters and not all of their pixels.
    """

    mean: list[float]
    std: list[float]
    sampling: WindowSampling


def band_normalization_stats_sampled(
    image_paths: Sequence[str | Path], num_channels: int, *, seed: int, window_size: int,
    max_windows_per_image: int, max_images: int = 50,
) -> SampledNormalizationStats | None:
    """Per-band ``(mean, std)`` in [0, 1] over seeded pixel windows of this dataset's rasters.

    The windowed sibling of :func:`band_normalization_stats`: the same statistic in the same unit
    system, for a source whose full decode is unaffordable (an orthomosaic, say). Each raster is
    opened through ``raster_source`` and only ``max_windows_per_image`` windows of ``window_size``
    pixels are read from it, chosen by ``raster_source.sample_windows`` from ``seed``; a
    ``max_windows_per_image`` covering every cell of a raster's grid reads all of it and gives the
    exact sibling's own answer.

    Pixels are scaled by ``image_utils.pil_to_tensor``, the same call the exact sibling and the
    dataset itself make, never a scale re-derived here. ``seed`` has no default: a sampled
    statistic is reproducible only when the caller states what chose the sample. Returns ``None``
    when no raster could be read, the honest underivable the exact sibling returns.
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
                label = str(getattr(path, "manifest_path", path))
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


# Every ``derived_from`` label ever stamped by ``resolution.derived()`` must appear here, mapped
# to the callable that actually computes it, or to an explicit non-derivation marker
# ("caller-input" / "placeholder") when the constructor is reused for a non-derived value.
# tests/test_provenance_honesty.py enforces this, so a data-sounding label can never again be
# stamped without an implementation behind it (the cross_tile_nms costume bug).
DERIVATION_IMPLEMENTATIONS: dict[str, object] = {
    "probed bands of": "tcip_mcp.pipelines.derivations.probe_channels",  # f-string prefix
    "max class id + 1 in the label set": "tcip_mcp.pipelines.derivations.num_classes_from_distribution",
    "count-unbiased center-match sweep": "tcip_mcp.pipelines.operating_point.sweep_operating_point",
    "count-unbiased center-match sweep over review verdicts": "tcip_mcp.pipelines.operating_point.sweep_operating_point",
    "F1-max center-match sweep": "tcip_mcp.pipelines.operating_point.sweep_operating_point",
    "F1-max center-match sweep over review verdicts": "tcip_mcp.pipelines.operating_point.sweep_operating_point",
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
    "caller override": "caller-input",
    "no GT for this dataset; unvalidated placeholder": "placeholder",
}
