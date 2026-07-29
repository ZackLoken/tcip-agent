"""Tier-A data/model derivations — read the artifact in hand, compute the value.

These are the "deterministic" and "distribution" derivations (CLAUDE.md "Parameters: derive, don't
pin"): channels from *this* raster, num_classes from *this* label set, anchor aspect ratios from
*this* dataset's GT box shapes. The agent's model builder / train(ctx) calls these to size a bespoke
model to the data in hand — never a value frozen from a different dataset.

Heavy deps (PIL/numpy/tifffile) are imported lazily so this stays cheap to import.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


def probe_channels(image_path: str | Path) -> int:
    """Band count of a raster read from disk — the artifact in hand, not a sensor-name guess."""
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
        import numpy as np
        import tifffile
        arr = np.asarray(tifffile.imread(str(path)))
        if arr.ndim == 2:
            return 1
        # channel-first (C,H,W) if the leading axis is the smallest, else channel-last.
        return int(arr.shape[0]) if arr.shape[0] < arr.shape[-1] else int(arr.shape[-1])
    from PIL import Image
    with Image.open(path) as im:
        return len(im.getbands())  # RGB->3, L->1, RGBA->4


def num_classes_from_distribution(class_distribution: dict[int, int]) -> int:
    """Number of classes implied by the label set = max class id + 1 (ids are 0-indexed)."""
    if not class_distribution:
        return 0
    return int(max(class_distribution)) + 1


def gt_aspect_ratios(class_distribution_boxes: list[tuple[float, float]],
                     quantiles: tuple[float, float] = (0.1, 0.9)) -> list[float] | None:
    """Aspect ratios (h/w) spanning the GT box-shape distribution — for anchor coverage.

    Returns a small ratio set covering the p10..p90 of GT box aspect ratios (plus 1.0), so anchors
    match the actual object shapes in this dataset rather than a fixed (0.5, 1, 2).
    ``class_distribution_boxes`` is a list of ``(w, h)`` in pixels. Returns ``None`` when no valid
    box gives a ratio — underivable, so the caller stamps an honest default rather than receiving
    this function's own pinned fallback dressed as a derivation.

    Wiring (derive -> pass; the agent's builder path, never auto-injected into build_detector so a
    method isn't re-pinned)::

        from tcip_mcp.pipelines.derivations import gt_aspect_ratios
        from tcip_mcp.pipelines.components.detectors import build_detector
        ratios = gt_aspect_ratios([(b.w, b.h) for b in gt_boxes])   # this dataset's shapes
        if ratios is None:
            ratios = [0.5, 1.0, 2.0]                                 # underivable — yours to stamp
        names = list(adapter(torch.zeros(1, in_chans, h, w)).keys())  # this adapter's actual keys
        model = build_detector("faster_rcnn", adapter, num_classes=n,
                               featmap_names=names, num_levels=len(names),
                               aspect_ratios=tuple(ratios))
    """
    import numpy as np
    ratios = [h / w for (w, h) in class_distribution_boxes if w > 0 and h > 0]
    if not ratios:
        # Underivable: no valid boxes. Return None rather than the fixed (0.5, 1, 2) this function
        # exists to replace — a pinned default returned as if derived is indistinguishable from a
        # real result. Same contract as derive_cross_tile_nms: the caller stamps an honest default.
        return None
    lo, hi = np.quantile(ratios, quantiles[0]), np.quantile(ratios, quantiles[1])
    out = sorted({round(float(lo), 2), 1.0, round(float(hi), 2)})
    return [r for r in out if r > 0]


def _neighbor_max_ious(boxes: Sequence[Sequence[float]]) -> list[float]:
    """Each box's max IoU with any OTHER box in the same image (xywh px); fewer than 2 boxes -> []."""
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
    nearest-neighbor spacing — or ``None`` if underivable.

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


def derive_sliver_frac(
    char_sizes: Sequence[float], *, percentile: float = 10.0,
    clamp: tuple[float, float] = (0.25, 0.9), min_samples: int = 5,
) -> float | None:
    """Tile-seam sliver cutoff, as a fraction of the class's characteristic size, from the GT's own
    size spread — or ``None`` if underivable.

    The cutoff has to tell a genuinely small-but-complete object (natural size variation, e.g. an
    earlier growth/bloom stage) from a real object a tile boundary clipped down to a fragment — a
    fixed fraction can't: a class with wide natural size variation needs a lower cutoff or it
    discards real small instances as slivers, while a tightly-sized class can use a higher one. So
    take a low percentile of this dataset's own characteristic-size distribution (the small end of
    genuinely complete objects) relative to its mean, clamped to a sane range. Fewer than
    ``min_samples`` boxes -> ``None``: a percentile from a handful of points is not a spread, it is
    noise (with 1-2 boxes the ratio is trivially ~1.0 regardless of the class's real variation) — the
    caller stamps an honest default, never a derivation label on that number.

    ``char_sizes`` is ``sqrt(w*h)`` per GT box (px), already filtered to the trait's own class.
    """
    import numpy as np
    sizes = [float(s) for s in char_sizes if s > 0]
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

    **Derive -> pass, never auto-injected** (the ``gt_aspect_ratios`` rule): the factory never reads
    the dataset. Pass the derived values through ``model_source.builder_kwargs`` rather than calling
    this from inside your builder body — torchvision keeps ``image_mean``/``image_std`` as plain
    lists on the transform, not as buffers, so they are absent from the checkpoint and a builder
    that re-derives them at load time will normalize differently at inference than at training.
    Returns ``None`` when no raster could be read — an honest underivable, not a stand-in constant.

    ``max_images`` caps the sample; band statistics converge long before a full orchard is read.
    """
    import numpy as np

    from tcip_mcp.pipelines.image_utils import load_image, pil_to_tensor

    # Accumulate over exactly the tensor the dataset yields — pil_to_tensor decides the [0, 1]
    # scaling from dtype, so re-deriving a scale here would put the statistics in a different unit
    # system than the pixels the detector normalizes (a float raster silently off by its own max).
    # Pixel-weighted, so rasters of different sizes compose into one dataset statistic.
    sums = np.zeros(num_channels, dtype=np.float64)
    sqs = np.zeros(num_channels, dtype=np.float64)
    pixels = 0
    for path in list(image_paths)[:max_images]:
        try:
            arr = pil_to_tensor(load_image(path, num_channels)).numpy().astype(np.float64)
        except Exception:  # noqa: BLE001 — an unreadable raster is skipped, not fatal
            continue
        if arr.shape[0] != num_channels:
            continue
        flat = arr.reshape(num_channels, -1)
        sums += flat.sum(axis=1)
        sqs += (flat ** 2).sum(axis=1)
        pixels += flat.shape[1]
    if not pixels:
        return None
    mean = sums / pixels
    var = np.maximum(sqs / pixels - mean ** 2, 0.0)
    return [float(m) for m in mean], [float(s) for s in np.sqrt(var)]


def derive_cross_tile_nms(gt_boxes_per_image: Sequence[Sequence[Sequence[float]]], *,
                          percentile: float = 99.0, margin: float = 0.05,
                          clamp: tuple[float, float] = (0.2, 0.8)) -> float | None:
    """Cross-tile NMS IoU threshold from the GT neighbor-overlap distribution — or None if underivable.

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
    tail: list[float] = []
    for boxes in gt_boxes_per_image:
        tail.extend(v for v in _neighbor_max_ious(boxes) if v > 0.0)
    if not tail:
        return None
    lo, hi = clamp
    return float(min(max(float(np.percentile(tail, percentile)) + margin, lo), hi))


# Every ``derived_from`` label ever stamped by ``resolution.derived()`` must appear here, mapped
# to the callable that actually computes it — or to an explicit non-derivation marker
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
    "~1.5x p99 GT objects/image": "tcip_mcp.pipelines.operating_point._max_dets_from_density",
    "model imgsz / persisted training geometry": "tcip_mcp.pipelines.resolution.raw_operating_point",
    "persisted training tile geometry": "tcip_mcp.pipelines.resolution.raw_operating_point",
    "caller override": "caller-input",
    "no GT for this dataset; unvalidated placeholder": "placeholder",
}
