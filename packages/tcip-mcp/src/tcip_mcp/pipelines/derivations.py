"""Tier-A data/model derivations — read the artifact in hand, compute the value.

These are the "deterministic" and "distribution" derivations (CLAUDE.md "Parameters: derive, don't
pin"): channels from *this* raster, num_classes from *this* label set, anchor aspect ratios from
*this* dataset's GT box shapes. Each runs at the pre-compose seam and fills a spec value the agent did
not pin — never a value frozen from a different dataset. Explicit agent values always win; a derived
value that contradicts an explicit one is surfaced (the runtime channel check already fails loud).

Heavy deps (PIL/numpy/tifffile) are imported lazily so this stays cheap to import.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from tcip_mcp.pipelines.resolution import ResolvedParam, derived

logger = logging.getLogger(__name__)


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
                     quantiles: tuple[float, float] = (0.1, 0.9)) -> list[float]:
    """Aspect ratios (h/w) spanning the GT box-shape distribution — for anchor coverage.

    Returns a small ratio set covering the p10..p90 of GT box aspect ratios (plus 1.0), so anchors
    match the actual object shapes in this dataset (e.g. elongated catkins) rather than a fixed
    (0.5, 1, 2). ``class_distribution_boxes`` is a list of ``(w, h)`` in pixels.
    """
    import numpy as np
    ratios = [h / w for (w, h) in class_distribution_boxes if w > 0 and h > 0]
    if not ratios:
        return [0.5, 1.0, 2.0]
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


def resolve_spec_derivations(model_spec: dict, *, sample_image: str | Path | None,
                             class_distribution: dict[int, int] | None) -> dict[str, ResolvedParam]:
    """Fill data-derived spec params the agent did not pin; return their ResolvedParams (provenance).

    - ``in_chans`` from the probed raster (only when the backbone did not set it explicitly).
    - ``num_classes`` from the label set (only when a detection head did not set it explicitly).

    Mutates ``model_spec`` in place at the pre-compose seam (mirrors ``_inject_imbalance_loss``).
    Explicit values are left untouched — the runtime channel/label checks catch a real mismatch.
    """
    provenance: dict[str, ResolvedParam] = {}

    bb = model_spec.get("backbone")
    if isinstance(bb, dict) and sample_image is not None and "in_chans" not in bb:
        # Isolate the probe: a raster-read failure must not suppress the num_classes derivation below
        # (which would then surface as an opaque compose-time TypeError with the real cause hidden).
        try:
            ch = probe_channels(sample_image)
            bb["in_chans"] = ch
            provenance["in_chans"] = derived("in_chans", ch, derivation_class="deterministic",
                                             derived_from=f"probed bands of {Path(sample_image).name}")
        except Exception:
            logger.warning("in_chans probe failed for %s; leaving backbone.in_chans unset",
                           sample_image, exc_info=True)

    if class_distribution:
        nc = num_classes_from_distribution(class_distribution)
        # Real detection head names (composer _DETECTION_HEADS + instance_seg) — not "detection".
        # Missing anchor_free_detection meant FCOS (the catkin detector) never derived num_classes.
        for h in model_spec.get("heads", []):
            if h.get("name") in ("anchor_detection", "anchor_free_detection", "instance_seg") and "num_classes" not in h:
                h["num_classes"] = nc
                provenance["num_classes"] = derived("num_classes", nc, derivation_class="deterministic",
                                                    derived_from="max class id + 1 in the label set")
    return provenance


# Every ``derived_from`` label ever stamped by ``resolution.derived()`` must appear here, mapped
# to the callable that actually computes it — or to an explicit non-derivation marker
# ("caller-input" / "placeholder") when the constructor is reused for a non-derived value.
# tests/test_provenance_honesty.py enforces this, so a data-sounding label can never again be
# stamped without an implementation behind it (the cross_tile_nms costume bug).
DERIVATION_IMPLEMENTATIONS: dict[str, object] = {
    "probed bands of": "tcip_mcp.pipelines.derivations.probe_channels",  # f-string prefix
    "max class id + 1 in the label set": "tcip_mcp.pipelines.derivations.num_classes_from_distribution",
    "count-unbiased center-match sweep": "tcip_mcp.pipelines.operating_point.sweep_operating_point",
    "GT neighbor-IoU distribution (p99 + margin)": "tcip_mcp.pipelines.derivations.derive_cross_tile_nms",
    "~1.5x p99 GT objects/image": "tcip_mcp.pipelines.operating_point._max_dets_from_density",
    "model imgsz / persisted training geometry": "tcip_mcp.pipelines.resolution.raw_operating_point",
    "persisted training tile geometry": "tcip_mcp.pipelines.resolution.raw_operating_point",
    "caller override": "caller-input",
    "no GT for this dataset; unvalidated placeholder": "placeholder",
}
