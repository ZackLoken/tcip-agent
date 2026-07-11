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
        # Isolate the probe: a raster-read failure must NOT suppress the num_classes derivation below
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
        # Real detection head names (composer _DETECTION_HEADS + instance_seg) — NOT "detection".
        # Missing anchor_free_detection meant FCOS (the catkin detector) never derived num_classes.
        for h in model_spec.get("heads", []):
            if h.get("name") in ("anchor_detection", "anchor_free_detection", "instance_seg") and "num_classes" not in h:
                h["num_classes"] = nc
                provenance["num_classes"] = derived("num_classes", nc, derivation_class="deterministic",
                                                    derived_from="max class id + 1 in the label set")
    return provenance
