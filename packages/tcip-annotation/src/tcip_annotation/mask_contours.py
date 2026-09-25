"""Mask -> polygon rings: the contour extractor behind every mask-derived shape. Every external
contour becomes its own ring, largest-area first; nothing is reduced to the largest component.
"""

from __future__ import annotations

from typing import Any

from tcip_annotation.state import is_ring

#: Simplification tolerance as a fraction of each ring's own perimeter, so the deviation a ring is
#: allowed is proportional to that ring's size: a small fragment is not flattened by the same
#: absolute tolerance that suits a large one.
DEFAULT_EPSILON_FRAC = 0.005


def mask_to_polygon_rings(
    mask: Any, *, threshold: float | None = None, epsilon_frac: float = DEFAULT_EPSILON_FRAC,
) -> list[list[tuple[float, float]]]:
    """One simplified polygon ring per connected component of ``mask``, in pixel coords.

    ``threshold=None`` treats the mask as already binary (any nonzero pixel is foreground): what a
    SAM boolean mask is. A soft/probability mask passes the binarization threshold its caller
    resolved.

    Rings are ordered largest-contour-area first. Each ring is simplified with ``cv2.approxPolyDP``
    at ``epsilon_frac`` of its own perimeter, and a ring left that is no ring
    (:func:`~tcip_annotation.state.is_ring`) is dropped. Returns ``[]`` when nothing is foreground.
    """
    import cv2
    import numpy as np

    arr = np.asarray(mask)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    binary = arr != 0 if threshold is None else arr.astype(np.float64) >= float(threshold)
    contours, _ = cv2.findContours(
        binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rings: list[list[tuple[float, float]]] = []
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        approx = cv2.approxPolyDP(c, epsilon_frac * cv2.arcLength(c, True), True)
        pts = [(float(p[0][0]), float(p[0][1])) for p in approx]
        if is_ring(pts):
            rings.append(pts)
    return rings
