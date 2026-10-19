"""Mask -> polygon rings.

Every external contour becomes its own ring, largest-area first; nothing is reduced to the
largest component.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tcip_annotation.state import RING_MIN_POINTS, is_ring

if TYPE_CHECKING:
    import numpy as np
    from scipy._lib._util import _RichResult

    from tcip_annotation.state import Rings


def mask_contours(mask: np.ndarray, *, threshold: float | None) -> list[Any]:
    """Return each connected component's outer contour of ``mask`` (``cv2.findContours``).

    The contours are in pixel coords, largest contour area first; ``[]`` when nothing is
    foreground.

    ``threshold=None`` treats the mask as already binary (any nonzero pixel is foreground): what a
    SAM boolean mask is. A soft/probability mask passes the binarization threshold its caller
    states or derives.
    """
    import cv2
    import numpy as np

    arr = mask[0] if mask.ndim == 3 and mask.shape[0] == 1 else mask
    binary = (
        arr != 0 if threshold is None else arr.astype(np.float64) >= threshold
    )
    contours, _ = cv2.findContours(
        binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return sorted(contours, key=cv2.contourArea, reverse=True)


def _simplified(contour: np.ndarray, vertex_spacing: float) -> np.ndarray:
    """Return ``contour`` simplified by ``cv2.approxPolyDP`` to the vertex count its perimeter sets.

    The count is what its own perimeter (``cv2.arcLength``, closed) implies at ``vertex_spacing``
    pixels per vertex, the nearest integer (a half rounding up) and at least
    :data:`~tcip_annotation.state.RING_MIN_POINTS`; the result has at most that count. The
    tolerance is the least fraction of the perimeter leaving at most that count, the upper end of
    the bracket ``scipy.optimize.elementwise.find_root`` narrows over ``[0, 1]`` to adjacent
    floats. A contour already at or under its count is returned at ``approxPolyDP`` tolerance
    zero, and at spacing zero the contour itself is returned.
    """
    import math

    import cv2
    import numpy as np
    from scipy.optimize.elementwise import find_root

    perimeter = cv2.arcLength(contour, True)

    def kept(frac: float) -> np.ndarray:
        return cv2.approxPolyDP(contour, frac * perimeter, True)

    def adjacent(res: _RichResult) -> None:
        low, high = res.bracket
        if np.nextafter(low, high) >= high:
            raise StopIteration

    if vertex_spacing == 0:
        return contour
    unsimplified = kept(0.0)
    vertices = len(unsimplified)
    count = max(
        RING_MIN_POINTS,
        math.floor(min(perimeter / vertex_spacing, vertices) + 0.5),
    )
    if vertices <= count:
        return unsimplified
    _low, high = find_root(
        np.vectorize(lambda frac: 1.0 if len(kept(frac)) > count else -1.0),
        (0.0, 1.0),
        tolerances={"xatol": 0, "xrtol": 0, "fatol": 0, "frtol": 0},
        callback=adjacent,
    ).bracket
    return kept(float(high))


def mask_to_polygon_rings(
    mask: np.ndarray, *, threshold: float | None, vertex_spacing: float
) -> Rings:
    """Return one simplified polygon ring per connected component of ``mask``.

    Components are :func:`mask_contours` at ``threshold``, in pixel coords, largest first: each
    simplified to at most the vertex count its own perimeter implies at ``vertex_spacing`` pixels
    of perimeter per vertex (:func:`_simplified`; zero keeps every contour vertex), and a ring
    left that is no ring (:func:`~tcip_annotation.state.is_ring`) dropped. Returns ``[]`` when
    nothing is foreground.
    """
    rings: Rings = []
    for c in mask_contours(mask, threshold=threshold):
        approx = _simplified(c, vertex_spacing)
        pts = [(float(p[0][0]), float(p[0][1])) for p in approx]
        if is_ring(pts):
            rings.append(pts)
    return rings
