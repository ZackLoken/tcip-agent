"""Mask -> polygon rings: the contour extractor behind every mask-derived shape.

Every external contour becomes its own ring, largest-area first; nothing is reduced to the
largest component.
"""

from __future__ import annotations

from typing import Any

from tcip_annotation.state import RING_MIN_POINTS, is_ring


def mask_contours(mask: Any, *, threshold: float | None) -> list[Any]:
    """Return each connected component's outer contour of ``mask`` (``cv2.findContours``).

    The contours are in pixel coords, largest contour area first; ``[]`` when nothing is
    foreground.

    ``threshold=None`` treats the mask as already binary (any nonzero pixel is foreground): what a
    SAM boolean mask is. A soft/probability mask passes the binarization threshold its caller
    states or derives.
    """
    import cv2
    import numpy as np

    arr = np.asarray(mask)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    binary = (
        arr != 0
        if threshold is None
        else arr.astype(np.float64) >= float(threshold)
    )
    contours, _ = cv2.findContours(
        binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return sorted(contours, key=cv2.contourArea, reverse=True)


def _simplified(contour: Any, vertex_spacing: float) -> Any:
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

    def kept(frac: float) -> Any:
        return cv2.approxPolyDP(contour, frac * perimeter, True)

    def adjacent(res: Any) -> None:
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
    mask: Any, *, threshold: float | None, vertex_spacing: float
) -> list[list[tuple[float, float]]]:
    """Return one simplified polygon ring per connected component of ``mask``.

    Components are :func:`mask_contours` at ``threshold``, in pixel coords, largest first: each
    simplified to at most the vertex count its own perimeter implies at ``vertex_spacing`` pixels
    of perimeter per vertex (:func:`_simplified`; zero keeps every contour vertex), and a ring
    left that is no ring (:func:`~tcip_annotation.state.is_ring`) dropped. Returns ``[]`` when
    nothing is foreground.
    """
    rings: list[list[tuple[float, float]]] = []
    for c in mask_contours(mask, threshold=threshold):
        approx = _simplified(c, vertex_spacing)
        pts = [(float(p[0][0]), float(p[0][1])) for p in approx]
        if is_ring(pts):
            rings.append(pts)
    return rings
