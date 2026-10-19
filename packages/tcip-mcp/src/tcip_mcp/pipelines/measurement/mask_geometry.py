"""Mask-geometry: dimensional measurements on a validated binary/instance mask.

From a mask compute, in pixels, the area, perimeter, the extents along the PCA principal and
secondary axes, and the centroid; when a physical ``scale`` (per-pixel, in a caller-stated
``unit``, never assumed to be millimeters) is supplied, the same quantities are also returned in
that unit. Numpy-first with no heavy imports.

An axis extent is not an anatomical span. It is the width of the mask's own footprint projected
onto a data-derived direction: it equals the anatomical dimension only when the structure is
straight and its long axis is the mask's principal axis. A curved, bent, forked or occlusion-split
structure has a principal-axis extent shorter than its arc length, and a structure whose visual
long axis is not its statistically dominant one has the two axes swapped outright. The returned
keys name the axis, not ``length``/``width``.

Conventions:
- Foreground is ``mask >= threshold``, the threshold the caller states or the pass's execution
  record carries, so a 0/1, bool, 0/255, or soft-probability mask binarizes at a value chosen
  for it.
- ``principal_axis_extent_px`` / ``secondary_axis_extent_px`` are pixel-inclusive extents along the
  PCA principal / secondary axis (extent + 1 px), so a solid ``W x H`` rectangle reports exactly
  ``W`` and ``H``.
- ``perimeter_px`` is the 4-connected boundary-edge length (exact for rectilinear masks; a
  staircase over-estimate on curved boundaries, as any pixel perimeter is).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy as np
    import torch
    from numpy.typing import NDArray


def _to_numpy(mask: np.ndarray | torch.Tensor) -> np.ndarray:
    """Accept a numpy array or a torch tensor; return a numpy array (heavy imports lazy)."""
    if hasattr(mask, "detach"):  # torch tensor
        mask = mask.detach().cpu().numpy()
    import numpy as np

    return np.asarray(mask)


def _perimeter_px(binary: NDArray[np.bool_]) -> float:
    """Return the 4-connected boundary length of a binary mask.

    The length is the sum over foreground pixels of edges facing background or outside.
    """
    import numpy as np

    b = binary.astype(np.int64)
    pad = np.pad(b, 1)
    neighbors = pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]
    return float(((4 - neighbors) * b).sum())


def _axes_from_points(
    pts: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float, float, NDArray[np.float64]]:
    """Return the PCA centroid, axis extents and major axis of a raw ``(x, y)`` point cloud.

    The result is ``(centroid_xy, principal, secondary, major_unit_vector)``, the two extents
    continuous chord spans with no pixel-inclusive correction.
    """
    import numpy as np

    centroid = pts.mean(axis=0)
    if (
        pts.shape[0] == 1
    ):  # a single point: no orientation to derive, by convention along x
        return centroid, 0.0, 0.0, np.array([1.0, 0.0])
    centered = pts - centroid
    cov = np.cov(centered, rowvar=False)
    evals, evecs = np.linalg.eigh(
        cov
    )  # ascending eigenvalues; columns are eigenvectors
    major_vec = evecs[:, -1]
    minor_vec = evecs[:, 0]
    if major_vec[0] < 0:  # fix sign so orientation is stable (major_x >= 0)
        major_vec = -major_vec
    proj_major = centered @ major_vec
    proj_minor = centered @ minor_vec
    principal = float(proj_major.max() - proj_major.min())
    secondary = float(proj_minor.max() - proj_minor.min())
    return centroid, principal, secondary, major_vec


def _axes(
    binary: NDArray[np.bool_],
) -> tuple[NDArray[np.float64], float, float, NDArray[np.float64]]:
    """Compute PCA geometry from foreground pixel coordinates.

    Returns ``(centroid_xy, principal, secondary, major_unit_vector)``; ``principal`` /
    ``secondary`` are pixel-inclusive extents (extent + 1) along the major / minor axis: chords of
    the mask's footprint, not anatomical spans.
    """
    import numpy as np

    ys, xs = np.nonzero(binary)
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    if (
        pts.shape[0] == 1
    ):  # a single pixel: 1x1, oriented along x by convention
        return pts.mean(axis=0), 1.0, 1.0, np.array([1.0, 0.0])
    centroid, principal, secondary, major_vec = _axes_from_points(pts)
    return centroid, principal + 1.0, secondary + 1.0, major_vec


def principal_axis_extent_of_points(
    points: Iterable[tuple[float, float]],
) -> float:
    """Return the principal-axis extent of a set of ``(x, y)`` points.

    The extent is the same orientation-independent PCA primitive :func:`_axes` computes for
    ``principal_axis_extent_px``, applied directly to a geometry's own vertices rather than a
    rasterized mask, so it carries no discretization error. A continuous chord span, not
    pixel-inclusive: the rasterized-mask convention ``_axes`` applies does not apply to raw
    coordinates.

    ``points`` is a flat sequence of ``(x, y)`` pairs, all treated as one object. Raises
    ``ValueError`` on no points.
    """
    import numpy as np

    pts = np.asarray(list(points), dtype=np.float64)
    if pts.shape[0] == 0:
        raise ValueError(
            "principal_axis_extent_of_points needs at least one point"
        )
    _centroid, principal, _secondary, _major_vec = _axes_from_points(pts)
    return principal


def _attach_physical(result: dict, scale: float, unit: str) -> None:
    """Add ``{unit}``-suffixed physical fields to ``result`` from a linear per-pixel scale.

    The scale is in ``unit`` and area scales by its square. The unit is the one the caller
    states, never assumed: a scale in cm/px passed with ``unit="cm"`` produces
    ``area_cm2``/``principal_axis_extent_cm``/etc.
    """
    s = float(scale)
    result[f"{unit}_per_px"] = s
    result[f"area_{unit}2"] = result["area_px"] * s * s
    result[f"perimeter_{unit}"] = result["perimeter_px"] * s
    result[f"principal_axis_extent_{unit}"] = (
        result["principal_axis_extent_px"] * s
    )
    result[f"secondary_axis_extent_{unit}"] = (
        result["secondary_axis_extent_px"] * s
    )
    c = result.get("centroid_px")
    result[f"centroid_{unit}"] = (
        (c[0] * s, c[1] * s) if c is not None else None
    )


# _attach_physical writes "{field}_{unit}[2]" (area squared, everything else linear), the naming
# convention any unit-bearing value_key, from mask_geometry or from bespoke agent-composed
# measurement code, follows to be recognized here. unit_from_value_key is the single reader of it
# (aggregation.py's delivery CSV calls this rather than re-deriving the pattern with its own regex:
# "when two paths must agree, call one from the other"). Deliberately not a field-name whitelist
# (mask_geometry has no monopoly on producing dimensional measurements: arc length, a skeleton path,
# a landmark distance are all real traits the agent composes elsewhere on the same validated mask):
# recognized units are derived from crops.yml's own declared vocabulary, the trait authority, plus
# each one's mechanically-derived squared form, never an invented category list.
def _known_units() -> set[str]:
    from tcip_mcp.traits import crops_units

    return set(crops_units().values())


def is_pixel_space_key(key: str) -> bool:
    """Return whether ``key`` explicitly names pixel space.

    It does when its last ``_``-separated token is ``px``, the bare key ``"px"`` included.
    """
    return key.rpartition("_")[2] == "px"


def unit_from_value_key(key: str) -> tuple[str, str] | None:
    """Return the physical unit ``key``'s trailing ``_<token>`` names, or ``None``.

    The unit is returned as ``(display_unit, linear_basis)``: ``"area_mm2"`` gives
    ``("mm2", "mm")``, ``"principal_axis_extent_cm"`` gives ``("cm", "cm")``. ``display_unit`` is
    the unit the value is in; ``linear_basis`` is its linear unit, comparable with crops.yml's
    declared (always-linear) unit: the two differ only for a squared (area-like) key. A
    pixel-suffixed key (``"principal_axis_extent_px"``) never implies a unit, and neither does any
    trailing token outside crops.yml's real declared units (``"elongated_fraction"`` does not
    imply a unit called "fraction").

    Raises when the key's own name says "area" but its unit isn't squared (``"area_mm"``, missing
    the ``2``).
    """
    root, sep, trailing = key.rpartition("_")
    if not sep or not trailing or is_pixel_space_key(key):
        return None
    known = _known_units()
    squared = {u + "2" for u in known}
    if trailing in known:
        display, linear_basis = trailing, trailing
    elif trailing in squared:
        display, linear_basis = trailing, trailing[:-1]
    else:
        return None
    if "area" in root.split("_") and display == linear_basis:
        raise ValueError(
            f"{key!r} names 'area' but its unit {trailing!r} isn't squared, an area is length^2, "
            f"expected a value_key ending in {trailing}2. Fix the value_key (or the computation it "
            "names) rather than shipping a linear label for a squared quantity."
        )
    return display, linear_basis


def mask_geometry(
    mask: np.ndarray | torch.Tensor,
    *,
    scale: float | None = None,
    unit: str,
    threshold: float,
) -> dict:
    """Return the dimensional geometry of a single validated 2D mask.

    The mask is ``[H, W]`` or ``[1, H, W]``, its foreground the pixels at or above ``threshold``;
    any other shape raises ``ValueError``. ``scale`` is a plain float (per-pixel, in ``unit``).

    Returns pixel measurements always, and ``{unit}``-suffixed physical measurements when a scale
    is given::

        {
            "empty",
            "area_px",
            "perimeter_px",
            "principal_axis_extent_px",
            "secondary_axis_extent_px",
            "centroid_px",
            "angle_deg",
            # when scale given, e.g. unit="mm":
            "mm_per_px",
            "area_mm2",
            "perimeter_mm",
            "principal_axis_extent_mm",
            "secondary_axis_extent_mm",
            "centroid_mm",
        }

    The two axis extents are chords of the mask's footprint along its own PCA axes, not anatomical
    spans.

    An empty mask returns zeros with ``empty=True`` and ``centroid_px=None``.
    """
    import numpy as np

    arr = _to_numpy(mask)
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        else:
            raise ValueError(
                "mask_geometry expects a single 2D mask ([H, W] or [1, H, W]); use "
                "instance_geometries for an [N, H, W] instance stack"
            )
    if arr.ndim != 2:
        raise ValueError(f"mask must be 2D (got shape {arr.shape})")

    binary = arr.astype(np.float64) >= float(threshold)
    area_px = float(binary.sum())
    result: dict = {
        "empty": area_px == 0.0,
        "area_px": area_px,
        "perimeter_px": 0.0,
        "principal_axis_extent_px": 0.0,
        "secondary_axis_extent_px": 0.0,
        "centroid_px": None,
        "angle_deg": None,
    }
    if area_px > 0.0:
        centroid, principal, secondary, major_vec = _axes(binary)
        result["perimeter_px"] = _perimeter_px(binary)
        result["principal_axis_extent_px"] = principal
        result["secondary_axis_extent_px"] = secondary
        result["centroid_px"] = (float(centroid[0]), float(centroid[1]))
        result["angle_deg"] = float(
            np.degrees(np.arctan2(major_vec[1], major_vec[0]))
        )
    if scale is not None:
        _attach_physical(result, scale, unit)
    return result


def instance_geometries(
    masks: np.ndarray | torch.Tensor,
    *,
    scale: float | None = None,
    unit: str,
    threshold: float,
) -> list[dict]:
    """Return :func:`mask_geometry` per instance of an ``[N, H, W]`` mask stack.

    A single ``[H, W]`` mask is one instance; any other shape raises ``ValueError``.
    """
    arr = _to_numpy(masks)
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim != 3:
        raise ValueError(
            f"masks must be [N, H, W] or [H, W] (got shape {arr.shape})"
        )
    return [
        mask_geometry(arr[i], scale=scale, unit=unit, threshold=threshold)
        for i in range(arr.shape[0])
    ]
