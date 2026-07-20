"""Mask-geometry — dimensional measurements on a validated binary/instance mask.

From a validated mask compute, in **pixels**, the area, perimeter, length (major axis) and width
(minor axis via PCA), plus the centroid; when a physical scale is supplied (``mm_per_px`` or an
equivalent ``gsd`` in mm/px) the same quantities are also returned in millimetres. Numpy-first with
no heavy imports — the toolkit primitive the agent composes for dimensional traits (organ area,
length, width). It measures whatever mask it is given; whether that mask is trustworthy is the
validation invariant's job, not this module's.

Conventions:
- Foreground is ``mask >= threshold`` (default 0.5), so a 0/1, bool, 0/255, or soft-probability mask
  all binarize correctly.
- ``length_px`` / ``width_px`` are pixel-inclusive extents along the PCA principal / secondary axis
  (extent + 1 px), so a solid ``W x H`` rectangle reports exactly ``W`` and ``H``.
- ``perimeter_px`` is the 4-connected boundary-edge length (exact for rectilinear masks;
  a staircase over-estimate on curved boundaries, as any pixel perimeter is).
"""

from __future__ import annotations

from typing import Any

# The mask-binarization threshold is a dimensional-phenotype knob: 0.5 is an honest engineering
# default, not a validated derivation. A calibrated mask-area measurement should derive it against
# validated masks (measured area vs GT); until then its provenance must travel as validated=false so
# a frozen 0.5 never silently defines every area/length number. Surfaced here as the one shared
# placeholder (resolve_binarize_threshold) so a delivery door stamps it rather than pinning it.
DEFAULT_MASK_BINARIZE_THRESHOLD = 0.5


def resolve_binarize_threshold(value: float | None = None):
    """The mask-binarization threshold as a firewalled ``ResolvedParam`` (default 0.5, validated=false).

    A calibration-class param: un-shippable as a bare number until derived/validated against validated
    masks (``.value`` raises), so a dimensional measurement can't silently freeze 0.5. An explicit
    ``value`` is honored but still stamped unvalidated until a door validates it.
    """
    from tcip_mcp.pipelines.resolution import VALIDATED_FALSE, ResolvedParam

    v = DEFAULT_MASK_BINARIZE_THRESHOLD if value is None else float(value)
    return ResolvedParam("mask_binarize_threshold", v,
                         source="explicit" if value is not None else "default",
                         derivation_class="calibration",
                         derived_from="documented default (derive against validated masks)",
                         validated_vs_gt=VALIDATED_FALSE)


def _to_numpy(mask: Any):
    """Accept a numpy array or a torch tensor; return a numpy array (heavy imports lazy)."""
    if hasattr(mask, "detach"):  # torch tensor
        mask = mask.detach().cpu().numpy()
    import numpy as np

    return np.asarray(mask)


def _perimeter_px(binary) -> float:
    """4-connected boundary length: sum over foreground pixels of edges facing background/outside."""
    import numpy as np

    b = binary.astype(np.int64)
    pad = np.pad(b, 1)
    neighbors = pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]
    return float(((4 - neighbors) * b).sum())


def _axes(binary):
    """PCA on the foreground pixel coords -> (centroid_xy, length, width, major_unit_vector).

    ``length`` / ``width`` are pixel-inclusive extents (extent + 1) along the major / minor axis.
    """
    import numpy as np

    ys, xs = np.nonzero(binary)
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    centroid = pts.mean(axis=0)
    if pts.shape[0] == 1:  # a single pixel: 1x1, oriented along x by convention
        return centroid, 1.0, 1.0, np.array([1.0, 0.0])
    centered = pts - centroid
    cov = np.cov(centered, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)  # ascending eigenvalues; columns are eigenvectors
    major_vec = evecs[:, -1]
    minor_vec = evecs[:, 0]
    if major_vec[0] < 0:  # fix sign so orientation is stable (major_x >= 0)
        major_vec = -major_vec
    proj_major = centered @ major_vec
    proj_minor = centered @ minor_vec
    length = float(proj_major.max() - proj_major.min()) + 1.0
    width = float(proj_minor.max() - proj_minor.min()) + 1.0
    return centroid, length, width, major_vec


def _attach_physical(result: dict, scale: float) -> None:
    """Add mm-unit fields from a linear mm-per-pixel scale (area scales by the square)."""
    s = float(scale)
    result["mm_per_px"] = s
    result["area_mm2"] = result["area_px"] * s * s
    result["perimeter_mm"] = result["perimeter_px"] * s
    result["length_mm"] = result["length_px"] * s
    result["width_mm"] = result["width_px"] * s
    c = result.get("centroid_px")
    result["centroid_mm"] = (c[0] * s, c[1] * s) if c is not None else None


def _resolve_scale(mm_per_px: float | None, gsd: float | None) -> float | None:
    """A single mm-per-pixel scale from either alias; ``gsd`` is a ground-sample-distance in mm/px."""
    if mm_per_px is not None and gsd is not None:
        raise ValueError("Provide only one of mm_per_px / gsd (gsd is mm-per-pixel), not both.")
    return mm_per_px if mm_per_px is not None else gsd


def mask_geometry(mask: Any, *, mm_per_px: float | None = None, gsd: float | None = None,
                  threshold: float = DEFAULT_MASK_BINARIZE_THRESHOLD) -> dict:
    """Dimensional geometry of a single validated 2D mask (``[H, W]`` or ``[1, H, W]``).

    Returns pixel measurements always, and mm-unit measurements when a scale is given::

        {"empty", "area_px", "perimeter_px", "length_px", "width_px", "centroid_px", "angle_deg",
         # when mm_per_px / gsd given:
         "mm_per_px", "area_mm2", "perimeter_mm", "length_mm", "width_mm", "centroid_mm"}

    An empty mask returns zeros with ``empty=True`` and ``centroid_px=None`` (measurement refuses to
    invent a location for nothing).
    """
    import numpy as np

    scale = _resolve_scale(mm_per_px, gsd)
    arr = _to_numpy(mask)
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        else:
            raise ValueError(
                "mask_geometry expects a single 2D mask ([H, W] or [1, H, W]); use "
                "instance_geometries for an [N, H, W] instance stack")
    if arr.ndim != 2:
        raise ValueError(f"mask must be 2D (got shape {arr.shape})")

    binary = arr.astype(np.float64) >= float(threshold)
    area_px = float(binary.sum())
    result: dict = {"empty": area_px == 0.0, "area_px": area_px, "perimeter_px": 0.0,
                    "length_px": 0.0, "width_px": 0.0, "centroid_px": None, "angle_deg": None}
    if area_px > 0.0:
        centroid, length, width, major_vec = _axes(binary)
        result["perimeter_px"] = _perimeter_px(binary)
        result["length_px"] = length
        result["width_px"] = width
        result["centroid_px"] = (float(centroid[0]), float(centroid[1]))
        result["angle_deg"] = float(np.degrees(np.arctan2(major_vec[1], major_vec[0])))
    if scale is not None:
        _attach_physical(result, scale)
    return result


def instance_geometries(masks: Any, *, mm_per_px: float | None = None, gsd: float | None = None,
                        threshold: float = DEFAULT_MASK_BINARIZE_THRESHOLD) -> list[dict]:
    """Per-instance :func:`mask_geometry` over an ``[N, H, W]`` mask stack (or a single ``[H, W]``)."""
    arr = _to_numpy(masks)
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim != 3:
        raise ValueError(f"masks must be [N, H, W] or [H, W] (got shape {arr.shape})")
    return [mask_geometry(arr[i], mm_per_px=mm_per_px, gsd=gsd, threshold=threshold)
            for i in range(arr.shape[0])]
