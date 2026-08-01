"""Mask-geometry: dimensional measurements on a validated binary/instance mask.

From a validated mask compute, in **pixels**, the area, perimeter, the extents along the PCA
principal and secondary axes, and the centroid; when a physical ``scale`` (per-pixel, in a
caller-stated ``unit`` (never assumed to be millimetres) is supplied, the same quantities are also
returned in that unit. Numpy-first with no heavy imports: the toolkit primitive the agent composes
for dimensional traits. It measures whatever mask it is given; whether that mask is trustworthy is
the validation invariant's job, not this module's. :func:`resolve_scale` firewalls a candidate
physical scale the same way :func:`resolve_binarize_threshold` firewalls the binarization threshold:
un-shippable until validated against a real reference for its kind.

An axis extent is not an anatomical span. It is the width of the mask's own footprint projected onto
a data-derived direction: it equals the anatomical dimension only when the structure is straight and
its long axis is the mask's principal axis. A curved, bent, forked or occlusion-split structure has a
principal-axis extent shorter than its arc length, and a structure whose visual long axis is not its
statistically dominant one has the two axes swapped outright. That is why the returned keys name the
axis rather than a body part: naming them ``length``/``width`` would assert an anatomy this
computation does not measure. An anatomical span that a chord cannot represent (an arc length, a
skeleton path, a span between two identified landmarks) is a different computation the agent
composes, on the same validated mask, and the expert's trait definition decides which of the two the
trait actually calls for.

Conventions:
- Foreground is ``mask >= threshold`` (default 0.5), so a 0/1, bool, 0/255, or soft-probability mask
  all binarize correctly.
- ``principal_axis_extent_px`` / ``secondary_axis_extent_px`` are pixel-inclusive extents along the
  PCA principal / secondary axis (extent + 1 px), so a solid ``W x H`` rectangle reports exactly
  ``W`` and ``H``.
- ``perimeter_px`` is the 4-connected boundary-edge length (exact for rectilinear masks;
  a staircase over-estimate on curved boundaries, as any pixel perimeter is).
"""

from __future__ import annotations

from typing import Any

from tcip_annotation.mask_contours import DEFAULT_EPSILON_FRAC

# The mask-binarization threshold is a dimensional-phenotype knob: 0.5 is an honest engineering
# default, not a validated derivation. A calibrated mask-area measurement should derive it against
# validated masks (measured area vs GT); until then its provenance must travel as validated=false so
# a frozen 0.5 never silently defines every area/extent number. Surfaced here as the one shared
# placeholder (resolve_binarize_threshold) so a delivery door stamps it rather than pinning it.
DEFAULT_MASK_BINARIZE_THRESHOLD = 0.5


def resolve_binarize_threshold(value: float | None = None):
    """The mask-binarization threshold as a firewalled ``ResolvedParam`` (default 0.5, validated=false).

    Requires validation (``validation_kind="annotations"``, a mask GT reference, the same kind
    ``conf`` is validated against, just for masks instead of boxes): un-shippable as a bare number
    until derived/validated against validated masks (``.value`` raises), so a dimensional measurement
    can't silently freeze 0.5. An explicit ``value`` is honored but still stamped unvalidated until a
    door validates it.
    """
    from tcip_mcp.pipelines.resolution import VALIDATED_FALSE, ResolvedParam

    v = DEFAULT_MASK_BINARIZE_THRESHOLD if value is None else float(value)
    return ResolvedParam("mask_binarize_threshold", v,
                         source="explicit" if value is not None else "default",
                         derived_from="documented default (derive against validated masks)",
                         requires_validation=True, validation_kind="annotations",
                         validated_against=VALIDATED_FALSE)


def resolve_scale(value: float | None = None, *, unit: str = "mm", capture_id: str | None = None):
    """A physical per-pixel scale as a firewalled ``ResolvedParam`` (validated=false by default).

    Requires validation (``validation_kind="physical"``), shippable only once ``validated_against``
    names a real physical-measurement reference (``VALIDATED_PHYSICAL_MEASUREMENT``). The derivation
    method (EXIF-derived geometry, a reference-object measurement, or anything else) is not this
    function's concern, a capability the agent composes per dataset, not a method this platform
    picks: it only wraps whatever candidate scale a caller has already derived, in whatever ``unit``
    that scale is actually in (never assumed to be mm), and refuses to let it ship until validated.

    ``capture_id`` marks the value as scoped to a single capture (a handheld standoff can vary image
    to image within one dataset) when the caller has one; no deriver for a stable capture_id exists
    yet, so most callers pass ``None`` and the scale is scoped no finer than the caller's own choice.
    """
    from tcip_mcp.pipelines.resolution import VALIDATED_FALSE, ResolvedParam

    return ResolvedParam(
        f"scale_{unit}_per_px", value, source="explicit" if value is not None else "default",
        derived_from="caller-supplied scale, unvalidated until a door confirms it against a real "
                     "physical reference",
        requires_validation=True, validation_kind="physical", validated_against=VALIDATED_FALSE,
        capture_scoped=capture_id is not None, capture_id=capture_id,
    )


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
    """PCA on the foreground pixel coords -> (centroid_xy, principal, secondary, major_unit_vector).

    ``principal`` / ``secondary`` are pixel-inclusive extents (extent + 1) along the major / minor
    axis: chords of the mask's footprint, not anatomical spans (see the module docstring).
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
    principal = float(proj_major.max() - proj_major.min()) + 1.0
    secondary = float(proj_minor.max() - proj_minor.min()) + 1.0
    return centroid, principal, secondary, major_vec


def _attach_physical(result: dict, scale: float, unit: str) -> None:
    """Add ``{unit}``-suffixed physical fields from a linear per-pixel scale in that unit (area
    scales by the square). The unit is real data the caller states, never assumed: a scale in cm/px
    passed with ``unit="cm"`` produces ``area_cm2``/``principal_axis_extent_cm``/etc, not a
    silently-wrong ``_mm`` label (the previous hardcoded-mm behavior mislabeled any trait whose real
    unit wasn't mm)."""
    s = float(scale)
    result[f"{unit}_per_px"] = s
    result[f"area_{unit}2"] = result["area_px"] * s * s
    result[f"perimeter_{unit}"] = result["perimeter_px"] * s
    result[f"principal_axis_extent_{unit}"] = result["principal_axis_extent_px"] * s
    result[f"secondary_axis_extent_{unit}"] = result["secondary_axis_extent_px"] * s
    c = result.get("centroid_px")
    result[f"centroid_{unit}"] = (c[0] * s, c[1] * s) if c is not None else None


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


def unit_from_value_key(key: str) -> tuple[str, str] | None:
    """``(display_unit, linear_basis)`` if ``key``'s trailing ``_<token>`` names a real physical unit
    (``"area_mm2"`` -> ``("mm2", "mm")``, ``"principal_axis_extent_cm"`` -> ``("cm", "cm")``), else
    ``None``.

    ``display_unit`` is what the units column should say; ``linear_basis`` is what crops.yml's
    declared (always-linear) unit is cross-checked against: the two differ only for a squared
    (area-like) key. A pixel-suffixed key (``"principal_axis_extent_px"``) never implies a unit,
    pixels are not one, and neither does any trailing token outside crops.yml's real declared units
    (however plausible it looks: ``"elongated_fraction"`` does not imply a unit called "fraction").

    Refuses (raises) rather than silently mislabeling when the key's own name says "area" but its
    unit isn't squared (``"area_mm"``, missing the ``2``), a real naming bug in the producing code,
    not a case to guess through: an area is length², and shipping it labeled with a bare linear unit
    is exactly the kind of silent dimensional mismatch this function exists to catch.
    """
    root, sep, trailing = key.rpartition("_")
    if not sep or not trailing or trailing == "px":
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


def mask_geometry(mask: Any, *, scale: float | None = None, unit: str = "mm",
                  threshold: float = DEFAULT_MASK_BINARIZE_THRESHOLD) -> dict:
    """Dimensional geometry of a single validated 2D mask (``[H, W]`` or ``[1, H, W]``).

    ``scale`` is a plain float (per-pixel, in ``unit``), never a ``ResolvedParam``. The firewall
    belongs at the delivery door that resolves/validates the scale (:func:`resolve_scale`), not
    inside this primitive: forcing every call (including diagnostics, visualization, and training-
    loop geometry that are never deliveries) through the firewall would train reflexive
    ``acknowledge_unvalidated=True`` boilerplate and degrade the escape hatch's signal value at the
    real delivery doors.

    Returns pixel measurements always, and ``{unit}``-suffixed physical measurements when a scale is
    given::

        {"empty", "area_px", "perimeter_px", "principal_axis_extent_px",
         "secondary_axis_extent_px", "centroid_px", "angle_deg",
         # when scale given, e.g. unit="mm":
         "mm_per_px", "area_mm2", "perimeter_mm", "principal_axis_extent_mm",
         "secondary_axis_extent_mm", "centroid_mm"}

    The two axis extents are chords of the mask's footprint along its own PCA axes, not anatomical
    spans: see the module docstring before treating one as a trait's length or width.

    An empty mask returns zeros with ``empty=True`` and ``centroid_px=None`` (measurement refuses to
    invent a location for nothing).
    """
    import numpy as np

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
                    "principal_axis_extent_px": 0.0, "secondary_axis_extent_px": 0.0,
                    "centroid_px": None, "angle_deg": None}
    if area_px > 0.0:
        centroid, principal, secondary, major_vec = _axes(binary)
        result["perimeter_px"] = _perimeter_px(binary)
        result["principal_axis_extent_px"] = principal
        result["secondary_axis_extent_px"] = secondary
        result["centroid_px"] = (float(centroid[0]), float(centroid[1]))
        result["angle_deg"] = float(np.degrees(np.arctan2(major_vec[1], major_vec[0])))
    if scale is not None:
        _attach_physical(result, scale, unit)
    return result


def mask_to_polygon_points(
    mask: Any, *, threshold: float = DEFAULT_MASK_BINARIZE_THRESHOLD,
    epsilon_frac: float = DEFAULT_EPSILON_FRAC,
) -> list[list[tuple[float, float]]]:
    """Binary/soft mask -> one simplified polygon ring per connected component (pixel coords).

    The extraction itself is :func:`tcip_annotation.mask_contours.mask_to_polygon_rings`, the same
    call SAM-assisted labeling makes, so a model's exported prediction and a breeder's SAM-assisted
    GT describe an occlusion-split object the same way instead of one of them silently keeping only
    the largest region. This entry point adds only what belongs to the measurement side: the
    tensor->numpy hop and the platform's mask-binarization threshold default.
    """
    from tcip_annotation.mask_contours import mask_to_polygon_rings

    return mask_to_polygon_rings(_to_numpy(mask), threshold=threshold, epsilon_frac=epsilon_frac)


def instance_geometries(masks: Any, *, scale: float | None = None, unit: str = "mm",
                        threshold: float = DEFAULT_MASK_BINARIZE_THRESHOLD) -> list[dict]:
    """Per-instance :func:`mask_geometry` over an ``[N, H, W]`` mask stack (or a single ``[H, W]``)."""
    arr = _to_numpy(masks)
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim != 3:
        raise ValueError(f"masks must be [N, H, W] or [H, W] (got shape {arr.shape})")
    return [mask_geometry(arr[i], scale=scale, unit=unit, threshold=threshold)
            for i in range(arr.shape[0])]
