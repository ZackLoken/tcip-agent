"""The reserved-region reference of a within-image split.

It holds the band geometry, completeness and feasibility for a mosaic's own reserved calibration
and holdout regions, for a raster training source too large or too singular to hold whole images
out from.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from tcip_annotation.json_io import LabelDocument
    from tcip_mcp.pipelines.schemas import SpatialManifest

logger = logging.getLogger(__name__)


def reserved_spatial_regions(
    spatial: SpatialManifest | None,
) -> SpatialManifest | None:
    """Return a run's within-image split ``spatial`` when it reserved both reference regions.

    ``spatial`` is ``experiments.RunResolution.spatial``; the regions are a calibration and a
    holdout region. ``None`` for a run that drew no within-image split, or one that reserved no
    calibration or no holdout region.
    """
    if (
        spatial is None
        or not spatial.calibration_region
        or not spatial.holdout_region
    ):
        return None
    return spatial


def band_rects(
    region_rect: tuple[int, int, int, int],
    k: int,
    tile_size: int,
    overlap: float,
    buffer_px: int,
    name_prefix: str,
) -> dict[str, tuple[int, int, int, int]]:
    """Return ``k`` non-overlapping, buffered bands over ``region_rect``, in mosaic coordinates.

    Recurses :func:`spatial_strip_split` over the region's own local extent (its lattice starts at
    local ``(0, 0)``), then translates every returned rect back by the region's own origin
    (``+ x0, + y0``).
    """
    from tcip_mcp.pipelines.data.splits import spatial_strip_split

    x0, y0, x1, y1 = region_rect
    width, height = x1 - x0, y1 - y0
    names = tuple(f"{name_prefix}_{i}" for i in range(k))
    fractions = tuple(1.0 / k for _ in range(k))
    spatial = spatial_strip_split(
        width,
        height,
        tile_size,
        overlap,
        fractions=fractions,
        split_names=names,
        buffer=buffer_px,
    )
    out: dict[str, tuple[int, int, int, int]] = {}
    for name in names:
        ((lx0, ly0, lx1, ly1),) = spatial.regions[name]
        out[name] = (lx0 + x0, ly0 + y0, lx1 + x0, ly1 + y0)
    return out


def centered_in(
    boxes: np.ndarray, rect: tuple[int, int, int, int]
) -> np.ndarray:
    """Return which of the xyxy ``boxes`` have their center inside the half-open ``rect``."""
    from tcip_annotation.matching import box_centers

    x0, y0, x1, y1 = rect
    cx, cy = box_centers(boxes).T
    return (cx >= x0) & (cx < x1) & (cy >= y0) & (cy < y1)


def select_gt_for_band(
    gt: dict[str, np.ndarray],
    band_rect: tuple[int, int, int, int],
) -> dict[str, np.ndarray]:
    """Return this band's own GT, the rows of ``gt`` :func:`centered_in` ``band_rect``.

    ``gt`` holds xyxy ``boxes`` with their ``labels`` and ``iscrowd``; each kept box stays at its
    full extent (never clipped), translated to the band's own local (inner-rect-relative) pixel
    space.
    """
    from tcip_mcp.pipelines.data.datasets import PER_BOX_KEYS

    keep = centered_in(gt["boxes"], band_rect)
    band = {k: gt[k][keep] for k in PER_BOX_KEYS if k in gt}
    band["boxes"] = band["boxes"].astype(np.float64, copy=True)
    band["boxes"][:, [0, 2]] -= band_rect[0]
    band["boxes"][:, [1, 3]] -= band_rect[1]
    return band


def check_completeness(
    document: LabelDocument,
    where: str,
    subject: str,
    rects: dict[str, tuple[int, int, int, int]],
) -> None:
    """Refuse unless ``document``'s completion marks for ``subject`` cover every rect in ``rects``.

    The marks are the live ones the label document holds; each rect is half-open. The
    :class:`~tcip_mcp.assessment.AssessmentRefusedError` names the regions, the subject and
    ``where``.
    """
    from tcip_annotation.json_io import covers
    from tcip_mcp.assessment import AssessmentRefusedError

    marks = document.marks.get(subject, [])
    uncovered = sorted(
        name for name, rect in rects.items() if not covers(marks, rect)
    )
    if uncovered:
        raise AssessmentRefusedError(
            f"the reserved region(s) {uncovered} are not marked complete for subject {subject!r} "
            f"in {where}: mark each region complete in the Annotate tab before its ground truth "
            "can be assessed against."
        )


def check_feasibility(
    gt_counts: dict[str, int], *, side: str, min_present: int = 2
) -> None:
    """Refuse when fewer than ``min_present`` bands on this side carry any GT at all.

    The refusal is an :class:`~tcip_mcp.assessment.AssessmentRefusedError` naming the side.
    """
    from tcip_mcp.assessment import AssessmentRefusedError

    n_present = sum(1 for c in gt_counts.values() if c > 0)
    if n_present < min_present:
        raise AssessmentRefusedError(
            f"the resolved {side} band layout "
            f"({len(gt_counts)} band(s)) leaves only {n_present} band(s) with any GT, fewer than "
            f"the {min_present} an equivalence check needs; reduce k_{side}, widen the reserved "
            f"fraction, or check region completeness/annotation density for this mosaic."
        )


def density_uniformity_flags(
    gt_counts: dict[str, int], *, factor: float = 3.0
) -> list[str]:
    """Return the band names whose GT count is a stark outlier relative to its siblings.

    An outlier holds more than ``factor``x the median, or less than ``1/factor``x it: a cheap
    smoke check for a plausible attestation error (a region marked complete after only part of it
    was actually annotated).
    """
    import statistics

    counts = [c for c in gt_counts.values() if c > 0]
    if len(counts) < 3:
        return []
    median = statistics.median(counts)
    if median <= 0:
        return []
    return sorted(
        name
        for name, c in gt_counts.items()
        if c > 0 and (c > factor * median or c < median / factor)
    )
