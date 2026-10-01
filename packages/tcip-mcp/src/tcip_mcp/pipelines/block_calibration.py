"""The reserved-region reference of a within-image split: the band geometry, completeness and
feasibility an assessment reads a mosaic's own reserved calibration and test regions through (see
``split_construction.spatial_single_source_split``'s four-way split,
``reserve_calibration_fraction``), for a raster training source too large or too singular to hold
whole images out from.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Enough bands to measure a per-band bias spread (the count criterion needs n >= 2 present images
# per side) without fragmenting a modest region into slivers.
DEFAULT_K_CAL = 3
DEFAULT_K_TEST = 3


def reserved_spatial_regions(resolved: dict) -> dict | None:
    """The spatial manifest of a run's resolution (``data.split.spatial_manifest``) when it
    reserved a calibration and a test region, else ``None``: a run that drew no within-image
    spatial split, or one that reserved no calibration or no test region."""
    spatial = resolved["data"]["split"].get("spatial_manifest")
    if not spatial or not spatial["calibration_region"] or not spatial["test_region"]:
        return None
    return spatial


def band_rects(
    region_rect: tuple[int, int, int, int], k: int, tile_size: int, overlap: float,
    buffer_px: int, name_prefix: str,
) -> dict[str, tuple[int, int, int, int]]:
    """``k`` non-overlapping, buffered bands over ``region_rect``, in full-mosaic coordinates.

    Recurses :func:`spatial_strip_split` over the region's own local extent (its lattice starts at
    local ``(0, 0)``), then translates every returned rect back by the region's own origin
    (``+ x0, + y0``), the clean, lattice-phase-safe translation.
    """
    from tcip_mcp.pipelines.data.splits import spatial_strip_split

    x0, y0, x1, y1 = region_rect
    width, height = x1 - x0, y1 - y0
    names = tuple(f"{name_prefix}_{i}" for i in range(k))
    fractions = tuple(1.0 / k for _ in range(k))
    spatial = spatial_strip_split(
        width, height, tile_size, overlap, fractions=fractions, split_names=names,
        buffer=buffer_px,
    )
    out: dict[str, tuple[int, int, int, int]] = {}
    for name in names:
        rects = spatial.regions.get(name) or []
        if not rects:
            continue
        # stripes_per_split defaults to 1 (never overridden here): one contiguous rect per band.
        lx0, ly0, lx1, ly1 = rects[0]
        out[name] = (lx0 + x0, ly0 + y0, lx1 + x0, ly1 + y0)
    return out


def centered_in(boxes: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    """Which of the xyxy ``boxes`` have their center inside the half-open ``rect``."""
    x0, y0, x1, y1 = rect
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
    return (cx >= x0) & (cx < x1) & (cy >= y0) & (cy < y1)


def select_gt_for_band(
    gt: dict[str, np.ndarray], band_rect: tuple[int, int, int, int],
) -> dict[str, np.ndarray]:
    """This band's own GT, the rows of ``gt`` (xyxy ``boxes`` with their ``labels`` and
    ``iscrowd``) :func:`centered_in` ``band_rect``, each box kept at its full extent (never
    clipped) and translated to the band's own local (inner-rect-relative) pixel space.
    """
    from tcip_mcp.pipelines.data.datasets import PER_BOX_KEYS

    keep = centered_in(gt["boxes"], band_rect)
    band = {k: gt[k][keep] for k in PER_BOX_KEYS if k in gt}
    band["boxes"] = band["boxes"].astype(np.float64, copy=True)
    band["boxes"][:, [0, 2]] -= band_rect[0]
    band["boxes"][:, [1, 3]] -= band_rect[1]
    return band


def check_completeness(
    dataset_root: str, subject: str, stem: str, rects: dict[str, tuple[int, int, int, int]],
) -> None:
    """Refuse by name (:class:`~tcip_mcp.assessment.AssessmentRefused`, naming the incomplete or
    stale cells and the subject) unless every rect in ``rects`` is fully covered by an
    attested-complete, non-stale region-completeness record.
    """
    from tcip_mcp.assessment import AssessmentRefused
    from tcip_mcp.pipelines.region_completeness import incomplete_cells_for_rect

    problems: list[str] = []
    for name, rect in sorted(rects.items()):
        missing = incomplete_cells_for_rect(dataset_root, subject, stem, rect)
        if missing is None:
            problems.append(f"{name}: no region-completeness record exists for subject {subject!r}")
        elif missing:
            problems.append(f"{name}: cells not attested complete for subject {subject!r}: {missing}")
    if problems:
        raise AssessmentRefused(
            "the reserved calibration/test regions are not fully attested complete for subject "
            f"{subject!r}: {'; '.join(problems)}. Attest every listed cell complete (the Annotate "
            "canvas's Attest control) before the regions' ground truth can be assessed against."
        )


def check_feasibility(gt_counts: dict[str, int], *, side: str, min_present: int = 2) -> None:
    """Refuse by name (:class:`~tcip_mcp.assessment.AssessmentRefused`) when fewer than
    ``min_present`` bands on this side carry any GT at all."""
    from tcip_mcp.assessment import AssessmentRefused

    n_present = sum(1 for c in gt_counts.values() if c > 0)
    if n_present < min_present:
        raise AssessmentRefused(
            f"the resolved {side} band layout "
            f"({len(gt_counts)} band(s)) leaves only {n_present} band(s) with any GT, fewer than "
            f"the {min_present} an equivalence check needs; reduce k_{side}, widen the reserved "
            f"fraction, or check region completeness/annotation density for this mosaic."
        )


def density_uniformity_flags(gt_counts: dict[str, int], *, factor: float = 3.0) -> list[str]:
    """Band names whose GT count is a stark outlier (more than ``factor``x the median, or less
    than ``1/factor``x it) relative to its siblings: a cheap smoke check for a plausible
    attestation error (a region marked complete after only part of it was actually annotated),
    flagged in provenance, never gating on its own."""
    import statistics

    counts = [c for c in gt_counts.values() if c > 0]
    if len(counts) < 3:
        return []
    median = statistics.median(counts)
    if median <= 0:
        return []
    return sorted(
        name for name, c in gt_counts.items()
        if c > 0 and (c > factor * median or c < median / factor)
    )
