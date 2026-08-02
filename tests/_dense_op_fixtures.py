"""Shared dense-reference fixture builder for operating-point gate tests.

Some gate conditions (a dispersion/localization floor, a reference-sufficiency and equivalence
criterion) cannot be exercised by a 1-2 object toy fixture: the failure modes are specifically
about per-image variance and tail behavior across a realistic, dense reference (tens to
hundreds of objects per image, per the catkin trait). ``dense_records`` builds such a reference with
an exactly hand-verifiable per-image miss/false-positive pattern (deterministic, not randomized) so
every derived statistic (count_bias_mean/std/p90, precision/recall) can be reasoned about directly
from ``miss_pattern``/``fp_pattern`` rather than pinned by re-running the code under test.
"""

from __future__ import annotations


def _box(cx: float, cy: float, s: float = 20.0) -> list[float]:
    return [cx - s / 2, cy - s / 2, s, s]


def dense_records(
    n_images: int = 20,
    objects_per_image: int = 80,
    *,
    id_prefix: str = "d",
    shift: float = 0.0,
    score: float = 0.9,
    fp_score: float | None = None,
    width: int = 4000,
    height: int = 4000,
    spacing: float = 40.0,
    miss_pattern: list[int] | None = None,
    fp_pattern: list[int] | None = None,
) -> list[dict]:
    """``n_images`` records, each with ``objects_per_image`` GT laid out on a grid (spaced well
    outside the ~half-class-avg-size center-match tolerance these 20x20 boxes derive, so there is no
    accidental cross-matching between neighboring GT).

    Per image ``i``: the first ``miss_pattern[i]`` GT boxes get no matching detection (false
    negatives); every other GT gets an exactly-matching detection at ``score``; ``fp_pattern[i]``
    extra, unmatched detections (at ``fp_score``, defaulting to ``score``; pass a distinct, lower
    value to model a realistic detector whose spurious detections skew low-confidence) are added far
    outside the grid (false positives). Per-image bias is therefore exactly ``fp_pattern[i] -
    miss_pattern[i]``, tp = ``objects_per_image - miss_pattern[i]``, fn = ``miss_pattern[i]``,
    fp = ``fp_pattern[i]``, hand-verifiable without re-running the sweep. ``shift`` offsets every GT
    box's center by that many px; the matching detection's center is not shifted, so it stays put at
    the original (unshifted) grid position, leaving miss/FP placement geometry unaffected; used to
    give a holdout fixture content genuinely distinct from calibration's, for a content-overlap gate.
    A caller must not raise ``shift`` past the center-match tolerance without also shifting the
    paired detection to match: past that point every "match" becomes a miss (the shifted GT) plus an
    unmatched detection (the un-shifted one) instead of a true positive, silently turning a clean
    fixture into an fp+fn pair. At the shipped default ``shift=5.0`` this is well inside tolerance and
    harmless.
    """
    miss_pattern = list(miss_pattern) if miss_pattern is not None else [0] * n_images
    fp_pattern = list(fp_pattern) if fp_pattern is not None else [0] * n_images
    if len(miss_pattern) != n_images or len(fp_pattern) != n_images:
        raise ValueError("miss_pattern/fp_pattern must have exactly n_images entries")
    fp_score = score if fp_score is None else fp_score

    cols = int(objects_per_image**0.5) + 2
    far_row = (objects_per_image // cols) + 2
    records: list[dict] = []
    for i in range(n_images):
        gt: list[dict] = []
        dt: list[dict] = []
        for k in range(objects_per_image):
            row, col = divmod(k, cols)
            cx, cy = 50.0 + col * spacing, 50.0 + row * spacing
            gt.append({"category_id": 0, "bbox": _box(cx + shift, cy)})  # only GT shifts, not the det
            if k < miss_pattern[i]:
                continue
            dt.append({"category_id": 0, "bbox": _box(cx, cy), "score": score})
        for j in range(fp_pattern[i]):
            fx, fy = 50.0 + j * spacing, 50.0 + (far_row + i) * spacing
            dt.append({"category_id": 0, "bbox": _box(fx, fy), "score": fp_score})
        records.append({"width": width, "height": height, "image_id": f"{id_prefix}_{i}",
                        "gt": gt, "dt": dt})
    return records
