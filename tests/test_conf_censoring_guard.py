"""G1 conf-censoring guard — a display-filtered reference cannot be stamped ``validated``.

The GT calibration path floors the detector to ``score_thresh=0.01`` so the count-unbiased sweep
sees the low-conf tail. Predictions a human reviewed are staged at display conf (``DEFAULT_CONF``),
so a review-derived reference is truncated above the display floor: its count-unbiased point can pass
the disjoint + count-bias holdout gate at an artificially high conf and stamp
``VALIDATED_HELD_OUT`` — a measurement-integrity hole. ``resolve_operating_point`` now refuses a
validated claim whenever the reference's lowest detection score sits at/above the display floor.

The A/B fixtures below hold geometry fixed (1 GT + 1 matching det per image, disjoint cal/holdout,
zero count bias — a reference that WOULD pass the holdout gate) and vary ONLY the detection score
level, proving the guard is specific to conf-censoring and not a blanket refusal.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")  # evaluation.py imports torch at module load

from tcip_mcp.pipelines.operating_point import (  # noqa: E402
    _conf_censored,
    _min_dt_score,
    resolve_operating_point,
)
from tcip_mcp.pipelines.resolution import DEFAULT_CONF  # noqa: E402


def _box(cx: float, cy: float, s: float = 20.0) -> list[float]:
    return [cx - s / 2, cy - s / 2, s, s]  # xywh centered at (cx, cy)


def _records(idp: str, score: float, *, shift: float = 0.0) -> list[dict]:
    """Two images, each 1 GT + 1 exactly-matching det at ``score`` — zero count bias at any conf.

    ``shift`` (K1): offsets the GT (and matching det) center by that many px, well inside the
    center-match tolerance, so a holdout's GT content genuinely differs from calibration's — a
    holdout identical in content (differing only by ``image_id``) now trips the content-overlap
    gate, which would otherwise mask what this guard is specifically testing.
    """
    return [
        {"width": 400, "height": 400, "image_id": f"{idp}_a",
         "gt": [{"category_id": 0, "bbox": _box(100 + shift, 100)}],
         "dt": [{"category_id": 0, "bbox": _box(100, 100), "score": score}]},
        {"width": 400, "height": 400, "image_id": f"{idp}_b",
         "gt": [{"category_id": 0, "bbox": _box(200 + shift, 200)}],
         "dt": [{"category_id": 0, "bbox": _box(200, 200), "score": score}]},
    ]


# ── unit: the censoring predicate ─────────────────────────────────────────

def test_min_dt_score():
    assert _min_dt_score(_records("c", 0.9)) == pytest.approx(0.9)
    assert _min_dt_score([{"gt": [], "dt": []}]) is None  # no detections


def test_conf_censored_predicate():
    # min score at/above the display floor -> censored; below it -> uncensored.
    assert _conf_censored(_records("c", 0.9), DEFAULT_CONF) is True
    assert _conf_censored(_records("c", DEFAULT_CONF), DEFAULT_CONF) is True  # exactly the floor
    assert _conf_censored(_records("c", 0.4), DEFAULT_CONF) is False
    assert _conf_censored([], DEFAULT_CONF) is False  # empty reference is not "censored"
    assert _conf_censored(None, DEFAULT_CONF) is False


# ── the guard: high-conf-only reference cannot be stamped validated ───────

def test_high_conf_only_reference_refused_via_holdout_gate():
    # Disjoint cal + holdout, zero count bias -> WOULD pass the holdout gate; but every detection is
    # at 0.9 (>= display floor), so the sweep never saw the tail -> the guard refuses the claim.
    b = resolve_operating_point("catkin", dataset_hash="h1",
                                calibration_records=_records("c", 0.9),
                                holdout_records=_records("h", 0.9))
    conf = b.get("conf")
    assert conf.validated_vs_gt == "false"
    assert b.is_shippable is False
    sweep = conf.sweep or {}
    assert sweep["conf_censored"] is True
    assert sweep["disjoint"] is True          # the guard, not disjointness, is what refused it
    assert sweep["passed_holdout"] is False


def test_low_conf_tail_reference_still_validates():
    # Identical geometry (same GT/det placement, same disjoint holdout, same zero bias) but the
    # detections carry a sub-floor score, so the reference shows the tail and validation stands.
    b = resolve_operating_point("catkin", dataset_hash="h1",
                                calibration_records=_records("c", 0.4),
                                holdout_records=_records("h", 0.4, shift=3.0))
    conf = b.get("conf")
    assert conf.validated_vs_gt == "validated_held_out"
    assert b.is_shippable is True
    sweep = conf.sweep or {}
    assert sweep["conf_censored"] is False
    assert sweep["passed_holdout"] is True


def test_censored_cal_uncensored_holdout_still_refused():
    # Even if only the calibration reference is censored, the derived conf is untrustworthy -> refuse.
    b = resolve_operating_point("catkin", dataset_hash="h1",
                                calibration_records=_records("c", 0.9),
                                holdout_records=_records("h", 0.4))
    assert b.get("conf").validated_vs_gt == "false"
    assert b.is_shippable is False
