"""The holdout's center-match tolerance is scaled by the holdout's own GT object size.

``count_criterion`` derives one localization policy (a fraction of a characteristic object size)
from the calibration GT's nearest-neighbor spacing and applies that fraction to both sides, each
side's absolute pixel tolerance being the fraction of its own object size.

The fixtures give the two sides different object scales.
"""

from __future__ import annotations

import pytest

from tests import _trait_fixtures as fx
from tests._training_values import COUNT_TEST

pytest.importorskip("torch")

from tcip_mcp.pipelines.operating_point import count_criterion  # noqa: E402

ENTRY = fx.with_fields(
    fx.BUD_OPENING, count_bias_tolerance_frac=0.1, count_error_tolerance=1.0
)
N_IMAGES = 4
OBJECTS_PER_IMAGE = 6
TOTAL_OBJECTS = N_IMAGES * OBJECTS_PER_IMAGE
SPACING = 120.0
"""The center spacing of both sides, which sets the localization fraction the calibration side
derives: ``120 * 0.5 / 80 = 0.75`` of an object's size, 60 px at 80 px objects and 15 px at 20 px
ones."""
DET_OFFSET = 30.0


def _records(
    prefix: str, *, size: float, x0: float, det_offset: float
) -> list[dict]:
    """Return ``N_IMAGES`` records of square GT boxes, each beside one offset detection.

    The GT boxes are ``size`` px square, one row per image at ``SPACING`` px centers, each
    carrying a detection whose center sits ``det_offset`` px to the right of its own GT center.
    """
    recs = []
    for i in range(N_IMAGES):
        y = 100.0 + 10.0 * i
        gt, dt = [], []
        for k in range(OBJECTS_PER_IMAGE):
            x = x0 + SPACING * k
            gt.append(
                {"bbox": [x, y, size, size], "category_id": 1, "iscrowd": 0}
            )
            dt.append(
                {
                    "bbox": [x + det_offset, y, size, size],
                    "category_id": 1,
                    "score": 0.9,
                }
            )
        recs.append(
            {
                "image_id": f"{prefix}{i}",
                "width": 200000,
                "height": 2000,
                "gt": gt,
                "dt": dt,
                "count": len(dt),
                "cap": len(dt) + 1,
            }
        )
    return recs


def _criterion(cal, hold, entry=ENTRY):
    return count_criterion(
        cal,
        hold,
        entry,
        staged_conf_floor=0.05,
        staged_conf_floor_attribute_path=None,
        **COUNT_TEST,
    )


def test_a_holdout_of_smaller_objects_is_judged_at_its_own_object_scale():
    """A holdout of smaller objects is judged at its own object scale.

    Calibration carries 80 px objects, the holdout 20 px ones, and each holdout detection sits
    30 px from its GT center, inside the 60 px tolerance calibration's own scale earns.

    Judged at the holdout's scale, nothing on that side matches, and the reference fails the
    localization floor while the count bias reads zero and raises no bias failure: the holdout
    loses exactly as many detections as it gains false ones.
    """
    cal = _records("c", size=80.0, x0=0.0, det_offset=0.0)
    hold = _records("h", size=20.0, x0=100000.0, det_offset=DET_OFFSET)

    _conf, evidence, failures = _criterion(cal, hold)
    hb = evidence["holdout_at_conf"]

    # Calibration's own tolerance is twice the offset, so the same 30 px displacement is a hit
    # there.
    assert evidence["calibration_curve"]["criterion"][
        "tolerance"
    ] == pytest.approx(60.0)
    assert evidence["calibration_at_conf"]["tp"] == TOTAL_OBJECTS

    assert hb["tp"] == 0
    assert hb["fp"] == TOTAL_OBJECTS and hb["fn"] == TOTAL_OBJECTS
    assert hb["count_bias_mean_present"] == pytest.approx(
        0.0
    )  # the bias terms cannot see this
    assert "count_bias_exceeds_tolerance" not in failures
    assert "localization_quality_floor_failed" in failures


def test_the_same_displacement_passes_when_both_sides_carry_the_same_object_scale():
    """A 30 px displacement passes with 80 px objects on both sides.

    The holdout's true positives equal its object count with no false positive or miss, and
    the criterion reports no failure.
    """
    cal = _records("c", size=80.0, x0=0.0, det_offset=DET_OFFSET)
    hold = _records("h", size=80.0, x0=100000.0, det_offset=DET_OFFSET)

    _conf, evidence, failures = _criterion(cal, hold)
    hb = evidence["holdout_at_conf"]

    assert hb["tp"] == TOTAL_OBJECTS
    assert hb["fp"] == 0 and hb["fn"] == 0
    assert failures == []


def test_an_authored_floor_the_holdout_does_not_clear_fails_by_name():
    """A floor stricter than what the reference's own precision and recall reach fails by name."""
    cal = _records("c", size=80.0, x0=0.0, det_offset=0.0)
    hold = _records("h", size=20.0, x0=100000.0, det_offset=DET_OFFSET)

    _conf, _evidence, failures = _criterion(cal, hold)

    assert ENTRY.holdout_match_quality_floor == 0.5
    assert "localization_quality_floor_failed" in failures
