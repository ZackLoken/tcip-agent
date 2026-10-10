"""The conf-censoring guard of the count criterion, built around the floor the reference's
predictions were collected at.

The criterion never infers censorship from the reference's own observed minimum score: every
surviving score is at or above the collection floor by construction. The floor is stated
(``staged_conf_floor``), and the conf the objective picks at or below it is censored; a floor far
below every observed score is reported beside the criterion and never fails it.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")  # evaluation.py imports torch at module load

from tests import _trait_fixtures as fx  # noqa: E402
from tests._dense_op_fixtures import dense_records, good_cal_holdout  # noqa: E402
from tcip_mcp.pipelines.operating_point import _min_dt_score, count_criterion  # noqa: E402


def _criterion(cal, hold, floor):
    return count_criterion(cal, hold, fx.COUNT_SPEC, staged_conf_floor=floor,
                           staged_conf_floor_attribute_path="self")


def test_min_dt_score():
    recs = dense_records(n_images=2, objects_per_image=3, score=0.9)
    assert _min_dt_score(recs) == pytest.approx(0.9)
    assert _min_dt_score([{"gt": [], "dt": []}]) is None


# Correct detections score 0.9 and one spurious detection per image scores low, so the
# count-unbiased pick lands at 0.9, comfortably above a real 0.01 collection floor.

def test_a_reference_collected_at_a_real_floor_passes():
    conf, evidence, failures = _criterion(*good_cal_holdout(fp_score=0.05), 0.01)

    assert conf == pytest.approx(0.9)
    assert evidence["conf_floor_mismatch"] is False
    assert failures == []


def test_a_conf_picked_at_or_below_the_collection_floor_is_censored():
    """The floor sits at the picked conf, so the sweep could not have seen anything below it: the
    criterion fails even though the holdout bias is zero."""
    _conf, evidence, failures = _criterion(*good_cal_holdout(fp_score=0.05), 0.95)

    assert "conf_censored" in failures
    assert evidence["conf_floor_mismatch"] is False


def test_a_floor_far_below_every_observed_score_is_reported_and_never_fails():
    """The stated 0.01 floor, and detections that never go below 0.5: the gap is reported beside
    the criterion for a person to notice, and a pick genuinely above the floor still passes."""
    _conf, evidence, failures = _criterion(*good_cal_holdout(fp_score=0.5), 0.01)

    assert evidence["conf_floor_mismatch"] is True
    assert failures == []
