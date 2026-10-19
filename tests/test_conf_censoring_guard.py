"""The count criterion's conf-censoring guard, built around the reference's collection floor.

The floor is the one the reference's predictions were collected at.

The floor is stated (``staged_conf_floor``), and a conf the objective picks at or below it is
censored; a stated 0.01 floor beneath detections no lower than 0.5 is reported beside the
criterion's observed minimum and does not fail it.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip(
    "torch"
)  # evaluation.py imports torch at module load

from tcip_mcp.pipelines.operating_point import _min_dt_score, count_criterion  # noqa: E402
from tests import _trait_fixtures as fx  # noqa: E402
from tests._dense_op_fixtures import dense_records, good_cal_holdout  # noqa: E402
from tests._training_values import COUNT_TEST  # noqa: E402


def _criterion(
    cal: list[dict], hold: list[dict], floor: float
) -> tuple[float, dict, list[str]]:
    return count_criterion(
        cal,
        hold,
        fx.COUNT_SPEC,
        staged_conf_floor=floor,
        staged_conf_floor_attribute_path="self",
        **COUNT_TEST,
    )


def test_min_dt_score() -> None:
    recs = dense_records(n_images=2, objects_per_image=3, score=0.9)
    assert _min_dt_score(recs) == pytest.approx(0.9)
    assert _min_dt_score([{"gt": [], "dt": []}]) is None


# Correct detections score 0.9 and one spurious detection per image scores low, so the
# count-unbiased pick lands at 0.9, comfortably above a real 0.01 collection floor.


def test_a_reference_collected_at_a_real_floor_passes() -> None:
    conf, evidence, failures = _criterion(
        *good_cal_holdout(fp_score=0.05), 0.01
    )

    assert conf == pytest.approx(0.9)
    assert evidence["staged_conf_floor"] == 0.01
    assert failures == []


def test_a_conf_picked_at_or_below_the_collection_floor_is_censored() -> None:
    """A conf picked at or below the collection floor is censored.

    With the stated floor at the picked conf, the criterion reports ``conf_censored``.
    """
    _conf, _evidence, failures = _criterion(
        *good_cal_holdout(fp_score=0.05), 0.95
    )

    assert "conf_censored" in failures


def test_a_floor_far_below_every_observed_score_is_reported_and_never_fails() -> (
    None
):
    """A floor far below the observed scores is reported beside the criterion and does not fail it.

    With the stated 0.01 floor and detections no lower than 0.5, the evidence records both the
    floor and the observed minimum, and the pick above the floor passes.
    """
    _conf, evidence, failures = _criterion(
        *good_cal_holdout(fp_score=0.5), 0.01
    )

    assert evidence["staged_conf_floor"] == 0.01
    assert evidence["observed_min_score"] == pytest.approx(0.5)
    assert failures == []
