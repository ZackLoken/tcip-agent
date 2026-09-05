"""The holdout's center-match tolerance is scaled by the holdout's own GT object size.

``resolve_operating_point`` derives one localization policy (``loc_frac``, a fraction of a
characteristic object size) from the calibration GT's nearest-neighbor spacing and applies that same
fraction to both sides. The absolute pixel tolerance it produces still has to be multiplied by the
object scale of the side it is applied to. Borrowing calibration's average object size for the
holdout redefines what counts as a hit on data whose objects are a different size, which moves
tp/fp/fn and therefore the count bias the whole held-out gate rests on.

The fixtures here deliberately give the two sides different object scales, which is the only shape
in which the two derivations produce different numbers at all.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from tcip_mcp.pipelines.operating_point import resolve_operating_point  # noqa: E402
from tcip_mcp.pipelines.resolution import VALIDATED_FALSE, VALIDATED_HELD_OUT  # noqa: E402

pytestmark = pytest.mark.usefixtures("seed_bud_trait_spec")

N_IMAGES = 4
OBJECTS_PER_IMAGE = 6
TOTAL_OBJECTS = N_IMAGES * OBJECTS_PER_IMAGE
SPACING = 400.0
DET_OFFSET = 30.0


def _records(prefix: str, *, size: float, x0: float, det_offset: float) -> list[dict]:
    """``N_IMAGES`` records whose GT boxes are ``size`` px square, one row per image at ``SPACING``
    px centers, each carrying a detection whose center sits ``det_offset`` px to the right of its
    own GT center.

    ``SPACING`` is far larger than any tolerance either object scale here derives, so a detection
    can only ever match its own GT box, never a neighbor.
    """
    recs = []
    for i in range(N_IMAGES):
        y = 100.0 + 10.0 * i
        gt, dt = [], []
        for k in range(OBJECTS_PER_IMAGE):
            x = x0 + SPACING * k
            gt.append({"bbox": [x, y, size, size], "category_id": 1})
            dt.append({"bbox": [x + det_offset, y, size, size], "category_id": 1, "score": 0.9})
        recs.append({"image_id": f"{prefix}{i}", "width": 200000, "height": 2000,
                     "gt": gt, "dt": dt})
    return recs


def test_a_holdout_of_smaller_objects_is_judged_at_its_own_object_scale():
    """Calibration carries 80 px objects, the holdout 20 px ones, and every holdout detection sits
    30 px from its GT center: comfortably inside the 60 px tolerance calibration's own scale earns,
    and far outside the 15 px the holdout's own scale earns.

    Judged at the holdout's scale, nothing on that side matches, and the reference is refused on the
    localization floor. Nothing about the count bias reveals this: every image loses exactly as many
    detections as it gains false ones, so the pooled and per-class bias terms both read zero.
    """
    cal = _records("c", size=80.0, x0=0.0, det_offset=0.0)
    hold = _records("h", size=20.0, x0=100000.0, det_offset=DET_OFFSET)

    b = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h", staged_conf_floor=0.05,
                                calibration_records=cal, holdout_records=hold)
    sweep = b.params["conf"].gate_evidence
    hb = sweep["holdout_bias"]

    # Calibration's own tolerance is twice the offset, so the same 30 px displacement is a hit there.
    assert sweep["calibration"]["tolerance"] == pytest.approx(60.0)
    assert sweep["calibration"]["curve"][0]["tp"] == TOTAL_OBJECTS

    assert hb["tp"] == 0
    assert hb["fp"] == TOTAL_OBJECTS and hb["fn"] == TOTAL_OBJECTS
    assert hb["count_bias_mean_present"] == pytest.approx(0.0)  # the bias terms cannot see this
    assert "count_bias_exceeds_tolerance" not in sweep["failures"]
    assert "localization_quality_floor_failed" in sweep["failures"]
    assert b.params["conf"].validated_against == VALIDATED_FALSE


def test_the_same_displacement_validates_when_both_sides_carry_the_same_object_scale():
    """The companion obligation: a 30 px displacement against 80 px objects is a genuine hit on both
    sides, and such a reference must still earn its held-out stamp.
    """
    cal = _records("c", size=80.0, x0=0.0, det_offset=DET_OFFSET)
    hold = _records("h", size=80.0, x0=100000.0, det_offset=DET_OFFSET)

    b = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h", staged_conf_floor=0.05,
                                calibration_records=cal, holdout_records=hold)
    sweep = b.params["conf"].gate_evidence
    hb = sweep["holdout_bias"]

    assert hb["tp"] == TOTAL_OBJECTS
    assert hb["fp"] == 0 and hb["fn"] == 0
    assert sweep["failures"] == []
    assert b.params["conf"].validated_against == VALIDATED_HELD_OUT


def test_a_trait_with_no_authored_floor_refuses_to_validate(tmp_path):
    """None means not yet authored: the gate refuses rather than substituting a floor of its own,
    even over a reference whose matches are otherwise perfect."""
    import dataclasses

    from tests._operationalization_fixtures import write_spec
    from tests._trait_fixtures import BUD_OPENING

    write_spec(tmp_path, dataclasses.replace(
        BUD_OPENING, name="no_floor_trait", holdout_match_quality_floor=None))
    cal = _records("c", size=80.0, x0=0.0, det_offset=DET_OFFSET)
    hold = _records("h", size=80.0, x0=100000.0, det_offset=DET_OFFSET)

    b = resolve_operating_point("no_floor_trait", tiled=True, dataset_hash="h",
                                staged_conf_floor=0.05, calibration_records=cal,
                                holdout_records=hold)
    sweep = b.params["conf"].gate_evidence

    assert sweep["holdout_match_quality_floor"] is None
    assert "holdout_match_quality_floor_unauthored" in sweep["failures"]
    assert b.params["conf"].validated_against == VALIDATED_FALSE


def test_an_authored_floor_the_holdout_does_not_clear_refuses():
    """A floor stricter than what the reference's own precision/recall reach refuses by name,
    distinct from the unauthored case."""
    cal = _records("c", size=80.0, x0=0.0, det_offset=0.0)
    hold = _records("h", size=20.0, x0=100000.0, det_offset=DET_OFFSET)

    b = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h", staged_conf_floor=0.05,
                                calibration_records=cal, holdout_records=hold)
    sweep = b.params["conf"].gate_evidence

    assert sweep["holdout_match_quality_floor"] == 0.5
    assert "localization_quality_floor_failed" in sweep["failures"]
    assert "holdout_match_quality_floor_unauthored" not in sweep["failures"]
