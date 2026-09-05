"""Each class's count-bias tolerance is scaled by that class's own typical per-image count.

The relative count-bias tolerance is a fraction of a derived density. Which density is the whole
point of the per-class gate: on a multi-class reference the pooled per-image count is the sum over
every class, so judging a rare class against it hands that class a tolerance inflated by every other
class's objects, and a real, systematic miscount of the rare class clears the gate while the stamped
per-class provenance still describes the tolerance it should have been held to.

The fixtures here pair a dense class with a rare one. A single-class reference cannot distinguish
the two scales at all, because there the pooled density is that one class's density.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from tcip_mcp.pipelines.operating_point import resolve_operating_point  # noqa: E402
from tcip_mcp.pipelines.resolution import VALIDATED_FALSE, VALIDATED_HELD_OUT  # noqa: E402

pytestmark = pytest.mark.usefixtures("seed_bud_trait_spec")

N_IMAGES = 6
DENSE_PER_IMAGE = 250
RARE_PER_IMAGE = 10
SPURIOUS_PER_IMAGE = 2


def _records(prefix: str, offset: float, *, spurious: bool) -> list[dict]:
    """``N_IMAGES`` records carrying a dense class 1 and a rare class 2, every real object matched
    exactly by one detection.

    With ``spurious``, each image also carries ``SPURIOUS_PER_IMAGE`` extra class-2 detections far
    from any GT, so class 2 is over-counted by exactly that much on every image and class 1 is never
    wrong. Boxes sit 50 px apart, well outside the tolerance this GT derives, so no detection can
    match a neighbor.
    """
    recs = []
    for i in range(N_IMAGES):
        gt, dt = [], []
        for k in range(DENSE_PER_IMAGE):
            box = [offset + 50.0 * k, 100.0 + 10.0 * i, 20.0, 20.0]
            gt.append({"bbox": box, "category_id": 1})
            dt.append({"bbox": box, "category_id": 1, "score": 0.9})
        for k in range(RARE_PER_IMAGE):
            box = [offset + 50.0 * k, 900.0 + 10.0 * i, 20.0, 20.0]
            gt.append({"bbox": box, "category_id": 2})
            dt.append({"bbox": box, "category_id": 2, "score": 0.9})
        if spurious:
            for k in range(SPURIOUS_PER_IMAGE):
                dt.append({"bbox": [offset + 50.0 * k, 1500.0 + 10.0 * i, 20.0, 20.0],
                           "category_id": 2, "score": 0.9})
        recs.append({"image_id": f"{prefix}{i}", "width": 20000, "height": 3000, "gt": gt, "dt": dt})
    return recs


def _resolve(*, spurious: bool):
    return resolve_operating_point(
        "bud_opening", tiled=True, dataset_hash="h", staged_conf_floor=0.05,
        calibration_records=_records("c", 0.0, spurious=spurious),
        holdout_records=_records("h", 100000.0, spurious=spurious))


def test_a_rare_class_is_not_granted_the_dense_class_density_as_its_tolerance():
    """Class 2 carries 10 objects an image and is over-counted by 2 on every one of them, a 20%
    systematic miscount of that class. Class 1 carries 250 an image and is never wrong, so the
    pooled per-image density is 260 and the pooled bias of 2 is well inside the pooled tolerance:
    only the class's own density refuses this reference.
    """
    b = _resolve(spurious=True)
    sweep = b.params["conf"].gate_evidence
    hb = sweep["holdout_bias"]
    rare = hb["per_class"]["2"]

    assert sweep["pooled_typical_count"] == pytest.approx(260.0)
    assert sweep["per_class_typical_count"]["2"] == pytest.approx(float(RARE_PER_IMAGE))
    assert rare["n_present"] == N_IMAGES
    assert rare["count_bias_mean_present"] == pytest.approx(float(SPURIOUS_PER_IMAGE))
    assert hb["per_class"]["1"]["count_bias_mean_present"] == pytest.approx(0.0)

    # The pooled scope is blind here: it passes, so nothing but the per-class scope can refuse.
    assert "count_bias_exceeds_tolerance" not in sweep["failures"]

    # The stamped per-class tolerance is what the gate compared against, an order of magnitude
    # below the pooled one, and the class's measured bias is far outside it.
    stamped = sweep["per_class_count_bias_tolerance"]["2"]
    assert stamped == pytest.approx(1.0 / N_IMAGES)
    assert stamped < sweep["pooled_count_bias_tolerance"] / 10
    assert rare["count_bias_mean_present"] > stamped

    assert sweep["per_class_count_bias_failures"] == ["2"]
    assert "count_bias_exceeds_tolerance_per_class" in sweep["failures"]
    assert b.params["conf"].validated_against == VALIDATED_FALSE


def test_the_same_dense_and_rare_class_pairing_validates_when_every_class_is_honest():
    """The companion obligation: the same two-class density profile, counted correctly, must still
    earn its held-out stamp. A per-class tolerance that refused a rare class merely for being rare
    would block every legitimate multi-class reference.
    """
    b = _resolve(spurious=False)
    sweep = b.params["conf"].gate_evidence

    assert set(sweep["holdout_bias"]["per_class"]) == {"1", "2"}
    assert sweep["per_class_count_bias_failures"] == []
    assert sweep["failures"] == []
    assert b.params["conf"].validated_against == VALIDATED_HELD_OUT
