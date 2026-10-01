"""What the count criterion's evidence says about itself has to match what actually happened.

Every field here is read by someone reconstructing a delivered number: which derivation produced
the conf, and what the reference's own scores looked like against the floor it was collected at.
A record that describes a different derivation or a different bar than the one that ran is a
silent loss of auditability, with no wrong number visible anywhere to signal it.
"""

from __future__ import annotations

import pytest

from tests import _trait_fixtures as fx

pytest.importorskip("torch")

from tcip_mcp.pipelines.operating_point import count_criterion  # noqa: E402

N_IMAGES = 4
OBJECTS_PER_IMAGE = 8
ENTRY = fx.with_fields(fx.BUD_OPENING, count_bias_tolerance_frac=0.1, count_error_tolerance=1.0)


def _records(prefix: str, offset: float, n_images: int = N_IMAGES) -> list[dict]:
    """One record per image, every object matched exactly by one detection at score 0.9. Boxes sit
    100 px apart, well outside the tolerance this GT derives.
    """
    recs = []
    for i in range(n_images):
        gt, dt = [], []
        for k in range(OBJECTS_PER_IMAGE):
            box = [offset + 100.0 * k, 50.0 + 10.0 * i, 40.0, 40.0]
            gt.append({"bbox": box, "category_id": 1, "iscrowd": 0})
            dt.append({"bbox": box, "category_id": 1, "score": 0.9})
        recs.append({"image_id": f"{prefix}{i}", "width": 4000, "height": 1000, "gt": gt,
                     "dt": dt, "cap_hit": False})
    return recs


def test_every_conf_label_the_registered_pickers_can_record_has_a_registered_implementation():
    """The conf label is built at runtime from whichever picker ran; drive every registered count
    objective and check the labels those runs produced against the derivation registry, which
    exists so no data-sounding provenance string names a derivation nothing implements."""
    from tcip_mcp.pipelines.derivations import DERIVATION_IMPLEMENTATIONS
    from tcip_mcp.pipelines.operating_point import COUNT_OBJECTIVE_PICKERS

    labels = set()
    for objective in sorted(COUNT_OBJECTIVE_PICKERS):
        _conf, evidence, _failures = count_criterion(
            _records("c", 0.0), _records("h", 100000.0),
            fx.with_fields(ENTRY, count_objective=objective), staged_conf_floor=0.05,
            staged_conf_floor_attribute_path=None)
        labels.add(evidence["conf_derived_from"])

    assert sorted(lbl for lbl in labels if lbl not in DERIVATION_IMPLEMENTATIONS) == []
    assert len(labels) == 2  # two pickers


def test_a_floor_mismatch_across_the_reference_is_surfaced_and_never_fails_it():
    """The reference's lowest observed score sits right at the floor it was collected at only on
    the holdout; the mismatch reads over both sides and is evidence, never a failure."""
    cal = _records("c", 0.0)
    hold = _records("h", 100000.0)
    hold[0]["dt"].append({"bbox": [900000.0, 50.0, 40.0, 40.0], "category_id": 1, "score": 0.05})

    _conf, evidence, failures = count_criterion(cal, hold, ENTRY, staged_conf_floor=0.05,
                                                staged_conf_floor_attribute_path=None)

    assert evidence["observed_min_score"] == pytest.approx(0.05)
    assert evidence["staged_conf_floor"] == pytest.approx(0.05)
    assert evidence["conf_floor_mismatch"] is False
    assert failures == []

    _conf, evidence, _failures = count_criterion(
        cal, _records("h", 100000.0), ENTRY, staged_conf_floor=0.05,
        staged_conf_floor_attribute_path=None)
    assert evidence["observed_min_score"] == pytest.approx(0.9)
    assert evidence["conf_floor_mismatch"] is True
