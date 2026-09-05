"""The conf-censoring guard, built around a caller-asserted staging floor.

``resolve_operating_point`` does not infer censorship from the reference's own observed minimum
score against a hardcoded display floor (that predicate was tautologically true for the GT path,
since every surviving score is >= the calibration floor by construction, and missed a display-
floored reference whose observed scores merely happened to dip below the display constant once).
Instead the caller asserts ``staged_conf_floor``, the floor the reference's predictions were
actually generated/filtered at, and ``censored = staged_conf_floor is not None and chosen_conf <=
staged_conf_floor``. An unstated floor is a distinct, separately-named gate
(``conf_floor_unstated``), never folded into ``conf_censored``, so the two causes are never
reported under one name; ``_floor_mismatch`` reconciles a stated floor against the reference's own
observed minimum score as an independent, non-gating check.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")  # evaluation.py imports torch at module load

from tests._dense_op_fixtures import dense_records, good_cal_holdout  # noqa: E402
from tcip_mcp.pipelines.operating_point import (  # noqa: E402
    _conf_censored,
    _floor_mismatch,
    _min_dt_score,
    resolve_operating_point,
)

# No built-in traits: seed_bud_trait_spec (conftest.py) writes a real bud.yml into this
# test's pinned platform state root so resolve_operating_point("bud_opening", ...) keeps resolving by default.
pytestmark = pytest.mark.usefixtures("seed_bud_trait_spec")


# ── unit: the two independent censoring predicates ─────────────────────────

def test_conf_censored_predicate():
    assert _conf_censored(0.6, None) is False                # no assertion -> conf_floor_unstated's job
    assert _conf_censored(0.6, 0.01) is False                # picked conf comfortably above the floor
    assert _conf_censored(0.6, 0.6) is True                  # picked conf AT the floor
    assert _conf_censored(0.4, 0.5) is True                  # picked conf BELOW the floor


def test_min_dt_score():
    recs = dense_records(n_images=2, objects_per_image=3, score=0.9)
    assert _min_dt_score(recs) == pytest.approx(0.9)
    assert _min_dt_score([{"gt": [], "dt": []}]) is None


def test_floor_mismatch_predicate():
    recs = dense_records(n_images=2, objects_per_image=3, score=0.02)
    assert _floor_mismatch(recs, None) is False               # no assertion -> nothing to reconcile
    assert _floor_mismatch(recs, 0.01) is False                # observed 0.02 vs asserted 0.01: <=0.05 gap
    recs_high = dense_records(n_images=2, objects_per_image=3, score=0.4)
    assert _floor_mismatch(recs_high, 0.01) is True            # observed 0.4 vs asserted 0.01: >0.05 gap
    assert _floor_mismatch([], 0.01) is False                  # no detections at all -> nothing to reconcile


# ── the full gate: a dense, realistic reference ────────────────────────────
# Correct detections score 0.9; one spurious detection per image scores low (a realistic detector's
# false positives skew low-confidence) so the count-unbiased pick lands at 0.9 (bias vanishes once
# the low-score FP is filtered out), comfortably above a real 0.01 calibration floor, exercising the
# "genuinely floored reference must still validate" direction.

def test_reference_floored_at_the_real_calibration_floor_still_validates():
    # A genuinely floored, honestly-asserted reference must still be able to validate: a naive fix
    # could otherwise make validation permanently unreachable.
    cal, hold = good_cal_holdout(fp_score=0.05)
    # tiled=False: this test is about conf-calibration shippability, not tiling (tile_size
    # only gates a bundle when tiled).
    b = resolve_operating_point("bud_opening", dataset_hash="h1", calibration_records=cal,
                                holdout_records=hold, tiled=False, staged_conf_floor=0.01)
    conf = b.get("conf")
    assert conf._raw == pytest.approx(0.9)
    assert conf.validated_against == "held_out_annotations"
    assert b.is_shippable is True
    sweep = conf.gate_evidence
    assert sweep["conf_censored"] is False
    assert sweep["conf_floor_mismatch"] is False
    assert sweep["failures"] == []


def test_no_staged_conf_floor_asserted_fails_closed():
    # No floor asserted must fail closed (the honest default), named conf_floor_unstated,
    # distinct from conf_censored (a stated floor the pick does not clear).
    cal, hold = good_cal_holdout(fp_score=0.05)
    b = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h1", calibration_records=cal,
                                holdout_records=hold)
    conf = b.get("conf")
    assert conf.validated_against == "false"
    assert b.is_shippable is False
    assert conf.gate_evidence["conf_censored"] is False
    assert "conf_floor_unstated" in conf.gate_evidence["failures"]
    assert "conf_censored" not in conf.gate_evidence["failures"]


def test_reference_truncated_above_the_picked_conf_is_refused():
    # Same geometry as the passing case, but the asserted floor sits at the picked conf, so the
    # sweep could not have seen anything below it, and it must refuse even though the holdout bias is 0.
    cal, hold = good_cal_holdout(fp_score=0.05)
    b = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h1", calibration_records=cal,
                                holdout_records=hold, staged_conf_floor=0.95)
    conf = b.get("conf")
    assert conf.validated_against == "false"
    assert b.is_shippable is False
    sweep = conf.gate_evidence
    assert sweep["conf_censored"] is True
    assert sweep["conf_floor_mismatch"] is False   # isolates: this is the pick-vs-floor check, not the mismatch one
    assert "conf_censored" in sweep["failures"]


def test_asserted_vs_observed_floor_mismatch_is_surfaced_but_never_gates():
    # The caller asserts the real 0.01 calibration floor, but the reference's own detections never
    # actually go below 0.5, a material gap between the assertion and the data. This check is
    # non-gating provenance only: it is still computed and stamped on the sweep for a human/agent to
    # notice, but a pinned +/-0.05 band is an ordinary property of a model's score distribution as
    # often as it is evidence of tampering, so it must not by itself refuse a reference whose pick
    # (0.9) is genuinely above the floor and whose count bias otherwise passes cleanly.
    cal, hold = good_cal_holdout(fp_score=0.5)
    # tiled=False: this test is about conf-calibration shippability, not tiling (tile_size
    # only gates a bundle when tiled).
    b = resolve_operating_point("bud_opening", dataset_hash="h1", calibration_records=cal,
                                holdout_records=hold, tiled=False, staged_conf_floor=0.01)
    conf = b.get("conf")
    assert conf.validated_against == "held_out_annotations"
    assert b.is_shippable is True
    sweep = conf.gate_evidence
    assert sweep["conf_censored"] is False          # isolates: pick (0.9) is genuinely above the floor
    assert sweep["conf_floor_mismatch"] is True      # surfaced...
    assert "conf_floor_mismatch" not in sweep["failures"]  # ...but never a named (gating) failure
    assert sweep["failures"] == []
