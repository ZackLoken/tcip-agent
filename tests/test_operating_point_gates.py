"""Coverage for the gate conditions in ``resolve_operating_point``: dispersion and
localization-quality floors, reference-sufficiency and equivalence criteria, pick-then-label plus
registry-driven objective, exact-conf holdout evaluation (not a nearest-neighbor snap),
cap-saturation provenance (non-gating), and the named-failure architecture. Plus an end-to-end
integration fixture: a realistic dense bud reference reaching ``VALIDATED_HELD_OUT`` with every
gate applied together.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")  # evaluation.py imports torch at module load

from tests._dense_op_fixtures import _box, dense_records  # noqa: E402
from tcip_mcp.pipelines.operating_point import (  # noqa: E402
    _cap_saturated_frac,
    _current_detections_cap,
    resolve_operating_point,
)
from tcip_mcp.pipelines.resolution import VALIDATED_REVIEW_CONFIRMED  # noqa: E402
from tcip_mcp.traits import COUNT_UNBIASED, DETECTION_F1, PRESENCE, TraitSpec  # noqa: E402
from tests._trait_fixtures import BUD_OPENING  # noqa: E402

# No built-in traits: seed_bud_trait_spec (conftest.py) writes a real bud_opening.yml into this
# test's pinned platform state root so resolve_operating_point("bud_opening", ...) keeps resolving by default.
pytestmark = pytest.mark.usefixtures("seed_bud_trait_spec")


def _ann(cx, cy, cid=0, score=None):
    a = {"category_id": cid, "bbox": _box(cx, cy)}
    if score is not None:
        a["score"] = score
    return a


def _records(idp="c", *, shift: float = 0.0):
    """The small (2-image) sparse fixture: count-unbiased conf 0.6, real per-image variance
    ([+1, -1] bias at that conf). Reused here, not as a fixture that must validate (the equivalence
    criterion correctly refuses it, see test_n_equals_2 below), but to exercise the
    reference-sufficiency checks precisely, where its small size and known bias distribution are
    exactly the point.
    """
    a = {"width": 400, "height": 400, "image_id": f"{idp}_a",
         "gt": [_ann(100 + shift, 100)],
         "dt": [_ann(100, 100, score=0.9), _ann(300, 300, score=0.6)]}
    b = {"width": 400, "height": 400, "image_id": f"{idp}_b",
         "gt": [_ann(100 + shift, 100), _ann(200 + shift, 200)],
         "dt": [_ann(100, 100, score=0.9), _ann(200, 200, score=0.3)]}
    return [a, b]


# ── Exact-conf holdout evaluation, not a nearest-neighbor snap ─────────────
#
# Every other holdout fixture in this module gives the holdout its own detection scores drawn from
# the same set calibration used, so the holdout's own auto-built conf grid (``derive_operating_point_curve``
# with no explicit ``conf_grid``) already contains the calibration-picked conf exactly, meaning the
# deleted nearest-neighbor snap (``count_bias_at``, no longer a symbol anywhere; reconstructed below
# as ``_old_nearest_neighbor_bias`` to prove the two approaches differ) and the current exact-conf
# call (``derive_operating_point_curve(holdout_records, conf_grid=[conf])``) would land on the identical
# curve point in every one of those tests. These two fixtures instead give the holdout a sparse
# detection-score set that deliberately excludes the calibration-picked conf (0.9), so the nearest
# grid point the old snap would have found is a genuinely different threshold with genuinely
# different tp/fp/fn: the scenario this exact-conf evaluation exists to catch.

def _cal_picks_conf_point_nine():
    """20-image dense calibration reference whose count-unbiased pick is exactly 0.9 (one low-conf
    spurious detection per image, filtered out once conf crosses its 0.05 score), the same
    pattern ``tests/_dense_op_fixtures.py``'s ``good_cal_holdout`` uses."""
    n, obj = 20, 80
    return dense_records(n_images=n, objects_per_image=obj, id_prefix="c",
                         miss_pattern=[0] * n, fp_pattern=[1] * n, score=0.9, fp_score=0.05)


def _old_nearest_neighbor_bias(holdout_records, tolerance, conf):
    """Reconstruction of the deleted ``count_bias_at``: the curve entry nearest ``conf`` on the
    holdout's own auto-built grid, not an exact evaluation at ``conf`` itself."""
    from tcip_mcp.pipelines.training.evaluation import derive_operating_point_curve

    sweep = derive_operating_point_curve(holdout_records, tolerance=tolerance)
    return min(sweep["curve"], key=lambda c: abs(c["conf"] - conf))


def test_exact_conf_eval_catches_a_catastrophic_bias_the_old_snap_would_have_missed():
    """Direction 1: the old snap would have misled, reading a validated-looking zero bias off a
    holdout whose true bias, at the conf that will actually ship, is catastrophic.

    Holdout detections all score 0.05 (well below the calibration-picked 0.9) with zero false
    positives. Evaluated exactly at 0.9, every detection is filtered out -> total miss, bias -80/image
    (80 objects/image). The old snap, with no holdout score anywhere near 0.9, would find its own
    grid's nearest point at 0.05 (closer to 0.9 than the grid's other point, 0.0) -- at which every
    detection survives and the bias reads as a perfect 0.0.
    """
    from tcip_mcp.pipelines.training.evaluation import gt_class_avg_size

    n, obj = 20, 80
    cal = _cal_picks_conf_point_nine()
    hold = dense_records(n_images=n, objects_per_image=obj, id_prefix="hA", shift=5.0,
                         miss_pattern=[0] * n, fp_pattern=[0] * n, score=0.05)
    assert {d["score"] for r in hold for d in r["dt"]} == {0.05}  # sparse, excludes the picked 0.9

    tol = 0.5 * gt_class_avg_size(hold)
    old = _old_nearest_neighbor_bias(hold, tol, 0.9)
    assert old["conf"] == pytest.approx(0.05)         # snapped to the nearest grid point, not 0.9
    assert old["count_bias_mean"] == pytest.approx(0.0)  # ...which misleadingly reads as unbiased

    b = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h1", calibration_records=cal,
                                holdout_records=hold, staged_conf_floor=0.01)
    conf = b.get("conf")
    hb = conf.gate_evidence["holdout_bias"]
    assert hb["conf"] == pytest.approx(0.9)           # evaluated at the conf that will actually ship
    assert hb["count_bias_mean"] == pytest.approx(-80.0)  # the TRUE bias at that conf: total miss
    assert hb["count_bias_mean"] != pytest.approx(old["count_bias_mean"])  # the two approaches differ
    assert conf.validated_against == "false"            # exact eval correctly refuses...
    assert "count_bias_exceeds_tolerance" in conf.gate_evidence["failures"]  # ...the old snap would not have


def test_exact_conf_eval_admits_a_reference_the_old_snap_would_have_unfairly_failed():
    """Direction 2: the old snap would have unfairly refused a reference the exact evaluation
    correctly admits.

    Holdout true-match detections score 0.99 (above 0.9); its false positives score 0.89 (just below
    0.9). Evaluated exactly at 0.9, the false positives are filtered out and the true matches survive
    -> zero bias, clean pass. The old snap's nearest grid point to 0.9 is 0.89 (closer than 0.99) --
    at which the false positives also survive, reading as a +2.0/image overcount that exceeds
    bud_opening's count-bias tolerance regardless of the exact value.
    """
    from tcip_mcp.pipelines.training.evaluation import gt_class_avg_size

    n, obj = 20, 80
    cal = _cal_picks_conf_point_nine()
    hold = dense_records(n_images=n, objects_per_image=obj, id_prefix="hB", shift=5.0,
                         miss_pattern=[0] * n, fp_pattern=[2] * n, score=0.99, fp_score=0.89)
    assert {d["score"] for r in hold for d in r["dt"]} == {0.99, 0.89}  # excludes the picked 0.9

    tol = 0.5 * gt_class_avg_size(hold)
    old = _old_nearest_neighbor_bias(hold, tol, 0.9)
    assert old["conf"] == pytest.approx(0.89)          # snapped to the nearest grid point, not 0.9
    assert old["count_bias_mean"] == pytest.approx(2.0)  # ...which reads as an over-tolerance bias

    b = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h1", calibration_records=cal,
                                holdout_records=hold, staged_conf_floor=0.01)
    conf = b.get("conf")
    hb = conf.gate_evidence["holdout_bias"]
    assert hb["conf"] == pytest.approx(0.9)            # evaluated at the conf that will actually ship
    assert hb["count_bias_mean"] == pytest.approx(0.0)  # the TRUE bias at that conf: clean
    assert hb["count_bias_mean"] != pytest.approx(old["count_bias_mean"])  # the two approaches differ
    assert conf.validated_against == "held_out_annotations"  # exact eval correctly admits...
    assert conf.gate_evidence["failures"] == []                  # ...what the old snap would have refused


# ── Dispersion and localization-quality floor ──────────────────────────────

def _tp_zero_bias_zero_records(id_prefix: str, *, n_images: int = 10, objects_per_image: int = 50):
    """Every image: N GT, N detections, but every detection sits far outside the center-match
    tolerance, so count bias is exactly 0 (fp == fn == N) while not one detection actually matches
    (tp=0). The degenerate case a naive count-bias-only gate cannot catch."""
    records = []
    cols = int(objects_per_image**0.5) + 2
    for i in range(n_images):
        gt, dt = [], []
        for k in range(objects_per_image):
            row, col = divmod(k, cols)
            cx, cy = 50.0 + col * 40, 50.0 + row * 40
            gt.append({"category_id": 0, "bbox": _box(cx, cy)})
            dt.append({"category_id": 0, "bbox": _box(cx + 100, cy), "score": 0.9})
        records.append({"width": 4000, "height": 4000, "image_id": f"{id_prefix}_{i}",
                        "gt": gt, "dt": dt})
    return records


def test_tp_zero_bias_zero_holdout_fails_the_localization_floor():
    cal = _tp_zero_bias_zero_records("c")
    hold = _tp_zero_bias_zero_records("h")
    b = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h1", calibration_records=cal,
                                holdout_records=hold, staged_conf_floor=0.0)
    conf = b.get("conf")
    hb = conf.gate_evidence["holdout_bias"]
    assert hb["tp"] == 0
    assert hb["count_bias_mean"] == pytest.approx(0.0)  # the degenerate case: bias vanishes...
    assert conf.validated_against == "false"               # ...but this must not pass silently
    assert "localization_quality_floor_failed" in conf.gate_evidence["failures"]


def test_dispersion_gate_skipped_when_unauthored_gates_when_authored(monkeypatch):
    import tcip_mcp.pipelines.operating_point as OP

    n_images, objects_per_image = 10, 50
    # One image drops 10 objects, none elsewhere -> mean bias -1.0, but the p90 tail is 1.0 (driven
    # by that single bad image among many good ones) -- exactly the "one bad plant among many"
    # scenario a population mean/SE alone can hide.
    miss = [0] * (n_images - 1) + [10]
    fp = [0] * n_images
    cal = dense_records(n_images=n_images, objects_per_image=objects_per_image, id_prefix="c",
                        miss_pattern=[0] * n_images, fp_pattern=[0] * n_images, score=0.9)
    hold = dense_records(n_images=n_images, objects_per_image=objects_per_image, id_prefix="h",
                         shift=5.0, miss_pattern=miss, fp_pattern=fp, score=0.9)

    strict = TraitSpec(name="bud_opening", count_objective=COUNT_UNBIASED, count_error_tolerance=0.5,
                       count_bias_tolerance_frac=1.0, delivers=BUD_OPENING.delivers)
    monkeypatch.setattr(OP, "get_trait", lambda name: strict)
    b_strict = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h1", calibration_records=cal,
                                       holdout_records=hold, staged_conf_floor=0.01)
    strict_sweep = b_strict.get("conf").gate_evidence
    assert strict_sweep["holdout_bias"]["count_error_p90"] == pytest.approx(1.0)
    assert "count_error_dispersion_too_high" in strict_sweep["failures"]

    # The same fixture, under a trait that has never authored count_error_tolerance (BUD_OPENING), the
    # dispersion term is skipped entirely, not gated on a platform-invented number.
    monkeypatch.setattr(OP, "get_trait", lambda name: BUD_OPENING)
    b_default = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h1", calibration_records=cal,
                                        holdout_records=hold, staged_conf_floor=0.01)
    default_sweep = b_default.get("conf").gate_evidence
    assert default_sweep["count_error_tolerance"] is None
    assert "count_error_dispersion_too_high" not in default_sweep["failures"]


def test_count_bias_tolerance_frac_source_platform_default_vs_trait(monkeypatch):
    """TraitSpec.count_bias_tolerance_frac, when unauthored (None, BUD_OPENING's own state), resolves to
    the platform's interim default fraction and stamps that provenance; a trait that authors its own
    value stamps ``"trait"`` instead, mirroring classifier_agreement_floor's own kappa_floor_source."""
    import tcip_mcp.pipelines.operating_point as OP

    n_images, objects_per_image = 10, 50
    cal = dense_records(n_images=n_images, objects_per_image=objects_per_image, id_prefix="c",
                        miss_pattern=[0] * n_images, fp_pattern=[0] * n_images, score=0.9)
    hold = dense_records(n_images=n_images, objects_per_image=objects_per_image, id_prefix="h",
                         shift=5.0, miss_pattern=[0] * n_images, fp_pattern=[0] * n_images, score=0.9)

    monkeypatch.setattr(OP, "get_trait", lambda name: BUD_OPENING)
    b_default = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h1", calibration_records=cal,
                                        holdout_records=hold, staged_conf_floor=0.01)
    default_sweep = b_default.get("conf").gate_evidence
    assert default_sweep["count_bias_tolerance_frac"] == pytest.approx(0.01)
    assert default_sweep["count_bias_tolerance_frac_source"] == "default"

    authored = TraitSpec(name="bud_opening", count_objective=COUNT_UNBIASED,
                         count_bias_tolerance_frac=0.2, delivers=BUD_OPENING.delivers)
    monkeypatch.setattr(OP, "get_trait", lambda name: authored)
    b_trait = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h1", calibration_records=cal,
                                      holdout_records=hold, staged_conf_floor=0.01)
    trait_sweep = b_trait.get("conf").gate_evidence
    assert trait_sweep["count_bias_tolerance_frac"] == pytest.approx(0.2)
    assert trait_sweep["count_bias_tolerance_frac_source"] == "trait"


# ── Reference-sufficiency and equivalence criterion ─────────────────────────

def test_all_negative_calibration_or_holdout_refused():
    real = _records("c")
    all_negative = [{"width": 400, "height": 400, "image_id": f"n_{i}", "gt": [], "dt": []}
                    for i in range(3)]

    b1 = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h1", calibration_records=all_negative,
                                 holdout_records=real, staged_conf_floor=0.0)
    assert "insufficient_calibration_gt" in b1.get("conf").gate_evidence["failures"]

    b2 = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h1", calibration_records=real,
                                 holdout_records=all_negative, staged_conf_floor=0.0)
    assert "insufficient_holdout_gt" in b2.get("conf").gate_evidence["failures"]


def test_single_image_holdout_fails_the_non_degeneracy_floor_alone():
    hold_one = [_records("h", shift=3.0)[0]]
    b = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h1", calibration_records=_records("c"),
                                holdout_records=hold_one, staged_conf_floor=0.3)
    sweep = b.get("conf").gate_evidence
    assert sweep["holdout_bias"]["n_images"] == 1
    assert "insufficient_holdout_images" in sweep["failures"]


def test_n_equals_2_holdout_with_real_variance_fails_equivalence_not_just_degeneracy():
    # n=2 clears the non-degeneracy floor but the mean+SE equivalence criterion still correctly
    # refuses it: a bare mean check would have passed this, since the per-image biases [+1, -1]
    # cancel exactly in the mean.
    b = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h1", calibration_records=_records("c"),
                                holdout_records=_records("h", shift=3.0), staged_conf_floor=0.3)
    sweep = b.get("conf").gate_evidence
    assert sweep["holdout_bias"]["count_bias_mean"] == pytest.approx(0.0)
    assert "insufficient_holdout_images" not in sweep["failures"]
    assert "count_bias_exceeds_tolerance" in sweep["failures"]


def test_zero_verdict_padding_cannot_dilute_the_gate_but_the_predicate_still_refuses_it(monkeypatch):
    """Padding a reference with zero-verdict, unadjudicated records must not make its own
    statistics easier to pass by silently diluting the variance denominator; wired here with a
    synthetic adjudication-coverage predicate standing in for a real one.

    The statistics themselves are immune to it: the equivalence test measures over the images that
    carry something, so eight empty records cannot shrink the standard error of a bias measured on
    two, and the same real n=2 disagreement (see
    ``test_n_equals_2_holdout_with_real_variance_fails_equivalence_not_just_degeneracy``) is refused
    with the padding exactly as it is without it. The coverage predicate is what names *why* such a
    reference is illegitimate rather than leaving it to be caught by its numbers, and it is a gate,
    never a filter that recomputes on a shrunk sample (a filter is itself fail-open: the excluded set
    correlates with the very quantity being measured, per ``resolve_operating_point``'s own
    docstring). The padding is never dropped before the statistics are computed (``n_images`` stays
    10, the full unfiltered set); the reference is refused outright because the coverage requirement
    failed.

    This fixture's own GT is sparse (typical count 1.5/image), so at the default 1% relative
    tolerance the derived tolerance is floor-dominated, an unrelated magnitude effect this test does
    not exist to demonstrate. The trait's fraction is loosened for this test only (mirrors
    ``test_dispersion_gate_skipped_when_unauthored_gates_when_authored``'s own pattern) so the
    padding's effect on the population is isolated from it.
    """
    import dataclasses

    import tcip_mcp.pipelines.operating_point as OP

    monkeypatch.setattr(OP, "get_trait", lambda name: dataclasses.replace(
        BUD_OPENING, count_bias_tolerance_frac=1.0))

    hold_real = _records("h", shift=3.0)  # n=2, real per-image variance ([+1, -1])
    padding = [{"width": 400, "height": 400, "image_id": f"h_pad_{i}", "gt": [], "dt": [],
               "padded": True} for i in range(8)]

    # Padded or not, the same real disagreement gets the same verdict: the padding carries no
    # evidence about count bias and so cannot buy any statistical confidence.
    b_padded = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h1", calibration_records=_records("c"),
                                       holdout_records=hold_real + padding, staged_conf_floor=0.3)
    b_bare = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h1", calibration_records=_records("c"),
                                     holdout_records=hold_real, staged_conf_floor=0.3)
    assert b_padded.get("conf").gate_evidence["holdout_bias"]["n_images"] == 10
    assert b_padded.get("conf").gate_evidence["holdout_bias"]["n_present"] == 2
    assert b_padded.get("conf").validated_against == "false"
    assert b_padded.get("conf").gate_evidence["failures"] == b_bare.get("conf").gate_evidence["failures"]
    assert "count_bias_exceeds_tolerance" in b_padded.get("conf").gate_evidence["failures"]

    # With the seam wired to a predicate that flags the padding as uncovered, the whole reference is
    # refused: statistics are still computed over the full, unfiltered 10-image set (never a
    # filter-then-recompute on a shrunk sample), and the failure now names the coverage requirement
    # rather than leaving the illegitimacy to be inferred from the numbers.
    covered = lambda r: not r.get("padded")  # noqa: E731
    b_covered = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h1", calibration_records=_records("c"),
                                        holdout_records=hold_real + padding, staged_conf_floor=0.3,
                                        adjudication_covered=covered)
    sweep = b_covered.get("conf").gate_evidence
    assert sweep["holdout_bias"]["n_images"] == 10  # unfiltered, a gate, not a filter
    assert sweep["adjudication_covered"] is False
    assert b_covered.get("conf").validated_against == "false"
    assert "insufficient_adjudication_coverage" in sweep["failures"]


# ── Pick-then-label and registry-driven objective ───────────────────────────

def test_detection_f1_objective_picks_f1_max_and_labels_it_accordingly(monkeypatch):
    import tcip_mcp.pipelines.operating_point as OP

    f1_trait = TraitSpec(name="bud_opening", count_objective=DETECTION_F1, delivers=BUD_OPENING.delivers)
    monkeypatch.setattr(OP, "get_trait", lambda name: f1_trait)

    b = OP.resolve_operating_point("bud_opening", tiled=True, dataset_hash="h1", calibration_records=_records())
    conf = b.get("conf")
    assert conf._raw == pytest.approx(0.0)  # F1-max pick for this fixture (recall-max, low conf)
    assert conf.derived_from == "F1-max center-match curve"


def test_presence_objective_deliberately_shares_the_f1_max_picker_and_label(monkeypatch):
    import tcip_mcp.pipelines.operating_point as OP

    presence_trait = TraitSpec(name="bud_opening", count_objective=PRESENCE, delivers=BUD_OPENING.delivers)
    monkeypatch.setattr(OP, "get_trait", lambda name: presence_trait)

    b = OP.resolve_operating_point("bud_opening", tiled=True, dataset_hash="h1", calibration_records=_records())
    conf = b.get("conf")
    assert conf._raw == pytest.approx(0.0)
    assert conf.derived_from == "F1-max center-match curve"  # same label as DETECTION_F1, deliberately


def test_f1_max_label_gets_the_review_suffix_when_review_confirmed(monkeypatch):
    import tcip_mcp.pipelines.operating_point as OP

    f1_trait = TraitSpec(name="bud_opening", count_objective=DETECTION_F1, delivers=BUD_OPENING.delivers)
    monkeypatch.setattr(OP, "get_trait", lambda name: f1_trait)

    b = OP.resolve_operating_point("bud_opening", tiled=True, dataset_hash="h1", calibration_records=_records(),
                                   validated_reference=VALIDATED_REVIEW_CONFIRMED)
    assert b.get("conf").derived_from == "F1-max center-match curve over review verdicts"


# ── Cap-saturation provenance (non-gating) ───────────────────────────────────

def test_current_detections_cap_reads_the_in_model_value():
    from types import SimpleNamespace
    m = SimpleNamespace(detector=SimpleNamespace(roi_heads=SimpleNamespace(detections_per_img=250)))
    assert _current_detections_cap(m) == 250


def test_cap_hit_stamped_by_records_from_detector():
    from tcip_mcp.pipelines.training.evaluation import records_from_detector

    target = {"boxes": torch.zeros((3, 4)), "labels": torch.zeros(3, dtype=torch.long), "image_id": "x"}
    output = {"boxes": torch.zeros((3, 4)), "labels": torch.zeros(3, dtype=torch.long),
             "scores": torch.tensor([0.9, 0.8, 0.7])}
    saturated = records_from_detector(target, output, width=10, height=10, detections_cap=3)
    assert saturated["cap_hit"] is True  # 3 raw detections >= cap of 3

    not_saturated = records_from_detector(target, output, width=10, height=10, detections_cap=10)
    assert not_saturated["cap_hit"] is False

    unknown_cap = records_from_detector(target, output, width=10, height=10)
    assert "cap_hit" not in unknown_cap


def test_cap_saturated_frac_excludes_records_with_no_flag():
    assert _cap_saturated_frac([{"cap_hit": True}, {"cap_hit": False}, {"no_flag": True}]) == pytest.approx(0.5)
    assert _cap_saturated_frac([{"no_flag": True}]) is None
    assert _cap_saturated_frac([]) is None
    assert _cap_saturated_frac(None) is None


def test_cap_saturation_is_surfaced_but_never_gates():
    cal = [dict(r, cap_hit=True) for r in _records("c")]  # every calibration record hit the cap
    hold = _records("h", shift=3.0)
    b = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h1", calibration_records=cal,
                                holdout_records=hold, staged_conf_floor=0.3)
    sweep = b.get("conf").gate_evidence
    assert sweep["calibration_cap_saturated_frac"] == pytest.approx(1.0)
    assert not any("cap" in f for f in sweep["failures"])  # non-gating: never a named failure


# ── Cross-cutting: named-failure architecture ───────────────────────────────

def test_passed_holdout_is_exactly_the_absence_of_named_failures():
    b = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h1", calibration_records=_records("c"),
                                holdout_records=_records("h", shift=3.0), staged_conf_floor=0.3)
    sweep = b.get("conf").gate_evidence
    assert sweep["passed_holdout"] == (not sweep["failures"])
    assert sweep["failures"]  # this fixture is a known-failing one (see test_n_equals_2 above)


# ── Genuine per-image dispersion in the admitting direction ─────────────────
#
# Every prior "good detector, should validate" fixture in this module (here and in
# test_operating_point.py / test_conf_censoring_guard.py / test_detection_measurement_integrity.py /
# test_char_goldens_measurement.py / test_calibration_holdout_disjointness.py / test_review_calibration.py /
# this file's own integration fixture above) has exactly zero per-image
# count-bias variance at the picked conf: a uniform miss/fp pattern, or a varying one whose
# spurious detections get filtered out identically on every image, so the
# ``abs(mean) + 1.645*SE <= tolerance`` equivalence criterion collapses to the bare mean check it
# replaced and is never actually exercised with real noise in the admitting direction.
#
# This fixture is different: every image gets a genuinely different miss count (1-5, cycling
# deterministically, not random, not uniform) representing a realistic ~97% recall detector, and a
# different false-positive count too, at the same high score as the true detections (0.9) so the
# false positives survive whatever conf gets picked, same as the true detections: spurious
# detections here are genuinely confusable, not filtered-out noise. The false-positive pattern is a
# rotation of the same miss values (same population, so the two means match exactly), which zeroes
# the systematic mean bias while leaving real, hand-traceable per-image dispersion behind (bias =
# fp[i] - miss[i] varies image-by-image even though sum(fp) == sum(miss)).
#
# Verified empirically against the actual code: at n=40 images / 100 objects/image this reaches
# count_bias_std ~= 2.03 and count_error_p90 = 4.0 (recall/precision ~0.97) against bud_opening's
# count-bias tolerance of 1.0 (the derived value at this reference's density:
# the platform's interim default 0.01 count_bias_tolerance_frac, bud_opening has not authored its own,
# times this fixture's 100-objects/image typical count),
# and correctly reaches VALIDATED_HELD_OUT. The identical per-image pattern at n=10 (fewer images,
# nothing else different) is correctly refused: the SE term grows enough that the equivalence
# criterion no longer clears the tolerance. Reference size is a real constraint the gate imposes on
# a genuinely noisy detector, not a formality: this is the accepted, known consequence of the gate,
# not every size validating the same detector.

def _rotating_noise_pattern(n: int, *, low: int = 1, high: int = 5, offset: int = 13
                            ) -> tuple[list[int], list[int]]:
    """``n`` per-image miss counts cycling deterministically through ``[low, high]`` (mean
    ``(low+high)/2``), and a false-positive pattern that is a rotation of the same values (the same
    population, so the two means match exactly and mean bias is exactly 0) offset by an amount that
    does not evenly divide ``n``, so it does not track the miss pattern image-by-image. Genuine,
    hand-traceable per-image variance (bias[i] = fp[i] - miss[i] differs per image), not randomness
    and not a uniform toy pattern.
    """
    span = high - low + 1
    miss = [low + (i * 3 + 1) % span for i in range(n)]
    fp = [miss[(i + offset) % n] for i in range(n)]
    return miss, fp


def test_realistic_dense_detector_with_genuine_per_image_dispersion_validates_at_n_equals_40():
    n, obj = 40, 100
    miss, fp = _rotating_noise_pattern(n)
    cal = dense_records(n_images=n, objects_per_image=obj, id_prefix="c",
                        miss_pattern=miss, fp_pattern=fp, score=0.9, fp_score=0.9)
    hold = dense_records(n_images=n, objects_per_image=obj, id_prefix="h", shift=5.0,
                         miss_pattern=miss, fp_pattern=fp, score=0.9, fp_score=0.9)

    # tiled=False: this test is about conf-calibration shippability, not tiling (tile_size
    # only gates a bundle when tiled).
    b = resolve_operating_point("bud_opening", dataset_hash="h1", calibration_records=cal,
                                holdout_records=hold, tiled=False, staged_conf_floor=0.01)
    conf = b.get("conf")
    hb = conf.gate_evidence["holdout_bias"]
    assert hb["count_bias_mean"] == pytest.approx(0.0)
    assert hb["count_bias_std"] == pytest.approx(2.0254787341673333, abs=1e-6)  # genuine dispersion
    assert hb["recall"] == pytest.approx(0.97, abs=1e-6)   # a realistic ~97% recall detector
    assert hb["precision"] == pytest.approx(0.97, abs=1e-6)
    assert conf.validated_against == "held_out_annotations"
    assert b.is_shippable is True
    assert conf.gate_evidence["failures"] == []


def test_same_noisy_detector_at_a_smaller_reference_size_correctly_fails_equivalence():
    """The identical per-image miss/fp pattern as the n=40 fixture above, truncated to its first 10
    images: same detector, same per-image behavior, just fewer held-out images. The SE term grows
    enough that abs(mean) + 1.645*SE no longer clears bud_opening's tolerance, so this correctly refuses:
    reference size is a real constraint, not every size validates the same noisy-but-good detector.
    """
    n_full, obj = 40, 100
    miss_full, fp_full = _rotating_noise_pattern(n_full)
    n = 10
    miss, fp = miss_full[:n], fp_full[:n]
    cal = dense_records(n_images=n, objects_per_image=obj, id_prefix="c",
                        miss_pattern=miss, fp_pattern=fp, score=0.9, fp_score=0.9)
    hold = dense_records(n_images=n, objects_per_image=obj, id_prefix="h", shift=5.0,
                         miss_pattern=miss, fp_pattern=fp, score=0.9, fp_score=0.9)

    b = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h1", calibration_records=cal,
                                holdout_records=hold, staged_conf_floor=0.01)
    conf = b.get("conf")
    assert conf.gate_evidence["holdout_bias"]["count_bias_mean"] == pytest.approx(0.0)  # same clean mean...
    assert conf.validated_against == "false"  # ...but too few images to clear the equivalence criterion
    assert "count_bias_exceeds_tolerance" in conf.gate_evidence["failures"]


def test_same_noisy_detector_reference_size_fixed_but_density_varied_crosses_admit_refuse():
    """The tests above vary image count at a fixed density (100 objects/image); this fixes n=40
    (already proven sufficient above) and varies density instead, the axis the relative tolerance
    actually introduces. The identical per-image noise pattern (mean bias exactly 0.0, std ~2.0255,
    unaffected by density) is refused at 30 objects/image (tolerance 0.30, comfortably under the
    ~0.527 the equivalence test needs) and admitted at 100 (tolerance 1.00); the true crossover is
    ~52.68 objects/image, so both chosen densities sit with real margin (43% below, 90% above), not
    on a fragile boundary.
    Not a defect: the explicit, disclosed consequence of "relative, not absolute" tolerance, a trait
    whose real density sits below where this default's tolerance would exceed its own noise floor
    needs a domain-authored (larger) fraction or more holdout images, not a platform-picked one.
    """
    n = 40
    miss, fp = _rotating_noise_pattern(n)

    def _run(obj):
        cal = dense_records(n_images=n, objects_per_image=obj, id_prefix="c",
                            miss_pattern=miss, fp_pattern=fp, score=0.9, fp_score=0.9)
        hold = dense_records(n_images=n, objects_per_image=obj, id_prefix="h", shift=5.0,
                             miss_pattern=miss, fp_pattern=fp, score=0.9, fp_score=0.9)
        return resolve_operating_point("bud_opening", tiled=True, dataset_hash="h1", calibration_records=cal,
                                       holdout_records=hold, staged_conf_floor=0.01)

    sparse = _run(30)
    dense = _run(100)
    # Same noise, same n, same mean/std -- density is the only thing that differs.
    assert sparse.get("conf").gate_evidence["holdout_bias"]["count_bias_std"] == pytest.approx(
        dense.get("conf").gate_evidence["holdout_bias"]["count_bias_std"])
    assert sparse.get("conf").gate_evidence["holdout_bias"]["count_bias_mean"] == pytest.approx(0.0)
    assert sparse.get("conf").validated_against == "false"
    assert "count_bias_exceeds_tolerance" in sparse.get("conf").gate_evidence["failures"]
    assert dense.get("conf").validated_against == "held_out_annotations"
    assert dense.get("conf").gate_evidence["failures"] == []


# ── Mandatory end-to-end integration fixture ────────────────────────────────

def test_integration_dense_realistic_reference_reaches_held_out_validation():
    """A realistic dense bud reference (perfect recall, a varying handful of low-conf spurious
    detections per image) must reach VALIDATED_HELD_OUT with every gate in this module applied
    together: the spurious detections are filtered out at the count-unbiased operating point,
    leaving zero bias and full recall/precision on the holdout.

    Despite the fp_pattern varying per image, all of that variation is filtered out uniformly at
    the picked conf (fp_score=0.05 never survives conf=0.9), so this fixture's measured
    count_bias_std at the operating point is exactly 0.0 (verified empirically): it does not
    exercise the equivalence criterion with genuine dispersion in the admitting direction, despite
    the varying input pattern suggesting otherwise. See
    ``test_realistic_dense_detector_with_genuine_per_image_dispersion_validates_at_n_equals_40``
    below for a fixture whose false positives survive at the same score as true detections and so
    leave real, nonzero dispersion behind at the picked conf.
    """
    n_images, objects_per_image = 24, 100
    miss = [0] * n_images
    fp = [2, 1, 3, 2, 1, 2, 3, 1, 2, 2, 1, 3, 2, 1, 2, 3, 1, 2, 2, 1, 3, 2, 1, 2]
    assert len(miss) == n_images and len(fp) == n_images

    cal = dense_records(n_images=n_images, objects_per_image=objects_per_image, id_prefix="c",
                        miss_pattern=miss, fp_pattern=fp, score=0.9, fp_score=0.05)
    hold = dense_records(n_images=n_images, objects_per_image=objects_per_image, id_prefix="h",
                         shift=5.0, miss_pattern=miss, fp_pattern=fp, score=0.9, fp_score=0.05)

    # tiled=False: this test is about conf-calibration shippability, not tiling (tile_size
    # only gates a bundle when tiled).
    b = resolve_operating_point("bud_opening", dataset_hash="h1", calibration_records=cal,
                                holdout_records=hold, tiled=False, staged_conf_floor=0.01)
    conf = b.get("conf")
    assert conf.validated_against == "held_out_annotations"
    assert b.is_shippable is True
    assert conf.gate_evidence["failures"] == []


# ── The gate's population: images that carry the thing being counted ────────
#
# A reference legitimately holds confirmed negatives (empty labels plus a human Complete), and a
# detector that finds nothing on one is right, not unbiased-by-evidence. Such an image contributes a
# certain zero to the per-image count bias and nothing to the density the relative tolerance is
# scaled by (``mean_of_present_counts`` counts only images that carry something). Averaging the bias
# over the wider population while scaling the tolerance to the narrower one divides the measured
# bias, shrinks its dispersion and inflates the equivalence test's sample size all at once, so the
# fraction the breeder authored gets enforced ``n_images / n_present`` times looser than it reads.

def _mixed_reference(id_prefix: str, *, n_loaded: int, n_negative: int, objects_per_image: int,
                     fp_per_loaded: int, shift: float = 0.0) -> list[dict]:
    """``n_loaded`` dense images plus ``n_negative`` confirmed negatives (no GT, no detections).

    Every loaded image carries the same systematic over-count, so the bias measured over the images
    that actually carry buds is exactly ``fp_per_loaded`` with no dispersion at all: whatever the
    gate concludes here is a statement about the population it measured, not about noise.
    """
    loaded = dense_records(n_images=n_loaded, objects_per_image=objects_per_image,
                           id_prefix=f"{id_prefix}L", shift=shift,
                           miss_pattern=[0] * n_loaded, fp_pattern=[fp_per_loaded] * n_loaded,
                           score=0.9, fp_score=0.9)
    negatives = dense_records(n_images=n_negative, objects_per_image=0,
                              id_prefix=f"{id_prefix}N")
    return loaded + negatives


def test_a_systematic_overcount_is_not_excused_by_the_negatives_beside_it():
    """A detector that finds two buds that are not there on every image that carries buds is
    over the trait's tolerance, and a holdout padded with correctly-empty images does not make it
    less so. The tolerance is 1.0 (0.01 of this reference's 100-objects-per-image typical count) and
    the measured over-count is 2.0 per loaded image."""
    cal = _mixed_reference("c", n_loaded=10, n_negative=40, objects_per_image=100, fp_per_loaded=2)
    hold = _mixed_reference("h", n_loaded=10, n_negative=40, objects_per_image=100,
                            fp_per_loaded=2, shift=5.0)

    b = resolve_operating_point("bud_opening", dataset_hash="h1", calibration_records=cal,
                                holdout_records=hold, tiled=False, staged_conf_floor=0.01)
    conf = b.get("conf")
    hb = conf.gate_evidence["holdout_bias"]
    assert hb["n_images"] == 50 and hb["n_present"] == 10
    assert hb["count_bias_mean_present"] == pytest.approx(2.0)
    assert conf.gate_evidence["pooled_typical_count"] == pytest.approx(100.0)
    assert conf.gate_evidence["pooled_count_bias_tolerance"] == pytest.approx(1.0)
    assert "count_bias_exceeds_tolerance" in conf.gate_evidence["failures"]
    assert conf.validated_against == "false"


def test_the_same_overcount_without_the_negatives_fails_identically():
    """The negatives are not what the refusal rests on: the identical loaded images alone, with no
    negatives beside them, reach the same verdict. Without this, the refusal above could be an
    artifact of the negatives rather than the over-count they were hiding."""
    cal = _mixed_reference("c", n_loaded=10, n_negative=0, objects_per_image=100, fp_per_loaded=2)
    hold = _mixed_reference("h", n_loaded=10, n_negative=0, objects_per_image=100,
                            fp_per_loaded=2, shift=5.0)

    conf = resolve_operating_point("bud_opening", dataset_hash="h1", calibration_records=cal,
                                   holdout_records=hold, tiled=False,
                                   staged_conf_floor=0.01).get("conf")
    assert conf.gate_evidence["holdout_bias"]["count_bias_mean_present"] == pytest.approx(2.0)
    assert "count_bias_exceeds_tolerance" in conf.gate_evidence["failures"]
    assert conf.validated_against == "false"


def test_a_clean_detector_still_validates_on_a_reference_full_of_negatives():
    """The rail must admit valid work: the same mixed reference, with a detector that is genuinely
    unbiased on the images that carry buds, validates. A reference holding four times as many
    confirmed negatives as loaded images is ordinary, not a reason to refuse."""
    cal = _mixed_reference("c", n_loaded=10, n_negative=40, objects_per_image=100, fp_per_loaded=0)
    hold = _mixed_reference("h", n_loaded=10, n_negative=40, objects_per_image=100,
                            fp_per_loaded=0, shift=5.0)

    b = resolve_operating_point("bud_opening", dataset_hash="h1", calibration_records=cal,
                                holdout_records=hold, tiled=False, staged_conf_floor=0.01)
    conf = b.get("conf")
    assert conf.gate_evidence["holdout_bias"]["n_present"] == 10
    assert conf.gate_evidence["failures"] == []
    assert conf.validated_against == "held_out_annotations"
    assert b.is_shippable is True


def test_a_negative_the_detector_hallucinates_on_is_evidence_not_a_discard():
    """Presence is "carries GT or a surviving detection", so a confirmed negative the detector fires
    on is counted: its false positives are exactly the kind of miscount this gate exists to catch,
    and dropping it would let a hallucinating detector shed the evidence against it."""
    def _hallucinating(id_prefix, shift=0.0):
        loaded = dense_records(n_images=10, objects_per_image=100, id_prefix=f"{id_prefix}L",
                               shift=shift, miss_pattern=[0] * 10, fp_pattern=[0] * 10,
                               score=0.9, fp_score=0.9)
        # Zero GT, but three surviving detections on each, so each negative carries a +3 bias.
        noisy = dense_records(n_images=10, objects_per_image=0, id_prefix=f"{id_prefix}N",
                              miss_pattern=[0] * 10, fp_pattern=[3] * 10, score=0.9, fp_score=0.9)
        return loaded + noisy

    conf = resolve_operating_point("bud_opening", dataset_hash="h1",
                                   calibration_records=_hallucinating("c"),
                                   holdout_records=_hallucinating("h", shift=5.0),
                                   tiled=False, staged_conf_floor=0.01).get("conf")
    hb = conf.gate_evidence["holdout_bias"]
    assert hb["n_present"] == 20  # the hallucinated-on negatives count as evidence
    assert hb["count_bias_mean_present"] == pytest.approx(1.5)  # (10 * 0 + 10 * 3) / 20
    assert "count_bias_exceeds_tolerance" in conf.gate_evidence["failures"]


def test_a_holdout_carrying_one_loaded_image_is_not_enough_evidence():
    """Reference sufficiency counts the images that carry something, not the reference's total size:
    ninety-nine correctly-empty images beside one loaded image are one image worth of evidence about
    count bias."""
    cal = _mixed_reference("c", n_loaded=4, n_negative=0, objects_per_image=100, fp_per_loaded=0)
    hold = _mixed_reference("h", n_loaded=1, n_negative=99, objects_per_image=100,
                            fp_per_loaded=0, shift=5.0)

    conf = resolve_operating_point("bud_opening", dataset_hash="h1", calibration_records=cal,
                                   holdout_records=hold, tiled=False,
                                   staged_conf_floor=0.01).get("conf")
    assert conf.gate_evidence["holdout_bias"]["n_images"] == 100
    assert "insufficient_holdout_images" in conf.gate_evidence["failures"]
    assert conf.validated_against == "false"
