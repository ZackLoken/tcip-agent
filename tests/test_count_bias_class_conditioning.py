"""The count-bias gate must be conditioned on class, not pooled across classes.

The pooled per-image bias ``E[FP-FN]`` is measured by a matcher that ignores ``category_id``, so a
detector that calls every object of one class another class scores TP-only with bias 0, and one that
over-detects class A exactly as much as it under-detects class B nets to 0 as well. Either way the
delivered phenotype (a per-class count, or a fraction built from two of them) is wrong while the
operating point earns a ``VALIDATED_HELD_OUT``/``VALIDATED_REVIEW_CONFIRMED`` stamp.

Every gate test here drives a real door: ``resolve_operating_point_from_review`` (what
``routes/review.py`` calls) or ``resolve_operating_point`` (what ``run_inference``'s
``calibrate_operating_point`` calls). The two at the end are about the conf sweep itself, so they
call ``derive_operating_point_curve`` (the thing under test) directly. None construct a sweep by hand.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("torch")  # operating_point imports evaluation, which imports torch

from tcip_mcp.pipelines.feedback import (  # noqa: E402
    describe_review_validation,
    resolve_operating_point_from_review,
)
from tcip_mcp.pipelines.operating_point import resolve_operating_point  # noqa: E402
from tcip_mcp.pipelines.resolution import (  # noqa: E402
    VALIDATED_FALSE,
    VALIDATED_HELD_OUT,
    VALIDATED_REVIEW_CONFIRMED,
)
from tcip_mcp.pipelines.training.evaluation import (  # noqa: E402
    gt_class_avg_size,
    pick_count_unbiased,
    derive_operating_point_curve,
)

_IDENTITY = {"checkpoint_sha256": "sha-model-a", "experiment_id": None}
N_IMAGES = 16
N_MATCHED = 12
N_SWAPPED = 3

# no built-in traits, seed_bud_trait_spec (conftest.py) writes a real bud.yml into this
# test's pinned platform state root so resolve_operating_point("bud_opening", ...) keeps resolving by default.
pytestmark = pytest.mark.usefixtures("seed_bud_trait_spec")


@pytest.fixture(autouse=True)
def _hermetic_platform_root(tmp_path):
    """The cal/holdout split locks under ``$TCIP_STATE_ROOT/.tcip``: keep it out of the repo."""
    os.environ["TCIP_STATE_ROOT"] = str(tmp_path)  # conftest restores the prior value


def _entry(action, cid, gt, pred, conf):
    return {"match_type": "TP" if pred and gt else ("FP" if pred else "FN"), "action": action,
            "class_id": cid, "gt_bbox_norm": gt, "pred_bbox_norm": pred, "conf": conf,
            "producer_identity": _IDENTITY, "conf_threshold": 0.01}


def _two_class_review_state(*, swap_classes: bool):
    """A dense two-class review reference: ``N_MATCHED`` correctly-called class-0 objects per image,
    plus ``N_SWAPPED`` objects the breeder confirmed as class 0.

    ``swap_classes=True`` records those last objects the way a class-confusing model's review reads:
    the class-0 object was missed (a gt-only verdict) and a class-1 detection was rejected at the
    same box. Pooled, the two cancel: the class-blind matcher pairs the class-1 prediction with the
    class-0 truth. ``False`` is the same geometry called correctly.

    The per-image jitter keeps every image's GT content byte-distinct, so the content-overlap gate
    doesn't read the holdout as a clone of calibration (mirrors ``test_review_calibration.py``'s
    dense fixture, whose geometry conventions this follows).
    """
    images = {}
    for i in range(N_IMAGES):
        dets = []
        jitter = i * 0.0001
        for k in range(N_MATCHED + N_SWAPPED):
            row, col = divmod(k, 5)
            box = [0.05 + col * 0.05 + jitter, 0.05 + row * 0.05, 0.01, 0.01]
            if k < N_MATCHED:
                dets.append(_entry("accepted", 0, box, box, 0.9))
            elif swap_classes:
                dets.append(_entry("edited", 0, box, None, None))       # class-0 truth, no detection
                dets.append(_entry("rejected", 1, None, box, 0.9))      # class-1 detection, no truth
            else:
                dets.append(_entry("accepted", 1, box, box, 0.9))
        images[f"A{i}.jpg"] = {"img_status": "completed", "gt_preexisting": True, "detections": dets}
    return {"image": images}


def _gt_records(prefix, n, *, swap_classes, classes=(1, 2), offset=0.0):
    """Per-image COCO records for the GT door: half the objects class ``classes[0]``, half
    ``classes[1]``; ``swap_classes`` makes the model call every one of them ``classes[1]``.

    ``offset`` shifts the whole layout so a calibration and a holdout set built here hold genuinely
    distinct geometry, since identical boxes on both sides would trip the content-overlap gate first, and the
    count-bias gate under test never runs.
    """
    recs = []
    for i in range(n):
        gt, dt = [], []
        for k in range(8):
            box = [100.0 * k + 3.0 * i + offset, 50.0 + i, 40.0, 40.0]
            cat = classes[0] if k < 4 else classes[1]
            gt.append({"bbox": box, "category_id": cat})
            dt.append({"bbox": box, "category_id": classes[1] if swap_classes else cat,
                       "score": 0.9})
        recs.append({"image_id": f"{prefix}{i}", "gt": gt, "dt": dt})
    return recs


# ── the review door (routes/review.py) ────────────────────────────────────────────────────────

def test_review_door_refuses_a_class_compensating_reference_the_pooled_bias_calls_unbiased(tmp_path):
    b = resolve_operating_point_from_review(_two_class_review_state(swap_classes=True), "bud_opening",
                                            tiled=True, staged_conf_floor=0.01,
                                            bucket_identities=[_IDENTITY], scope_root=tmp_path)
    conf = b.params["conf"]
    sweep = conf.gate_evidence
    hb = sweep["holdout_bias"]
    # The pooled term the gate used to judge on reads perfectly unbiased, on a reference where every
    # swapped object is counted under the wrong class.
    assert hb["count_bias_mean"] == pytest.approx(0.0)
    assert hb["fp"] == 0 and hb["fn"] == 0
    # Per class it is anything but: class 1 under-counted, class 2 over-counted, by N_SWAPPED each.
    assert hb["per_class"]["1"]["count_bias_mean"] == pytest.approx(-float(N_SWAPPED))
    assert hb["per_class"]["2"]["count_bias_mean"] == pytest.approx(float(N_SWAPPED))
    assert sweep["per_class_count_bias_failures"] == ["1", "2"]
    assert "count_bias_exceeds_tolerance_per_class" in sweep["failures"]
    assert conf.validated_against == VALIDATED_FALSE


def test_review_door_class_failure_has_its_own_breeder_message(tmp_path):
    # The named-failure vocabulary is exhaustive by construction (describe_review_validation raises
    # on a name it can't translate), so a new gate name without a message is a loud error here.
    b = resolve_operating_point_from_review(_two_class_review_state(swap_classes=True), "bud_opening",
                                            tiled=True, staged_conf_floor=0.01,
                                            bucket_identities=[_IDENTITY], scope_root=tmp_path)
    out = describe_review_validation(b, reviewed_image_count=N_IMAGES)
    assert out["validated"] is False
    assert "kind" in out["reason"].lower()
    # Not the pooled message, which would send the breeder to review more images for a mismatch
    # more reviewing cannot fix.
    assert "didn't agree closely enough" not in out["reason"]


def test_review_door_still_validates_a_multi_class_reference_that_is_honest_per_class(tmp_path):
    b = resolve_operating_point_from_review(_two_class_review_state(swap_classes=False), "bud_opening",
                                            tiled=True, staged_conf_floor=0.01,
                                            bucket_identities=[_IDENTITY], scope_root=tmp_path)
    conf = b.params["conf"]
    assert conf.gate_evidence["failures"] == []
    assert conf.gate_evidence["per_class_count_bias_failures"] == []
    assert set(conf.gate_evidence["holdout_bias"]["per_class"]) == {"1", "2"}
    assert conf.validated_against == VALIDATED_REVIEW_CONFIRMED


# ── the GT door (run_inference -> calibrate_operating_point) ─────────────────────────────────

def test_gt_door_refuses_a_class_compensating_reference():
    b = resolve_operating_point(
        "bud_opening", tiled=True, dataset_hash="h", staged_conf_floor=0.05,
        calibration_records=_gt_records("cal", 4, swap_classes=True),
        holdout_records=_gt_records("hold", 4, swap_classes=True, offset=5000.0))
    sweep = b.params["conf"].gate_evidence
    assert sweep["holdout_bias"]["count_bias_mean"] == pytest.approx(0.0)
    assert sweep["failures"] == ["count_bias_exceeds_tolerance_per_class"]
    assert b.params["conf"].validated_against == VALIDATED_FALSE


def test_gt_door_validates_the_same_geometry_called_correctly():
    b = resolve_operating_point(
        "bud_opening", tiled=True, dataset_hash="h", staged_conf_floor=0.05,
        calibration_records=_gt_records("cal", 4, swap_classes=False),
        holdout_records=_gt_records("hold", 4, swap_classes=False, offset=5000.0))
    assert b.params["conf"].gate_evidence["failures"] == []
    assert b.params["conf"].validated_against == VALIDATED_HELD_OUT


def test_single_class_reference_is_unaffected_by_the_conditioning():
    # Bud's shipped shape today: one detection class ({subject: 0} -> category_id 1), the
    # opening call made by a separate classifier. Conditioning on class must be a no-op here:
    # a rail that fail-closed the only trait the platform ships would be worse than the hole.
    cal = _gt_records("cal", 4, swap_classes=False, classes=(1, 1))
    hold = _gt_records("hold", 4, swap_classes=False, classes=(1, 1), offset=5000.0)
    b = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h", staged_conf_floor=0.05,
                                calibration_records=cal, holdout_records=hold)
    sweep = b.params["conf"].gate_evidence
    hb = sweep["holdout_bias"]
    assert b.params["conf"].validated_against == VALIDATED_HELD_OUT
    assert sweep["per_class_count_bias_failures"] == []
    # The one class's statistics are the pooled ones, reused rather than recomputed.
    assert list(hb["per_class"]) == ["1"]
    assert hb["per_class"]["1"]["count_bias_mean"] == hb["count_bias_mean"]
    assert hb["per_class"]["1"]["count_bias_std"] == hb["count_bias_std"]
    # ...and that reuse has to be worth trusting: an independently class-filtered sweep over the
    # same records must produce the same statistics the shortcut hands back.
    explicit = derive_operating_point_curve(hold, tolerance=sweep["calibration"]["tolerance"],
                                     class_id=1, conf_grid=[hb["conf"]])["curve"][0]
    for key in ("tp", "fp", "fn", "count_bias_mean", "count_bias_std", "n_images", "n_present"):
        assert hb["per_class"]["1"][key] == pytest.approx(explicit[key])


def test_a_class_the_holdout_never_carries_cannot_be_validated_by_its_absence():
    # This gate has a hole reachable through the split: the model confuses classes 1 and 2 in
    # calibration, the holdout draw happens to hold only class 1, every per-class entry the gate
    # can see reads bias 0.0, and the reference was stamped validated on no class-2 evidence.
    cal = _gt_records("cal", 4, swap_classes=True)
    hold = _gt_records("hold", 4, swap_classes=False, classes=(1, 1), offset=5000.0)
    b = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h", staged_conf_floor=0.05,
                                calibration_records=cal, holdout_records=hold)
    sweep = b.params["conf"].gate_evidence
    assert sweep["per_class_count_bias_failures"] == []   # the holdout has nothing to fail on
    assert sweep["holdout_missing_classes"] == ["2"]
    assert "holdout_missing_class" in sweep["failures"]
    assert b.params["conf"].validated_against == VALIDATED_FALSE


def _sparse_class_records(prefix, n_images, *, offset=0.0):
    """Class 1 present and correctly detected on every image; class 2 present (and missed
    entirely) on only the first two: thin evidence diluted toward zero by images that say nothing
    about the class.
    """
    recs = []
    for i in range(n_images):
        gt, dt = [], []
        for k in range(2):
            box = [100.0 * k + offset, 50.0 + i, 40.0, 40.0]
            gt.append({"bbox": box, "category_id": 1})
            dt.append({"bbox": box, "category_id": 1, "score": 0.9})
        if i < 2:
            for k in range(4):
                box = [500.0 + 100.0 * k + offset, 50.0 + i, 40.0, 40.0]
                gt.append({"bbox": box, "category_id": 2})  # no dt: guaranteed FN
        recs.append({"image_id": f"{prefix}{i}", "gt": gt, "dt": dt})
    return recs


def _sparse_class_calibration(prefix, n, *, offset):
    recs = []
    for i in range(n):
        gt, dt = [], []
        for cat in (1, 2):
            box = [100.0 * cat + offset, 50.0 + i, 40.0, 40.0]
            gt.append({"bbox": box, "category_id": cat})
            dt.append({"bbox": box, "category_id": cat, "score": 0.9})
        recs.append({"image_id": f"{prefix}{i}", "gt": gt, "dt": dt})
    return recs


def test_a_class_scarce_in_the_holdout_cannot_be_diluted_to_a_pass():
    # Class 2 is present on 2 of 20 holdout images and missed outright both times, wrong every
    # time there is anything to be wrong about. Pooled over all 20 images the mean reads -0.4,
    # comfortably inside bud's tolerance of 1.0; only weighting the equivalence test's standard
    # error by the 2 images that actually carried the class (not the 20 that said nothing about
    # it) surfaces that this reference has almost no real evidence backing that mean.
    cal = _sparse_class_calibration("cal", 4, offset=0.0)
    hold = _sparse_class_records("hold", 20, offset=5000.0)
    b = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h", staged_conf_floor=0.05,
                                calibration_records=cal, holdout_records=hold)
    sweep = b.params["conf"].gate_evidence
    c2 = sweep["holdout_bias"]["per_class"]["2"]
    assert c2["n_images"] == 20
    assert c2["n_present"] == 2
    assert c2["count_bias_mean"] == pytest.approx(-0.4)
    assert "2" in sweep["per_class_count_bias_failures"]
    assert "count_bias_exceeds_tolerance_per_class" in sweep["failures"]
    assert b.params["conf"].validated_against == VALIDATED_FALSE


def test_holdout_class_coverage_admits_a_reference_that_evidences_every_class():
    # The companion obligation: the coverage rule must not refuse a holdout that does carry every
    # class, including one whose objects the model correctly finds on only some images.
    cal = _gt_records("cal", 4, swap_classes=False)
    hold = _gt_records("hold", 4, swap_classes=False, offset=5000.0)
    b = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h", staged_conf_floor=0.05,
                                calibration_records=cal, holdout_records=hold)
    assert b.params["conf"].gate_evidence["holdout_missing_classes"] == []
    assert b.params["conf"].gate_evidence["failures"] == []
    assert b.params["conf"].validated_against == VALIDATED_HELD_OUT


def test_missing_class_failure_has_its_own_breeder_message():
    # describe_review_validation raises on a gate failure it cannot translate, so this is what
    # proves the new name is in the breeder-facing vocabulary at all. Driven off a real refusal
    # rather than a hand-built bundle; the review door reaches the same message through the same
    # lookup, but which images its locked split holds back is not the fixture's to choose.
    b = resolve_operating_point(
        "bud_opening", tiled=True, dataset_hash="h", staged_conf_floor=0.05,
        calibration_records=_gt_records("cal", 4, swap_classes=True),
        holdout_records=_gt_records("hold", 4, swap_classes=False, classes=(1, 1), offset=5000.0))
    out = describe_review_validation(b, reviewed_image_count=N_IMAGES)
    assert out["validated"] is False
    assert "held back" in out["reason"] and "no independent evidence" in out["reason"]


def test_per_class_keys_survive_the_sweep_artifact_round_trip():
    # run_inference persists the whole curve to .tcip/artifacts/operating_point_sweep_<hash>.json,
    # so the gate's per-class breakdown is only reconstructable later if its keys are JSON-stable:
    # int keys would come back as strings and silently stop matching an in-memory read.
    b = resolve_operating_point(
        "bud_opening", tiled=True, dataset_hash="h", staged_conf_floor=0.05,
        calibration_records=_gt_records("cal", 4, swap_classes=True),
        holdout_records=_gt_records("hold", 4, swap_classes=True, offset=5000.0))
    path = Path(os.environ["TCIP_STATE_ROOT"]) / "sweep.json"
    path.write_text(json.dumps({"gate_evidence": b.params["conf"].gate_evidence}), encoding="utf-8")
    reloaded = json.loads(path.read_text(encoding="utf-8"))["gate_evidence"]
    assert reloaded["holdout_bias"]["per_class"] == b.params["conf"].gate_evidence["holdout_bias"]["per_class"]
    assert reloaded["per_class_count_bias_failures"] == ["1", "2"]


# ── the conf pick, which has to optimize what the gate judges ─────────────────────────────────

def test_no_conf_in_the_sweep_escapes_a_wholesale_class_swap():
    """For a model that calls every class-1 object class 2, the refusal is not an artifact of which
    conf was picked: every conf on the curve leaves a class over tolerance.

    Also pins the identity the pick and the gate both rest on: a class's per-image bias is exactly
    ``|dt_c| - |gt_c|`` at that conf, because the matched pairs cancel out of ``fp - fn``.
    """
    recs = _gt_records("cal", 4, swap_classes=True)
    for i, r in enumerate(recs):                 # a real score spread, so confs actually filter
        for k, d in enumerate(r["dt"]):
            d["score"] = 0.15 + 0.1 * k
    sweep = derive_operating_point_curve(recs, tolerance=0.5 * gt_class_avg_size(recs))
    assert len(sweep["curve"]) > 4
    filtered = [c for c in sweep["curve"] if c["fp"] + c["tp"] < 32]
    assert filtered, "fixture is vacuous: no conf on the curve filters any detection"
    for entry in sweep["curve"]:
        for cid, stats in entry["per_class"].items():
            expected = sum(
                len([d for d in r["dt"] if d["category_id"] == int(cid)
                     and d["score"] >= entry["conf"]])
                - len([a for a in r["gt"] if a["category_id"] == int(cid)])
                for r in recs) / len(recs)
            assert stats["count_bias_mean"] == pytest.approx(expected)
        assert max(abs(s["count_bias_mean"]) for s in entry["per_class"].values()) > 1.0


def test_pick_serves_the_worst_class_not_the_pooled_total():
    """With three classes the pooled-unbiased conf can be the one the gate must refuse: at conf 0.9
    the pooled bias is 0 while class 3 sits at -2.0; at conf 0.4, every class's bias magnitude is
    smaller. A pooled pick refuses a model that has a valid operating point.

    The gate's tolerance is relative to each class's (and the pooled scope's) own typical
    per-image count (derived from the holdout), not a flat absolute default. Every class here has
    only 1-3 real objects/image, so classes 1/2's permanent +1 spurious detection and class 3's
    permanent single miss (unavoidable at any conf: nothing ever detects it) are each a large
    relative miss no conf can fix, correctly refused regardless of which conf is picked. An
    absolute tolerance of 1.0 would let every one of them through at exactly its own boundary (a
    bias of 1.0 clearing a tolerance of 1.0 via ``<=``); see
    ``test_pick_serves_the_worst_class_not_the_pooled_total_admits_it_when_dense_enough`` below for
    the same picker mechanism validating end-to-end once every class is dense enough that its own
    identical permanent miscount is a small relative fraction, the "rail admits valid work, not
    only reject invalid work" case CLAUDE.md requires for a strengthened gate.
    """
    def build(prefix, offset):
        recs = []
        for i in range(6):
            gt, dt = [], []
            for cat in (1, 2):
                # one real object each, found; plus one spurious detection that survives everywhere.
                box = [200.0 * cat + offset, 50.0, 40.0, 40.0]
                gt.append({"bbox": box, "category_id": cat})
                dt.append({"bbox": box, "category_id": cat, "score": 0.95})
                dt.append({"bbox": [200.0 * cat + offset, 400.0, 40.0, 40.0],
                           "category_id": cat, "score": 0.95})
            for k in range(3):                      # three real class-3 objects...
                gt.append({"bbox": [700.0 + 100.0 * k + offset, 50.0, 40.0, 40.0],
                           "category_id": 3})
            # ...one found confidently, one only hesitantly, one missed outright. Raising conf past
            # the hesitant one buys pooled balance by dropping a real class-3 object.
            dt.append({"bbox": [700.0 + offset, 50.0, 40.0, 40.0], "category_id": 3, "score": 0.95})
            dt.append({"bbox": [800.0 + offset, 50.0, 40.0, 40.0], "category_id": 3, "score": 0.4})
            recs.append({"image_id": f"{prefix}{i}", "gt": gt, "dt": dt})
        return recs

    recs = build("w", 0.0)

    sweep = derive_operating_point_curve(recs, tolerance=0.5 * gt_class_avg_size(recs))
    at = {round(c["conf"], 2): c for c in sweep["curve"]}
    assert at[0.95]["count_bias_mean"] == pytest.approx(0.0)       # the pooled trap
    assert at[0.95]["per_class"]["3"]["count_bias_mean"] == pytest.approx(-2.0)
    assert max(abs(s["count_bias_mean"]) for s in at[0.4]["per_class"].values()) == pytest.approx(1.0)
    assert pick_count_unbiased(sweep) == pytest.approx(0.4)

    # ...end to end: the picker still avoids the pooled trap (conf 0.4, not 0.95's misleadingly
    # unbiased-looking pooled total), but at this toy density every class's permanent per-image
    # miscount (classes 1/2's +1 spurious detection, class 3's -1 permanent miss) is a large relative
    # error, correctly refused under the default relative tolerance (0.01, i.e. 1%) even at the
    # least-bad conf. An absolute tolerance of 1.0 would have let every one of these through at
    # this same picked conf (each bias magnitude was exactly 1.0, clearing a tolerance of 1.0 via
    # ``<=``); the relative tolerance does not, because none of them was ever a trustworthy
    # 1%-relative claim at this reference's actual density.
    b = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h", staged_conf_floor=0.05,
                                calibration_records=recs, holdout_records=build("h", 5000.0))
    assert b.params["conf"]._raw == pytest.approx(0.4)  # still the worst-class-aware pick
    assert "count_bias_exceeds_tolerance" in b.params["conf"].gate_evidence["failures"]        # pooled
    assert "count_bias_exceeds_tolerance_per_class" in b.params["conf"].gate_evidence["failures"]
    assert set(b.params["conf"].gate_evidence["per_class_count_bias_failures"]) == {"1", "2", "3"}
    assert b.params["conf"].validated_against == VALIDATED_FALSE


def test_pick_serves_the_worst_class_not_the_pooled_total_admits_it_when_dense_enough():
    """Same picker mechanism and the same qualitative shape as the test above (a pooled-unbiased
    conf that traps a naive pick, a worst-class-aware pick that avoids it), but every class also
    carries 200 always-correctly-detected background objects: real GT, always matched at every conf
    on this curve, so they contribute zero bias at any threshold and don't touch the
    pooled-trap-at-0.95 or worst-class-at-0.4 arithmetic the test above already pins. They exist only
    to make every class's typical per-image count large enough (~201-203) that its existing permanent
    miscount (class 3's single missed object, and classes 1/2's own permanent +1 spurious detection,
    present in the test above too, masked there by that test's boundary-exact absolute
    tolerance of 1.0, which a bias of exactly 1.0 cleared via ``<=``) is a small relative fraction
    of it, comfortably inside the default 1% relative tolerance. This is the "a rail must admit
    valid work, not only reject invalid work" case CLAUDE.md requires alongside the refusal test
    above: the relative gate is not merely stricter everywhere, it correctly admits a reference
    whose real misses are genuinely small relative to how much of each class there is.
    """
    def build(prefix, offset):
        recs = []
        for i in range(6):
            gt, dt = [], []
            for cat in (1, 2):
                box = [200.0 * cat + offset, 50.0, 40.0, 40.0]
                gt.append({"bbox": box, "category_id": cat})
                dt.append({"bbox": box, "category_id": cat, "score": 0.95})
                dt.append({"bbox": [200.0 * cat + offset, 400.0, 40.0, 40.0],
                           "category_id": cat, "score": 0.95})
                # 200 background objects for this class too, always found -- classes 1/2 carry their
                # own permanent +1 bias (the spurious detection above), so they need the same density
                # boost as class 3 to clear the new relative tolerance.
                for k in range(200):
                    box2 = [3000.0 + 4000.0 * cat + 60.0 * k + offset, 900.0, 40.0, 40.0]
                    gt.append({"bbox": box2, "category_id": cat})
                    dt.append({"bbox": box2, "category_id": cat, "score": 0.95})
            for k in range(3):
                gt.append({"bbox": [700.0 + 100.0 * k + offset, 50.0, 40.0, 40.0],
                           "category_id": 3})
            dt.append({"bbox": [700.0 + offset, 50.0, 40.0, 40.0], "category_id": 3, "score": 0.95})
            dt.append({"bbox": [800.0 + offset, 50.0, 40.0, 40.0], "category_id": 3, "score": 0.4})
            # 200 background class-3 objects, always found (score 0.95, survives every conf on this
            # curve) -- spaced well outside center-match tolerance from everything above and each
            # other, in their own row far from the rest of the layout.
            for k in range(200):
                box = [1500.0 + 60.0 * k + offset, 900.0, 40.0, 40.0]
                gt.append({"bbox": box, "category_id": 3})
                dt.append({"bbox": box, "category_id": 3, "score": 0.95})
            recs.append({"image_id": f"{prefix}{i}", "gt": gt, "dt": dt})
        return recs

    recs = build("w", 0.0)
    sweep = derive_operating_point_curve(recs, tolerance=0.5 * gt_class_avg_size(recs))
    at = {round(c["conf"], 2): c for c in sweep["curve"]}
    # The pinned arithmetic from the test above is unchanged by the background objects.
    assert at[0.95]["per_class"]["3"]["count_bias_mean"] == pytest.approx(-2.0)
    assert max(abs(s["count_bias_mean"]) for s in at[0.4]["per_class"].values()) == pytest.approx(1.0)
    assert pick_count_unbiased(sweep) == pytest.approx(0.4)

    b = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h", staged_conf_floor=0.05,
                                calibration_records=recs, holdout_records=build("h", 5000.0))
    assert b.params["conf"].value == pytest.approx(0.4)
    assert b.params["conf"].gate_evidence["failures"] == []
    assert b.params["conf"].validated_against == VALIDATED_HELD_OUT


# ── the relative tolerance's own derived floor ───────────────────────
#
# Nothing above pins ``_effective_count_bias_tolerance`` itself: deleting the floor outright left
# the whole suite green. These test the floor directly (the pure function) and its effect on the
# provenance a caller actually reads, not just an indirect pass/fail outcome that a small-n
# equivalence test's own SE term can obscure.

def test_effective_count_bias_tolerance_floor_governs_a_near_zero_fraction():
    from tcip_mcp.pipelines.operating_point import _effective_count_bias_tolerance

    # A rare class (typical_count=1) at the platform default fraction (0.01) alone would demand a
    # tolerance of 0.01 -- an impossible standard for any integer count. The derived floor (1/n) is
    # what actually governs here, not the fraction term.
    tol = _effective_count_bias_tolerance(0.01, typical_count=1.0, n=5)
    assert tol == pytest.approx(0.2)
    assert tol > 0.01 * 1.0


def test_effective_count_bias_tolerance_floor_shrinks_as_evidence_grows():
    from tcip_mcp.pipelines.operating_point import _effective_count_bias_tolerance

    tol_n5 = _effective_count_bias_tolerance(0.0, typical_count=0.0, n=5)
    tol_n50 = _effective_count_bias_tolerance(0.0, typical_count=0.0, n=50)
    assert tol_n5 == pytest.approx(0.2)
    assert tol_n50 == pytest.approx(0.02)
    assert tol_n50 < tol_n5  # more evidence behind the mean -> a tighter floor, never looser


def test_effective_count_bias_tolerance_floor_never_loosens_past_d12_default_at_n_ge_2():
    from tcip_mcp.pipelines.operating_point import _effective_count_bias_tolerance

    # The old absolute default was 1.0 -- at every n the reference-sufficiency gates actually let
    # through (n >= 2), the floor alone stays at or below half that, so it can never be the reason a
    # reference passes today that the old flat gate would have refused.
    for n in range(2, 60):
        assert _effective_count_bias_tolerance(0.0, typical_count=0.0, n=n) <= 0.5


def test_effective_count_bias_tolerance_fraction_term_dominates_a_dense_reference():
    from tcip_mcp.pipelines.operating_point import _effective_count_bias_tolerance

    # At the density this platform's own test suite's dense fixtures use (~100 objects/image, not
    # verified against real production imagery), the fraction term -- not the floor -- sets the
    # tolerance, and reproduces the old absolute default of 1.0 exactly at the new 0.01 default
    # fraction (the property the default was chosen for).
    assert _effective_count_bias_tolerance(0.01, typical_count=100.0, n=40) == pytest.approx(1.0)


def _floor_matters_records(prefix, offset):
    """Class 1: dense (20/image), perfect, present every image -- keeps the reference from tripping
    unrelated gates. Class 2: sparse (2/image), present every image, with one extra spurious
    detection on exactly one of the 5 images (a real, non-uniform per-image bias, mean 0.2)."""
    recs = []
    for i in range(5):
        gt, dt = [], []
        for k in range(20):
            box = [50.0 + 30.0 * k + offset, 50.0, 20.0, 20.0]
            gt.append({"bbox": box, "category_id": 1})
            dt.append({"bbox": box, "category_id": 1, "score": 0.95})
        for k in range(2):
            box = [50.0 + 30.0 * k + offset, 900.0, 20.0, 20.0]
            gt.append({"bbox": box, "category_id": 2})
            dt.append({"bbox": box, "category_id": 2, "score": 0.95})
        if i == 0:
            dt.append({"bbox": [2000.0 + offset, 900.0, 20.0, 20.0], "category_id": 2, "score": 0.95})
        recs.append({"image_id": f"{prefix}{i}", "gt": gt, "dt": dt})
    return recs


def test_per_class_stamped_tolerance_reflects_the_floor_not_just_the_fraction_term():
    """End-to-end: the sparse class's stamped tolerance (what the gate actually compared its bias
    against) must be the floor (1/n_present == 0.2), not the fraction term alone (0.01 * 2 == 0.02) --
    a direct pin on ``_effective_count_bias_tolerance`` being live inside ``resolve_operating_point``,
    not inferred indirectly from a pass/fail outcome an unrelated SE term could also explain.
    """
    b = resolve_operating_point(
        "bud_opening", tiled=True, dataset_hash="h", staged_conf_floor=0.05,
        calibration_records=_floor_matters_records("c", 0.0),
        holdout_records=_floor_matters_records("h", 5000.0))
    sweep = b.params["conf"].gate_evidence
    assert sweep["per_class_typical_count"]["2"] == pytest.approx(2.0)
    # The floor (1/5 == 0.2), not the fraction term alone (0.01 * 2 == 0.02) -- an order of magnitude
    # apart, so this is not a rounding coincidence.
    assert sweep["per_class_count_bias_tolerance"]["2"] == pytest.approx(0.2)
    assert sweep["per_class_count_bias_tolerance"]["2"] > 0.01 * sweep["per_class_typical_count"]["2"]


def test_a_class_present_on_exactly_one_holdout_image_cannot_be_validated_by_it_alone():
    """Without a per-class minimum-presence gate, a class present on exactly one holdout image gets
    a relative tolerance derived from that one image's own density (reachable with no adversarial
    construction: an ordinary rare class that happens to show up once, in a denser-than-typical
    frame), and could admit a reference an absolute tolerance of 1.0 would have refused on the
    identical numbers. ``insufficient_holdout_images_per_class`` closes this the same way
    ``insufficient_holdout_images`` already does for the pooled scope.
    """
    def build(prefix, offset):
        recs = []
        for i in range(10):
            gt, dt = [], []
            for k in range(20):
                box = [50.0 + 30.0 * k + offset, 50.0, 20.0, 20.0]
                gt.append({"bbox": box, "category_id": 1})
                dt.append({"bbox": box, "category_id": 1, "score": 0.95})
            if i == 0:
                # class 2 exists only on this one image: 150 real objects, a small (2%) real
                # over-count -- small enough that a legitimately-derived tolerance from 150 objects
                # of density would admit it, if one image were enough evidence to trust at all.
                for k in range(150):
                    box = [50.0 + 30.0 * k + offset, 900.0, 20.0, 20.0]
                    gt.append({"bbox": box, "category_id": 2})
                    dt.append({"bbox": box, "category_id": 2, "score": 0.95})
                for k in range(3):
                    dt.append({"bbox": [10000.0 + 30.0 * k + offset, 900.0, 20.0, 20.0],
                              "category_id": 2, "score": 0.95})
            recs.append({"image_id": f"{prefix}{i}", "gt": gt, "dt": dt})
        return recs

    b = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h", staged_conf_floor=0.05,
                                calibration_records=build("c", 0.0), holdout_records=build("h", 5000.0))
    sweep = b.params["conf"].gate_evidence
    assert sweep["holdout_bias"]["per_class"]["2"]["n_present"] == 1
    assert sweep["per_class_insufficient_images"] == ["2"]
    assert "insufficient_holdout_images_per_class" in sweep["failures"]
    assert b.params["conf"].validated_against == VALIDATED_FALSE
    # The named-failure vocabulary is exhaustive by construction (describe_review_validation raises
    # on a name it can't translate), so this is what proves the new name is actually in the
    # breeder-facing vocabulary, not just the internal failures list, the same obligation
    # test_missing_class_failure_has_its_own_breeder_message already meets for holdout_missing_class.
    out = describe_review_validation(b, reviewed_image_count=10)
    assert out["validated"] is False
    assert "single image" in out["reason"]


def test_a_class_missing_entirely_gets_the_missing_class_message_not_the_single_image_one():
    """``n_present == 0`` (a class evidenced in calibration but with no holdout presence at all, not
    even one image) must keep getting ``holdout_missing_class``'s message, not
    ``insufficient_holdout_images_per_class``'s "held back in exactly one image", which would be
    false for a class held back in zero. The per-class insufficient-images conjunct is scoped to
    ``n_present == 1`` exactly, not ``< 2``, so it never fires here.
    """
    cal = _gt_records("cal", 4, swap_classes=True)  # classes 1 and 2 both evidenced in calibration
    hold = _gt_records("hold", 4, swap_classes=False, classes=(1, 1), offset=5000.0)  # holdout: only 1
    b = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h", staged_conf_floor=0.05,
                                calibration_records=cal, holdout_records=hold)
    sweep = b.params["conf"].gate_evidence
    assert "2" not in sweep["holdout_bias"]["per_class"]  # not present at all -- no n_present==0 key
    assert sweep["per_class_insufficient_images"] == []
    assert sweep["holdout_missing_classes"] == ["2"]
    out = describe_review_validation(b, reviewed_image_count=N_IMAGES)
    assert "held back" in out["reason"] and "no independent evidence" in out["reason"]
    assert "single image" not in out["reason"]


def test_gate_evidence_summary_surfaces_per_class_tolerance_and_typical_count():
    """Per-class provenance fields were added to `gate_evidence_summary` (the agent-facing compact view)
    but nothing asserted they actually reach its output -- deleting them left the suite green.
    Drives a real refusal through the real door end to end, then checks the compact view a caller
    (e.g. run_inference's response) actually sees, not just the full sidecar.
    """
    from tcip_mcp.pipelines.calibration import gate_evidence_summary

    b = resolve_operating_point(
        "bud_opening", tiled=True, dataset_hash="h", staged_conf_floor=0.05,
        calibration_records=_gt_records("cal", 4, swap_classes=True),
        holdout_records=_gt_records("hold", 4, swap_classes=True, offset=5000.0))
    sweep = b.params["conf"].gate_evidence
    out = gate_evidence_summary(b.params["conf"])
    assert out["per_class_count_bias_tolerance"] == sweep["per_class_count_bias_tolerance"]
    assert out["pooled_typical_count"] == sweep["pooled_typical_count"]
    assert out["per_class_typical_count"] == sweep["per_class_typical_count"]
    assert out["per_class_insufficient_images"] == sweep["per_class_insufficient_images"]
    # Not vacuously equal to None on both sides -- the sidecar actually carries real values here.
    assert out["per_class_typical_count"] == {"1": 4.0, "2": 4.0}
