"""K4 #4 — the count-bias gate must be conditioned on class, not pooled across classes.

The pooled per-image bias ``E[FP-FN]`` is measured by a matcher that ignores ``category_id``, so a
detector that calls every object of one class another class scores TP-only with bias 0, and one that
over-detects class A exactly as much as it under-detects class B nets to 0 as well. Either way the
delivered phenotype — a per-class count, or a fraction built from two of them — is wrong while the
operating point earns a ``validated_held_out``/``review_confirmed`` stamp.

Every gate test here drives a real door: ``resolve_operating_point_from_review`` (what
``routes/review.py`` calls) or ``resolve_operating_point`` (what ``run_inference``'s
``_calibrate_operating_point`` calls). The two at the end are about the conf sweep itself, so they
call ``sweep_operating_point`` — the thing under test — directly. None construct a sweep by hand.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("torch")  # operating_point imports evaluation, which imports torch

from tcip_mcp.utils.atomic_io import atomic_write_json  # noqa: E402

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
    sweep_operating_point,
)

_IDENTITY = {"checkpoint_sha256": "sha-model-a", "experiment_id": None}
N_IMAGES = 16
N_MATCHED = 12
N_SWAPPED = 3


@pytest.fixture(autouse=True)
def _hermetic_platform_root(tmp_path):
    """The cal/holdout split locks under ``$TCIP_PROJECT_ROOT/.tcip`` — keep it out of the repo."""
    os.environ["TCIP_PROJECT_ROOT"] = str(tmp_path)  # conftest restores the prior value


def _entry(action, cid, gt, pred, conf):
    return {"match_type": "TP" if pred and gt else ("FP" if pred else "FN"), "action": action,
            "class_id": cid, "gt_bbox_norm": gt, "pred_bbox_norm": pred, "conf": conf,
            "producer_identity": _IDENTITY, "conf_threshold": 0.01}


def _two_class_review_state(*, swap_classes: bool):
    """A dense two-class review reference: ``N_MATCHED`` correctly-called class-0 objects per image,
    plus ``N_SWAPPED`` objects the breeder confirmed as class 0.

    ``swap_classes=True`` records those last objects the way a class-confusing model's review reads:
    the class-0 object was missed (a gt-only verdict) and a class-1 detection was rejected at the
    same box. Pooled, the two cancel — the class-blind matcher pairs the class-1 prediction with the
    class-0 truth. ``False`` is the same geometry called correctly.

    The per-image jitter keeps every image's GT content byte-distinct, so K1's content-overlap gate
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
    distinct geometry — identical boxes on both sides trip K1's content-overlap gate first, and the
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

def test_review_door_refuses_a_class_compensating_reference_the_pooled_bias_calls_unbiased():
    b = resolve_operating_point_from_review(_two_class_review_state(swap_classes=True), "catkin",
                                            staged_conf_floor=0.01, bucket_identities=[_IDENTITY])
    conf = b.params["conf"]
    sweep = conf.sweep
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
    assert conf.validated_vs_gt == VALIDATED_FALSE


def test_review_door_class_failure_has_its_own_breeder_message():
    # The named-failure vocabulary is exhaustive by construction (describe_review_validation raises
    # on a name it can't translate), so a new gate name without a message is a loud error here.
    b = resolve_operating_point_from_review(_two_class_review_state(swap_classes=True), "catkin",
                                            staged_conf_floor=0.01, bucket_identities=[_IDENTITY])
    out = describe_review_validation(b, reviewed_image_count=N_IMAGES)
    assert out["validated"] is False
    assert "kind" in out["reason"].lower()
    # Not the pooled message, which would send the breeder to review more images for a mismatch
    # more reviewing cannot fix.
    assert "didn't agree closely enough" not in out["reason"]


def test_review_door_still_validates_a_multi_class_reference_that_is_honest_per_class():
    b = resolve_operating_point_from_review(_two_class_review_state(swap_classes=False), "catkin",
                                            staged_conf_floor=0.01, bucket_identities=[_IDENTITY])
    conf = b.params["conf"]
    assert conf.sweep["failures"] == []
    assert conf.sweep["per_class_count_bias_failures"] == []
    assert set(conf.sweep["holdout_bias"]["per_class"]) == {"1", "2"}
    assert conf.validated_vs_gt == VALIDATED_REVIEW_CONFIRMED


# ── the GT door (run_inference -> _calibrate_operating_point) ─────────────────────────────────

def test_gt_door_refuses_a_class_compensating_reference():
    b = resolve_operating_point(
        "catkin", dataset_hash="h", staged_conf_floor=0.05,
        calibration_records=_gt_records("cal", 4, swap_classes=True),
        holdout_records=_gt_records("hold", 4, swap_classes=True, offset=5000.0))
    sweep = b.params["conf"].sweep
    assert sweep["holdout_bias"]["count_bias_mean"] == pytest.approx(0.0)
    assert sweep["failures"] == ["count_bias_exceeds_tolerance_per_class"]
    assert b.params["conf"].validated_vs_gt == VALIDATED_FALSE


def test_gt_door_validates_the_same_geometry_called_correctly():
    b = resolve_operating_point(
        "catkin", dataset_hash="h", staged_conf_floor=0.05,
        calibration_records=_gt_records("cal", 4, swap_classes=False),
        holdout_records=_gt_records("hold", 4, swap_classes=False, offset=5000.0))
    assert b.params["conf"].sweep["failures"] == []
    assert b.params["conf"].validated_vs_gt == VALIDATED_HELD_OUT


def test_single_class_reference_is_unaffected_by_the_conditioning():
    # Catkin's shipped shape today: one detection class ({subject: 0} -> category_id 1), the
    # elongation call made by a separate classifier. Conditioning on class must be a no-op here —
    # a rail that fail-closed the only trait the platform ships would be worse than the hole.
    cal = _gt_records("cal", 4, swap_classes=False, classes=(1, 1))
    hold = _gt_records("hold", 4, swap_classes=False, classes=(1, 1), offset=5000.0)
    b = resolve_operating_point("catkin", dataset_hash="h", staged_conf_floor=0.05,
                                calibration_records=cal, holdout_records=hold)
    sweep = b.params["conf"].sweep
    hb = sweep["holdout_bias"]
    assert b.params["conf"].validated_vs_gt == VALIDATED_HELD_OUT
    assert sweep["per_class_count_bias_failures"] == []
    # The one class's statistics ARE the pooled ones, reused rather than recomputed.
    assert list(hb["per_class"]) == ["1"]
    assert hb["per_class"]["1"]["count_bias_mean"] == hb["count_bias_mean"]
    assert hb["per_class"]["1"]["count_bias_std"] == hb["count_bias_std"]
    # ...and that reuse has to be worth trusting: an independently class-filtered sweep over the
    # same records must produce the same statistics the shortcut hands back.
    explicit = sweep_operating_point(hold, tolerance=sweep["calibration"]["tolerance"],
                                     class_id=1, conf_grid=[hb["conf"]])["curve"][0]
    for key in ("tp", "fp", "fn", "count_bias_mean", "count_bias_std", "n_images"):
        assert hb["per_class"]["1"][key] == pytest.approx(explicit[key])


def test_a_class_the_holdout_never_carries_cannot_be_validated_by_its_absence():
    # Stage-6 review reached this gate's own hole through the split: the model confuses classes 1
    # and 2 in calibration, the holdout draw happens to hold only class 1, every per-class entry the
    # gate can see reads bias 0.0, and the reference was stamped validated on no class-2 evidence.
    cal = _gt_records("cal", 4, swap_classes=True)
    hold = _gt_records("hold", 4, swap_classes=False, classes=(1, 1), offset=5000.0)
    b = resolve_operating_point("catkin", dataset_hash="h", staged_conf_floor=0.05,
                                calibration_records=cal, holdout_records=hold)
    sweep = b.params["conf"].sweep
    assert sweep["per_class_count_bias_failures"] == []   # the holdout has nothing to fail on
    assert sweep["holdout_missing_classes"] == ["2"]
    assert "holdout_missing_class" in sweep["failures"]
    assert b.params["conf"].validated_vs_gt == VALIDATED_FALSE


def test_holdout_class_coverage_admits_a_reference_that_evidences_every_class():
    # The companion obligation: the coverage rule must not refuse a holdout that does carry every
    # class, including one whose objects the model correctly finds on only some images.
    cal = _gt_records("cal", 4, swap_classes=False)
    hold = _gt_records("hold", 4, swap_classes=False, offset=5000.0)
    b = resolve_operating_point("catkin", dataset_hash="h", staged_conf_floor=0.05,
                                calibration_records=cal, holdout_records=hold)
    assert b.params["conf"].sweep["holdout_missing_classes"] == []
    assert b.params["conf"].sweep["failures"] == []
    assert b.params["conf"].validated_vs_gt == VALIDATED_HELD_OUT


def test_missing_class_failure_has_its_own_breeder_message():
    # describe_review_validation raises on a gate failure it cannot translate, so this is what
    # proves the new name is in the breeder-facing vocabulary at all. Driven off a real refusal
    # rather than a hand-built bundle; the review door reaches the same message through the same
    # lookup, but which images its locked split holds back is not the fixture's to choose.
    b = resolve_operating_point(
        "catkin", dataset_hash="h", staged_conf_floor=0.05,
        calibration_records=_gt_records("cal", 4, swap_classes=True),
        holdout_records=_gt_records("hold", 4, swap_classes=False, classes=(1, 1), offset=5000.0))
    out = describe_review_validation(b, reviewed_image_count=N_IMAGES)
    assert out["validated"] is False
    assert "held back" in out["reason"] and "no independent evidence" in out["reason"]


def test_per_class_keys_survive_the_sweep_artifact_round_trip():
    # run_inference persists the whole sweep to .tcip/artifacts/operating_point_sweep_<hash>.json,
    # so the gate's per-class breakdown is only reconstructable later if its keys are JSON-stable —
    # int keys would come back as strings and silently stop matching an in-memory read.
    b = resolve_operating_point(
        "catkin", dataset_hash="h", staged_conf_floor=0.05,
        calibration_records=_gt_records("cal", 4, swap_classes=True),
        holdout_records=_gt_records("hold", 4, swap_classes=True, offset=5000.0))
    path = Path(os.environ["TCIP_PROJECT_ROOT"]) / "sweep.json"
    atomic_write_json(path, {"sweep": b.params["conf"].sweep})
    reloaded = json.loads(path.read_text(encoding="utf-8"))["sweep"]
    assert reloaded["holdout_bias"]["per_class"] == b.params["conf"].sweep["holdout_bias"]["per_class"]
    assert reloaded["per_class_count_bias_failures"] == ["1", "2"]


# ── the conf pick, which has to optimize what the gate judges ─────────────────────────────────

def test_no_conf_in_the_sweep_escapes_a_wholesale_class_swap():
    """For a model that calls every class-1 object class 2, the refusal is not an artifact of which
    conf was picked — every conf on the curve leaves a class over tolerance.

    Also pins the identity the pick and the gate both rest on: a class's per-image bias is exactly
    ``|dt_c| - |gt_c|`` at that conf, because the matched pairs cancel out of ``fp - fn``.
    """
    recs = _gt_records("cal", 4, swap_classes=True)
    for i, r in enumerate(recs):                 # a real score spread, so confs actually filter
        for k, d in enumerate(r["dt"]):
            d["score"] = 0.15 + 0.1 * k
    sweep = sweep_operating_point(recs, tolerance=0.5 * gt_class_avg_size(recs))
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
    """With three classes the pooled-unbiased conf can be the one the gate must refuse.

    Stage-6 review's case: at conf 0.9 the pooled bias is 0 while class 3 sits at -2.0 (over
    catkin's 1.0 tolerance); at conf 0.4, on the same curve, every class is within tolerance. A
    pooled pick refuses a model that has a valid operating point.
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

    sweep = sweep_operating_point(recs, tolerance=0.5 * gt_class_avg_size(recs))
    at = {round(c["conf"], 2): c for c in sweep["curve"]}
    assert at[0.95]["count_bias_mean"] == pytest.approx(0.0)       # the pooled trap
    assert at[0.95]["per_class"]["3"]["count_bias_mean"] == pytest.approx(-2.0)
    assert max(abs(s["count_bias_mean"]) for s in at[0.4]["per_class"].values()) == pytest.approx(1.0)
    assert pick_count_unbiased(sweep) == pytest.approx(0.4)

    # ...and end to end: the conf the picker now chooses is one the gate validates.
    b = resolve_operating_point("catkin", dataset_hash="h", staged_conf_floor=0.05,
                                calibration_records=recs, holdout_records=build("h", 5000.0))
    assert b.params["conf"].sweep["failures"] == []
    assert b.params["conf"].validated_vs_gt == VALIDATED_HELD_OUT
