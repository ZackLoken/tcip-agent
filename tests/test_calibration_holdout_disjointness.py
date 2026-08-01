"""The held-out calibration reference is genuinely held out — not trained on, not reused.

Five real, reproduced defects found in the calibration/holdout split's initial implementation, each
guarded here by a test that fails against the pre-fix code (see the fix commit for the exact
before/after) and passes after, per CLAUDE.md's fail-before-test discipline:

  1. The train-disjointness gate permanently blocked the explicit-val_images_dir and
     group_key_map training routes.
  2. The review-confirmation path reported a clean disjointness check for images the model
     actually trained on (extensioned review ids never matched unextensioned training stems).
  3. Both new refusals (train-provenance unresolvable/leaked) were invisible to the agent and
     misreported to the breeder as a generic "review more images".
  4. A locked cal/holdout split could go stale (a missing image) or be silently redrawn from a
     corrupt lock file.
  5. The agent could declare group_by/group_key_map but not seed/holdout_ratio for the FIRST
     (locking) calibration draw.

Plus the minor model_registry.resolve_model_identity checkpoint-deserialization finding.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

# Round 10 (2026-07-29): no built-in traits — seed_catkin_trait_spec (conftest.py) writes a real
# catkin.yml into this test's pinned project root so trait="catkin" call sites keep resolving.
pytestmark = pytest.mark.usefixtures("seed_catkin_trait_spec")

torch = pytest.importorskip("torch")

from tcip_annotation import json_io  # noqa: E402
from tcip_annotation.state import Annotation, BBox  # noqa: E402

IMG = 32


def _save_png(path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (IMG, IMG), color=(128, 128, 128)).save(path)


def _detection_dataset(root: Path, stems: list[str]) -> tuple[Path, Path]:
    """One image + one foreground annotation per stem — enough for build_dataset/_auto_train_val."""
    images_dir = root / "images"
    labels_dir = root / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    for s in stems:
        _save_png(images_dir / f"{s}.png")
        json_io.write_annotations(
            str(labels_dir / f"{s}.json"),
            [Annotation(subject="catkin", geometry=BBox(2, 2, 10, 10))], IMG, IMG, keep_empty=True,
        )
    return images_dir, labels_dir


class _CalStub:
    """Predictor stub with the mutable operating-point surface run_inference/_calibrate_operating_point
    set — every prediction comes back empty, which is enough to exercise the split/provenance
    machinery without a real model forward pass."""

    def __init__(self) -> None:
        from types import SimpleNamespace

        self.model = SimpleNamespace(score_thresh=0.5, nms_thresh=0.5, detections_per_img=100)
        self.device = "cpu"
        self.score_threshold = 0.5
        self.train_tile_size = None
        self.train_overlap = None

    def predict_batch(self, paths, **kw):
        return [{"image": p, "width": IMG, "height": IMG,
                 "boxes": [], "scores": [], "labels": [], "count": 0} for p in paths]


# ===========================================================================
# Finding 1 — the train-disjointness gate no longer permanently blocks the
# explicit-val_images_dir ("external") and group_key_map training routes.
# ===========================================================================

def test_finding1_external_marker_not_permanently_blocked_when_disjoint(tmp_path, monkeypatch):
    """The pre-fix headline bug: group_by="external" (the explicit-val_images_dir route) used to
    map to unresolvable=True forever. It must now fall back to the exact-stem check and validate
    when genuinely disjoint."""
    from tcip_mcp.pipelines.operating_point import resolve_operating_point

    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    exp_dir = tmp_path / ".tcip" / "experiments" / "exp_ext"
    exp_dir.mkdir(parents=True)
    (exp_dir / "split.json").write_text(
        json.dumps({"train": ["train_a", "train_b"], "group_by": "external"}), encoding="utf-8")

    cal, hold = _good_dense_op_records()
    b = resolve_operating_point("catkin", dataset_hash="h1",
                                calibration_records=cal, holdout_records=hold,
                                staged_conf_floor=0.01, experiment_id="exp_ext")
    conf = b.get("conf")
    assert conf.validated_against == "held_out_annotations"  # NOT permanently blocked
    td = conf.sweep["train_disjointness"]
    assert td["unresolvable"] is False
    assert td["group_check"] == "not_performed"
    assert td["leaked_stems"] == []


def test_finding1_external_marker_still_catches_a_real_leak(tmp_path, monkeypatch):
    """The exact-stem fallback must still refuse a genuine leak, not just always pass."""
    from tcip_mcp.pipelines.operating_point import resolve_operating_point

    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    exp_dir = tmp_path / ".tcip" / "experiments" / "exp_ext2"
    exp_dir.mkdir(parents=True)
    # Training trained on "c_a" — the SAME stem the calibration reference uses below.
    (exp_dir / "split.json").write_text(
        json.dumps({"train": ["c_a", "other_stem"], "group_by": "external"}), encoding="utf-8")

    b = resolve_operating_point("catkin", dataset_hash="h1",
                                calibration_records=_op_records("c"),
                                holdout_records=_op_records("h", shift=3.0),
                                experiment_id="exp_ext2")
    conf = b.get("conf")
    assert conf.validated_against == "false"
    td = conf.sweep["train_disjointness"]
    assert td["unresolvable"] is False
    assert td["group_check"] == "not_performed"
    assert td["leaked_stems"] == ["c_a"]  # caught even with no group policy at all


def test_finding1_group_key_map_end_to_end_not_permanently_blocked(tmp_path):
    """group_key_map, exercised through _auto_train_val -> _persist_split_manifest ->
    _train_disjointness, must not permanently block the model — and the persisted map must
    actually be used for a real group-level leak check, not just declared unresolvable."""
    from tcip_mcp.experiments import create_experiment, experiments_dir
    from tcip_mcp.pipelines.operating_point import _train_disjointness
    from tcip_mcp.tools.training_tools import _auto_train_val, _persist_split_manifest

    stems = ["imgA0", "imgA1", "imgB0", "imgB1"]
    images_dir, labels_dir = _detection_dataset(tmp_path / "ds", stems)
    group_key_map = {"imgA0": "gA", "imgA1": "gA", "imgB0": "gB", "imgB1": "gB"}
    data_cfg = {
        "images_dir": str(images_dir), "labels_dir": str(labels_dir), "subject": "catkin",
        "auto_val": True,
        "split": {"val_ratio": 0.5, "seed": 1, "group_key_map": dict(group_key_map)},
    }
    train_ds, val_ds = _auto_train_val("detection", data_cfg, None)
    assert val_ds is not None
    # The two groups (gA/gB) never straddle train/val — group-coherent by construction.
    train_groups = {group_key_map[s] for s in train_ds.stems}
    val_groups = {group_key_map[s] for s in val_ds.stems}
    assert train_groups.isdisjoint(val_groups)

    create_experiment("e1", {})
    _persist_split_manifest("e1", train_ds, val_ds, data_cfg)
    split = json.loads((experiments_dir() / "e1" / "split.json").read_text())
    assert split["group_by"] == "explicit_map"
    assert split["group_key_map"] == group_key_map  # the map itself, not just the policy name

    # A calibration reference drawn from val's own stems must NOT be permanently blocked.
    td = _train_disjointness("e1", set(val_ds.stems), set())
    assert td["unresolvable"] is False
    assert td["group_check"] == "performed"  # every stem covered by the persisted map
    assert td["leaked_groups"] == []

    # The mechanism genuinely checks GROUPS, not just exact stems: a calibration id that is a
    # DIFFERENT stem mapped to the SAME group as a training stem must be caught.
    train_group = group_key_map[train_ds.stems[0]]
    data_cfg["split"]["group_key_map"]["extra_leak_stem"] = train_group
    _persist_split_manifest("e1", train_ds, val_ds, data_cfg)
    td_leak = _train_disjointness("e1", {"extra_leak_stem"}, set())
    assert td_leak["unresolvable"] is False
    assert td_leak["leaked_groups"] == [train_group]


# ===========================================================================
# Finding 2 — review-confirmation image ids are stemmed, matching training stems.
# ===========================================================================

_IDENTITY = {"checkpoint_sha256": "deadbeef", "experiment_id": None}


def test_finding2_review_to_records_stems_the_image_id():
    from tcip_mcp.pipelines.feedback.review_calibration import review_to_records

    review_state = {"image": {"srcA_0_0.jpg": {"img_status": "completed", "detections": [
        {"action": "accepted", "class_id": 0,
         "gt_bbox_norm": [0.5, 0.5, 0.1, 0.1], "pred_bbox_norm": [0.5, 0.5, 0.1, 0.1], "conf": 0.9,
         "producer_identity": _IDENTITY},
    ]}}}
    recs = review_to_records(review_state, bucket_identities=[_IDENTITY])
    assert recs[0]["image_id"] == "srcA_0_0"  # stemmed, not "srcA_0_0.jpg"


def test_finding2_review_confirmed_leak_now_detected(tmp_path, monkeypatch):
    """The exact reviewer-reproduced defect: extensioned review ids in the SAME tile group as
    training stems must now be caught, not silently reported clean."""
    from tcip_mcp.pipelines.feedback.review_calibration import resolve_operating_point_from_review

    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    exp_dir = tmp_path / ".tcip" / "experiments" / "exp_review"
    exp_dir.mkdir(parents=True)
    # Trained on two tiles of source "srcA".
    (exp_dir / "split.json").write_text(
        json.dumps({"train": ["srcA_0_0", "srcA_0_1"], "group_by": "tile_prefix"}), encoding="utf-8")

    def _entry(gt, pred, conf):
        return {"action": "accepted", "class_id": 0, "gt_bbox_norm": gt, "pred_bbox_norm": pred,
                "conf": conf, "producer_identity": _IDENTITY}

    # Two reviewed images — both further tiles of the SAME source the model trained on — keyed
    # WITH an extension, exactly as review state stores them. gt_preexisting=True (Fix H) so these
    # records aren't excluded from the gate as unadjudicated — this test is about train
    # disjointness, not FN-coverage.
    review_state = {"image": {
        "srcA_0_2.jpg": {"img_status": "completed", "gt_preexisting": True,
                         "detections": [_entry([0.25, 0.25, 0.05, 0.05], [0.25, 0.25, 0.05, 0.05], 0.05)]},
        "srcA_0_3.jpg": {"img_status": "completed", "gt_preexisting": True,
                         "detections": [_entry([0.5, 0.5, 0.05, 0.05], [0.5, 0.5, 0.05, 0.05], 0.05)]},
    }}
    bundle = resolve_operating_point_from_review(
        review_state, "catkin", group_by="stem", experiment_id="exp_review",
        bucket_identities=[_IDENTITY])
    td = bundle.get("conf").sweep["train_disjointness"]
    assert td["leaked_groups"] == ["srcA"]  # would be [] before the fix
    assert bundle.get("conf").validated_against == "false"


# ===========================================================================
# Finding 3 — the new refusals are visible to the agent and honestly described.
# ===========================================================================

def _review_bundle(sweep: dict):
    from tcip_mcp.pipelines.resolution import VALIDATED_FALSE, ResolvedBundle, derived

    conf = derived("conf", 0.42, requires_validation=True, validation_kind="annotations",
                   derived_from="count-unbiased center-match sweep over review verdicts",
                   validated_against=VALIDATED_FALSE, dataset_scoped=True, dataset_hash="abc",
                   sweep=sweep)
    return ResolvedBundle(trait="catkin", dataset_hash="abc", params={"conf": conf})


def test_finding3_describe_review_validation_unresolvable_message():
    from tcip_mcp.pipelines.feedback import describe_review_validation

    b = _review_bundle({"conf_censored": False, "disjoint": True, "passed_holdout": False,
                        "train_disjointness": {"unresolvable": True},
                        "failures": ["train_disjointness_unresolvable"]})
    out = describe_review_validation(b, reviewed_image_count=4)
    assert out["validated"] is False
    assert "training record" in out["reason"]


def test_finding3_describe_review_validation_leaked_message():
    from tcip_mcp.pipelines.feedback import describe_review_validation

    b = _review_bundle({"conf_censored": False, "disjoint": True, "passed_holdout": False,
                        "train_disjointness": {"unresolvable": False, "leaked_groups": ["srcA"],
                                                "leaked_stems": []},
                        "failures": ["train_disjointness_leaked"]})
    out = describe_review_validation(b, reviewed_image_count=4)
    assert out["validated"] is False
    assert "also used to train" in out["reason"]


def test_finding3_describe_review_validation_content_duplicated_message():
    from tcip_mcp.pipelines.feedback import describe_review_validation

    b = _review_bundle({"conf_censored": False, "disjoint": True, "passed_holdout": False,
                        "content_duplicated": True, "failures": ["content_duplicated"]})
    out = describe_review_validation(b, reviewed_image_count=4)
    assert out["validated"] is False
    assert "duplicate" in out["reason"]


def test_finding3_sweep_summary_surfaces_disjointness_fields():
    from tcip_mcp.pipelines.resolution import VALIDATED_FALSE, derived
    from tcip_mcp.tools.inference_tools import _sweep_summary

    conf = derived("conf", 0.4, requires_validation=True, validation_kind="annotations", derived_from="x",
                   validated_against=VALIDATED_FALSE,
                   sweep={"disjoint": True, "content_overlap_frac": 0.0, "content_duplicated": False,
                          "train_disjointness": {"unresolvable": False, "leaked_groups": ["g1"]},
                          "passed_holdout": False, "conf_censored": False, "count_bias_tolerance_frac": 1.0,
                          "pooled_count_bias_tolerance": 4.0})
    out = _sweep_summary(conf)
    assert out["disjoint"] is True
    assert out["content_overlap_frac"] == 0.0
    assert out["train_disjointness"]["leaked_groups"] == ["g1"]  # visible, not silently dropped
    # K4 residual, stage-6 review Finding F3: the renamed/new fields must actually reach the OUTPUT,
    # not just be present somewhere in the input sweep dict — the earlier version of this test did
    # the latter, which a key-name drift in `_sweep_summary`'s own `.get(...)` calls could not fail.
    assert out["count_bias_tolerance_frac"] == 1.0
    assert out["pooled_count_bias_tolerance"] == 4.0


def test_finding5_sweep_summary_surfaces_split_policy_divergence():
    """attach_split_policy_provenance writes into conf.sweep; _sweep_summary must forward those
    keys too, or run_inference's actual response never shows a caller their declared seed/ratio
    didn't take effect against an existing lock -- only the persisted sweep artifact would."""
    from tcip_mcp.pipelines.resolution import VALIDATED_FALSE, derived
    from tcip_mcp.tools.inference_tools import _sweep_summary

    conf = derived("conf", 0.4, requires_validation=True, validation_kind="annotations", derived_from="x",
                   validated_against=VALIDATED_FALSE,
                   sweep={"passed_holdout": False, "conf_censored": False, "count_bias_tolerance_frac": 1.0,
                          "split_policy_divergence": {"requested": {"seed": 7}, "locked": {"seed": 0}},
                          "split_unlocked_stems": ["new_stem_0_0"]})
    out = _sweep_summary(conf)
    assert out["split_policy_divergence"] == {"requested": {"seed": 7}, "locked": {"seed": 0}}
    assert out["split_unlocked_stems"] == ["new_stem_0_0"]


# ===========================================================================
# Finding 4 — a locked split can't go stale silently, and a corrupt lock refuses.
# ===========================================================================

def test_finding4_stale_locked_stem_refuses_cleanly():
    from tcip_mcp.pipelines.data.splits import resolve_locked_cal_holdout_split

    stems_full = ["a_0_0", "a_0_1", "b_0_0", "b_0_1"]
    resolve_locked_cal_holdout_split(stems_full, identity_hash="stale-test", seed=1)

    # One stem's image/label vanished since the lock was drawn.
    stems_now = ["a_0_0", "a_0_1", "b_0_0"]
    with pytest.raises(ValueError, match="no longer present"):
        resolve_locked_cal_holdout_split(stems_now, identity_hash="stale-test", seed=1)


def test_finding4_corrupt_lock_file_refuses_instead_of_silent_redraw():
    from tcip_mcp.pipelines.data.splits import cal_holdout_lock_path, resolve_locked_cal_holdout_split

    lock_path = cal_holdout_lock_path("corrupt-test")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(ValueError, match="corrupt"):
        resolve_locked_cal_holdout_split(["a_0_0", "b_0_0"], identity_hash="corrupt-test", seed=1)

    # force_redraw=True IS the deliberate, audited fix and proceeds past the corrupt file — but
    # its redraw_history honestly starts fresh (nothing recoverable from an unreadable file).
    redrawn = resolve_locked_cal_holdout_split(
        ["a_0_0", "b_0_0"], identity_hash="corrupt-test", seed=1, force_redraw=True,
        timestamp="2026-01-01T00:00:00Z")
    assert len(redrawn["redraw_history"]) == 1
    assert redrawn["redraw_history"][0]["old_content_hash"] is None


def test_finding4_missing_image_refuses_cleanly_not_keyerror(tmp_path):
    """The KeyError-risk finding, at the tool level: a locked stem whose image was later deleted
    must produce a clean ValueError through _calibrate_operating_point, never a bare KeyError from
    a stale stem_to_image lookup."""
    import tcip_mcp.tools.inference_tools as itools

    stems = ["a_0_0", "a_0_1", "b_0_0", "b_0_1"]
    images_dir, labels_dir = _detection_dataset(tmp_path / "ds", stems)

    kwargs = dict(tile=False, tile_size=IMG, overlap=0.2, tile_batch_size=8,
                  global_nms_iou=0.3, postprocess="nms", cross_tile_nms=None, max_dets=None)
    # First call locks the split over all 4 stems.
    itools._calibrate_operating_point(_CalStub(), "catkin", str(labels_dir), str(images_dir), **kwargs)

    (images_dir / "b_0_1.png").unlink()  # an image vanishes after the lock

    with pytest.raises(ValueError, match="no longer present"):
        itools._calibrate_operating_point(_CalStub(), "catkin", str(labels_dir), str(images_dir), **kwargs)


def test_finding4_force_redraw_shares_the_labels_intersect_images_scan(tmp_path):
    """force_redraw_cal_holdout_split(images_dir=...) must use the SAME labels-intersect-images
    scan _calibrate_operating_point uses, not a second independent labels-only glob — a stem
    with no image on disk must not enter the redraw's stem universe."""
    from tcip_mcp.tools.inference_tools import force_redraw_cal_holdout_split

    stems = ["a_0_0", "a_0_1", "b_0_0", "b_0_1"]
    images_dir, labels_dir = _detection_dataset(tmp_path / "ds", stems)
    (images_dir / "b_0_1.png").unlink()  # labeled but no image

    result = force_redraw_cal_holdout_split(
        labels_dir=str(labels_dir), images_dir=str(images_dir), seed=1,
        reason="finding 4 coverage test")
    assert "error" not in result
    all_new = result["new_membership"]["calibration"] + result["new_membership"]["holdout"]
    assert "b_0_1" not in all_new


# ===========================================================================
# Finding 5 — a declared seed/holdout_ratio reaches the FIRST (locking) draw.
# ===========================================================================

def test_finding5_declared_seed_and_holdout_ratio_reach_the_first_draw(tmp_path):
    import tcip_mcp.tools.inference_tools as itools

    stems = [f"src{g}_{t}_0" for g in range(4) for t in range(2)]
    images_dir, labels_dir = _detection_dataset(tmp_path / "ds", stems)

    bundle, _dh, _n_excluded = itools._calibrate_operating_point(
        _CalStub(), "catkin", str(labels_dir), str(images_dir),
        tile=False, tile_size=IMG, overlap=0.2, tile_batch_size=8,
        global_nms_iou=0.3, postprocess="nms", cross_tile_nms=None, max_dets=None,
        seed=7, holdout_ratio=0.75,
    )
    policy = bundle.get("conf").sweep["split_policy"]
    assert policy["seed"] == 7
    assert policy["holdout_ratio"] == pytest.approx(0.75)  # not the 0/0.5 defaults


# ===========================================================================
# Round 4 (task #26) — the calibration record builder's whole-image exclusion
# for an unlabeled attribute instance must be COUNTED and disclosed, not a
# silent filter (matching evaluation.py's n_excluded_incomplete_attribute).
# ===========================================================================

def test_calibration_discloses_excluded_incomplete_attribute_count(tmp_path):
    """A stem with any instance unlabeled for `attribute` is dropped whole from the cal/holdout
    record set (the missing-label-file precedent) — the count must travel back to the caller,
    not vanish, so a caller can see the reference shrank rather than assume every stem measured."""
    import tcip_mcp.tools.inference_tools as itools
    from tcip_mcp.class_registry import Attribute, ClassRegistry, Subject, write_registry

    root = tmp_path / "ds"
    images_dir, labels_dir = root / "images", root / "labels"
    write_registry(root / "classes.json", ClassRegistry(subjects=(
        Subject(name="catkin", attributes=(
            Attribute(name="state", type="categorical", values=("elongated", "dormant")),)),)))

    stems = ["complete_a", "complete_b", "partial_a", "partial_b"]
    for s in stems:
        _save_png(images_dir / f"{s}.png")
    for s in ("complete_a", "complete_b"):
        json_io.write_annotations(str(labels_dir / f"{s}.json"), [
            Annotation(subject="catkin", geometry=BBox(2, 2, 10, 10), attributes={"state": "elongated"}),
        ], IMG, IMG)
    for s in ("partial_a", "partial_b"):
        json_io.write_annotations(str(labels_dir / f"{s}.json"), [
            Annotation(subject="catkin", geometry=BBox(2, 2, 10, 10), attributes={"state": "elongated"}),
            Annotation(subject="catkin", geometry=BBox(15, 15, 20, 20)),  # no `state` -- unlabeled
        ], IMG, IMG)

    stub = _CalStub()
    stub.config = {"data": {"subject": "catkin", "attribute": "state"}}

    _bundle, _dh, n_excluded = itools._calibrate_operating_point(
        stub, "catkin", str(labels_dir), str(images_dir),
        tile=False, tile_size=IMG, overlap=0.2, tile_batch_size=8,
        global_nms_iou=0.3, postprocess="nms", cross_tile_nms=None, max_dets=None,
        group_by="stem", seed=0, holdout_ratio=0.5,
    )

    assert n_excluded == 2  # partial_a + partial_b, wherever the split put them


def test_k18_calibration_attribute_registry_refusal_reaches_the_caller(tmp_path):
    """K18 B2: before this fix, _calibrate_operating_point's bare `except Exception` around
    _resolve_registry_id_map silently degraded an attribute-classification calibration to a
    single-class GT read when the registry read failed for a real reason — sitting directly on
    the calibration/operating-point rail, worse than the delivery-grade-eval instance of the same
    bug. No classes.json exists here for an attribute-scoped config, so this must now refuse."""
    import tcip_mcp.tools.inference_tools as itools

    stems = ["a_0_0", "a_0_1"]
    images_dir, labels_dir = _detection_dataset(tmp_path / "ds", stems)

    stub = _CalStub()
    stub.config = {"data": {"subject": "catkin", "attribute": "state"}}  # no classes.json written

    with pytest.raises(ValueError, match="classes.json"):
        itools._calibrate_operating_point(
            stub, "catkin", str(labels_dir), str(images_dir),
            tile=False, tile_size=IMG, overlap=0.2, tile_batch_size=8,
            global_nms_iou=0.3, postprocess="nms", cross_tile_nms=None, max_dets=None,
            group_by="stem", seed=0, holdout_ratio=0.5,
        )


# ===========================================================================
# Minor — resolve_model_identity's checkpoint deserialization.
# ===========================================================================

class _UnsafeGlobal:
    """A plain object with no torch.load(weights_only=True) allowlist entry — module-level so
    torch.save/pickle can reference it, unlike a function-local class."""


def test_minor_resolve_model_identity_reads_the_codebase_own_stamped_checkpoints(tmp_path):
    """weights_only=True must still read a checkpoint saved the way stamp_model_ref produces it —
    the rail must admit valid work, not only reject foreign payloads."""
    from tcip_mcp.model_registry import resolve_model_identity

    ckpt = tmp_path / "m.pt"
    torch.save({"model_state_dict": {"w": torch.zeros(2, 2)},
               "optimizer_state_dict": {"state": {}, "param_groups": [{"lr": 1e-3}]},
               "experiment_id": "exp_abc", "kind": "tcip_module"}, ckpt)
    identity = resolve_model_identity(ckpt)
    assert identity["experiment_id"] == "exp_abc"


def test_minor_resolve_model_identity_logs_unreadable_stamp_distinctly(tmp_path, caplog):
    """An unreadable (not merely absent) stamp is logged distinctly rather than falling through to
    the same silent 'pass' a genuinely foreign checkpoint gets."""
    from tcip_mcp.model_registry import resolve_model_identity

    ckpt = tmp_path / "m.pt"
    torch.save({"model_state_dict": {}, "weird": _UnsafeGlobal()}, ckpt)
    with caplog.at_level(logging.WARNING, logger="tcip_mcp.model_registry"):
        identity = resolve_model_identity(ckpt)
    assert identity["experiment_id"] is None
    assert any("could not read a stamped experiment_id" in r.message for r in caplog.records)


# --- shared record fixture (mirrors test_operating_point.py's _records) -----------------------

def _op_box(cx: float, cy: float, s: float = 20.0) -> list[float]:
    return [cx - s / 2, cy - s / 2, s, s]


def _op_ann(cx, cy, cid=0, score=None):
    a = {"category_id": cid, "bbox": _op_box(cx, cy)}
    if score is not None:
        a["score"] = score
    return a


def _op_records(idp="c", *, shift: float = 0.0):
    a = {"width": 400, "height": 400, "image_id": f"{idp}_a",
         "gt": [_op_ann(100 + shift, 100)],
         "dt": [_op_ann(100, 100, score=0.9), _op_ann(300, 300, score=0.6)]}
    b = {"width": 400, "height": 400, "image_id": f"{idp}_b",
         "gt": [_op_ann(100 + shift, 100), _op_ann(200 + shift, 200)],
         "dt": [_op_ann(100, 100, score=0.9), _op_ann(200, 200, score=0.3)]}
    return [a, b]


def _good_dense_op_records():
    """K2 rule 17: a realistic dense reference for tests that expect the holdout gate to VALIDATE —
    the 2-image ``_op_records`` toy's per-image variance now trips Fix C's equivalence criterion at
    n=2 (a genuine, intended tightening; see test_operating_point.py for the same fixture idiom)."""
    from tests._dense_op_fixtures import dense_records

    n_images, objects_per_image = 20, 80
    miss, fp = [0] * n_images, [1] * n_images
    cal = dense_records(n_images=n_images, objects_per_image=objects_per_image, id_prefix="c",
                        miss_pattern=miss, fp_pattern=fp, score=0.9, fp_score=0.05)
    hold = dense_records(n_images=n_images, objects_per_image=objects_per_image, id_prefix="h",
                         shift=5.0, miss_pattern=miss, fp_pattern=fp, score=0.9, fp_score=0.05)
    return cal, hold
