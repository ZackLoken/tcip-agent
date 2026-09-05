"""The held-out calibration reference is genuinely held out: not trained on, not reused.

Covers: the train-disjointness gate must not permanently block the explicit-val_images_dir and
group_key_map training routes; the review-confirmation path must detect a training leak even when
review image ids carry an extension review state stores them with, unlike unextensioned training
stems; an unresolvable or leaked train-disjointness check must be visible to the agent, not
misreported as a generic "review more images"; a locked cal/holdout split must refuse rather than
silently redraw when stale (a missing image) or when its lock file is corrupt; and a declared
seed/holdout_ratio must reach the first (locking) calibration draw, not only later redraws. Also
covers resolve_model_identity reading the codebase's own stamped checkpoints off a verified load.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# no built-in traits, seed_bud_trait_spec (conftest.py) writes a real bud.yml into this
# test's pinned platform state root so trait="bud_opening" call sites keep resolving.
pytestmark = pytest.mark.usefixtures("seed_bud_trait_spec")

torch = pytest.importorskip("torch")

from tcip_annotation import json_io  # noqa: E402
from tcip_annotation.state import Annotation, BBox  # noqa: E402

IMG = 32


def _save_png(path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (IMG, IMG), color=(128, 128, 128)).save(path)


def _detection_dataset(root: Path, stems: list[str]) -> tuple[Path, Path]:
    """One image + one foreground annotation per stem: enough for build_dataset/auto_train_val."""
    images_dir = root / "images"
    labels_dir = root / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    for s in stems:
        _save_png(images_dir / f"{s}.png")
        json_io.write_annotations(
            str(labels_dir / f"{s}.json"),
            [Annotation(subject="bud", geometry=BBox(2, 2, 10, 10))], IMG, IMG, keep_empty=True,
        )
    return images_dir, labels_dir


class _CalStub:
    """Predictor stub with the mutable operating-point surface run_inference/calibrate_operating_point
    set: every prediction comes back empty, which is enough to exercise the split/provenance
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
# The train-disjointness gate must not permanently block the explicit-val_images_dir
# ("external") and group_key_map training routes.
# ===========================================================================

def test_external_marker_not_permanently_blocked_when_disjoint(tmp_path, monkeypatch):
    """group_by="external" (the explicit-val_images_dir route) falls back to the exact-stem check
    and validates when genuinely disjoint, rather than mapping to unresolvable=True forever."""
    import tcip_store

    from tcip_mcp.experiments import split_key
    from tcip_mcp.pipelines.operating_point import resolve_operating_point

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    tcip_store.replace(split_key("exp_ext"), {"train": ["train_a", "train_b"], "group_by": "external"})

    cal, hold = _good_dense_op_records()
    b = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h1",
                                calibration_records=cal, holdout_records=hold,
                                staged_conf_floor=0.01, experiment_id="exp_ext")
    conf = b.get("conf")
    assert conf.validated_against == "held_out_annotations"  # not permanently blocked
    td = conf.gate_evidence["train_disjointness"]
    assert td["unresolvable"] is False
    assert td["group_check"] == "not_performed"
    assert td["leaked_stems"] == []


def test_external_marker_still_catches_a_real_leak(tmp_path, monkeypatch):
    """The exact-stem fallback must still refuse a genuine leak, not just always pass."""
    import tcip_store

    from tcip_mcp.experiments import split_key
    from tcip_mcp.pipelines.operating_point import resolve_operating_point

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    # Training trained on "c_a", the same stem the calibration reference uses below.
    tcip_store.replace(split_key("exp_ext2"), {"train": ["c_a", "other_stem"], "group_by": "external"})

    b = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h1",
                                calibration_records=_op_records("c"),
                                holdout_records=_op_records("h", shift=3.0),
                                experiment_id="exp_ext2")
    conf = b.get("conf")
    assert conf.validated_against == "false"
    td = conf.gate_evidence["train_disjointness"]
    assert td["unresolvable"] is False
    assert td["group_check"] == "not_performed"
    assert td["leaked_stems"] == ["c_a"]  # caught even with no group policy at all


def test_group_key_map_end_to_end_not_permanently_blocked(tmp_path):
    """group_key_map, exercised through auto_train_val -> persist_split_manifest ->
    _train_disjointness, must not permanently block the model, and the persisted map must
    actually be used for a real group-level leak check, not just declared unresolvable."""
    from tcip_mcp.experiments import create_experiment, read_split_manifest
    from tcip_mcp.pipelines.operating_point import _train_disjointness
    from tcip_mcp.pipelines.data.split_construction import auto_train_val, persist_split_manifest

    stems = ["imgA0", "imgA1", "imgB0", "imgB1"]
    images_dir, labels_dir = _detection_dataset(tmp_path / "ds", stems)
    group_key_map = {"imgA0": "gA", "imgA1": "gA", "imgB0": "gB", "imgB1": "gB"}
    data_cfg = {
        "images_dir": str(images_dir), "labels_dir": str(labels_dir), "subject": "bud",
        "auto_val": True,
        "split": {"val_ratio": 0.5, "seed": 1, "group_key_map": dict(group_key_map)},
    }
    train_ds, val_ds, _ = auto_train_val("detection", data_cfg, None)
    assert val_ds is not None
    # The two groups (gA/gB) never straddle train/val: group-coherent by construction.
    train_groups = {group_key_map[s] for s in train_ds.stems}
    val_groups = {group_key_map[s] for s in val_ds.stems}
    assert train_groups.isdisjoint(val_groups)

    create_experiment("e1", {})
    persist_split_manifest("e1", train_ds, val_ds, data_cfg)
    split = read_split_manifest("e1")
    assert split["group_by"] == "explicit_map"
    assert split["group_key_map"] == group_key_map  # the map itself, not just the policy name

    # A calibration reference drawn from val's own stems must not be permanently blocked.
    td = _train_disjointness("e1", set(val_ds.stems), set())
    assert td["unresolvable"] is False
    assert td["group_check"] == "performed"  # every stem covered by the persisted map
    assert td["leaked_groups"] == []

    # The mechanism genuinely checks groups, not just exact stems: a calibration id that is a
    # different stem mapped to the same group as a training stem must be caught.
    train_group = group_key_map[train_ds.stems[0]]
    data_cfg["split"]["group_key_map"]["extra_leak_stem"] = train_group
    persist_split_manifest("e1", train_ds, val_ds, data_cfg)
    td_leak = _train_disjointness("e1", {"extra_leak_stem"}, set())
    assert td_leak["unresolvable"] is False
    assert td_leak["leaked_groups"] == [train_group]


# a spatial split's manifest never reads as a bare-stem leak, and _train_disjointness still
# catches a genuine same-source reference.

def test_train_disjointness_named_group_by_output_unchanged(tmp_path, monkeypatch):
    """Byte-identical to the pre-spatial-split behavior: a tile_prefix split.json is untouched
    by the spatial_strip branch in _train_disjointness."""
    import tcip_store

    from tcip_mcp.experiments import split_key
    from tcip_mcp.pipelines.operating_point import _train_disjointness

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    tcip_store.replace(split_key("exp_named"),
                       {"train": ["srcA_0_0", "srcA_0_1", "srcB_0_0"], "group_by": "tile_prefix"})
    result = _train_disjointness("exp_named", {"srcB_0_0"}, set())
    # leaked_stems stays empty here: a named group_by resolves every train and reference stem,
    # so nothing falls through to the exact-stem fallback; leaked_groups is the real signal.
    assert result == {
        "checked": True, "unresolvable": False,
        "leaked_groups": ["srcB"], "leaked_stems": [], "group_check": "performed",
    }


def test_train_disjointness_spatial_strip_detects_same_source_leak(tmp_path, monkeypatch):
    import tcip_store

    from tcip_mcp.experiments import split_key
    from tcip_mcp.pipelines.operating_point import _train_disjointness

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    tcip_store.replace(split_key("exp_spatial"), {
        "train": ["mosaic::strip_x_1"],
        "val": ["mosaic::strip_x_0"], "group_by": "spatial_strip",
    })

    # A reference drawn from the same source is a real leak: region-scoping aside, the trained
    # pixels and the reference still share one source image.
    leaked = _train_disjointness("exp_spatial", {"mosaic"}, set())
    assert leaked["group_check"] == "spatial_strip"
    assert leaked["leaked_groups"] == ["mosaic"]
    assert leaked["unresolvable"] is False

    clean = _train_disjointness("exp_spatial", {"other_mosaic"}, set())
    assert clean["leaked_groups"] == []


def test_train_disjointness_rects_kwargs_default_none_is_byte_identical(tmp_path, monkeypatch):
    """cal_rects/hold_rects default to None: passing them explicitly as None must produce
    byte-identical output to every existing caller, which omits them entirely."""
    import tcip_store

    from tcip_mcp.experiments import split_key
    from tcip_mcp.pipelines.operating_point import _train_disjointness

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    tcip_store.replace(split_key("exp_noop"),
                       {"train": ["mosaic::strip_x_1"], "group_by": "spatial_strip"})

    omitted = _train_disjointness("exp_noop", {"mosaic"}, set())
    explicit_none = _train_disjointness(
        "exp_noop", {"mosaic"}, set(), cal_rects=None, hold_rects=None)
    assert omitted == explicit_none == {
        "checked": True, "unresolvable": False,
        "leaked_groups": ["mosaic"], "leaked_stems": [], "group_check": "spatial_strip",
    }


def test_train_disjointness_spatial_strip_geometric_containment(tmp_path, monkeypatch):
    """When a caller supplies cal_rects/hold_rects against a spatial_strip split, the check
    becomes geometric containment (fully inside a persisted non-train region, disjoint from
    every persisted train region) instead of the lexical same-source check, and catches a leak
    the lexical check alone would miss: a rect that spills into train from a source stem that
    isn't literally the training source's own name."""
    import tcip_store

    from tcip_mcp.experiments import split_key
    from tcip_mcp.pipelines.operating_point import _train_disjointness

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    tcip_store.replace(split_key("exp_geo"), {
        "train": ["mosaic::strip_x_1"], "group_by": "spatial_strip",
        "spatial": {
            "train_region": [[0, 0, 500, 1000]],
            "val_region": [[500, 0, 750, 1000]],
            "test_region": [[750, 0, 1000, 1000]],
        },
    })

    clean = _train_disjointness(
        "exp_geo", {"mosaic"}, set(), cal_rects={"mosaic": (550, 100, 700, 300)})
    assert clean["leaked_groups"] == []
    assert clean["group_check"] == "spatial_strip_geometric"
    assert clean["unresolvable"] is False

    # A rect from a source whose name is not "mosaic" at all -- the lexical check has nothing
    # to key off, but the region it actually covers spills into the persisted train area.
    leaked = _train_disjointness(
        "exp_geo", set(), {"other_mosaic"}, hold_rects={"other_mosaic": (10, 10, 100, 100)})
    assert leaked["leaked_groups"] == ["other_mosaic"]

    # Straddling the train/val boundary: not fully contained in any single non-train region.
    straddling = _train_disjointness(
        "exp_geo", {"mosaic"}, set(), cal_rects={"mosaic": (400, 100, 600, 300)})
    assert straddling["leaked_groups"] == ["mosaic"]


def test_train_disjointness_spatial_strip_geometric_admits_calibration_region(tmp_path, monkeypatch):
    """A rect fully inside a persisted calibration_region (the four-way split's own reserved
    calibration side, distinct from val/test) must clear the geometric check exactly like a
    val/test rect does -- calibration_region was omitted from the non-train set once and every
    block calibration failed as a result; this pins it against regression."""
    import tcip_store

    from tcip_mcp.experiments import split_key
    from tcip_mcp.pipelines.operating_point import _train_disjointness

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    tcip_store.replace(split_key("exp_geo4"), {
        "train": ["mosaic::strip_x_1"], "group_by": "spatial_strip",
        "spatial": {
            "train_region": [[0, 0, 500, 1000]],
            "val_region": [[500, 0, 650, 1000]],
            "calibration_region": [[650, 0, 800, 1000]],
            "test_region": [[800, 0, 1000, 1000]],
        },
    })

    clean = _train_disjointness(
        "exp_geo4", {"mosaic"}, {"mosaic"},
        cal_rects={"mosaic": (680, 100, 780, 300)}, hold_rects={"mosaic": (850, 100, 950, 300)})
    assert clean["leaked_groups"] == []
    assert clean["group_check"] == "spatial_strip_geometric"

    leaked = _train_disjointness(
        "exp_geo4", {"mosaic"}, set(), cal_rects={"mosaic": (100, 100, 300, 300)})
    assert leaked["leaked_groups"] == ["mosaic"]


def test_train_disjointness_geometric_check_end_to_end_with_persisted_regions(tmp_path):
    """The real pipeline: auto_train_val -> persist_split_manifest persists train_region/
    val_region (this phase's own addition), and _train_disjointness's geometric check reads
    them back correctly -- a calibration rect drawn from inside the persisted val region reads
    clean, and one drawn from inside the persisted train region is caught."""
    from tcip_mcp.experiments import create_experiment, read_split_manifest
    from tcip_mcp.pipelines.operating_point import _train_disjointness
    from tcip_mcp.pipelines.data.split_construction import auto_train_val, persist_split_manifest

    images_dir, labels_dir, stem = _big_single_source(tmp_path / "ds", 4000, 3000)
    data_cfg = {
        "images_dir": str(images_dir), "labels_dir": str(labels_dir), "subject": "bud",
        "auto_val": True, "tiling": {"enabled": True, "tile_size": 128, "overlap": 0.2},
        "split": {"val_ratio": 0.25, "test_ratio": 0.1, "seed": 1},
    }
    train_ds, val_ds, _ = auto_train_val("detection", data_cfg, None)
    assert val_ds is not None

    create_experiment("exp_geo_e2e", {})
    persist_split_manifest("exp_geo_e2e", train_ds, val_ds, data_cfg)
    split = read_split_manifest("exp_geo_e2e")
    train_region = split["spatial"]["train_region"]
    val_region = split["spatial"]["val_region"]
    assert train_region and val_region

    def _shrunk(rect):
        x0, y0, x1, y1 = rect
        return (x0 + 1, y0 + 1, x1 - 1, y1 - 1)

    clean = _train_disjointness(
        "exp_geo_e2e", {stem}, set(), cal_rects={stem: _shrunk(val_region[0])})
    assert clean["leaked_groups"] == []

    leaked = _train_disjointness(
        "exp_geo_e2e", set(), {stem}, hold_rects={stem: _shrunk(train_region[0])})
    assert leaked["leaked_groups"] == [stem]


def _big_single_source(root: Path, width: int, height: int) -> tuple[Path, Path, str]:
    from torchvision.utils import save_image

    images_dir, labels_dir = root / "images", root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    stem = "mosaic"
    save_image(torch.rand(3, height, width) * 0.3, str(images_dir / f"{stem}.png"))
    boxes = [Annotation(subject="bud", geometry=BBox(x, y, x + 20, y + 20))
            for x in range(20, width - 20, 200) for y in range(20, height - 20, 200)]
    json_io.write_annotations(str(labels_dir / f"{stem}.json"), boxes, width, height, keep_empty=True)
    return images_dir, labels_dir, stem


def test_spatial_manifest_never_reads_as_a_bare_stem_leak(tmp_path):
    """The manifest for a spatial split lists per-region identities, not the bare stem: the
    identity mechanism is provenance for a future region-aware reader, not something today's
    disjointness gate depends on (it already resolves a same-source reference correctly by
    mapping an identity back to its stem, per the other test in this section) - but the bare
    stem must still never appear as a member on its own, and a different-source reference
    must still read clean end to end through the real training-launch path."""
    from tcip_mcp.experiments import create_experiment, read_split_manifest
    from tcip_mcp.pipelines.operating_point import _train_disjointness
    from tcip_mcp.pipelines.data.split_construction import auto_train_val, persist_split_manifest

    images_dir, labels_dir, stem = _big_single_source(tmp_path / "ds", 4000, 3000)
    data_cfg = {
        "images_dir": str(images_dir), "labels_dir": str(labels_dir), "subject": "bud",
        "auto_val": True, "tiling": {"enabled": True, "tile_size": 128, "overlap": 0.2},
        "split": {"val_ratio": 0.25, "test_ratio": 0.1, "seed": 1},
    }
    train_ds, val_ds, _ = auto_train_val("detection", data_cfg, None)
    assert val_ds is not None

    create_experiment("exp_spatial_e2e", {})
    persist_split_manifest("exp_spatial_e2e", train_ds, val_ds, data_cfg)
    split = read_split_manifest("exp_spatial_e2e")
    assert split["group_by"] == "spatial_strip"
    assert stem not in split["train"]  # the bare stem itself is never a member
    assert all("::strip_" in s for s in split["train"])

    clean = _train_disjointness("exp_spatial_e2e", {"a_different_mosaic"}, set())
    assert clean["leaked_groups"] == []


# review-confirmation image ids are stemmed, matching training stems.
# ===========================================================================

_IDENTITY = {"checkpoint_sha256": "deadbeef", "experiment_id": None}


def test_review_to_records_stems_the_image_id():
    from tcip_mcp.pipelines.feedback.review_calibration import review_to_records

    review_state = {"image": {"srcA_0_0.jpg": {"img_status": "completed", "detections": [
        {"action": "accepted", "class_id": 0,
         "gt_bbox_norm": [0.5, 0.5, 0.1, 0.1], "pred_bbox_norm": [0.5, 0.5, 0.1, 0.1], "conf": 0.9,
         "producer_identity": _IDENTITY},
    ]}}}
    recs = review_to_records(review_state, bucket_identities=[_IDENTITY])
    assert recs[0]["image_id"] == "srcA_0_0"  # stemmed, not "srcA_0_0.jpg"


def test_review_confirmed_leak_now_detected(tmp_path, monkeypatch):
    """Extensioned review ids in the same tile group as training stems must be caught, not
    silently reported clean."""
    import tcip_store

    from tcip_mcp.experiments import split_key
    from tcip_mcp.pipelines.feedback.review_calibration import resolve_operating_point_from_review

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    # Trained on two tiles of source "srcA".
    tcip_store.replace(split_key("exp_review"),
                       {"train": ["srcA_0_0", "srcA_0_1"], "group_by": "tile_prefix"})

    def _entry(gt, pred, conf):
        return {"action": "accepted", "class_id": 0, "gt_bbox_norm": gt, "pred_bbox_norm": pred,
                "conf": conf, "producer_identity": _IDENTITY}

    # Two reviewed images, both further tiles of the same source the model trained on, keyed
    # with an extension, exactly as review state stores them. gt_preexisting=True so these
    # records aren't excluded from the gate as unadjudicated: this test is about train
    # disjointness, not FN-coverage.
    review_state = {"image": {
        "srcA_0_2.jpg": {"img_status": "completed", "gt_preexisting": True,
                         "detections": [_entry([0.25, 0.25, 0.05, 0.05], [0.25, 0.25, 0.05, 0.05], 0.05)]},
        "srcA_0_3.jpg": {"img_status": "completed", "gt_preexisting": True,
                         "detections": [_entry([0.5, 0.5, 0.05, 0.05], [0.5, 0.5, 0.05, 0.05], 0.05)]},
    }}
    bundle = resolve_operating_point_from_review(
        review_state, "bud_opening", tiled=True, group_by="stem", experiment_id="exp_review",
        bucket_identities=[_IDENTITY], scope_root=tmp_path)
    td = bundle.get("conf").gate_evidence["train_disjointness"]
    assert td["leaked_groups"] == ["srcA"]  # matched despite the .jpg extension on the review id
    assert bundle.get("conf").validated_against == "false"


# ===========================================================================
# unresolvable/leaked train-disjointness refusals are visible to the agent and honestly described.
# ===========================================================================

def _review_bundle(gate_evidence: dict):
    from tcip_mcp.pipelines.resolution import VALIDATED_FALSE, ResolvedBundle, derived

    conf = derived("conf", 0.42, requires_validation=True, validation_kind="annotations",
                   derived_from="count-unbiased center-match curve over review verdicts",
                   validated_against=VALIDATED_FALSE, dataset_scoped=True, dataset_hash="abc",
                   gate_evidence=gate_evidence)
    return ResolvedBundle(trait="bud_opening", dataset_hash="abc", params={"conf": conf})


def test_describe_review_validation_unresolvable_message():
    from tcip_mcp.pipelines.feedback import describe_review_validation

    b = _review_bundle({"conf_censored": False, "disjoint": True, "passed_holdout": False,
                        "train_disjointness": {"unresolvable": True},
                        "failures": ["train_disjointness_unresolvable"]})
    out = describe_review_validation(b, reviewed_image_count=4)
    assert out["validated"] is False
    assert "training record" in out["reason"]


def test_describe_review_validation_leaked_message():
    from tcip_mcp.pipelines.feedback import describe_review_validation

    b = _review_bundle({"conf_censored": False, "disjoint": True, "passed_holdout": False,
                        "train_disjointness": {"unresolvable": False, "leaked_groups": ["srcA"],
                                                "leaked_stems": []},
                        "failures": ["train_disjointness_leaked"]})
    out = describe_review_validation(b, reviewed_image_count=4)
    assert out["validated"] is False
    assert "also used to train" in out["reason"]


def test_describe_review_validation_content_shared_with_calibration_message():
    from tcip_mcp.pipelines.feedback import describe_review_validation

    b = _review_bundle({"conf_censored": False, "disjoint": True, "passed_holdout": False,
                        "content_shared_with_calibration": True,
                        "failures": ["content_shared_with_calibration"]})
    out = describe_review_validation(b, reviewed_image_count=4)
    assert out["validated"] is False
    assert "share content" in out["reason"]


def test_gate_evidence_summary_surfaces_disjointness_fields():
    from tcip_mcp.pipelines.calibration import gate_evidence_summary
    from tcip_mcp.pipelines.resolution import VALIDATED_FALSE, derived

    conf = derived("conf", 0.4, requires_validation=True, validation_kind="annotations", derived_from="x",
                   validated_against=VALIDATED_FALSE,
                   gate_evidence={"disjoint": True, "content_overlap_frac": 0.0,
                          "content_shared_with_calibration": False,
                          "train_disjointness": {"unresolvable": False, "leaked_groups": ["g1"]},
                          "passed_holdout": False, "conf_censored": False, "count_bias_tolerance_frac": 1.0,
                          "pooled_count_bias_tolerance": 4.0})
    out = gate_evidence_summary(conf)
    assert out["disjoint"] is True
    assert out["content_overlap_frac"] == 0.0
    assert out["train_disjointness"]["leaked_groups"] == ["g1"]  # visible, not silently dropped
    # The renamed/new fields must actually reach gate_evidence_summary's output, not just be present
    # in the input sweep dict, catching a key-name drift in its own `.get(...)` calls.
    assert out["count_bias_tolerance_frac"] == 1.0
    assert out["pooled_count_bias_tolerance"] == 4.0


def test_gate_evidence_summary_surfaces_split_policy_divergence():
    """attach_split_policy_provenance writes into conf.gate_evidence; gate_evidence_summary must forward those
    keys too, or run_inference's actual response never shows a caller their declared seed/ratio
    didn't take effect against an existing lock -- only the persisted sweep artifact would."""
    from tcip_mcp.pipelines.calibration import gate_evidence_summary
    from tcip_mcp.pipelines.resolution import VALIDATED_FALSE, derived

    conf = derived("conf", 0.4, requires_validation=True, validation_kind="annotations", derived_from="x",
                   validated_against=VALIDATED_FALSE,
                   gate_evidence={"passed_holdout": False, "conf_censored": False, "count_bias_tolerance_frac": 1.0,
                          "split_policy_divergence": {"requested": {"seed": 7}, "locked": {"seed": 0}},
                          "split_unlocked_stems": ["new_stem_0_0"]})
    out = gate_evidence_summary(conf)
    assert out["split_policy_divergence"] == {"requested": {"seed": 7}, "locked": {"seed": 0}}
    assert out["split_unlocked_stems"] == ["new_stem_0_0"]


# ===========================================================================
# a locked split can't go stale silently, and a corrupt lock refuses.
# ===========================================================================

def test_stale_locked_stem_refuses_cleanly(tmp_path):
    from tcip_mcp.pipelines.data.splits import resolve_locked_cal_holdout_split

    stems_full = ["a_0_0", "a_0_1", "b_0_0", "b_0_1"]
    resolve_locked_cal_holdout_split(
        stems_full, identity_hash="stale-test", scope_root=tmp_path, seed=1)

    # One stem's image/label vanished since the lock was drawn.
    stems_now = ["a_0_0", "a_0_1", "b_0_0"]
    with pytest.raises(ValueError, match="no longer present"):
        resolve_locked_cal_holdout_split(
            stems_now, identity_hash="stale-test", scope_root=tmp_path, seed=1)


def test_corrupt_lock_file_refuses_instead_of_silent_redraw(tmp_path):
    """Bound to the file backend: undecodable bytes behind a record have no seam expression (a
    write always encodes a valid value), so this reaches the file the seam's own locator places
    them at. What is under test, catching DecodeError, is the store's own concern, not the file
    backend's, so this exercises it identically to a corruption reached any other way."""
    import tcip_store
    from tcip_store.file_backend import FileBackend

    from tcip_mcp.pipelines.data.splits import cal_holdout_lock_path, resolve_locked_cal_holdout_split

    tcip_store.bind(FileBackend())
    lock_path = cal_holdout_lock_path("corrupt-test", scope_root=tmp_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(ValueError, match="corrupt"):
        resolve_locked_cal_holdout_split(
            ["a_0_0", "b_0_0"], identity_hash="corrupt-test", scope_root=tmp_path, seed=1)

    # force_redraw=True is the deliberate, audited path past the corrupt file, but its
    # redraw_history honestly starts fresh (nothing recoverable from an unreadable file).
    redrawn = resolve_locked_cal_holdout_split(
        ["a_0_0", "b_0_0"], identity_hash="corrupt-test", scope_root=tmp_path, seed=1,
        force_redraw=True, timestamp="2026-01-01T00:00:00Z")
    assert len(redrawn["redraw_history"]) == 1
    assert redrawn["redraw_history"][0]["old_content_hash"] is None


def test_a_dataset_identity_cannot_place_a_lock_outside_the_artifact_store(tmp_path):
    """An identity names one lock, so an identity spelled as a path locks nothing elsewhere.

    A lock written outside the artifact store is a held-out split no later call would find,
    which is the silent redraw this lock exists to prevent.
    """
    from tcip_store import BadKey

    from tcip_mcp.pipelines.data.splits import resolve_locked_cal_holdout_split

    with pytest.raises(BadKey):
        resolve_locked_cal_holdout_split(
            ["a_0_0", "b_0_0"], identity_hash="../escaped-identity", scope_root=tmp_path, seed=1)


def test_a_first_draw_locks_a_split_an_ordinary_identity_can_read_back(tmp_path):
    """The refusal above must leave the ordinary path intact: draw once, read the same split."""
    from tcip_mcp.pipelines.data.splits import resolve_locked_cal_holdout_split

    first = resolve_locked_cal_holdout_split(
        ["a_0_0", "a_0_1", "b_0_0", "b_0_1"], identity_hash="d41d8cd98f00b204",
        scope_root=tmp_path, seed=1, timestamp="2026-01-01T00:00:00Z")
    again = resolve_locked_cal_holdout_split(
        ["a_0_0", "a_0_1", "b_0_0", "b_0_1"], identity_hash="d41d8cd98f00b204",
        scope_root=tmp_path, seed=1)

    assert first["calibration"] and first["holdout"]
    assert again["calibration"] == first["calibration"]
    assert again["holdout"] == first["holdout"]


def test_missing_image_refuses_cleanly_not_keyerror(tmp_path):
    """At the tool level: a locked stem whose image was later deleted must produce a clean
    ValueError through calibrate_operating_point, never a bare KeyError from a stale
    stem_to_image lookup."""
    import tcip_mcp.pipelines.calibration as calibration

    stems = ["a_0_0", "a_0_1", "b_0_0", "b_0_1"]
    images_dir, labels_dir = _detection_dataset(tmp_path / "ds", stems)

    kwargs = dict(tile=False, tile_size=IMG, overlap=0.2, tile_batch_size=8,
                  global_nms_iou=0.3, postprocess="nms", cross_tile_nms=None, max_dets=None)
    # First call locks the split over all 4 stems.
    calibration.calibrate_operating_point(_CalStub(), "bud_opening", str(labels_dir), str(images_dir), **kwargs)

    (images_dir / "b_0_1.png").unlink()  # an image vanishes after the lock

    with pytest.raises(ValueError, match="no longer present"):
        calibration.calibrate_operating_point(_CalStub(), "bud_opening", str(labels_dir), str(images_dir), **kwargs)


def test_calibrate_operating_point_lock_balances_on_the_checkpoints_own_subject(tmp_path, monkeypatch):
    """The locked cal/holdout draw balances on the checkpoint's own subject's annotation count,
    the same subject-aware scope the manifest draw itself applies: a stem carrying only another
    subject's annotation counts zero foreground here, not the file's raw record count."""
    import tcip_mcp.pipelines.calibration as calibration
    import tcip_mcp.pipelines.data.splits as splits_mod

    stems = ["a_0_0", "a_0_1", "b_0_0", "b_0_1"]
    images_dir, labels_dir = _detection_dataset(tmp_path / "ds", stems)
    # b_0_1 carries only a different subject's annotation: zero bud foreground.
    json_io.write_annotations(
        str(labels_dir / "b_0_1.json"),
        [Annotation(subject="leaf", geometry=BBox(2, 2, 10, 10))], IMG, IMG, keep_empty=True,
    )

    captured: dict = {}
    real_resolve = splits_mod.resolve_locked_cal_holdout_split

    def _capture(stems, **kwargs):
        captured["annotation_counts"] = dict(kwargs.get("annotation_counts") or {})
        return real_resolve(stems, **kwargs)

    monkeypatch.setattr(splits_mod, "resolve_locked_cal_holdout_split", _capture)

    stub = _CalStub()
    stub.config = {"data": {"subject": "bud"}}
    calibration.calibrate_operating_point(
        stub, "bud_opening", str(labels_dir), str(images_dir),
        tile=False, tile_size=IMG, overlap=0.2, tile_batch_size=8,
        global_nms_iou=0.3, postprocess="nms", cross_tile_nms=None, max_dets=None,
        seed=1, holdout_ratio=0.5,
    )
    assert captured["annotation_counts"]["b_0_1"] == 0


def test_force_redraw_shares_the_labels_intersect_images_scan(tmp_path):
    """redraw_calibration_holdout(images_dir=...) must use the same labels-intersect-images
    scan calibrate_operating_point uses, not a second independent labels-only glob: a stem
    with no image on disk must not enter the redraw's stem universe."""
    from tcip_mcp.tools.calibration_tools import redraw_calibration_holdout

    stems = ["a_0_0", "a_0_1", "b_0_0", "b_0_1"]
    images_dir, labels_dir = _detection_dataset(tmp_path / "ds", stems)
    (images_dir / "b_0_1.png").unlink()  # labeled but no image

    result = redraw_calibration_holdout(
        dataset_root=str(tmp_path / "ds"), labels_dir=str(labels_dir),
        images_dir=str(images_dir), seed=1,
        reason="labels-intersect-images coverage test")
    assert "error" not in result
    all_new = result["new_membership"]["calibration"] + result["new_membership"]["holdout"]
    assert "b_0_1" not in all_new


# ===========================================================================
# a declared seed/holdout_ratio reaches the first (locking) draw.
# ===========================================================================

def test_declared_seed_and_holdout_ratio_reach_the_first_draw(tmp_path):
    import tcip_mcp.pipelines.calibration as calibration

    stems = [f"src{g}_{t}_0" for g in range(4) for t in range(2)]
    images_dir, labels_dir = _detection_dataset(tmp_path / "ds", stems)

    bundle, _dh, _n_excluded, _evidence = calibration.calibrate_operating_point(
        _CalStub(), "bud_opening", str(labels_dir), str(images_dir),
        tile=False, tile_size=IMG, overlap=0.2, tile_batch_size=8,
        global_nms_iou=0.3, postprocess="nms", cross_tile_nms=None, max_dets=None,
        seed=7, holdout_ratio=0.75,
    )
    policy = bundle.get("conf").gate_evidence["split_policy"]
    assert policy["seed"] == 7
    assert policy["holdout_ratio"] == pytest.approx(0.75)  # not the 0/0.5 defaults


def test_the_calibration_door_keeps_its_lock_across_an_active_project_repin(tmp_path, monkeypatch):
    """The count-calibration door reads one lock for a labeled dir, before and after an adoption.

    Adopting a project repins the platform state root inside a live process. A lock scoped to that
    root reads as absent once it moves, so a second calibration over the same labels cuts a fresh
    holdout and the held-out claim rests on a split the first pass never held back.
    """
    import shutil

    import tcip_mcp.pipelines.calibration as calibration

    stems = [f"src{g}_{t}_0" for g in range(4) for t in range(2)]
    images_dir, labels_dir = _detection_dataset(tmp_path / "ds", stems)
    kwargs = dict(tile=False, tile_size=IMG, overlap=0.2, tile_batch_size=8,
                  global_nms_iou=0.3, postprocess="nms", cross_tile_nms=None, max_dets=None)
    # Each root carries the trait spec an adopted project of its own would hold.
    for root in (tmp_path / "before_adoption", tmp_path / "adopted_project"):
        shutil.copytree(tmp_path / ".tcip", root / ".tcip")

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path / "before_adoption"))
    first, _dh, _n_excluded, _evidence = calibration.calibrate_operating_point(
        _CalStub(), "bud_opening", str(labels_dir), str(images_dir), seed=1, **kwargs)

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path / "adopted_project"))
    second, _dh2, _n_excluded2, _evidence2 = calibration.calibrate_operating_point(
        _CalStub(), "bud_opening", str(labels_dir), str(images_dir), seed=2, **kwargs)

    assert first.get("conf").gate_evidence["split_policy"]["seed"] == 1
    assert second.get("conf").gate_evidence["split_policy"]["seed"] == 1
    assert second.get("conf").gate_evidence["split_policy_divergence"]["requested"]["seed"] == 2


# ===========================================================================
# the calibration record builder's whole-image exclusion for an unlabeled attribute instance
# must be counted and disclosed, not a silent filter (matching evaluation.py's
# n_excluded_incomplete_attribute).
# ===========================================================================

def test_calibration_discloses_excluded_incomplete_attribute_count(tmp_path):
    """A stem with any instance unlabeled for `attribute` is dropped whole from the cal/holdout
    record set (the missing-label-file precedent): the count must travel back to the caller,
    not vanish, so a caller can see the reference shrank rather than assume every stem measured."""
    import tcip_mcp.pipelines.calibration as calibration
    from tcip_mcp.class_registry import Attribute, ClassRegistry, Subject, write_registry

    root = tmp_path / "ds"
    images_dir, labels_dir = root / "images", root / "labels"
    write_registry(root / "classes.json", ClassRegistry(subjects=(
        Subject(name="bud", attributes=(
            Attribute(name="state", type="categorical", values=("open", "closed")),)),)))

    stems = ["complete_a", "complete_b", "partial_a", "partial_b"]
    for s in stems:
        _save_png(images_dir / f"{s}.png")
    for s in ("complete_a", "complete_b"):
        json_io.write_annotations(str(labels_dir / f"{s}.json"), [
            Annotation(subject="bud", geometry=BBox(2, 2, 10, 10), attributes={"state": "open"}),
        ], IMG, IMG)
    for s in ("partial_a", "partial_b"):
        json_io.write_annotations(str(labels_dir / f"{s}.json"), [
            Annotation(subject="bud", geometry=BBox(2, 2, 10, 10), attributes={"state": "open"}),
            Annotation(subject="bud", geometry=BBox(15, 15, 20, 20)),  # no `state` -- unlabeled
        ], IMG, IMG)

    stub = _CalStub()
    stub.config = {"data": {"subject": "bud", "attribute": "state"}}

    _bundle, _dh, n_excluded, _evidence = calibration.calibrate_operating_point(
        stub, "bud_opening", str(labels_dir), str(images_dir),
        tile=False, tile_size=IMG, overlap=0.2, tile_batch_size=8,
        global_nms_iou=0.3, postprocess="nms", cross_tile_nms=None, max_dets=None,
        group_by="stem", seed=0, holdout_ratio=0.5,
    )

    assert n_excluded == 2  # partial_a + partial_b, wherever the split put them


def test_calibration_attribute_registry_refusal_reaches_the_caller(tmp_path):
    """calibrate_operating_point's bare `except Exception` around resolve_registry_id_map must
    not silently degrade an attribute-classification calibration to a single-class GT read when
    the registry read fails for a real reason, sitting directly on the calibration/
    operating-point rail, worse than the delivery-grade-eval instance of the same bug. No
    classes.json exists here for an attribute-scoped config, so this must refuse."""
    import tcip_mcp.pipelines.calibration as calibration

    stems = ["a_0_0", "a_0_1"]
    images_dir, labels_dir = _detection_dataset(tmp_path / "ds", stems)

    stub = _CalStub()
    stub.config = {"data": {"subject": "bud", "attribute": "state"}}  # no classes.json written

    with pytest.raises(ValueError, match="classes.json"):
        calibration.calibrate_operating_point(
            stub, "bud_opening", str(labels_dir), str(images_dir),
            tile=False, tile_size=IMG, overlap=0.2, tile_batch_size=8,
            global_nms_iou=0.3, postprocess="nms", cross_tile_nms=None, max_dets=None,
            group_by="stem", seed=0, holdout_ratio=0.5,
        )


# ===========================================================================
# Calibration's GT-side id-map resolution must prefer the training-recorded map over a fresh
# registry read, the same preference resolve_decode_id_map already applies to decode: a
# classes.json whose declared attribute-value order was edited since training must not silently
# relabel the calibration GT.
# ===========================================================================

def test_calibration_gt_id_map_prefers_the_training_recorded_map_over_a_fresh_registry_read(
    tmp_path, monkeypatch,
):
    import tcip_mcp.pipelines.calibration as calibration

    stems = ["a_0_0", "a_0_1"]
    images_dir, labels_dir = tmp_path / "ds" / "images", tmp_path / "ds" / "labels"
    for s in stems:
        _save_png(images_dir / f"{s}.png")
        json_io.write_annotations(str(labels_dir / f"{s}.json"), [
            Annotation(subject="bud", geometry=BBox(2, 2, 10, 10), attributes={"state": "open"}),
        ], IMG, IMG)

    stub = _CalStub()
    # A recorded map present: the registry read must never even be attempted, regardless of what
    # a fresh classes.json (absent here) would derive.
    stub.config = {"data": {"subject": "bud", "attribute": "state",
                            "id_map": {"open": 0, "closed": 1}}}

    def _boom(*a, **kw):
        raise AssertionError(
            "resolve_registry_id_map must not be called when the checkpoint carries its own "
            "recorded id_map")

    monkeypatch.setattr("tcip_mcp.pipelines.data.label_queries.resolve_registry_id_map", _boom)

    # No classes.json exists for this dataset, so the pre-fix code (which always re-derived from
    # the registry when `subject` was set) would have raised the ValueError
    # test_calibration_attribute_registry_refusal_reaches_the_caller pins -- this must instead
    # succeed, using only the recorded map.
    bundle, _dh, n_excluded, _evidence = calibration.calibrate_operating_point(
        stub, "bud_opening", str(labels_dir), str(images_dir),
        tile=False, tile_size=IMG, overlap=0.2, tile_batch_size=8,
        global_nms_iou=0.3, postprocess="nms", cross_tile_nms=None, max_dets=None,
        group_by="stem", seed=0, holdout_ratio=0.5,
    )
    assert n_excluded == 0
    assert bundle is not None


def test_recorded_training_id_map_helper_is_none_when_config_carries_no_map():
    import tcip_mcp.tools.inference_tools as itools

    stub = _CalStub()
    stub.config = {"data": {"subject": "bud", "attribute": "state"}}
    assert itools._recorded_training_id_map(stub) is None

    stub.config["data"]["id_map"] = {"open": 0, "closed": 1}
    assert itools._recorded_training_id_map(stub) == {"open": 0, "closed": 1}


# --- Minor: resolve_model_identity off a load_registered_checkpoint object. -------------------

def test_minor_resolve_model_identity_reads_the_codebase_own_stamped_checkpoints(tmp_path):
    """weights_only=True must still read a checkpoint saved the way stamp_model_ref produces it:
    the rail must admit valid work, not only reject foreign payloads."""
    from tcip_mcp.model_registry import (
        ModelRegistry, load_registered_checkpoint, resolve_model_identity,
    )

    ckpt = tmp_path / "m.pt"
    torch.save({"model_state_dict": {"w": torch.zeros(2, 2)},
               "optimizer_state_dict": {"state": {}, "param_groups": [{"lr": 1e-3}]},
               "experiment_id": "exp_abc", "kind": "tcip_module"}, ckpt)
    ModelRegistry(str(tmp_path)).register_model("m", str(ckpt), {}, metrics_source=None)
    checkpoint = load_registered_checkpoint(ckpt, project_path=str(tmp_path))
    identity = resolve_model_identity(checkpoint)
    assert identity["experiment_id"] == "exp_abc"


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
    """A realistic dense reference for tests that expect the holdout gate to validate: the
    2-image ``_op_records`` toy's per-image variance trips the equivalence criterion at n=2 (an
    intended tightening; see test_operating_point.py for the same fixture idiom)."""
    from tests._dense_op_fixtures import dense_records

    n_images, objects_per_image = 20, 80
    miss, fp = [0] * n_images, [1] * n_images
    cal = dense_records(n_images=n_images, objects_per_image=objects_per_image, id_prefix="c",
                        miss_pattern=miss, fp_pattern=fp, score=0.9, fp_score=0.05)
    hold = dense_records(n_images=n_images, objects_per_image=objects_per_image, id_prefix="h",
                         shift=5.0, miss_pattern=miss, fp_pattern=fp, score=0.9, fp_score=0.05)
    return cal, hold
