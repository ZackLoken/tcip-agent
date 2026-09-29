"""The held-out calibration reference is genuinely held out: not trained on, not reused.

Covers: the train-disjointness gate must not permanently block the drawn-validation and
group_key_map training routes; the review-confirmation path must detect a training leak even when
review image ids carry an extension review state stores them with, unlike unextensioned training
stems; an unresolvable or leaked train-disjointness check must be visible to the agent, not
misreported as a generic "review more images"; a locked cal/holdout split must refuse rather than
silently redraw when stale (a missing image) or when its lock file is corrupt; and a declared
seed/holdout_ratio must reach the first (locking) calibration draw, not only later redraws. Also
covers resolve_model_identity reading the codebase's own stamped checkpoints off a verified load.
"""

from __future__ import annotations

import copy
from functools import partial
from pathlib import Path

import pytest

from tests._clear_prediction_bucket_fixtures import write_image
from tests._regime_fixtures import stub_pass, tiled_regime

# no built-in traits, seed_bud_trait_spec (conftest.py) writes a real bud.yml into this
# test's pinned platform state root so trait="bud_opening" call sites keep resolving.
pytestmark = pytest.mark.usefixtures("seed_bud_trait_spec")

torch = pytest.importorskip("torch")

from tcip_annotation import json_io  # noqa: E402
from tcip_annotation.state import Annotation, BBox  # noqa: E402

IMG = 32


_save_png = partial(write_image, size=IMG)


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
        # The run's recorded scope, the one its admission wrote.
        self.config: dict = {"data": {"scope": {"subject": "bud", "id_map": {"bud": 0}}}}

    def predict_batch(self, paths, **kw):
        return [{"image": p, "width": IMG, "height": IMG,
                 "boxes": [], "scores": [], "labels": [], "count": 0} for p in paths]


def _resolved(run_dir: Path) -> dict:
    """What the run's launch record says it resolved."""
    from tcip_mcp.experiments import RUN_FILE, read_record

    return read_record(run_dir / RUN_FILE)["resolved"]


def _side(run_dir: Path, side: str) -> list:
    """The samples the run's resolved partition put on ``side``."""
    from tcip_mcp.pipelines.data.split_construction import partition_samples

    return [s for s in partition_samples(_resolved(run_dir)["partition"]) if s.side == side]


def test_group_key_map_end_to_end_not_permanently_blocked(tmp_path):
    """group_key_map, exercised through the launcher's own resolution -> _train_disjointness, must
    not permanently block the model: the recorded groups resolve every calibration stem."""
    from tcip_mcp.pipelines.operating_point import _train_disjointness
    from tests._verified_checkpoint_fixtures import resolved_run

    stems = ["imgA0", "imgA1", "imgB0", "imgB1"]
    images_dir, labels_dir = _detection_dataset(tmp_path / "ds", stems)
    group_key_map = {"imgA0": "gA", "imgA1": "gA", "imgB0": "gB", "imgB1": "gB"}
    run_dir = resolved_run(tmp_path, {
        "images_dir": str(images_dir), "labels_dir": str(labels_dir), "scope": {"subject": "bud"},
        "auto_val": True,
        "split": {"val_ratio": 0.5, "seed": 1, "group_key_map": dict(group_key_map)},
    }, experiment_id="e1")
    assert _resolved(run_dir)["partition"]["group_by"] == "explicit_map"
    train, val = _side(run_dir, "train"), _side(run_dir, "val")
    # The two groups (gA/gB) never straddle train/val: group-coherent by construction.
    assert {s.group for s in train}.isdisjoint({s.group for s in val})

    # A calibration reference drawn from val's own stems must not be permanently blocked.
    td = _train_disjointness("e1", {s.member for s in val}, set(),
                             calibration_labels_dir=str(labels_dir))
    assert td["unresolvable"] is False
    assert td["group_check"] == "performed"  # every stem covered by the recorded groups
    assert td["leaked_groups"] == []

    # A training stem named as calibration is caught at its recorded group.
    td_leak = _train_disjointness("e1", {train[0].member}, set(),
                                  calibration_labels_dir=str(labels_dir))
    assert td_leak["leaked_groups"] == [train[0].group]


def test_a_drawn_validation_side_is_checked_end_to_end(tmp_path):
    """A run's own drawn validation side, driven through the launcher's own resolution ->
    _selection_disjointness: the record names those val members and the scope they live under,
    so a calibration reading that same directory is checked against them like any other side. A
    reader that skipped this route would report no overlap where the calibration is measuring on
    exactly the images the checkpoint was chosen against."""
    from tcip_mcp.pipelines.operating_point import _selection_disjointness
    from tests._verified_checkpoint_fixtures import resolved_run

    images_dir, labels_dir = _detection_dataset(tmp_path / "ds", ["t0", "t1", "v0", "v1"])
    run_dir = resolved_run(tmp_path, {
        "images_dir": str(images_dir), "labels_dir": str(labels_dir), "scope": {"subject": "bud"},
        "split": {"group_by": "stem", "val_ratio": 0.5, "seed": 1},
    }, experiment_id="exp-drawn-val")
    val_member = _side(run_dir, "val")[0].member

    leaked = _selection_disjointness(
        "exp-drawn-val", {val_member}, set(), selection_dir="some/selection",
        calibration_labels_dir=str(labels_dir))
    assert leaked["applicable"] is True
    assert leaked["leaked_stems"] == [val_member] or leaked["leaked_groups"] == [val_member]

    # Admits valid work: a calibration over that same directory sharing no member is clean.
    clean = _selection_disjointness(
        "exp-drawn-val", {"unrelated"}, set(), selection_dir="some/selection",
        calibration_labels_dir=str(labels_dir))
    assert clean["applicable"] is True
    assert clean["leaked_stems"] == [] and clean["leaked_groups"] == []


DATE = "2-11-26"
PARENTS = ("p1", "p2", "p3", "p4")


def _dated_detection_dataset(root: Path) -> tuple[Path, Path, list[str]]:
    """The canonical ``images/<date>/`` plus ``annotations/<date>/`` layout, two crops of each of
    four parents, every crop carrying foreground.

    Dated because that is where two spellings of a group key could differ: ``admission_date``
    reads a capture date out of this layout, and a key that carries one and a key that does not
    are the two vocabularies a leak check would have to compare across.
    """
    images_dir, labels_dir = root / "images" / DATE, root / "annotations" / DATE
    labels_dir.mkdir(parents=True, exist_ok=True)
    stems = [f"{parent}_{x}_0" for parent in PARENTS for x in (0, 1)]
    for stem in stems:
        _save_png(images_dir / f"{stem}.png")
        json_io.write_annotations(
            str(labels_dir / f"{stem}.json"),
            [Annotation(subject="bud", geometry=BBox(2, 2, 10, 10))], IMG, IMG, keep_empty=True,
        )
    return images_dir, labels_dir, stems


def _dated_run_config(images_dir: Path, labels_dir: Path) -> dict:
    return {"images_dir": str(images_dir), "labels_dir": str(labels_dir), "scope": {"subject": "bud"},
            "auto_val": True, "split": {"val_ratio": 0.5, "seed": 1}}


def test_a_drawn_run_and_a_drawn_selection_spell_one_group_key(tmp_path):
    """A run that draws its own split and a selection ``draw_splits`` writes record the same group
    key for one stem under one directory."""
    from tcip_mcp.pipelines.data.selection import read_selection
    from tcip_mcp.pipelines.data.split_construction import auto_train_val, partition_samples
    from tcip_mcp.tools.data_tools import draw_splits

    root = tmp_path / "ds"
    images_dir, labels_dir, _stems = _dated_detection_dataset(root)

    _train_ds, val_ds, partition = auto_train_val(
        "detection", _dated_run_config(images_dir, labels_dir), None)
    assert val_ds is not None
    drawn_keys = {Path(s.ground_truth).stem: s.group for s in partition_samples(partition)}

    out = tmp_path / "m"
    result = draw_splits(str(root), output_path=str(out), subject="bud", seed=1,
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)
    assert "error" not in result, result
    selection_keys = {Path(s.ground_truth).stem: s.group for s in read_selection(out).samples}

    assert drawn_keys == selection_keys
    assert set(drawn_keys.values()) == {f"{DATE}/{parent}" for parent in PARENTS}


def test_the_launch_door_admits_the_map_the_draw_requires(tmp_path, monkeypatch):
    """Preflight and the launch door admit the group_key_map the run's own draw accepts."""
    import subprocess

    from tcip_mcp.pipelines.data.split_construction import auto_train_val
    from tcip_mcp.pipelines.data.splits import member_identity
    from tcip_mcp.tools.training_tools import launch_training, preflight_config

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.tensorboard_manager.launch_tensorboard", lambda *a, **k: {})

    class _StubChild:
        def __init__(self, *a, **k) -> None:
            self.pid = 4242

        def __class_getitem__(cls, item):
            return cls

    monkeypatch.setattr(subprocess, "Popen", _StubChild)

    root = tmp_path / "ds"
    images_dir, labels_dir, stems = _dated_detection_dataset(root)
    group_key_map = {member_identity(DATE, stem): f"g{index}"
                     for index, stem in enumerate(stems)}
    config = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"min_size": 64, "max_size": 64},
                         "task": "detection"},
        "data": {**_dated_run_config(images_dir, labels_dir),
                 "split": {"val_ratio": 0.5, "seed": 1, "group_key_map": group_key_map}},
        "batch_size": 1, "stages": [{"freeze_to": -1, "epochs": 1}],
        "mixed_precision": False, "device": "cpu",
        "checkpoint_every_n_epochs": 0, "early_stopping": {"enabled": False},
    }

    assert preflight_config(copy.deepcopy(config))["issues"] == []
    launched = launch_training(copy.deepcopy(config))
    assert "error" not in launched, launched

    # The same config the door admitted is one the real draw accepts.
    _train_ds, val_ds, _partition = auto_train_val(
        "detection", copy.deepcopy(config)["data"], None)
    assert val_ds is not None


def test_a_crop_annotated_after_a_drawn_run_is_caught_as_its_parents_group(tmp_path):
    """A sibling crop of a training parent, annotated after the run and in no recorded map, is
    caught as a leak of that parent's own recorded group."""
    from tcip_mcp.pipelines.operating_point import _train_disjointness
    from tests._verified_checkpoint_fixtures import resolved_run

    root = tmp_path / "ds"
    images_dir, labels_dir, _stems = _dated_detection_dataset(root)
    run_dir = resolved_run(tmp_path, _dated_run_config(images_dir, labels_dir),
                           experiment_id="e-dated-drawn")

    trained = _side(run_dir, "train")
    assert trained, "the fixture must train on this directory for the leak to be a leak"
    parent = trained[0].member.rsplit("_", 2)[0]
    late_crop = f"{parent}_9_0"
    _save_png(images_dir / f"{late_crop}.png")
    json_io.write_annotations(
        str(labels_dir / f"{late_crop}.json"),
        [Annotation(subject="bud", geometry=BBox(2, 2, 10, 10))], IMG, IMG, keep_empty=True)

    resolved = _train_disjointness(
        "e-dated-drawn", {late_crop}, set(), calibration_labels_dir=str(labels_dir))

    assert resolved["leaked_groups"] == [trained[0].group]


def test_no_group_is_reproduced_for_a_directory_the_record_names_nothing_under(tmp_path):
    """A per-directory record's flat member lists are the union across every directory its samples
    live under, so a bare stem of them belongs to no one capture date. Read against a directory the
    record names nothing under, the check reports what it can prove, that no member is that stem,
    rather than grouping two unrelated dates' images into one and refusing legitimate work."""
    from tcip_mcp.pipelines.operating_point import _train_disjointness
    from tests._verified_checkpoint_fixtures import resolved_run

    root = tmp_path / "ds"
    images_dir, labels_dir, _stems = _dated_detection_dataset(root)
    run_dir = resolved_run(tmp_path, _dated_run_config(images_dir, labels_dir),
                           experiment_id="e-other-date")

    parent = _side(run_dir, "train")[0].member.rsplit("_", 2)[0]
    elsewhere = root / "annotations" / "2-12-01"
    elsewhere.mkdir(parents=True)

    resolved = _train_disjointness(
        "e-other-date", {f"{parent}_9_0"}, set(), calibration_labels_dir=str(elsewhere))

    assert resolved["unresolvable"] is False
    assert resolved["leaked_groups"] == []
    assert resolved["leaked_stems"] == []
    assert resolved["group_check"] == "not_performed"


# a spatial split's manifest never reads as a bare-stem leak, and _train_disjointness still
# catches a genuine same-source reference.

def test_train_disjointness_named_group_by_resolves_every_stem_to_its_recorded_group(tmp_path):
    """A tile_prefix run's recorded groups answer the check: a calibration stem of a trained
    source is caught at that source's group, with nothing falling through to the exact-stem
    fallback."""
    from tcip_mcp.pipelines.operating_point import _train_disjointness
    from tests._verified_checkpoint_fixtures import resolved_run

    images_dir, labels_dir = _detection_dataset(
        tmp_path / "ds", ["srcA_0_0", "srcA_0_1", "srcB_0_0", "srcB_0_1"])
    run_dir = resolved_run(tmp_path, {
        "images_dir": str(images_dir), "labels_dir": str(labels_dir), "scope": {"subject": "bud"},
        "split": {"group_by": "tile_prefix", "val_ratio": 0.5, "seed": 1},
    }, experiment_id="exp_named")
    trained = _side(run_dir, "train")[0]

    result = _train_disjointness(
        "exp_named", {trained.member}, set(), calibration_labels_dir=str(labels_dir))
    assert result == {
        "checked": True, "unresolvable": False,
        "leaked_groups": [trained.group], "leaked_stems": [], "group_check": "performed",
    }


def _spatial_run(tmp_path: Path, experiment_id: str) -> tuple[Path, str]:
    """A within-image spatial split over one large source, resolved by the launcher's own
    producer; the run directory and the source's stem."""
    from tests._verified_checkpoint_fixtures import resolved_run

    images_dir, labels_dir, stem = _big_single_source(tmp_path / "ds", 4000, 3000)
    run_dir = resolved_run(tmp_path, {
        "images_dir": str(images_dir), "labels_dir": str(labels_dir), "scope": {"subject": "bud"},
        "auto_val": True, "tiling": {"enabled": True, "tile_size": 128, "overlap": 0.2},
        "split": {"val_ratio": 0.25, "test_ratio": 0.1, "seed": 1},
    }, experiment_id=experiment_id)
    return run_dir, stem


def test_train_disjointness_spatial_strip_detects_same_source_leak(tmp_path):
    from tcip_mcp.pipelines.operating_point import _train_disjointness

    _run_dir, stem = _spatial_run(tmp_path, "exp_spatial")

    # A reference drawn from the same source is a real leak: region-scoping aside, the trained
    # pixels and the reference still share one source image.
    leaked = _train_disjointness("exp_spatial", {stem}, set())
    assert leaked == {
        "checked": True, "unresolvable": False,
        "leaked_groups": [stem], "leaked_stems": [], "group_check": "spatial_strip",
    }
    # cal_rects/hold_rects default to None: stating them as None answers the same.
    assert _train_disjointness("exp_spatial", {stem}, set(),
                               cal_rects=None, hold_rects=None) == leaked

    clean = _train_disjointness("exp_spatial", {"other_mosaic"}, set())
    assert clean["leaked_groups"] == []


def test_the_geometric_check_is_containment_in_a_non_train_region():
    """Given cal/hold rects against a spatial manifest, the check is geometric containment (fully
    inside a non-train region, the four-way split's calibration region included, and disjoint from
    every train region), and catches a leak the lexical check alone would miss: a rect that spills
    into train from a source stem that isn't literally the training source's own name."""
    from tcip_mcp.pipelines.operating_point import _spatial_strip_geometric_disjointness

    spatial = {
        "train_region": [[0, 0, 500, 1000]],
        "val_region": [[500, 0, 650, 1000]],
        "calibration_region": [[650, 0, 800, 1000]],
        "test_region": [[800, 0, 1000, 1000]],
    }

    def _check(cal=None, hold=None) -> list[str]:
        return _spatial_strip_geometric_disjointness(spatial, cal, hold)["leaked_groups"]

    assert _check({"mosaic": (550, 100, 600, 300)}) == []
    assert _check({"mosaic": (680, 100, 780, 300)}, {"mosaic": (850, 100, 950, 300)}) == []
    assert _check(hold={"other_mosaic": (10, 10, 100, 100)}) == ["other_mosaic"]
    # Straddling the train/val boundary: not fully contained in any single non-train region.
    assert _check({"mosaic": (400, 100, 600, 300)}) == ["mosaic"]


def test_train_disjointness_geometric_check_end_to_end_with_persisted_regions(tmp_path):
    """The real pipeline: the launcher's own resolution records train_region/val_region, and
    _train_disjointness's geometric check reads them back correctly: a calibration rect drawn from
    inside the recorded val region reads clean, and one drawn from inside the recorded train
    region is caught."""
    from tcip_mcp.pipelines.operating_point import _train_disjointness

    run_dir, stem = _spatial_run(tmp_path, "exp_geo_e2e")
    spatial = _resolved(run_dir)["data"]["split"]["spatial_manifest"]
    train_region = spatial["train_region"]
    val_region = spatial["val_region"]
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
    from tcip_mcp.pipelines.operating_point import _train_disjointness
    from tests._verified_checkpoint_fixtures import partition_side

    run_dir, stem = _spatial_run(tmp_path, "exp_spatial_e2e")
    resolved = _resolved(run_dir)
    assert partition_side(resolved["partition"], "train") == [stem]
    trained = resolved["data"]["split"]["spatial_manifest"]["train_identities"]
    assert stem not in trained  # the bare stem itself is never a member
    assert all("::strip_" in s for s in trained)

    clean = _train_disjointness("exp_spatial_e2e", {"a_different_mosaic"}, set())
    assert clean["leaked_groups"] == []


# review-confirmation image ids are stemmed, matching training stems.
# ===========================================================================

_IDENTITY = {"checkpoint_sha256": "deadbeef", "experiment_id": None}


def test_review_to_records_stems_the_image_id():
    from tcip_mcp.pipelines.feedback.review_calibration import review_to_records

    review_state = {"image": {"srcA_0_0.jpg": {"img_status": "completed", "detections": [
        {"action": "accepted", "class_id": 0,
         "iscrowd": False, "reviewed_by": "", "class_name": "", "conf_threshold": None, "missed_object_attested": False, "gt_bbox_norm": [0.5, 0.5, 0.1, 0.1], "pred_bbox_norm": [0.5, 0.5, 0.1, 0.1], "conf": 0.9,
         "producer_identity": _IDENTITY},
    ]}}}
    recs = review_to_records(review_state, bucket_identities=[_IDENTITY])
    assert recs[0]["image_id"] == "srcA_0_0"  # stemmed, not "srcA_0_0.jpg"


def test_train_disjointness_matches_extensioned_review_ids_to_train_group(tmp_path, monkeypatch):
    """Extensioned review ids in the same tile group as training stems must be caught, not
    silently reported clean."""
    from tcip_mcp.pipelines.feedback.review_calibration import resolve_operating_point_from_review
    from tests._verified_checkpoint_fixtures import resolved_run

    # Trained on the tiles of one source; the review below names further tiles of it.
    images, labels = _detection_dataset(
        tmp_path / "ds", ["srcA_0_0", "srcA_0_1", "srcB_0_0", "srcB_0_1"])
    labels_dir = str(labels)
    run_dir = resolved_run(tmp_path, {
        "images_dir": str(images), "labels_dir": labels_dir, "scope": {"subject": "bud"},
        "split": {"group_by": "tile_prefix", "val_ratio": 0.5, "seed": 1},
    }, experiment_id="exp_review")
    source = _side(run_dir, "train")[0].group

    def _entry(gt, pred, conf):
        return {"action": "accepted", "class_id": 0, "iscrowd": False, "reviewed_by": "", "class_name": "", "conf_threshold": None, "missed_object_attested": False, "gt_bbox_norm": gt, "pred_bbox_norm": pred,
                "conf": conf, "producer_identity": _IDENTITY}

    # Two reviewed images, both further tiles of the same source the model trained on, keyed
    # with an extension, exactly as review state stores them. gt_preexisting=True so these
    # records aren't excluded from the gate as unadjudicated: this test is about train
    # disjointness, not FN-coverage.
    review_state = {"image": {
        f"{source}_0_2.jpg": {"img_status": "completed", "gt_preexisting": True, "detections": [
            _entry([0.25, 0.25, 0.05, 0.05], [0.25, 0.25, 0.05, 0.05], 0.05)]},
        f"{source}_0_3.jpg": {"img_status": "completed", "gt_preexisting": True, "detections": [
            _entry([0.5, 0.5, 0.05, 0.05], [0.5, 0.5, 0.05, 0.05], 0.05)]},
    }}
    bundle = resolve_operating_point_from_review(
        review_state, "bud_opening", **tiled_regime(), group_by="stem", experiment_id="exp_review",
        bucket_identities=[_IDENTITY], scope_root=tmp_path,
        calibration_labels_dir=labels_dir)
    td = bundle.get("conf").gate_evidence["train_disjointness"]
    assert td["leaked_groups"] == [source]  # matched despite the .jpg extension on the review id
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
    didn't take effect against an existing lock: only the persisted sweep artifact would."""
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


def test_corrupt_lock_file_refuses_instead_of_silent_redraw(tmp_path, monkeypatch):
    """Bound to the file backend: undecodable bytes behind a record have no seam expression (a
    write always encodes a valid value), so this reaches the file the seam's own locator places
    them at. What is under test, catching DecodeError, is the store's own concern, not the file
    backend's, so this exercises it identically to a corruption reached any other way."""
    import tcip_store
    from tcip_store.file_backend import FileBackend

    from tcip_mcp.pipelines.data.splits import cal_holdout_lock_path, resolve_locked_cal_holdout_split

    tcip_store.bind(FileBackend())
    # The module's seeded trait put the pinned root in the database, so this case uses its own.
    scope = tmp_path / "file_backend_scope"
    monkeypatch.setenv("TCIP_STATE_ROOT", str(scope))
    lock_path = cal_holdout_lock_path("corrupt-test", scope_root=scope)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(ValueError, match="corrupt"):
        resolve_locked_cal_holdout_split(
            ["a_0_0", "b_0_0"], identity_hash="corrupt-test", scope_root=scope, seed=1)

    # force_redraw=True is the deliberate, audited path past the corrupt file, but its
    # redraw_history honestly starts fresh (nothing recoverable from an unreadable file).
    redrawn = resolve_locked_cal_holdout_split(
        ["a_0_0", "b_0_0"], identity_hash="corrupt-test", scope_root=scope, seed=1,
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

    # First call locks the split over all 4 stems.
    calibration.calibrate_operating_point(
        stub_pass(_CalStub()), "bud_opening", str(labels_dir), str(images_dir))

    (images_dir / "b_0_1.png").unlink()  # an image vanishes after the lock

    with pytest.raises(ValueError, match="no longer present"):
        calibration.calibrate_operating_point(
            stub_pass(_CalStub()), "bud_opening", str(labels_dir), str(images_dir))


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
    calibration.calibrate_operating_point(
        stub_pass(stub), "bud_opening", str(labels_dir), str(images_dir),
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
        stub_pass(_CalStub()), "bud_opening", str(labels_dir), str(images_dir),
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
    # Each root carries the trait spec an adopted project of its own would hold.
    for root in (tmp_path / "before_adoption", tmp_path / "adopted_project"):
        shutil.copytree(tmp_path / ".tcip", root / ".tcip")

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path / "before_adoption"))
    first, _dh, _n_excluded, _evidence = calibration.calibrate_operating_point(
        stub_pass(_CalStub()), "bud_opening", str(labels_dir), str(images_dir), seed=1)

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path / "adopted_project"))
    second, _dh2, _n_excluded2, _evidence2 = calibration.calibrate_operating_point(
        stub_pass(_CalStub()), "bud_opening", str(labels_dir), str(images_dir), seed=2)

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
    from tcip_mcp.subject_registry import Attribute, SubjectRegistry, Subject, write_registry

    root = tmp_path / "ds"
    images_dir, labels_dir = root / "images", root / "labels"
    write_registry(root / "subjects.json", SubjectRegistry(subjects=(
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
            Annotation(subject="bud", geometry=BBox(15, 15, 20, 20)),  # no `state`: unlabeled
        ], IMG, IMG)

    stub = _CalStub()
    stub.config = {"data": {"scope": {"subject": "bud", "attribute": "state",
                                      "id_map": {"open": 0, "closed": 1}}}}

    _bundle, _dh, n_excluded, _evidence = calibration.calibrate_operating_point(
        stub_pass(stub), "bud_opening", str(labels_dir), str(images_dir),
        group_by="stem", seed=0, holdout_ratio=0.5,
    )

    assert n_excluded == 2  # partial_a + partial_b, wherever the split put them


# ===========================================================================
# Calibration's GT-side id-map resolution must prefer the training-recorded map over a fresh
# registry read, the same map decode reads through run_scope: a
# subjects.json whose declared attribute-value order was edited since training must not silently
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
    # a fresh subjects.json (absent here) would derive.
    stub.config = {"data": {"scope": {"subject": "bud", "attribute": "state",
                                      "id_map": {"open": 0, "closed": 1}}}}

    def _boom(*a, **kw):
        raise AssertionError(
            "resolve_registry_id_map must not be called when the checkpoint carries its own "
            "recorded id_map")

    monkeypatch.setattr("tcip_mcp.pipelines.data.label_queries.resolve_registry_id_map", _boom)

    # No subjects.json exists for this dataset: calibrate_operating_point must still succeed,
    # using only the recorded map, never re-deriving from the registry when `subject` is set.
    bundle, _dh, n_excluded, _evidence = calibration.calibrate_operating_point(
        stub_pass(stub), "bud_opening", str(labels_dir), str(images_dir),
        group_by="stem", seed=0, holdout_ratio=0.5,
    )
    assert n_excluded == 0
    assert bundle is not None


# --- Minor: resolve_model_identity off a load_registered_checkpoint object. -------------------

def test_minor_resolve_model_identity_reads_the_codebase_own_stamped_checkpoints(tmp_path):
    """weights_only=True must still read a checkpoint the envelope saves, and the identity names
    the run that completed it: the rail must admit valid work, not only reject foreign
    payloads."""
    from tcip_mcp.model_registry import load_registered_checkpoint, resolve_model_identity
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    ckpt = registered_checkpoint(tmp_path, experiment_id="exp_abc")
    checkpoint = load_registered_checkpoint(ckpt, project_path=str(tmp_path))
    identity = resolve_model_identity(checkpoint)
    assert identity["experiment_id"] == "exp_abc"

