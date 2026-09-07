"""clear_prediction_bucket: every D5 refusal, each asserted on its own sentence, and D7's bracket
refusal naming the door. Interrupted clears, fault injection and the review-landed-during-move
race live in test_clear_prediction_bucket_resume.py instead."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tests._clear_prediction_bucket_fixtures import (
    build_published_bucket, mark_bulk_accepted, record_review_verdict,
)
from tests._record_damage_fixtures import damage_record


def _set_stamp_field(bucket: Path, **fields) -> None:
    from tcip_mcp.pipelines.resolution import update_sidecar

    update_sidecar(bucket, lambda stamp: {**stamp, **fields})


def test_empty_reason_refuses(tmp_path, monkeypatch):
    from tcip_mcp.tools.inference_tools import clear_prediction_bucket

    built = build_published_bucket(tmp_path, monkeypatch, experiment_id="expEmptyReason")
    result = clear_prediction_bucket(str(built["bucket"]), "   ")
    assert "error" in result
    assert "non-empty reason" in result["error"]


def test_source_under_cleared_tree_refuses(tmp_path, monkeypatch):
    from tcip_mcp.tools.inference_tools import clear_prediction_bucket

    built = build_published_bucket(tmp_path, monkeypatch, experiment_id="expAlreadyCleared")
    first = clear_prediction_bucket(str(built["bucket"]), "first clear")
    assert "error" not in first, first

    second = clear_prediction_bucket(first["cleared_bucket"], "clearing the archive itself")
    assert "error" in second
    assert "already under the cleared archive" in second["error"]


def test_bespoke_bucket_refuses_naming_the_layout(tmp_path, monkeypatch):
    from tcip_mcp.tools.inference_tools import clear_prediction_bucket

    bespoke = tmp_path / "somewhere" / "not_a_dataset_tree"
    bespoke.mkdir(parents=True)
    result = clear_prediction_bucket(str(bespoke), "a bespoke bucket")
    assert "error" in result
    assert "not a canonical prediction bucket" in result["error"]
    assert "predictions/<model>" in result["error"]


def test_no_stamp_and_no_candidate_refuses_as_not_published(tmp_path, monkeypatch):
    """A staged bucket (stage_prediction_shapes, no stamp of any kind) is not a published bucket
    and no cleared archive names a remedy for it."""
    from tcip_annotation import Annotation, BBox
    from tcip_mcp.prediction_buckets import stage_prediction_shapes
    from tcip_mcp.tools.inference_tools import clear_prediction_bucket

    dataset_root = tmp_path / "ds"
    staged = stage_prediction_shapes(
        str(dataset_root), "staged_model", "2026-03-02", "img",
        annotations=[Annotation(subject="bud", geometry=BBox(1, 1, 5, 5))], img_w=32, img_h=32,
    )
    bucket = Path(staged["path"]).parent

    result = clear_prediction_bucket(str(bucket), "no stamp at all")
    assert "error" in result
    assert "not a published bucket" in result["error"]


def test_undecodable_operating_point_stamp_refuses_as_unreadable(tmp_path, monkeypatch):
    from tcip_mcp.pipelines.resolution import sidecar_key
    from tcip_mcp.tools.inference_tools import clear_prediction_bucket

    built = build_published_bucket(tmp_path, monkeypatch, experiment_id="expUndecodableOp")
    key = sidecar_key(built["bucket"], "operating_point")
    damage_record(key, b"{not json")

    result = clear_prediction_bucket(str(built["bucket"]), "should refuse: undecodable")
    assert "error" in result
    assert "will not decode" in result["error"]
    assert "unreadable" in result["error"]


def test_unstated_scope_refuses_naming_the_conform_script(tmp_path, monkeypatch):
    """A stamp written straight to the store (bypassing operating_point_stamp's own rail) with no
    subject/attribute pair raises StampScopeUnstated, named as the conform script's own case."""
    import tcip_store as ts
    from tcip_mcp.dataset_layout import prediction_dir
    from tcip_mcp.experiments import create_experiment, update_lineage, update_status
    from tcip_mcp.pipelines.resolution import sidecar_key
    from tcip_mcp.tools.inference_tools import clear_prediction_bucket

    dataset_root = tmp_path / "ds"
    bucket = prediction_dir(dataset_root, "m", "2026-03-02")
    from tcip_annotation.json_io import write_annotations
    write_annotations(bucket / "img.json", [], 100, 100)

    exp_id = "expUnstatedScope"
    create_experiment(exp_id, {"model_source": {"builder": "x:y"}})
    update_status(exp_id, "running")
    update_lineage(exp_id, predictions=str(bucket))
    update_status(exp_id, "completed")

    ts.replace(
        sidecar_key(bucket, "operating_point"),
        {"conf": {"value": 0.5, "validated": False, "validated_against": None},
         "experiment_id": exp_id, "checkpoint_sha256": "abc"},
        expect=ts.Version.ABSENT,
    )

    result = clear_prediction_bucket(str(bucket), "should refuse: unstated scope")
    assert "error" in result
    assert "tcip repair-classified-predictions" in result["error"]


def test_undecodable_secondary_stamp_refuses(tmp_path, monkeypatch):
    import tcip_store as ts
    from tcip_mcp.pipelines.resolution import sidecar_key
    from tcip_mcp.tools.inference_tools import clear_prediction_bucket

    built = build_published_bucket(tmp_path, monkeypatch, experiment_id="expUndecodableSecondary")
    key = sidecar_key(built["bucket"], "resolve_scale")
    ts.replace(key, {"placeholder": True}, expect=ts.Version.ABSENT)
    damage_record(key, b"{not json")

    result = clear_prediction_bucket(str(built["bucket"]), "should refuse: secondary undecodable")
    assert "error" in result
    assert "resolve_scale.json will not decode" in result["error"]


def test_raster_stamp_refuses_naming_the_regime_out_of_scope(tmp_path, monkeypatch):
    built = build_published_bucket(tmp_path, monkeypatch, experiment_id="expRaster")
    _set_stamp_field(built["bucket"], raster_path="/some/mosaic.tif")

    from tcip_mcp.tools.inference_tools import clear_prediction_bucket

    result = clear_prediction_bucket(str(built["bucket"]), "should refuse: raster regime")
    assert "error" in result
    assert "raster_path" in result["error"]
    assert ".tcip/" in result["error"]


def test_no_experiment_refuses(tmp_path, monkeypatch):
    built = build_published_bucket(tmp_path, monkeypatch, experiment_id="expNoExperimentField")
    _set_stamp_field(built["bucket"], experiment_id=None)

    from tcip_mcp.tools.inference_tools import clear_prediction_bucket

    result = clear_prediction_bucket(str(built["bucket"]), "should refuse: no experiment_id")
    assert "error" in result
    assert "names no experiment_id" in result["error"]


def test_running_experiment_refuses_naming_the_variant_remedy(tmp_path, monkeypatch):
    built = build_published_bucket(
        tmp_path, monkeypatch, experiment_id="expRunning", state=None)

    from tcip_mcp.tools.inference_tools import clear_prediction_bucket

    result = clear_prediction_bucket(str(built["bucket"]), "should refuse: not terminal")
    assert "error" in result
    assert "not terminal" in result["error"]
    assert "@r<n>" in result["error"]


def test_no_document_refuses_as_already_republishable(tmp_path, monkeypatch):
    """A stamp-only bucket (an empty first pass) has nothing to clear: it is already
    re-publishable in place through run_inference."""
    from tcip_mcp.experiments import create_experiment, update_status
    from tcip_mcp.dataset_layout import prediction_dir
    from tests._clear_prediction_bucket_fixtures import stub_checkpoint_verification, stub_predictor
    from tcip_mcp.tools.inference_tools import clear_prediction_bucket, run_inference

    stub_predictor(monkeypatch)
    stub_checkpoint_verification(monkeypatch)
    dataset_root = tmp_path / "ds"
    empty_images = tmp_path / "empty_images"
    empty_images.mkdir()
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"stub")

    exp_id = "expNoDocument"
    create_experiment(exp_id, {"model_source": {"builder": "x:y"}})
    update_status(exp_id, "running")
    bucket = prediction_dir(dataset_root, "m", "2026-03-02")
    r1 = run_inference(str(ckpt), str(empty_images), output_dir=str(bucket), tile=False,
                       experiment_id=exp_id)
    assert "error" not in r1, r1
    update_status(exp_id, "completed")

    result = clear_prediction_bucket(str(bucket), "should refuse: no document")
    assert "error" in result
    assert "holds no prediction document" in result["error"]


def test_detection_verdict_refuses_carrying_review_state(tmp_path, monkeypatch):
    from tcip_mcp.prediction_buckets import review_state_dir_of
    from tcip_mcp.tools.inference_tools import clear_prediction_bucket

    built = build_published_bucket(tmp_path, monkeypatch, experiment_id="expVerdict")
    review_state_dir = review_state_dir_of(built["dataset_root"])
    record_review_verdict(built["bucket"], review_state_dir, "img.png")

    result = clear_prediction_bucket(str(built["bucket"]), "should refuse: review state")
    assert "error" in result
    assert "review" in result["error"]


def test_zero_verdict_complete_refuses_carrying_review_state(tmp_path, monkeypatch):
    """A bulk-accepted image carries no detection verdict, but Q34 refuses the bucket, not its
    documents: review_state_count sees it where verdict_count would not."""
    from tcip_mcp.prediction_buckets import review_state_dir_of
    from tcip_mcp.tools.inference_tools import clear_prediction_bucket

    built = build_published_bucket(tmp_path, monkeypatch, experiment_id="expBulkAccept")
    review_state_dir = review_state_dir_of(built["dataset_root"])
    mark_bulk_accepted(built["bucket"], review_state_dir, "img.json")

    result = clear_prediction_bucket(str(built["bucket"]), "should refuse: bulk accept")
    assert "error" in result
    assert "review" in result["error"]


def test_review_on_removed_document_still_refuses(tmp_path, monkeypatch):
    """The bucket's review state survives a removed document: Q34 refuses the bucket, so this
    still refuses even though the stem no longer holds a document."""
    import tcip_store as ts
    from tcip_annotation.json_io import annotation_record_key
    from tcip_mcp.prediction_buckets import review_state_dir_of
    from tcip_mcp.tools.inference_tools import clear_prediction_bucket

    built = build_published_bucket(tmp_path, monkeypatch, experiment_id="expRemovedDoc")
    review_state_dir = review_state_dir_of(built["dataset_root"])
    record_review_verdict(built["bucket"], review_state_dir, "img.png")
    ts.delete(annotation_record_key(built["bucket"], "img"))

    result = clear_prediction_bucket(str(built["bucket"]), "should refuse: review on removed doc")
    assert "error" in result
    assert "review" in result["error"]


def test_existing_destination_without_cleared_bucket_refuses(tmp_path, monkeypatch):
    from tcip_mcp.dataset_layout import cleared_prediction_dir
    from tcip_mcp.tools.inference_tools import clear_prediction_bucket
    import tcip_mcp.dataset_layout as dataset_layout_mod

    built = build_published_bucket(tmp_path, monkeypatch, experiment_id="expExistingDest")
    fixed_stamp = "20260101T000000Z"
    monkeypatch.setattr(dataset_layout_mod, "current_cleared_stamp", lambda: fixed_stamp)
    collide = cleared_prediction_dir(built["dataset_root"], "m", "2026-03-02", fixed_stamp)
    collide.mkdir(parents=True)

    result = clear_prediction_bucket(str(built["bucket"]), "should refuse: destination exists")
    assert "error" in result
    assert "already exists" in result["error"]


def test_cleared_bucket_naming_the_source_itself_refuses(tmp_path, monkeypatch):
    from tcip_mcp.tools.inference_tools import clear_prediction_bucket

    built = build_published_bucket(tmp_path, monkeypatch, experiment_id="expCBSource")
    result = clear_prediction_bucket(
        str(built["bucket"]), "should refuse", cleared_bucket=str(built["bucket"]))
    assert "error" in result
    assert "does not name" in result["error"]


def test_cleared_bucket_naming_another_models_cleared_bucket_refuses(tmp_path, monkeypatch):
    from tcip_mcp.tools.inference_tools import clear_prediction_bucket

    a = build_published_bucket(
        tmp_path, monkeypatch, experiment_id="expCBOtherModelA", model="modelA")
    b = build_published_bucket(
        tmp_path, monkeypatch, experiment_id="expCBOtherModelB", model="modelB",
        dataset_root=a["dataset_root"])
    cleared_b = clear_prediction_bucket(str(b["bucket"]), "clear b")
    assert "error" not in cleared_b, cleared_b

    result = clear_prediction_bucket(
        str(a["bucket"]), "should refuse: wrong model", cleared_bucket=cleared_b["cleared_bucket"])
    assert "error" in result
    assert "does not name" in result["error"]


def test_cleared_bucket_naming_a_path_no_artifact_names_refuses(tmp_path, monkeypatch):
    from tcip_mcp.dataset_layout import cleared_prediction_dir
    from tcip_mcp.tools.inference_tools import clear_prediction_bucket

    built = build_published_bucket(tmp_path, monkeypatch, experiment_id="expCBNoArtifact")
    phantom = cleared_prediction_dir(built["dataset_root"], "m", "2026-03-02", "20200101T000000Z")

    result = clear_prediction_bucket(
        str(built["bucket"]), "should refuse: no artifact names it",
        cleared_bucket=str(phantom))
    assert "error" in result
    assert "no cleared: artifact" in result["error"]


def test_cleared_bucket_naming_a_nonexistent_path_no_artifact_names_refuses(tmp_path, monkeypatch):
    """Same refusal as the path-shaped-but-unrecorded case: cleared_bucket_of works on strings,
    never the filesystem, so a non-existent path is refused the identical way."""
    from tcip_mcp.dataset_layout import cleared_prediction_dir
    from tcip_mcp.tools.inference_tools import clear_prediction_bucket

    built = build_published_bucket(tmp_path, monkeypatch, experiment_id="expCBNonExistent")
    phantom = cleared_prediction_dir(built["dataset_root"], "m", "2026-03-02", "20200101T000000Z")
    assert not phantom.exists()

    result = clear_prediction_bucket(
        str(built["bucket"]), "should refuse: non-existent, unrecorded",
        cleared_bucket=str(phantom))
    assert "error" in result
    assert "no cleared: artifact" in result["error"]


def test_a_matching_pointer_never_reaches_the_brackets_own_refusal(tmp_path, monkeypatch):
    """The boundary D7's own naming condition sits on: a terminal experiment whose recorded
    pointer equals the resolved out is the pointer's own same-value admission (pointer_frozen
    never refuses an equal write), never the bracket's refusal at all. A second run in place
    reaches this admitted branch, not the refusal the door's name would otherwise be attached to.
    See the implementer's report for why the "names the door" branch could not be reached by any
    composition of shipped callers."""
    from tcip_mcp.tools.inference_tools import run_inference

    built = build_published_bucket(tmp_path, monkeypatch, experiment_id="expBracketAdmitted",
                                   stems=())
    second = run_inference(
        str(built["checkpoint"]), str(built["images_dir"]), output_dir=str(built["bucket"]),
        tile=False, experiment_id="expBracketAdmitted")
    assert "error" not in second, second


def test_bracket_names_no_route_when_the_pointer_names_another_path(tmp_path, monkeypatch):
    """A terminal experiment whose recorded pointer is a *different* path than the one this call
    resolved to (redirected around review verdicts on the original) refuses without naming the
    door: the door would refuse it too, since its own pointer check names only the recorded path."""
    from tcip_mcp.prediction_buckets import review_state_dir_of
    from tcip_mcp.tools.inference_tools import run_inference

    built = build_published_bucket(tmp_path, monkeypatch, experiment_id="expBracketNoRoute")
    record_review_verdict(built["bucket"], review_state_dir_of(built["dataset_root"]), "img.png")

    second = run_inference(
        str(built["checkpoint"]), str(built["images_dir"]), output_dir=str(built["bucket"]),
        tile=False, experiment_id="expBracketNoRoute")
    assert "error" in second
    assert "clear_prediction_bucket" not in second["error"]
