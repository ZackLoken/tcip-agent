"""clear_prediction_bucket: every refusal it makes before any write, each asserted on its own
sentence. Interrupted clears, fault injection and the review-landed-during-move race live in
test_clear_prediction_bucket_resume.py instead."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tests._clear_prediction_bucket_fixtures import (
    build_published_bucket, mark_bulk_accepted, record_review_verdict,
)
from tests._record_damage_fixtures import damage_record


def _set_stamp_field(project: Path, bucket: Path, **fields) -> None:
    from tcip_mcp.pipelines.resolution import update_sidecar

    update_sidecar(bucket, lambda stamp: {**stamp, **fields}, project=project)


def test_empty_reason_refuses(tmp_path, monkeypatch):
    from tcip_mcp.tools.inference_tools import clear_prediction_bucket

    built = build_published_bucket(tmp_path, monkeypatch, experiment_id="expEmptyReason")
    result = clear_prediction_bucket(tmp_path, str(built["bucket"]), "   ")
    assert "error" in result
    assert "non-empty reason" in result["error"]


def test_source_under_cleared_tree_refuses(tmp_path, monkeypatch):
    from tcip_mcp.tools.inference_tools import clear_prediction_bucket

    built = build_published_bucket(tmp_path, monkeypatch, experiment_id="expAlreadyCleared")
    first = clear_prediction_bucket(tmp_path, str(built["bucket"]), "first clear")
    assert "error" not in first, first

    second = clear_prediction_bucket(tmp_path, first["cleared_bucket"], "clearing the archive itself")
    assert "error" in second
    assert "already under the cleared archive" in second["error"]


def test_bespoke_bucket_refuses_naming_the_layout(tmp_path, monkeypatch):
    from tcip_mcp.tools.inference_tools import clear_prediction_bucket

    bespoke = tmp_path / "somewhere" / "not_a_dataset_tree"
    bespoke.mkdir(parents=True)
    result = clear_prediction_bucket(tmp_path, str(bespoke), "a bespoke bucket")
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

    result = clear_prediction_bucket(tmp_path, str(bucket), "no stamp at all")
    assert "error" in result
    assert "not a published bucket" in result["error"]


def test_undecodable_operating_point_stamp_refuses_as_unreadable(tmp_path, monkeypatch):
    from tcip_mcp.pipelines.resolution import sidecar_key
    from tcip_mcp.tools.inference_tools import clear_prediction_bucket

    built = build_published_bucket(tmp_path, monkeypatch, experiment_id="expUndecodableOp")
    key = sidecar_key(built["bucket"], "operating_point")
    damage_record(key, b"{not json")

    result = clear_prediction_bucket(tmp_path, str(built["bucket"]), "should refuse: undecodable")
    assert "error" in result
    assert "will not decode" in result["error"]
    assert "unreadable" in result["error"]


def test_undecodable_secondary_stamp_refuses(tmp_path, monkeypatch):
    import tcip_store as ts
    from tcip_mcp.pipelines.resolution import sidecar_key
    from tcip_mcp.tools.inference_tools import clear_prediction_bucket

    built = build_published_bucket(tmp_path, monkeypatch, experiment_id="expUndecodableSecondary")
    key = sidecar_key(built["bucket"], "resolve_scale")
    ts.replace(key, {"placeholder": True}, expect=ts.Version.ABSENT)
    damage_record(key, b"{not json")

    result = clear_prediction_bucket(tmp_path, str(built["bucket"]), "should refuse: secondary undecodable")
    assert "error" in result
    assert "resolve_scale.json will not decode" in result["error"]


def test_raster_stamp_refuses_naming_the_regime_out_of_scope(tmp_path, monkeypatch):
    built = build_published_bucket(tmp_path, monkeypatch, experiment_id="expRaster")
    _set_stamp_field(tmp_path, built["bucket"], raster_path="/some/mosaic.tif")

    from tcip_mcp.tools.inference_tools import clear_prediction_bucket

    result = clear_prediction_bucket(tmp_path, str(built["bucket"]), "should refuse: raster regime")
    assert "error" in result
    assert "raster_path" in result["error"]
    assert ".tcip/" in result["error"]


def test_no_document_refuses_as_already_republishable(tmp_path, monkeypatch):
    """A stamp-only bucket (an empty first pass) has nothing to clear: it is already
    re-publishable in place through run_inference."""
    from tcip_mcp.dataset_layout import prediction_dir
    from tests._clear_prediction_bucket_fixtures import stub_predictor
    from tests._verified_checkpoint_fixtures import registered_checkpoint
    from tcip_mcp.tools.inference_tools import clear_prediction_bucket, run_inference

    stub_predictor(monkeypatch)
    dataset_root = tmp_path / "ds"
    empty_images = tmp_path / "empty_images"
    empty_images.mkdir()
    ckpt = registered_checkpoint(tmp_path, experiment_id="expNoDocument")

    bucket = prediction_dir(dataset_root, "m", "2026-03-02")
    r1 = run_inference(tmp_path, ckpt, str(empty_images), output_dir=str(bucket), tile=False)
    assert "error" not in r1, r1

    result = clear_prediction_bucket(tmp_path, str(bucket), "should refuse: no document")
    assert "error" in result
    assert "holds no prediction document" in result["error"]


def test_detection_verdict_refuses_carrying_review_state(tmp_path, monkeypatch):
    from tcip_mcp.project_paths import project_state_dir
    from tcip_mcp.tools.inference_tools import clear_prediction_bucket

    built = build_published_bucket(tmp_path, monkeypatch, experiment_id="expVerdict")
    review_state_dir = project_state_dir(built["dataset_root"])
    record_review_verdict(built["bucket"], review_state_dir, "img.png")

    result = clear_prediction_bucket(tmp_path, str(built["bucket"]), "should refuse: review state")
    assert "error" in result
    assert "review" in result["error"]


def test_zero_verdict_complete_refuses_carrying_review_state(tmp_path, monkeypatch):
    """A bulk-accepted image carries no detection verdict, but this door refuses a bucket carrying
    review state, not only its documents: review_state_count sees it where verdict_count would
    not."""
    from tcip_mcp.project_paths import project_state_dir
    from tcip_mcp.tools.inference_tools import clear_prediction_bucket

    built = build_published_bucket(tmp_path, monkeypatch, experiment_id="expBulkAccept")
    review_state_dir = project_state_dir(built["dataset_root"])
    mark_bulk_accepted(built["bucket"], review_state_dir, "img.json")

    result = clear_prediction_bucket(tmp_path, str(built["bucket"]), "should refuse: bulk accept")
    assert "error" in result
    assert "review" in result["error"]


def test_review_on_removed_document_still_refuses(tmp_path, monkeypatch):
    """The bucket's review state survives a removed document, so this still refuses even though
    the stem no longer holds a document."""
    import tcip_store as ts
    from tcip_annotation.json_io import annotation_record_key
    from tcip_mcp.project_paths import project_state_dir
    from tcip_mcp.tools.inference_tools import clear_prediction_bucket

    built = build_published_bucket(tmp_path, monkeypatch, experiment_id="expRemovedDoc")
    review_state_dir = project_state_dir(built["dataset_root"])
    record_review_verdict(built["bucket"], review_state_dir, "img.png")
    ts.delete(annotation_record_key(built["bucket"], "img"))

    result = clear_prediction_bucket(tmp_path, str(built["bucket"]), "should refuse: review on removed doc")
    assert "error" in result
    assert "review" in result["error"]


def test_an_existing_empty_destination_reads_as_an_unfinished_clear(tmp_path, monkeypatch):
    """The destination directory is a clear's own record, so one already standing empty is an
    unfinished clear of this source, refused naming it as the resume."""
    from tcip_mcp.dataset_layout import cleared_prediction_dir
    from tcip_mcp.tools.inference_tools import clear_prediction_bucket
    import tcip_mcp.dataset_layout as dataset_layout_mod

    built = build_published_bucket(tmp_path, monkeypatch, experiment_id="expExistingDest")
    fixed_stamp = "20260101T000000Z"
    monkeypatch.setattr(dataset_layout_mod, "current_cleared_stamp", lambda: fixed_stamp)
    collide = cleared_prediction_dir(built["dataset_root"], "m", "2026-03-02", fixed_stamp)
    collide.mkdir(parents=True)

    result = clear_prediction_bucket(tmp_path, str(built["bucket"]), "should refuse: destination exists")
    assert "error" in result
    assert "is on record and unfinished" in result["error"]
    assert f"cleared_bucket={str(collide)!r}" in result["error"]


def test_cleared_bucket_naming_the_source_itself_refuses(tmp_path, monkeypatch):
    from tcip_mcp.tools.inference_tools import clear_prediction_bucket

    built = build_published_bucket(tmp_path, monkeypatch, experiment_id="expCBSource")
    result = clear_prediction_bucket(
        tmp_path, str(built["bucket"]), "should refuse", cleared_bucket=str(built["bucket"]))
    assert "error" in result
    assert "does not name" in result["error"]


def test_cleared_bucket_naming_another_models_cleared_bucket_refuses(tmp_path, monkeypatch):
    from tcip_mcp.tools.inference_tools import clear_prediction_bucket

    a = build_published_bucket(
        tmp_path, monkeypatch, experiment_id="expCBOtherModelA", model="modelA")
    b = build_published_bucket(
        tmp_path, monkeypatch, experiment_id="expCBOtherModelB", model="modelB",
        dataset_root=a["dataset_root"])
    cleared_b = clear_prediction_bucket(tmp_path, str(b["bucket"]), "clear b")
    assert "error" not in cleared_b, cleared_b

    result = clear_prediction_bucket(
        tmp_path, str(a["bucket"]), "should refuse: wrong model", cleared_bucket=cleared_b["cleared_bucket"])
    assert "error" in result
    assert "does not name" in result["error"]


def test_cleared_bucket_naming_a_path_no_clear_created_refuses(tmp_path, monkeypatch):
    """A destination-shaped path no clear created is not the newest cleared bucket on record:
    cleared_bucket_of works on strings, so the same refusal meets it whether or not it exists."""
    from tcip_mcp.dataset_layout import cleared_prediction_dir
    from tcip_mcp.tools.inference_tools import clear_prediction_bucket

    built = build_published_bucket(tmp_path, monkeypatch, experiment_id="expCBNoArtifact")
    phantom = cleared_prediction_dir(built["dataset_root"], "m", "2026-03-02", "20200101T000000Z")
    assert not phantom.exists()

    result = clear_prediction_bucket(
        tmp_path, str(built["bucket"]), "should refuse: no clear created it",
        cleared_bucket=str(phantom))
    assert "error" in result
    assert "not the newest cleared bucket on record" in result["error"]


def test_a_run_into_the_suggested_variant_publishes(tmp_path, monkeypatch):
    """The refusal a caller meets for a run's own published bucket is the document refusal at
    resolution, naming the suggested @r<n> variant; no run record pins a bucket, so a run into
    that variant publishes."""
    from tcip_mcp.tools.inference_tools import run_inference

    built = build_published_bucket(tmp_path, monkeypatch, experiment_id="expBracketNamesDoor")
    source = built["bucket"]

    first_retry = run_inference(
        tmp_path, str(built["checkpoint"]), str(built["images_dir"]), output_dir=str(source), tile=False)
    assert "error" in first_retry
    suggested = first_retry["suggested_bucket"]
    assert suggested is not None

    second = run_inference(
        tmp_path, str(built["checkpoint"]), str(built["images_dir"]), output_dir=suggested, tile=False)
    assert "error" not in second, second
