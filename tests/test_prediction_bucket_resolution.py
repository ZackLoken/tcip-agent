"""resolve_prediction_bucket: which dir a run may write for (dataset_root, model, date)."""

from __future__ import annotations

import json

import pytest
from tcip_annotation import Annotation, BBox
from tcip_annotation.json_io import write_annotations
from tcip_annotation.review_engine import ReviewContext, ReviewDetection, ReviewEngine

from tcip_mcp.dataset_layout import models_with_predictions, prediction_dir
from tcip_mcp.prediction_buckets import bucket_key_of, resolve_prediction_bucket

DATE = "2026-02-11"


def _record_verdict(review_state_dir, bucket_dir, stem: str) -> None:
    review_state_dir.mkdir(parents=True, exist_ok=True)
    engine = ReviewEngine(review_state_dir)
    ctx = ReviewContext(
        img_name=f"{stem}.png",
        img_width=100,
        img_height=100,
        preds=[Annotation(subject="bud", geometry=BBox(10.0, 10.0, 30.0, 30.0), score=0.9)],
    )
    det = ReviewDetection(det_type="fp", class_name="bud", conf=0.9, iou=None, gt_idx=None,
                          pred_idx=0, bbox=(10.0, 10.0, 30.0, 30.0))
    engine.record_detection_action(bucket_key_of(bucket_dir), det, ctx, action="accepted")


def _write_bucket(dataset_root, model: str, stem: str):
    d = prediction_dir(dataset_root, model, DATE)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{stem}.json").write_text(
        json.dumps({"image": stem, "width": 100, "height": 100, "annotations": []})
    )
    return d


def test_unreviewed_bucket_is_used_as_named(tmp_path):
    dataset_root = tmp_path / "data"
    bucket, resolution = resolve_prediction_bucket(
        dataset_root, "baseline", DATE, review_state_dir=tmp_path / "state"
    )
    assert bucket == prediction_dir(dataset_root, "baseline", DATE)
    assert resolution.redirected is False


def test_verdicted_bucket_redirects_to_a_discoverable_model_named_sibling(tmp_path):
    dataset_root = tmp_path / "data"
    review_state_dir = tmp_path / "state"
    _write_bucket(dataset_root, "baseline", "img")
    _record_verdict(review_state_dir, prediction_dir(dataset_root, "baseline", DATE), "img")

    bucket, resolution = resolve_prediction_bucket(
        dataset_root, "baseline", DATE, review_state_dir=review_state_dir
    )
    assert resolution.redirected is True
    # The model segment varies, never the date: a date-named sibling would be invisible to
    # models_with_predictions, which is how every reader finds a bucket.
    assert bucket == prediction_dir(dataset_root, "baseline@r2", DATE)
    _write_bucket(dataset_root, resolution.name, "img")
    assert "baseline@r2" in models_with_predictions(dataset_root, DATE)


def test_verdict_redirect_skips_a_variant_that_already_holds_a_document(tmp_path):
    """A caller that opts into refuse_documents redirects around a verdicted bucket the same way
    it always has, but the variant search now also skips a candidate that holds a document with
    no verdict of its own: a redirect must never land on a bucket a prior, unreviewed publish
    already filled. refuse_documents is a keyword this family adds: run against a baseline
    without it, prove_test_fails_before.py records the call's own TypeError rather than a failed
    assertion, so this is new-API coverage, not a guard proof."""
    dataset_root = tmp_path / "data"
    review_state_dir = tmp_path / "state"
    _write_bucket(dataset_root, "baseline", "img")
    _record_verdict(review_state_dir, prediction_dir(dataset_root, "baseline", DATE), "img")
    # @r2 already holds a document, but no verdict: a candidate the search must pass over.
    _write_bucket(dataset_root, "baseline@r2", "img")

    bucket, resolution = resolve_prediction_bucket(
        dataset_root, "baseline", DATE, review_state_dir=review_state_dir, refuse_documents=True
    )
    assert resolution.redirected is True
    assert bucket == prediction_dir(dataset_root, "baseline@r3", DATE)


def test_document_holding_bucket_with_no_verdicts_refuses_naming_a_free_suggestion(tmp_path):
    """A rail must admit valid work: the suggested bucket a document refusal names is itself
    free of both a verdict and a document, and writing into it (the platform's own producer)
    succeeds. refuse_documents is a keyword this family adds: run against a baseline without it,
    prove_test_fails_before.py records the call's own TypeError rather than a failed assertion,
    so this is new-API coverage, not a guard proof."""
    dataset_root = tmp_path / "data"
    review_state_dir = tmp_path / "state"
    _write_bucket(dataset_root, "baseline", "img")

    with pytest.raises(Exception) as excinfo:
        resolve_prediction_bucket(
            dataset_root, "baseline", DATE, review_state_dir=review_state_dir,
            refuse_documents=True,
        )
    assert type(excinfo.value).__name__ == "BucketHoldsDocuments"
    assert excinfo.value.document_stem_count == 1
    assert excinfo.value.suggested == "baseline@r2"

    # The admitting case: the suggested bucket is free, so writing into it succeeds.
    bucket, resolution = resolve_prediction_bucket(
        dataset_root, excinfo.value.suggested, DATE, review_state_dir=review_state_dir,
        refuse_documents=True,
    )
    assert resolution.redirected is False
    bucket.mkdir(parents=True, exist_ok=True)
    write_annotations(bucket / "img2.json", [], img_w=100, img_h=100, keep_empty=True)
    assert (bucket / "img2.json").is_file()


def test_document_refusal_exhaustion_names_no_suggestion(tmp_path):
    """Coverage of the exhausted variant search: when the requested bucket and every
    <name>@r<n> variant up to the ceiling already hold a document, the resolver refuses by name
    with no suggestion, rather than handing back an unchecked, never-searched directory.
    refuse_documents is a keyword this family adds: run against a baseline without it,
    prove_test_fails_before.py records the call's own TypeError rather than a failed assertion,
    so this too is new-API coverage."""
    from tcip_mcp.prediction_buckets import resolve_writable_bucket

    dataset_root = tmp_path / "data"
    review_state_dir = tmp_path / "state"
    max_variants = 3

    names = ["baseline"] + [f"baseline@r{n}" for n in range(2, max_variants + 1)]
    for name in names:
        d = prediction_dir(dataset_root, name, DATE)
        d.mkdir(parents=True, exist_ok=True)
        write_annotations(d / "img.json", [], img_w=100, img_h=100, keep_empty=True)

    def _dirs_for(name: str):
        return [prediction_dir(dataset_root, name, DATE)]

    with pytest.raises(Exception) as excinfo:
        resolve_writable_bucket(
            review_state_dir, "baseline", _dirs_for,
            refuse_documents=True, max_variants=max_variants,
        )
    assert type(excinfo.value).__name__ == "BucketHoldsDocuments"
    assert excinfo.value.suggested is None
    assert "baseline" in str(excinfo.value)


def test_verdict_exhaustion_refuses_by_name_with_the_document_keyword_off(tmp_path):
    """Coverage for a caller that never opts into refuse_documents (stage_prediction_shapes, the
    web route's own resolve_prediction_bucket call): both resolve through this same function, so
    the refusal proven here at the keyword's own default stands in for either rather than
    re-deriving their own setup. Before this family, an exhausted variant search silently fell
    back to an unchecked <name>@r100 even here; now it raises by name with no suggestion, since a
    redirect onto an unchecked directory is the overwrite this guard exists to refuse."""
    from tcip_mcp.prediction_buckets import resolve_writable_bucket

    dataset_root = tmp_path / "data"
    review_state_dir = tmp_path / "state"
    max_variants = 3

    for name in ["baseline"] + [f"baseline@r{n}" for n in range(2, max_variants + 1)]:
        _write_bucket(dataset_root, name, "img")
        _record_verdict(review_state_dir, prediction_dir(dataset_root, name, DATE), "img")

    def _dirs_for(name: str):
        return [prediction_dir(dataset_root, name, DATE)]

    with pytest.raises(Exception) as excinfo:
        resolve_writable_bucket(review_state_dir, "baseline", _dirs_for, max_variants=max_variants)
    assert type(excinfo.value).__name__ == "BucketHasVerdicts"
    assert excinfo.value.suggested is None
