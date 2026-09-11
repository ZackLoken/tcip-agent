"""resolve_prediction_bucket: which dir a run may write for (dataset_root, model, date)."""

from __future__ import annotations

import json

import pytest
from tcip_annotation import Annotation, BBox
from tcip_annotation.json_io import write_annotations
from tcip_annotation.review_engine import ReviewContext, ReviewDetection, ReviewEngine

from tcip_mcp.dataset_layout import models_with_predictions, prediction_dir
from tcip_mcp.prediction_buckets import (
    bucket_key_of, resolve_prediction_bucket, review_state_count, verdict_count,
)

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
    already filled. Coverage, not a guard."""
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
    succeeds. Coverage, not a guard."""
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
    with no suggestion, rather than handing back an unchecked, never-searched directory."""
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
    re-deriving their own setup. An exhausted variant search raises by name with no suggestion,
    rather than falling back to an unchecked <name>@r100: a redirect onto an unchecked directory
    is the overwrite this guard exists to refuse."""
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


def test_review_state_count_counts_review_decisions_verdict_count_does_not(tmp_path):
    """review_state_count is bucket-wide over the engine's own image states without ``names``:
    a bulk-accepted image and a reviewed image whose document has since been removed both count
    for it, and for neither does verdict_count, which counts detection entries alone. A
    not_started shard an unmark leaves behind, and an image reviewed under another bucket, count
    for neither."""
    dataset_root = tmp_path / "data"
    review_state_dir = tmp_path / "state"
    for stem in ("bulk_accepted", "not_started_shard", "reviewed_elsewhere", "doc_removed"):
        _write_bucket(dataset_root, "baseline", stem)
    bucket_dir = prediction_dir(dataset_root, "baseline", DATE)
    bucket_key = bucket_key_of(bucket_dir)
    other_bucket_dir = prediction_dir(dataset_root, "other", DATE)
    other_bucket_dir.mkdir(parents=True, exist_ok=True)

    engine = ReviewEngine(review_state_dir)
    engine.mark_image_reviewed(bucket_key, "bulk_accepted.json")
    engine.mark_image_reviewed(bucket_key, "not_started_shard.json")
    engine.unmark_image_reviewed(bucket_key, "not_started_shard.json")
    engine.mark_image_reviewed(bucket_key, "doc_removed.json")
    (bucket_dir / "doc_removed.json").unlink()
    engine.mark_image_reviewed(bucket_key_of(other_bucket_dir), "reviewed_elsewhere.json")

    names = {"bulk_accepted", "not_started_shard", "reviewed_elsewhere", "doc_removed"}
    assert verdict_count(review_state_dir, bucket_key, names) == 0
    assert review_state_count(review_state_dir, bucket_key) == 2


def test_review_state_count_stem_scoped_form_matches_the_detection_count_rule(tmp_path):
    """review_state_count's stem-scoped form (``names`` given) applies the identical stem-match
    rule verdict_count_for_images does, over the same fixture the bucket-wide test above builds:
    a stem the resolver actually asks about (one holding a document) reads the engine's state for
    it, an unreviewed or elsewhere-reviewed stem reads 0, and an empty ``names`` (identity against
    ``None``, never truthiness) reads 0 without asking the store anything. Coverage, a new
    function argument."""
    dataset_root = tmp_path / "data"
    review_state_dir = tmp_path / "state"
    for stem in ("bulk_accepted", "not_started_shard", "reviewed_elsewhere", "doc_removed"):
        _write_bucket(dataset_root, "baseline", stem)
    bucket_dir = prediction_dir(dataset_root, "baseline", DATE)
    bucket_key = bucket_key_of(bucket_dir)
    other_bucket_dir = prediction_dir(dataset_root, "other", DATE)
    other_bucket_dir.mkdir(parents=True, exist_ok=True)

    engine = ReviewEngine(review_state_dir)
    engine.mark_image_reviewed(bucket_key, "bulk_accepted.json")
    engine.mark_image_reviewed(bucket_key, "not_started_shard.json")
    engine.unmark_image_reviewed(bucket_key, "not_started_shard.json")
    engine.mark_image_reviewed(bucket_key, "doc_removed.json")
    (bucket_dir / "doc_removed.json").unlink()
    engine.mark_image_reviewed(bucket_key_of(other_bucket_dir), "reviewed_elsewhere.json")

    assert review_state_count(review_state_dir, bucket_key, names={"bulk_accepted"}) == 1
    assert review_state_count(review_state_dir, bucket_key, names={"not_started_shard"}) == 0
    assert review_state_count(review_state_dir, bucket_key, names={"reviewed_elsewhere"}) == 0
    assert review_state_count(review_state_dir, bucket_key, names={"doc_removed"}) == 1
    assert review_state_count(review_state_dir, bucket_key, names=()) == 0
    assert review_state_count(review_state_dir, bucket_key) == 2


def test_resolve_prediction_bucket_count_review_state_redirects_a_bulk_accept(tmp_path):
    """Coverage: the keyword does not exist at the baseline, so this call fails on a TypeError
    there, not the assertion named below; proved on the landed tree by flipping the keyword's
    reading in a scratch copy of the module, never the shipped one.

    A bucket whose one document's image was bulk-accepted (no detection entry at all) redirects
    to @r2 under count_review_state=True, with verdict_count == 1 on the default and
    BucketHasVerdicts on overwrite=True; the same call with the keyword off returns the bucket in
    place, since the publishers' own reading never counts a bulk accept."""
    from tcip_mcp.prediction_buckets import BucketHasVerdicts

    dataset_root = tmp_path / "data"
    review_state_dir = tmp_path / "state"
    bucket_dir = _write_bucket(dataset_root, "baseline", "img")
    ReviewEngine(review_state_dir).mark_image_reviewed(bucket_key_of(bucket_dir), "img.json")

    bucket, resolution = resolve_prediction_bucket(
        dataset_root, "baseline", DATE, review_state_dir=review_state_dir,
        count_review_state=True,
    )
    assert resolution.redirected is True
    assert resolution.verdict_count == 1
    assert bucket == prediction_dir(dataset_root, "baseline@r2", DATE)

    with pytest.raises(BucketHasVerdicts) as excinfo:
        resolve_prediction_bucket(
            dataset_root, "baseline", DATE, review_state_dir=review_state_dir,
            overwrite=True, count_review_state=True,
        )
    assert excinfo.value.count == 1

    bucket_off, resolution_off = resolve_prediction_bucket(
        dataset_root, "baseline", DATE, review_state_dir=review_state_dir,
    )
    assert resolution_off.redirected is False
    assert bucket_off == prediction_dir(dataset_root, "baseline", DATE)


def test_variant_search_under_count_review_state_skips_a_bulk_accepted_candidate(tmp_path):
    """Coverage, for the same reason as above: the variant search under count_review_state=True
    treats a bulk-accepted @r2 as taken, same as a verdicted one, and redirects past it to @r3;
    without the keyword @r2 (no detection verdict) is free and is where the redirect lands."""
    dataset_root = tmp_path / "data"
    review_state_dir = tmp_path / "state"
    bucket_dir = _write_bucket(dataset_root, "baseline", "img")
    _record_verdict(review_state_dir, bucket_dir, "img")
    r2_dir = _write_bucket(dataset_root, "baseline@r2", "img")
    ReviewEngine(review_state_dir).mark_image_reviewed(bucket_key_of(r2_dir), "img.json")

    bucket, resolution = resolve_prediction_bucket(
        dataset_root, "baseline", DATE, review_state_dir=review_state_dir,
        count_review_state=True,
    )
    assert resolution.redirected is True
    assert bucket == prediction_dir(dataset_root, "baseline@r3", DATE)

    bucket_off, resolution_off = resolve_prediction_bucket(
        dataset_root, "baseline", DATE, review_state_dir=review_state_dir,
    )
    assert resolution_off.redirected is True
    assert bucket_off == prediction_dir(dataset_root, "baseline@r2", DATE)


def test_count_review_state_leaves_not_started_and_verdicted_buckets_unaffected(tmp_path):
    """Coverage: a bucket carrying only a not_started shard (an unmark) is unaffected by
    count_review_state, since a not_started shard is not in _REVIEWED_IMAGE_STATUSES; a bucket
    with one detection verdict per image redirects with the same count either way, since every
    image with a detection entry has img_status started or completed and so counts under both
    readings."""
    dataset_root = tmp_path / "data"
    review_state_dir = tmp_path / "state"

    unmarked_dir = _write_bucket(dataset_root, "unmarked", "img")
    engine = ReviewEngine(review_state_dir)
    key = bucket_key_of(unmarked_dir)
    engine.mark_image_reviewed(key, "img.json")
    engine.unmark_image_reviewed(key, "img.json")
    bucket, resolution = resolve_prediction_bucket(
        dataset_root, "unmarked", DATE, review_state_dir=review_state_dir,
        count_review_state=True,
    )
    assert resolution.redirected is False
    assert bucket == prediction_dir(dataset_root, "unmarked", DATE)

    verdicted_dir = _write_bucket(dataset_root, "verdicted", "img")
    _record_verdict(review_state_dir, verdicted_dir, "img")
    _, resolution_on = resolve_prediction_bucket(
        dataset_root, "verdicted", DATE, review_state_dir=review_state_dir,
        count_review_state=True,
    )
    _, resolution_off = resolve_prediction_bucket(
        dataset_root, "verdicted", DATE, review_state_dir=review_state_dir,
    )
    assert resolution_on.redirected is True and resolution_off.redirected is True
    assert resolution_on.verdict_count == resolution_off.verdict_count == 1
