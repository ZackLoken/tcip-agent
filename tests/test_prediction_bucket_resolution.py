"""resolve_prediction_bucket: which dir a run may write for (dataset_root, model, date)."""

from __future__ import annotations

import json

from tcip_annotation import Annotation, BBox
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
        preds=[Annotation(subject="catkin", geometry=BBox(10.0, 10.0, 30.0, 30.0), score=0.9)],
    )
    det = ReviewDetection(det_type="fp", class_name="catkin", conf=0.9, iou=None, gt_idx=None,
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
