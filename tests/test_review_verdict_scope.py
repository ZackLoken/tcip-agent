"""Review verdicts are scoped to the prediction bucket they were recorded against.

A camera that restarts its numbering gives two capture dates the same filename. Verdicts keyed by
image name alone put both dates' reviews in one place: the reference for one date is fed the
other's adjudications, and the immutability guard freezes a bucket nobody reviewed. These drive
the real staging path and the real review route, so what freezes a bucket is what a reviewer
actually recorded against it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tcip_annotation import Annotation, BBox
from tcip_mcp.dataset_layout import prediction_dir
from tcip_mcp.prediction_buckets import stage_prediction_shapes
from tcip_web.app import app

DATE_A = "2026-02-11"
DATE_B = "2026-03-04"
STEM = "IMG_0007"
IMG_W, IMG_H = 640, 480
BOX = (12.0, 20.0, 52.0, 44.0)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _stage(dataset_root: Path, model: str, date: str) -> dict:
    """Stage one prediction into ``model``'s bucket for ``date``."""
    return stage_prediction_shapes(
        str(dataset_root), model, date, STEM,
        annotations=[Annotation(subject="catkin", geometry=BBox(*BOX), score=0.83)],
        img_w=IMG_W, img_h=IMG_H,
    )


def _image(dataset_root: Path) -> Path:
    """One readable source image the review route can size its context from."""
    img_dir = dataset_root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    path = img_dir / f"{STEM}.jpg"
    Image.new("RGB", (IMG_W, IMG_H), color=(80, 120, 90)).save(path)
    return path


def _review(client: TestClient, dataset_root: Path, img: Path, pred_path: str, gt_path: Path):
    """Record one accepted verdict through the route a breeder's canvas uses."""
    return client.post("/api/review/action", json={
        "project_root": str(dataset_root),
        "image_name": f"{STEM}.jpg",
        "image_path": str(img),
        "gt_path": str(gt_path),
        "pred_path": pred_path,
        "det_type": "fp", "class_name": "catkin",
        "conf": 0.83, "iou": None,
        "gt_idx": None, "pred_idx": 0,
        "bbox": list(BOX),
        "action": "accepted",
    })


def test_same_basename_on_two_dates_keeps_separate_verdicts(
    client: TestClient, tmp_path: Path
) -> None:
    """Two dates of one camera filename, one dataset, one model: a verdict on one date's bucket
    freezes that bucket and leaves the other date's alone."""
    dataset_root = tmp_path / "data"
    img = _image(dataset_root)
    staged_a = _stage(dataset_root, "detector", DATE_A)
    staged_b = _stage(dataset_root, "detector", DATE_B)
    assert staged_a["redirected"] is False and staged_b["redirected"] is False

    resp = _review(client, dataset_root, img, staged_a["path"], tmp_path / "gt.json")
    assert resp.status_code == 200

    # The reviewed date's bucket is frozen and redirects.
    again_a = _stage(dataset_root, "detector", DATE_A)
    assert again_a["redirected"] is True
    assert again_a["bucket"] == "detector@r2"
    assert again_a["verdict_count"] == 1

    # The other date shares the filename and nothing else: it is written where it was asked for.
    again_b = _stage(dataset_root, "detector", DATE_B)
    assert again_b["redirected"] is False
    assert again_b["bucket"] == "detector"
    assert again_b["verdict_count"] == 0
    assert again_b["path"] == str(Path(prediction_dir(dataset_root, "detector", DATE_B))
                                 / f"{STEM}.json")


def test_a_multi_bucket_datasets_verdicts_enumerate_under_the_bucket_they_were_recorded_on(
    client: TestClient, tmp_path: Path
) -> None:
    """The admit case: a dataset holding several reviewed and unreviewed buckets reads back one
    set of verdicts per bucket, and the unreviewed bucket keeps a count of zero."""
    from tcip_annotation.review_engine import ReviewEngine
    from tcip_mcp.prediction_buckets import bucket_key_of, review_state_dir_of, verdict_count

    dataset_root = tmp_path / "data"
    img = _image(dataset_root)
    staged_a = _stage(dataset_root, "detector", DATE_A)
    staged_b = _stage(dataset_root, "detector", DATE_B)

    assert _review(client, dataset_root, img, staged_a["path"], tmp_path / "gt.json").status_code == 200

    key_a = bucket_key_of(prediction_dir(dataset_root, "detector", DATE_A))
    key_b = bucket_key_of(prediction_dir(dataset_root, "detector", DATE_B))
    state_dir = review_state_dir_of(dataset_root)
    engine = ReviewEngine(state_dir)

    assert engine.reviewed_buckets() == [key_a]
    assert list(engine.image_states(key_a)) == [f"{STEM}.jpg"]
    assert engine.image_states(key_b) == {}
    assert verdict_count(state_dir, key_a, [STEM]) == 1
    assert verdict_count(state_dir, key_b, [STEM]) == 0
    # The reviewed image reads as reviewed under the bucket it was reviewed on, and only there.
    assert engine.get_image_review_status(key_a, f"{STEM}.jpg") != "not_started"
    assert engine.get_image_review_status(key_b, f"{STEM}.jpg") == "not_started"
    # Both buckets' prediction files are still on disk, neither overwritten by the other's review.
    assert json.loads(Path(staged_b["path"]).read_text(encoding="utf-8"))["annotations"]
