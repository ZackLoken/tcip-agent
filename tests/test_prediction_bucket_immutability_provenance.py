"""What a redirected prediction bucket records, and how far the variant search looks.

A bucket whose images carry human verdicts is frozen, and the resolution returned in its place
carries the provenance of that freeze: which bucket was asked for and how many verdicts stood in
the way. The search for a free name must keep walking while each candidate is itself reviewed,
since a variant that was reviewed in an earlier round is as immutable as the original. The verdict
lookup is also a two-sided agreement: the writer of a verdict (the GUI review route) and the reader
that guards a bucket must land on the same review state without either being told where it is.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tcip_annotation import Annotation, BBox
from tcip_annotation.review_engine import ReviewContext, ReviewDetection, ReviewEngine
from tcip_mcp.dataset_layout import prediction_dir
from tcip_mcp.prediction_buckets import (
    bucket_key_of, resolve_prediction_bucket, stage_prediction_shapes,
)
from tcip_web.app import app

DATE = "2026-03-04"
IMG_W, IMG_H = 640, 480


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


def _write_bucket(dataset_root: Path, model: str, stems: list[str]) -> Path:
    """Give ``model``'s bucket one per-image prediction file per stem."""
    d = Path(prediction_dir(dataset_root, model, DATE))
    d.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        (d / f"{stem}.json").write_text(
            json.dumps(
                {
                    "image": stem,
                    "width": IMG_W,
                    "height": IMG_H,
                    "annotations": [
                        {"subject": "bud", "bbox": [10.0, 12.0, 40.0, 60.0], "score": 0.7}
                    ],
                }
            ),
            encoding="utf-8",
        )
    return d


def _record_verdicts(review_state_dir: Path, bucket_dir: Path, stem: str,
                     boxes: list[tuple[float, ...]]) -> None:
    """Record one human verdict per box against ``stem`` under ``bucket_dir``, each on its own
    detection."""
    review_state_dir.mkdir(parents=True, exist_ok=True)
    engine = ReviewEngine(review_state_dir)
    preds = [Annotation(subject="bud", geometry=BBox(*b), score=0.7) for b in boxes]
    ctx = ReviewContext(img_name=f"{stem}.jpg", img_width=IMG_W, img_height=IMG_H, preds=preds)
    for i, box in enumerate(boxes):
        det = ReviewDetection(
            det_type="fp", class_name="bud", conf=0.7, iou=None,
            gt_idx=None, pred_idx=i, bbox=box,
        )
        engine.record_detection_action(bucket_key_of(bucket_dir), det, ctx, action="accepted")


def test_redirect_reports_how_many_verdicts_froze_the_requested_bucket(tmp_path: Path) -> None:
    """The resolution returned in place of a frozen bucket carries the verdict count that froze
    it, so an operator reading the redirect sees the human work being protected rather than an
    unexplained change of destination."""
    dataset_root = tmp_path / "data"
    review_state_dir = tmp_path / "state"
    detector = _write_bucket(dataset_root, "detector", ["IMG_0007", "IMG_0021"])
    _record_verdicts(review_state_dir, detector, "IMG_0007",
                     [(10.0, 12.0, 40.0, 60.0), (300.0, 40.0, 360.0, 90.0)])
    _record_verdicts(review_state_dir, detector, "IMG_0021", [(88.0, 200.0, 130.0, 260.0)])

    _bucket, resolution = resolve_prediction_bucket(
        dataset_root, "detector", DATE, review_state_dir=review_state_dir
    )

    assert resolution.redirected is True
    assert resolution.requested == "detector"
    assert resolution.name == "detector@r2"
    # Three verdicts were recorded across the two reviewed images of this bucket.
    assert resolution.verdict_count == 3


def test_variant_search_walks_past_every_reviewed_variant(tmp_path: Path) -> None:
    """Each already-reviewed variant is as immutable as the bucket originally asked for, so the
    search keeps going until it reaches a name no verdict is attached to. Stopping at the first
    candidate would hand back a variant a reviewer had already adjudicated."""
    dataset_root = tmp_path / "data"
    review_state_dir = tmp_path / "state"
    detector = _write_bucket(dataset_root, "detector", ["IMG_0007", "IMG_0021"])
    detector_r2 = _write_bucket(dataset_root, "detector@r2", ["IMG_0055"])
    detector_r3 = _write_bucket(dataset_root, "detector@r3", ["IMG_0090"])
    _record_verdicts(review_state_dir, detector, "IMG_0007",
                     [(10.0, 12.0, 40.0, 60.0), (300.0, 40.0, 360.0, 90.0)])
    _record_verdicts(review_state_dir, detector, "IMG_0021", [(88.0, 200.0, 130.0, 260.0)])
    _record_verdicts(review_state_dir, detector_r2, "IMG_0055",
                     [(20.0, 20.0, 70.0, 70.0), (400.0, 300.0, 440.0, 350.0)])
    _record_verdicts(review_state_dir, detector_r3, "IMG_0090", [(120.0, 130.0, 180.0, 190.0)])

    bucket, resolution = resolve_prediction_bucket(
        dataset_root, "detector", DATE, review_state_dir=review_state_dir
    )

    assert resolution.name not in {"detector", "detector@r2", "detector@r3"}
    assert resolution.name == "detector@r4"
    assert bucket == prediction_dir(dataset_root, "detector@r4", DATE)
    assert not bucket.exists()
    # The count reported is the requested bucket's own, not a later variant's.
    assert resolution.verdict_count == 3


def test_verdicts_on_a_neighbouring_bucket_do_not_freeze_this_one(tmp_path: Path) -> None:
    """Review state is shared by every bucket of a dataset, and a bucket is frozen only by
    verdicts against its own images. A populated bucket whose images nobody reviewed is written
    as named, even while another bucket's images carry verdicts."""
    dataset_root = tmp_path / "data"
    review_state_dir = tmp_path / "state"
    detector = _write_bucket(dataset_root, "detector", ["IMG_0007"])
    _write_bucket(dataset_root, "second_pass", ["IMG_0333"])
    _record_verdicts(review_state_dir, detector, "IMG_0007", [(10.0, 12.0, 40.0, 60.0)])

    bucket, resolution = resolve_prediction_bucket(
        dataset_root, "second_pass", DATE, review_state_dir=review_state_dir
    )

    assert resolution.redirected is False
    assert resolution.name == "second_pass"
    assert resolution.verdict_count == 0
    assert bucket == prediction_dir(dataset_root, "second_pass", DATE)


def test_staging_sees_the_verdicts_the_review_route_recorded(
    client: TestClient, tmp_path: Path
) -> None:
    """Neither side is told where the review state lives: the GUI review route derives it from the
    project it was given, and the staging path derives it from the dataset it writes to. A verdict
    recorded through the route must be the same verdict that freezes the bucket on the next stage,
    so a re-run lands in a fresh variant instead of overwriting adjudicated predictions."""
    root = tmp_path / "proj"
    img_dir = root / "images"
    img_dir.mkdir(parents=True)
    img = img_dir / "IMG_0007.jpg"
    Image.new("RGB", (IMG_W, IMG_H), color=(80, 120, 90)).save(img)
    preds = [Annotation(subject="bud", geometry=BBox(12.0, 20.0, 52.0, 44.0), score=0.83)]

    first = stage_prediction_shapes(
        str(root), "detector", DATE, "IMG_0007",
        annotations=preds, img_w=IMG_W, img_h=IMG_H,
    )
    assert first["bucket"] == "detector"
    assert first["redirected"] is False

    resp = client.post("/api/review/action", json={
        "dataset_root": str(root),
        "image_name": "IMG_0007.jpg",
        "image_path": str(img),
        "gt_path": str(tmp_path / "gt.json"),
        "pred_path": first["path"],
        "det_type": "fp", "class_name": "bud",
        "conf": 0.83, "iou": None,
        "gt_idx": None, "pred_idx": 0,
        "bbox": [12.0, 20.0, 52.0, 44.0],
        "action": "accepted",
    })
    assert resp.status_code == 200

    second = stage_prediction_shapes(
        str(root), "detector", DATE, "IMG_0007",
        annotations=preds, img_w=IMG_W, img_h=IMG_H,
    )
    assert second["redirected"] is True
    assert second["bucket"] == "detector@r2"
    assert second["verdict_count"] == 1
    assert Path(first["path"]).is_file()
    assert not Path(first["path"]).samefile(Path(second["path"]))
