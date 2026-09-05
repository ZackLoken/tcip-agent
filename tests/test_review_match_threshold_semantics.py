"""The review routes' two operating-point knobs, and the population each reported count covers.

``iou_threshold`` decides whether a prediction overlaps a ground-truth object enough to be the
same object; ``conf_threshold`` decides whether a prediction is confident enough to be shown at
all. They address different axes and are never substitutable, so the fixtures here are built so
that reading one as the other changes the TP/FP/FN split rather than leaving it alone.

``n_tp``/``n_fp``/``n_fn`` count the whole match set for the image, which is also the population
the auto-complete check walks; ``detections`` is the filtered walk list. A filter narrows the
second without narrowing the first.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tcip_annotation.json_io import write_annotations
from tcip_annotation.state import Annotation, BBox
from tcip_web.app import app

IMG_W, IMG_H = 160, 100

# Three ground-truth objects with distinct widths and heights, on a non-square frame.
GT_BOXES = [
    (10.0, 10.0, 50.0, 30.0),    # 40 wide, 20 tall
    (100.0, 40.0, 130.0, 76.0),  # 30 wide, 36 tall
    (5.0, 60.0, 25.0, 88.0),     # 20 wide, 28 tall
]
# One low-confidence prediction at a partial overlap, one confident exact hit, one confident
# partial overlap. Overlaps and scores straddle the thresholds the tests request.
PRED_BOXES = [
    (10.0, 10.0, 70.0, 30.0, 0.5),    # intersection over union with the first GT is 2/3
    (100.0, 40.0, 130.0, 76.0, 0.9),  # exact hit on the second GT
    (5.0, 60.0, 45.0, 88.0, 0.9),     # intersection over union with the third GT is 1/2
]
IOU_THRESHOLD = 0.7
CONF_THRESHOLD = 0.2


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


def _image(tmp_path: Path) -> Path:
    d = tmp_path / "images"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "IMG_0000.JPG"
    Image.new("RGB", (IMG_W, IMG_H), color=(120, 130, 140)).save(p)
    return p


def _write_gt(path: Path, boxes) -> None:
    write_annotations(str(path), [Annotation(subject="bud", geometry=BBox(*b)) for b in boxes],
                      IMG_W, IMG_H, keep_empty=True)


def _write_pred(path: Path, preds) -> None:
    write_annotations(
        str(path),
        [Annotation(subject="bud", geometry=BBox(p[0], p[1], p[2], p[3]), score=p[4])
         for p in preds],
        IMG_W, IMG_H,
    )


def _scene(tmp_path: Path):
    """The three-object scene, its prediction file, and an empty project root."""
    img = _image(tmp_path)
    gt = tmp_path / "gt.json"
    pred = tmp_path / "pred.json"
    _write_gt(gt, GT_BOXES)
    _write_pred(pred, PRED_BOXES)
    dataset_root = tmp_path / "proj"
    (dataset_root / ".tcip" / "state").mkdir(parents=True)
    return img, gt, pred, dataset_root


def test_iou_and_confidence_thresholds_are_not_interchangeable(
    client: TestClient, tmp_path: Path
) -> None:
    """At an overlap floor of 0.7 and a confidence floor of 0.2, only the exact hit is a true
    positive: the two partial overlaps stay false positives and their objects stay missed. Reading
    the confidence floor as the overlap floor would admit both partial overlaps as hits instead,
    and reading the overlap floor as the confidence floor would hide the 0.5-scored prediction.
    """
    img, gt, pred, dataset_root = _scene(tmp_path)

    body = client.post("/api/review/matches", json={
        "dataset_root": str(dataset_root),
        "image_name": "IMG_0000.JPG",
        "image_path": str(img),
        "gt_path": str(gt),
        "pred_path": str(pred),
        "iou_threshold": IOU_THRESHOLD,
        "conf_threshold": CONF_THRESHOLD,
    }).json()

    assert body["n_tp"] == 1
    assert body["n_fp"] == 2
    assert body["n_fn"] == 2
    assert len(body["detections"]) == 5


def test_recorded_verdict_returns_matches_at_the_requested_operating_point(
    client: TestClient, tmp_path: Path
) -> None:
    """The fresh match set a verdict returns is scoped to the thresholds the verdict was recorded
    at, so the canvas installs it without a second fetch and without silently changing scope."""
    img, gt, pred, dataset_root = _scene(tmp_path)

    resp = client.post("/api/review/action", json={
        "dataset_root": str(dataset_root),
        "image_name": "IMG_0000.JPG",
        "image_path": str(img),
        "gt_path": str(gt),
        "pred_path": str(pred),
        "det_type": "tp", "class_name": "bud", "conf": 0.9, "iou": 1.0,
        "gt_idx": 1, "pred_idx": 1,
        "bbox": [100.0, 40.0, 130.0, 76.0],
        "action": "accepted",
        "iou_threshold": IOU_THRESHOLD,
        "conf_threshold": CONF_THRESHOLD,
    })
    assert resp.status_code == 200
    fresh = resp.json()["matches"]
    assert fresh["n_tp"] == 1
    assert fresh["n_fp"] == 2
    assert fresh["n_fn"] == 2


def test_completion_check_counts_detections_at_the_requested_operating_point(
    client: TestClient, tmp_path: Path
) -> None:
    """One partial-overlap prediction below the overlap floor leaves two things to adjudicate, the
    unmatched prediction and the missed object, so a single verdict must not finish the image.
    Applying the confidence floor as the overlap floor would drop the prediction entirely and let
    that one verdict flip the image to completed."""
    img = _image(tmp_path)
    gt = tmp_path / "gt.json"
    pred = tmp_path / "pred.json"
    _write_gt(gt, [GT_BOXES[0]])
    _write_pred(pred, [PRED_BOXES[0]])
    dataset_root = tmp_path / "proj"
    (dataset_root / ".tcip" / "state").mkdir(parents=True)

    resp = client.post("/api/review/action", json={
        "dataset_root": str(dataset_root),
        "image_name": "IMG_0000.JPG",
        "image_path": str(img),
        "gt_path": str(gt),
        "pred_path": str(pred),
        "det_type": "fn", "class_name": "bud", "conf": None, "iou": None,
        "gt_idx": 0, "pred_idx": None,
        "bbox": list(GT_BOXES[0]),
        "action": "accepted",
        "iou_threshold": IOU_THRESHOLD,
        "conf_threshold": CONF_THRESHOLD,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["matches"]["n_fp"] == 1
    assert body["matches"]["n_fn"] == 1
    assert body["image_status"] == "started"


def test_detection_filter_narrows_the_walk_list_without_rescoping_the_counts(
    client: TestClient, tmp_path: Path
) -> None:
    """A type filter is a navigation aid over the walk list. The reported counts stay the whole
    image's match set, the same population the completion check reads, so a filtered session never
    reports a smaller image than the platform considers unfinished."""
    img, gt, pred, dataset_root = _scene(tmp_path)

    body = client.post("/api/review/matches", json={
        "dataset_root": str(dataset_root),
        "image_name": "IMG_0000.JPG",
        "image_path": str(img),
        "gt_path": str(gt),
        "pred_path": str(pred),
        "iou_threshold": IOU_THRESHOLD,
        "conf_threshold": CONF_THRESHOLD,
        "filter_type": "fn",
    }).json()

    dets = body["detections"]
    assert len(dets) == 2
    assert {d["det_type"] for d in dets} == {"fn"}
    assert body["n_tp"] == 1
    assert body["n_fp"] == 2
    assert body["n_fn"] == 2
