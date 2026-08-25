"""One rule for a reviewed detection, shared by the ticks, the wheel and the completion gate.

Three predictions, three different confidences, no ground truth (every one is a false positive).
Reviewing at a low confidence threshold sees all three; raising the threshold afterward drops the
low-confidence ones out of the current match set, but the stored verdict entries for them do not
disappear. The completion gate must read "every detection still in the current set has an entry",
never "at least this many entries exist somewhere in the shard": the latter lets a raised threshold
complete an image whose one remaining current detection nobody ever reviewed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tcip_annotation.json_io import write_annotations
from tcip_web.app import app

IMG_W, IMG_H = 200, 100

# No ground truth at all: every prediction below is an unmatched false positive, so raising the
# confidence threshold only ever removes entries from the current set, never re-scores them.
PRED_BOXES = [
    (10.0, 10.0, 30.0, 30.0, 0.9),   # the high-confidence one, deliberately left unreviewed
    (60.0, 10.0, 80.0, 30.0, 0.3),   # reviewed at the low threshold
    (110.0, 10.0, 130.0, 30.0, 0.2),  # reviewed at the low threshold
]
LOW_CONF = 0.1
HIGH_CONF = 0.5


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


def _scene(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    images = tmp_path / "images"
    images.mkdir(parents=True, exist_ok=True)
    img = images / "IMG_0000.JPG"
    Image.new("RGB", (IMG_W, IMG_H), color=(120, 130, 140)).save(img)
    gt = tmp_path / "gt.json"
    write_annotations(str(gt), [], IMG_W, IMG_H, keep_empty=True)
    pred = tmp_path / "pred.json"
    from tcip_annotation.state import Annotation, BBox
    write_annotations(
        str(pred),
        [Annotation(subject="chestnut_bud", geometry=BBox(p[0], p[1], p[2], p[3]), score=p[4])
         for p in PRED_BOXES],
        IMG_W, IMG_H,
    )
    dataset_root = tmp_path / "proj"
    (dataset_root / ".tcip" / "state").mkdir(parents=True)
    return img, gt, pred, dataset_root


def _matches(client: TestClient, img: Path, gt: Path, pred: Path, dataset_root: Path,
            conf_threshold: float) -> dict:
    return client.post("/api/review/matches", json={
        "dataset_root": str(dataset_root),
        "image_name": "IMG_0000.JPG",
        "image_path": str(img),
        "gt_path": str(gt),
        "pred_path": str(pred),
        "iou_threshold": 0.5,
        "conf_threshold": conf_threshold,
    }).json()


def _accept(client: TestClient, img: Path, gt: Path, pred: Path, dataset_root: Path,
           det: dict, conf_threshold: float) -> dict:
    return client.post("/api/review/action", json={
        "dataset_root": str(dataset_root),
        "image_name": "IMG_0000.JPG",
        "image_path": str(img),
        "gt_path": str(gt),
        "pred_path": str(pred),
        "det_type": det["det_type"], "class_name": det["class_name"],
        "conf": det["conf"], "iou": det["iou"],
        "gt_idx": det["gt_idx"], "pred_idx": det["pred_idx"],
        "bbox": list(det["bbox"]),
        "action": "rejected",
        "iou_threshold": 0.5,
        "conf_threshold": conf_threshold,
    }).json()


def test_raising_the_threshold_does_not_complete_an_image_with_one_unreviewed_current_detection(
    client: TestClient, tmp_path: Path,
) -> None:
    img, gt, pred, dataset_root = _scene(tmp_path)

    at_low = _matches(client, img, gt, pred, dataset_root, LOW_CONF)
    assert at_low["n_total"] == 3 and at_low["n_reviewed"] == 0
    low_conf_dets = [d for d in at_low["detections"] if d["conf"] < HIGH_CONF]
    assert len(low_conf_dets) == 2

    for det in low_conf_dets:
        resp = _accept(client, img, gt, pred, dataset_root, det, LOW_CONF)
    assert resp["image_status"] == "started"  # the high-confidence one is still unreviewed

    at_high = _matches(client, img, gt, pred, dataset_root, HIGH_CONF)
    assert at_high["n_total"] == 1, at_high["detections"]  # only the high-conf prediction remains
    assert at_high["n_reviewed"] == 0  # it was never itself reviewed
    assert at_high["image_status"] == "started"  # not completed by the two now-invisible entries

    # Reviewing the one remaining current detection completes the image at this threshold.
    remaining = at_high["detections"][0]
    final = _accept(client, img, gt, pred, dataset_root, remaining, HIGH_CONF)
    assert final["matches"]["n_reviewed"] == 1 and final["matches"]["n_total"] == 1
    assert final["image_status"] == "completed"


def test_the_wheels_two_numbers_reflect_the_whole_image_under_a_filter(
    client: TestClient, tmp_path: Path,
) -> None:
    img, gt, pred, dataset_root = _scene(tmp_path)

    unfiltered = _matches(client, img, gt, pred, dataset_root, LOW_CONF)
    filtered = client.post("/api/review/matches", json={
        "dataset_root": str(dataset_root),
        "image_name": "IMG_0000.JPG",
        "image_path": str(img),
        "gt_path": str(gt),
        "pred_path": str(pred),
        "iou_threshold": 0.5,
        "conf_threshold": LOW_CONF,
        "filter_type": "tp",  # matches nothing: every detection here is a false positive
    }).json()

    assert filtered["detections"] == []
    assert filtered["n_total"] == unfiltered["n_total"] == 3
    assert filtered["n_reviewed"] == unfiltered["n_reviewed"] == 0
