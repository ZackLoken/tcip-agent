"""What the review routes write when a verdict authors ground truth.

Two standing properties. A confirmed detection becomes ground truth, and the on-disk schema
separates ground truth from predictions by the presence of ``score``, so a promoted annotation
carries none. And a label file is baselined into its ``.original`` sibling directory before the
first verdict rewrites it, so the pristine copy survives and stays out of the label census.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
import tcip_store
from fastapi.testclient import TestClient
from PIL import Image
from tcip_store import StoreBusy
from tcip_store.file_backend import FileBackend, path_lock

from tcip_annotation.json_io import read_annotations, write_annotations
from tcip_annotation.review_engine import label_baseline_key
from tcip_annotation.state import Annotation, BBox
from tcip_web.app import app
from tcip_web.routes import review

IMG_W, IMG_H = 160, 100
PREDICTED_BOX = (12.0, 20.0, 52.0, 44.0)  # 40 wide, 24 tall
PREDICTED_SCORE = 0.83
EDITED_BOX = (60.0, 8.0, 90.0, 66.0)      # 30 wide, 58 tall


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


def _image(tmp_path: Path) -> Path:
    d = tmp_path / "images"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "IMG_0000.JPG"
    Image.new("RGB", (IMG_W, IMG_H), color=(90, 110, 130)).save(p)
    return p


def _dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / ".tcip" / "state").mkdir(parents=True)
    return root


def test_accepted_false_positive_becomes_ground_truth_without_a_model_score(
    client: TestClient, tmp_path: Path
) -> None:
    """Accepting a false positive promotes the predicted shape into the label file as the human's
    own confirmed object. It keeps the geometry and loses the model score, because a score-bearing
    record reads as a prediction to every later reader of this schema."""
    img = _image(tmp_path)
    gt = tmp_path / "gt.json"
    pred = tmp_path / "pred.json"
    write_annotations(str(gt), [], IMG_W, IMG_H, keep_empty=True)
    write_annotations(
        str(pred),
        [Annotation(subject="bud", geometry=BBox(*PREDICTED_BOX), score=PREDICTED_SCORE)],
        IMG_W, IMG_H,
    )
    dataset_root = _dataset_root(tmp_path)

    resp = client.post("/api/review/action", json={
        "dataset_root": str(dataset_root),
        "image_name": "IMG_0000.JPG",
        "image_path": str(img),
        "gt_path": str(gt),
        "pred_path": str(pred),
        "det_type": "fp", "class_name": "bud",
        "conf": PREDICTED_SCORE, "iou": None,
        "gt_idx": None, "pred_idx": 0,
        "bbox": list(PREDICTED_BOX),
        "action": "accepted",
    })
    assert resp.status_code == 200

    (written,) = read_annotations(str(gt))
    assert written.score is None
    assert (written.geometry.x1, written.geometry.y1,
            written.geometry.x2, written.geometry.y2) == PREDICTED_BOX

    (record,) = json.loads(gt.read_text(encoding="utf-8"))["annotations"]
    assert "score" not in record


def test_editing_an_fp_with_a_stray_gt_idx_appends_rather_than_overwrites(
    client: TestClient, tmp_path: Path
) -> None:
    """Editing an ``fp`` is only ever a fresh record: a ``gt_idx`` carried on this det_type is not
    this branch's paired-fp case (that exists only under a classified scope), so it must not be
    read as naming an existing record to overwrite. The pre-existing GT record at that index stays
    exactly as it was, and the edit lands as a new record instead."""
    img = _image(tmp_path)
    gt = tmp_path / "gt.json"
    pred = tmp_path / "pred.json"
    original = (10.0, 10.0, 50.0, 30.0)
    write_annotations(str(gt), [Annotation(subject="bud", geometry=BBox(*original))],
                      IMG_W, IMG_H)
    write_annotations(
        str(pred),
        [Annotation(subject="bud", geometry=BBox(*PREDICTED_BOX), score=PREDICTED_SCORE)],
        IMG_W, IMG_H,
    )
    dataset_root = _dataset_root(tmp_path)

    resp = client.post("/api/review/action", json={
        "dataset_root": str(dataset_root),
        "image_name": "IMG_0000.JPG",
        "image_path": str(img),
        "gt_path": str(gt),
        "pred_path": str(pred),
        "det_type": "fp", "class_name": "bud",
        "conf": PREDICTED_SCORE, "iou": None,
        "gt_idx": 0, "pred_idx": 0,
        "bbox": list(PREDICTED_BOX),
        "action": "edited",
        "edited_box": list(EDITED_BOX),
    })
    assert resp.status_code == 200, resp.text

    written = read_annotations(str(gt))
    assert len(written) == 2
    assert (written[0].geometry.x1, written[0].geometry.y1,
            written[0].geometry.x2, written[0].geometry.y2) == original
    assert (written[1].geometry.x1, written[1].geometry.y1,
            written[1].geometry.x2, written[1].geometry.y2) == EDITED_BOX
    assert written[1].score is None


def test_first_verdict_baselines_the_label_file_under_its_original_directory(
    client: TestClient, tmp_path: Path
) -> None:
    """Before a verdict rewrites a label file, its pristine content is copied to
    ``<label dir>/.original/<name>``. The baseline holds the pre-edit geometry, and it lands in
    that hidden subdirectory rather than beside the label, where it would be counted as another
    label file of the dataset."""
    img = _image(tmp_path)
    label_dir = tmp_path / "annotations" / "2-11-26"
    label_dir.mkdir(parents=True)
    gt = label_dir / "IMG_0000.json"
    write_annotations(str(gt), [Annotation(subject="bud", geometry=BBox(10.0, 10.0, 50.0, 30.0))],
                      IMG_W, IMG_H)
    dataset_root = _dataset_root(tmp_path)

    resp = client.post("/api/review/action", json={
        "dataset_root": str(dataset_root),
        "image_name": "IMG_0000.JPG",
        "image_path": str(img),
        "gt_path": str(gt),
        "det_type": "fn", "class_name": "bud",
        "conf": None, "iou": None,
        "gt_idx": 0, "pred_idx": None,
        "bbox": [10.0, 10.0, 50.0, 30.0],
        "action": "edited",
        "edited_box": list(EDITED_BOX),
    })
    assert resp.status_code == 200

    baseline = label_dir / ".original" / "IMG_0000.json"
    assert baseline.is_file()
    (kept,) = read_annotations(str(baseline))
    assert (kept.geometry.x1, kept.geometry.y1,
            kept.geometry.x2, kept.geometry.y2) == (10.0, 10.0, 50.0, 30.0)

    (live,) = read_annotations(str(gt))
    assert (live.geometry.x1, live.geometry.y1,
            live.geometry.x2, live.geometry.y2) == EDITED_BOX

    assert sorted(p.name for p in label_dir.glob("*.json")) == ["IMG_0000.json"]


def _verdict(client: TestClient, dataset_root: Path, img: Path, gt: Path,
             box: tuple[float, float, float, float],
             edited: tuple[float, float, float, float]) -> None:
    resp = client.post("/api/review/action", json={
        "dataset_root": str(dataset_root),
        "image_name": "IMG_0000.JPG",
        "image_path": str(img),
        "gt_path": str(gt),
        "det_type": "fn", "class_name": "bud",
        "conf": None, "iou": None,
        "gt_idx": 0, "pred_idx": None,
        "bbox": list(box),
        "action": "edited",
        "edited_box": list(edited),
    })
    assert resp.status_code == 200, resp.text


def test_a_second_verdict_keeps_the_baseline_the_first_one_captured(
    client: TestClient, tmp_path: Path
) -> None:
    """The baseline is captured once and then left alone.

    A second verdict rewrites the label file again, and the pristine copy must still hold the
    content the file had before any verdict touched it, not the state the previous edit left
    behind. This is the ordinary reviewing path, so the capture has to keep succeeding on it.
    """
    img = _image(tmp_path)
    label_dir = tmp_path / "annotations" / "2-11-26"
    label_dir.mkdir(parents=True)
    gt = label_dir / "IMG_0000.json"
    original = (10.0, 10.0, 50.0, 30.0)
    write_annotations(str(gt), [Annotation(subject="bud", geometry=BBox(*original))],
                      IMG_W, IMG_H)
    dataset_root = _dataset_root(tmp_path)

    _verdict(client, dataset_root, img, gt, original, EDITED_BOX)
    second = (5.0, 5.0, 25.0, 35.0)
    _verdict(client, dataset_root, img, gt, EDITED_BOX, second)

    baseline = label_dir / ".original" / "IMG_0000.json"
    (kept,) = read_annotations(str(baseline))
    assert (kept.geometry.x1, kept.geometry.y1,
            kept.geometry.x2, kept.geometry.y2) == original

    (live,) = read_annotations(str(gt))
    assert (live.geometry.x1, live.geometry.y1,
            live.geometry.x2, live.geometry.y2) == second

    # Lock files outlive writes on POSIX; only the documents are the store's contents.
    assert sorted(p.name for p in (label_dir / ".original").glob("*.json")) == ["IMG_0000.json"]
    assert not list((label_dir / ".original").glob("*.tmp"))


def test_the_baseline_capture_waits_on_the_lock_its_record_is_written_under(
    tmp_path: Path
) -> None:
    """The capture takes the baseline record's own lock, and reports the contention instead of
    writing past it.

    A copy that ignores the lock can land on top of bytes another writer is part way through
    replacing, which is the one thing a pristine copy must never be.

    Bound to the file backend on purpose: the contention is staged by holding the baseline's
    own path lock, and ``path_for`` names that path, both of which only the file backend has.
    """
    label_dir = tmp_path / "annotations" / "2-11-26"
    label_dir.mkdir(parents=True)
    gt = label_dir / "IMG_0000.json"
    write_annotations(str(gt), [Annotation(subject="bud", geometry=BBox(10.0, 10.0, 50.0, 30.0))],
                      IMG_W, IMG_H)

    backend = FileBackend(lock_timeout_s=0.2)
    tcip_store.bind(backend)
    baseline = backend.path_for(label_baseline_key(label_dir, gt.stem))
    baseline.parent.mkdir(parents=True, exist_ok=True)

    holding, release = threading.Event(), threading.Event()

    def hold() -> None:
        with path_lock(baseline, timeout_s=30):
            holding.set()
            release.wait(30)

    holder = threading.Thread(target=hold)
    holder.start()
    try:
        assert holding.wait(30)
        with pytest.raises(StoreBusy):
            review._ensure_original_backup(str(gt))
        assert not baseline.exists()
    finally:
        release.set()
        holder.join(30)
