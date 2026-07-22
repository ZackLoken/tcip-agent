"""Integration tests for tcip-web HTTP routes (Slice 1 backend)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tcip_annotation.json_io import read_detect, write_detect
from tcip_annotation.state import BBox, PredBBox
from tcip_web.app import app
from tcip_web.paths import safe_join


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


# ── per-image JSON label fixtures (canonical on-disk format) ─────────────────


def _write_gt_detect(path, boxes, *, w: int = 100, h: int = 80, keep_empty: bool = False) -> None:
    """Author a per-image JSON GT label; each box is a pixel-xyxy ``(x1, y1, x2, y2, class_id)``."""
    write_detect(str(path), [BBox(*b) for b in boxes], w, h, keep_empty=keep_empty)


def _write_pred_detect(path, preds, *, w: int = 100, h: int = 80) -> None:
    """Author a per-image JSON prediction label; each pred is ``(x1, y1, x2, y2, class_id, conf)``."""
    write_detect(
        str(path),
        [PredBBox(p[0], p[1], p[2], p[3], p[4], confidence=p[5]) for p in preds],
        w,
        h,
    )


# ── paths.safe_join ──────────────────────────────────────────────────────


class TestSafeJoin:
    def test_joins_under_root(self, tmp_path: Path) -> None:
        base = tmp_path / "project"
        base.mkdir()
        result = safe_join(base, "images", "2-11-26")
        assert result == (base / "images" / "2-11-26").resolve()

    def test_rejects_parent_traversal(self, tmp_path: Path) -> None:
        base = tmp_path / "project"
        base.mkdir()
        with pytest.raises(ValueError):
            safe_join(base, "..", "escaped")

    def test_rejects_absolute(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            safe_join(tmp_path, "/etc/passwd")

    def test_accepts_forward_slashes(self, tmp_path: Path) -> None:
        base = tmp_path
        result = safe_join(base, "images/2-11-26/IMG.jpg")
        assert result == (base / "images" / "2-11-26" / "IMG.jpg").resolve()


# ── /api/dataset ─────────────────────────────────────────────────────────


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "Valley_Farm"
    (root / "images" / "2-11-26").mkdir(parents=True)
    (root / "images" / "3-2-26").mkdir(parents=True)
    (root / "annotations" / "catkin" / "2-11-26" / "detect").mkdir(parents=True)
    (root / "annotations" / "catkin" / "2-11-26" / "segment").mkdir(parents=True)
    (root / "annotations" / "bush" / "3-2-26" / "detect").mkdir(parents=True)
    (root / "models" / "baseline").mkdir(parents=True)
    # Add some images
    for i in range(3):
        img = Image.new("RGB", (100, 80), color=(128, 128, 128))
        img.save(root / "images" / "2-11-26" / f"IMG_{i:04d}.JPG")
    return root


def test_dataset_tree(client: TestClient, dataset_root: Path) -> None:
    resp = client.get("/api/dataset/tree", params={"dataset_root": str(dataset_root)})
    assert resp.status_code == 200
    body = resp.json()
    assert "2-11-26" in body["dates_with_images"]
    assert "3-2-26" in body["dates_with_images"]
    assert sorted(body["subjects"]) == ["bush", "catkin"]
    assert "baseline" in body["model_names"]
    # Per-date maps present for every image date (empty here — the fixture makes subject
    # dirs but no label files).
    assert set(body["subjects_by_date"]) == {"2-11-26", "3-2-26"}
    assert body["subjects_by_date"]["2-11-26"] == []


def test_dataset_tree_per_date_reflects_actual_labels(client: TestClient, tmp_path: Path) -> None:
    root = tmp_path / "ds"
    (root / "images" / "2026-02-11").mkdir(parents=True)
    (root / "images" / "2026-03-24").mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(root / "images" / "2026-02-11" / "IMG_1.JPG")
    Image.new("RGB", (8, 8)).save(root / "images" / "2026-03-24" / "IMG_2.JPG")
    # catkin labelled + baseline predicted on 02-11; nothing on 03-24.
    det = root / "annotations" / "catkin" / "2026-02-11" / "detect"
    det.mkdir(parents=True)
    _write_gt_detect(det / "IMG_1.json", [(1, 1, 3, 3, 0)], w=8, h=8)
    pdet = root / "predictions" / "baseline" / "2026-02-11" / "detect"
    pdet.mkdir(parents=True)
    _write_pred_detect(pdet / "IMG_1.json", [(1, 1, 3, 3, 0, 0.9)], w=8, h=8)

    body = client.get("/api/dataset/tree", params={"dataset_root": str(root)}).json()
    assert body["subjects_by_date"]["2026-02-11"] == ["catkin"]
    assert body["subjects_by_date"]["2026-03-24"] == []
    assert body["models_by_date"]["2026-02-11"] == ["baseline"]
    assert body["models_by_date"]["2026-03-24"] == []


def test_dataset_list_images(client: TestClient, dataset_root: Path) -> None:
    resp = client.get(
        "/api/dataset/images",
        params={"dataset_root": str(dataset_root), "date": "2-11-26"},
    )
    body = resp.json()
    assert body["count"] == 3
    assert body["images"][0].startswith("IMG_")


def test_dataset_select_populates_state(client: TestClient, dataset_root: Path, tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    resp = client.post(
        "/api/dataset/select",
        json={
            "project_root": str(project),
            "dataset_root": str(dataset_root),
            "subject": "catkin",
            "date": "2-11-26",
            "model_name": "baseline",
        },
    )
    assert resp.status_code == 200
    sel = resp.json()["selection"]
    assert sel["subject"] == "catkin"
    assert sel["date"] == "2-11-26"
    assert len(sel["image_list"]) == 3
    assert sel["annotations_detect_dir"].endswith("catkin/2-11-26/detect") or sel[
        "annotations_detect_dir"
    ].endswith("catkin\\2-11-26\\detect")


def test_dataset_select_advisory_reflects_actual_labels(
    client: TestClient, dataset_root: Path, tmp_path: Path
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    body = {
        "project_root": str(project),
        "dataset_root": str(dataset_root),
        "subject": "catkin",
        "date": "2-11-26",
        "model_name": "baseline",
    }
    # The fixture makes the annotation dirs but no label files → advisory says "starts empty".
    r1 = client.post("/api/dataset/select", json=body).json()
    assert r1["annotations_present"] is False
    assert r1["predictions_present"] is False

    # Drop in a real label + prediction; the advisory flips to present (never rejects either way).
    _write_gt_detect(
        dataset_root / "annotations" / "catkin" / "2-11-26" / "detect" / "IMG_0000.json",
        [(40, 32, 60, 48, 0)],
    )
    pdet = dataset_root / "predictions" / "baseline" / "2-11-26" / "detect"
    pdet.mkdir(parents=True, exist_ok=True)
    _write_pred_detect(pdet / "IMG_0000.json", [(40, 32, 60, 48, 0, 0.9)])
    r2 = client.post("/api/dataset/select", json=body).json()
    assert r2["annotations_present"] is True
    assert r2["predictions_present"] is True


def test_dataset_nav_persists_current_index(
    client: TestClient, dataset_root: Path, tmp_path: Path
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    client.post(
        "/api/dataset/select",
        json={
            "project_root": str(project),
            "dataset_root": str(dataset_root),
            "subject": "catkin",
            "date": "2-11-26",
        },
    )
    # A valid position is accepted and shows up in the live GuiState the agent reads.
    ok = client.post("/api/dataset/nav", json={"current_image_index": 2})
    assert ok.status_code == 200
    assert ok.json()["current_image_index"] == 2
    assert client.get("/api/state").json()["dataset"]["current_image_index"] == 2
    # Out of range (3 images → valid 0..2) is rejected, not silently clamped.
    assert client.post("/api/dataset/nav", json={"current_image_index": 9}).status_code == 400


# ── /api/images ──────────────────────────────────────────────────────────


def test_images_serve(client: TestClient, dataset_root: Path) -> None:
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    resp = client.get("/api/images", params={"path": str(img_path)})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    im = Image.open(io.BytesIO(resp.content))
    assert im.size == (100, 80)


def test_images_downsample(client: TestClient, dataset_root: Path) -> None:
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    resp = client.get("/api/images", params={"path": str(img_path), "max_width": 50})
    im = Image.open(io.BytesIO(resp.content))
    assert im.size[0] == 50


def test_images_etag_revalidation(client: TestClient, dataset_root: Path) -> None:
    # First fetch carries an ETag + Cache-Control; re-requesting with If-None-Match gets a
    # cheap 304 (no re-decode/re-encode), and the ETag varies with the render params.
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    first = client.get("/api/images", params={"path": str(img_path)})
    etag = first.headers.get("etag")
    assert etag and "cache-control" in first.headers

    again = client.get(
        "/api/images", params={"path": str(img_path)}, headers={"If-None-Match": etag}
    )
    assert again.status_code == 304
    assert again.content == b""

    # A different max_width is a different variant -> different ETag -> full 200.
    variant = client.get(
        "/api/images",
        params={"path": str(img_path), "max_width": 50},
        headers={"If-None-Match": etag},
    )
    assert variant.status_code == 200
    assert variant.headers["etag"] != etag


def test_images_dimensions(client: TestClient, dataset_root: Path) -> None:
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    resp = client.get("/api/images/dimensions", params={"path": str(img_path)})
    body = resp.json()
    assert body["width"] == 100
    assert body["height"] == 80


def test_images_not_found(client: TestClient) -> None:
    resp = client.get("/api/images", params={"path": "/does/not/exist.jpg"})
    assert resp.status_code == 404


# ── /api/annotate ────────────────────────────────────────────────────────


def test_annotate_load_and_save_roundtrip(client: TestClient, dataset_root: Path, tmp_path: Path) -> None:
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    det_path = tmp_path / "detect" / "IMG_0000.json"
    seg_path = tmp_path / "segment" / "IMG_0000.json"

    # Save
    resp = client.post(
        "/api/annotate/labels",
        json={
            "image_path": str(img_path),
            "detect_path": str(det_path),
            "segment_path": str(seg_path),
            "boxes": [{"x1": 10, "y1": 20, "x2": 50, "y2": 60, "class_id": 0}],
            "polygons": [
                {
                    "points": [[5, 5], [10, 5], [10, 10], [5, 10]],
                    "class_id": 1,
                }
            ],
        },
    )
    assert resp.status_code == 200
    assert det_path.exists()
    assert seg_path.exists()

    # Load
    resp = client.get(
        "/api/annotate/labels",
        params={
            "image_path": str(img_path),
            "detect_path": str(det_path),
            "segment_path": str(seg_path),
        },
    )
    body = resp.json()
    # Detect is derived from the polygon, so the box takes the polygon's class (1),
    # not the independently-sent box's class (0). The polygon round-trips as-is.
    assert len(body["boxes"]) == 1
    assert body["boxes"][0]["class_id"] == 1
    assert len(body["polygons"]) == 1
    assert body["polygons"][0]["class_id"] == 1


def test_annotate_save_empty_preserves_negative(
    client: TestClient, dataset_root: Path, tmp_path: Path
) -> None:
    # Clearing all boxes and saving must keep a 0-byte label file, not delete it — the empty file
    # is a valid on-disk state (it becomes a confirmed negative once the image is explicitly completed).
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    det_path = tmp_path / "detect" / "IMG_0000.json"

    client.post(
        "/api/annotate/labels",
        json={
            "image_path": str(img_path),
            "detect_path": str(det_path),
            "boxes": [{"x1": 10, "y1": 20, "x2": 50, "y2": 60, "class_id": 0}],
            "polygons": [],
        },
    )
    assert det_path.exists()

    resp = client.post(
        "/api/annotate/labels",
        json={
            "image_path": str(img_path),
            "detect_path": str(det_path),
            "boxes": [],
            "polygons": [],
        },
    )
    assert resp.status_code == 200
    # A present file with no objects is a confirmed negative (kept, not deleted).
    assert det_path.exists()
    boxes, _ = read_detect(str(det_path))
    assert boxes == []


def test_annotate_save_label_path_outside_allowed_root_403(
    client: TestClient, dataset_root: Path, tmp_path: Path, monkeypatch
) -> None:
    # With an allow-list configured, a label path outside it must be rejected —
    # write_detect_labels is otherwise an arbitrary file write/delete primitive.
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    monkeypatch.setenv("TCIP_IMAGE_ROOTS", str(dataset_root.resolve()))
    outside = tmp_path / "evil" / "IMG_0000.json"
    resp = client.post(
        "/api/annotate/labels",
        json={
            "image_path": str(img_path),
            "detect_path": str(outside),
            "boxes": [],
            "polygons": [],
        },
    )
    assert resp.status_code == 403
    assert not outside.exists()


def _save_box(client: TestClient, img_path, det_path, **extra) -> dict:
    resp = client.post(
        "/api/annotate/labels",
        json={
            "image_path": str(img_path),
            "detect_path": str(det_path),
            "boxes": [{"x1": 10, "y1": 20, "x2": 50, "y2": 60, "class_id": 0}],
            "polygons": [],
            **extra,
        },
    )
    return resp


def test_annotate_save_stale_mtime_conflicts(
    client: TestClient, dataset_root: Path, tmp_path: Path
) -> None:
    import os

    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    det_path = tmp_path / "detect" / "IMG_0000.json"
    base = _save_box(client, img_path, det_path).json()["base_mtimes"]

    # A concurrent writer changes the file after our client loaded it.
    det_path.write_text('{"image": "IMG_0000", "width": 100, "height": 80, "objects": []}')
    bumped = int(base["detect"]) + 1_000_000
    os.utime(det_path, ns=(bumped, bumped))

    resp = client.post(
        "/api/annotate/labels",
        json={
            "image_path": str(img_path),
            "detect_path": str(det_path),
            "boxes": [],
            "polygons": [],
            "base_mtimes": base,
        },
    )
    assert resp.status_code == 409


def test_annotate_save_matching_mtime_ok(
    client: TestClient, dataset_root: Path, tmp_path: Path
) -> None:
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    det_path = tmp_path / "detect" / "IMG_0000.json"
    base = _save_box(client, img_path, det_path).json()["base_mtimes"]

    # No external change → the current mtime still matches → the save is accepted
    # and returns fresh version tokens.
    resp = _save_box(client, img_path, det_path, base_mtimes=base)
    assert resp.status_code == 200
    token = resp.json()["base_mtimes"]["detect"]
    assert token is not None
    # Tokens must be strings: the ns value exceeds JavaScript's 2**53 exact-integer
    # range, so a numeric token gets rounded by the browser and every save 409s.
    assert isinstance(token, str)


def test_annotate_save_derives_detect_from_polygons(client, dataset_root, tmp_path) -> None:
    # When polygons exist, detect is their bbox — a drawn box that disagrees is ignored,
    # so editing a polygon can never leave a stale box twin behind. Image is 100x80.
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    det_path = tmp_path / "detect" / "IMG_0000.json"
    seg_path = tmp_path / "segment" / "IMG_0000.json"
    resp = client.post(
        "/api/annotate/labels",
        json={
            "image_path": str(img_path),
            "detect_path": str(det_path),
            "segment_path": str(seg_path),
            # A box the client sent that disagrees with the polygon — must be discarded.
            "boxes": [{"x1": 50, "y1": 50, "x2": 60, "y2": 60, "class_id": 0}],
            "polygons": [{"points": [[10, 10], [30, 10], [30, 30], [10, 30]], "class_id": 0}],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["detect_derived"] is True
    boxes, _ = read_detect(str(det_path))
    assert len(boxes) == 1
    # Detect is the polygon bbox (10,10)-(30,30), not the drawn box.
    b = boxes[0]
    assert (b.x1, b.y1, b.x2, b.y2) == (10.0, 10.0, 30.0, 30.0)


def test_annotate_save_keeps_boxes_when_no_polygons(client, dataset_root, tmp_path) -> None:
    # No polygons → detect is authoritative, boxes written as drawn (detect-primary project).
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    det_path = tmp_path / "detect" / "IMG_0000.json"
    seg_path = tmp_path / "segment" / "IMG_0000.json"
    resp = client.post(
        "/api/annotate/labels",
        json={
            "image_path": str(img_path),
            "detect_path": str(det_path),
            "segment_path": str(seg_path),
            "boxes": [{"x1": 50, "y1": 40, "x2": 70, "y2": 60, "class_id": 0}],
            "polygons": [],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["detect_derived"] is False
    boxes, _ = read_detect(str(det_path))
    assert len(boxes) == 1
    b = boxes[0]
    assert (b.x1, b.y1, b.x2, b.y2) == (50.0, 40.0, 70.0, 60.0)  # drawn box, written as-is


def test_annotate_save_writes_audit_entry(
    client: TestClient, dataset_root: Path, tmp_path: Path
) -> None:
    proj = tmp_path / "proj"
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    det_path = tmp_path / "detect" / "IMG_0000.json"
    resp = _save_box(client, img_path, det_path, project_root=str(proj))
    assert resp.status_code == 200

    audit = proj / ".tcip" / "audit.jsonl"
    assert audit.exists()
    assert "gui_save_labels" in audit.read_text()


# ── /api/review ─────────────────────────────────────────────────────────


def test_review_matches_end_to_end(client: TestClient, dataset_root: Path, tmp_path: Path) -> None:
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    det_gt = tmp_path / "gt_detect.json"
    # Image is 100x80; one GT covering the center (pixel xyxy [40,32,60,48]).
    _write_gt_detect(det_gt, [(40, 32, 60, 48, 0)])
    pred_det = tmp_path / "pred_detect.json"
    # One prediction matching the GT (TP) + one off-center prediction (FP).
    _write_pred_detect(pred_det, [(40, 32, 60, 48, 0, 0.9), (75, 60, 85, 68, 0, 0.8)])
    project_root = tmp_path / "proj"
    project_root.mkdir()

    resp = client.post(
        "/api/review/matches",
        json={
            "project_root": str(project_root),
            "image_name": "IMG_0000.JPG",
            "image_path": str(img_path),
            "gt_detect_path": str(det_gt),
            "pred_detect_path": str(pred_det),
            "iou_threshold": 0.3,
            "conf_threshold": 0.1,
        },
    )
    body = resp.json()
    assert body["n_tp"] == 1
    assert body["n_fp"] == 1
    assert body["n_fn"] == 0
    assert body["image_status"] == "not_started"


def test_review_image_statuses_batch(client: TestClient, dataset_root: Path, tmp_path: Path) -> None:
    # A prediction file with a box (reviewable), a confirmed-negative empty file (nothing to review),
    # and a third image with no file at all — only the first should surface as a detection stem.
    pred_dir = tmp_path / "predictions" / "detect"
    pred_dir.mkdir(parents=True)
    _write_pred_detect(pred_dir / "IMG_0000.json", [(40, 32, 60, 48, 0, 0.9)])
    write_detect(str(pred_dir / "IMG_0001.json"), [], 100, 80, keep_empty=True)  # empty negative
    project_root = tmp_path / "proj"
    (project_root / ".tcip" / "state").mkdir(parents=True)

    # Give one image a review status so the engine has state to return.
    det_gt = tmp_path / "gt_detect.json"
    _write_gt_detect(det_gt, [(40, 32, 60, 48, 0)])
    client.post(
        "/api/review/mark_complete",
        json={"project_root": str(project_root), "image_name": "IMG_0000.JPG", "gt_detect_path": str(det_gt)},
    )

    resp = client.get(
        "/api/review/image_statuses",
        params={"project_root": str(project_root), "pred_dir": str(pred_dir)},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["detection_stems"] == ["IMG_0000"]  # empty + missing files excluded
    assert body["statuses"]["IMG_0000.JPG"] == "completed"


def test_review_action_persists(client: TestClient, dataset_root: Path, tmp_path: Path) -> None:
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    det_gt = tmp_path / "gt_detect.json"
    _write_gt_detect(det_gt, [(40, 32, 60, 48, 0)])
    pred_det = tmp_path / "pred_detect.json"
    _write_pred_detect(pred_det, [(40, 32, 60, 48, 0, 0.9)])
    project_root = tmp_path / "proj"
    (project_root / ".tcip" / "state").mkdir(parents=True)

    resp = client.post(
        "/api/review/action",
        json={
            "project_root": str(project_root),
            "image_name": "IMG_0000.JPG",
            "image_path": str(img_path),
            "gt_detect_path": str(det_gt),
            "pred_detect_path": str(pred_det),
            "det_type": "tp",
            "class_id": 0,
            "conf": 0.9,
            "iou": 0.95,
            "gt_type": "box",
            "gt_idx": 0,
            "pred_type": "box",
            "pred_idx": 0,
            "bbox": [40.0, 32.0, 60.0, 48.0],
            "action": "accepted",
        },
    )
    assert resp.status_code == 200
    # the image's review shard should now exist (per-image file, not one whole-state file)
    shard_path = project_root / ".tcip" / "state" / "review" / "IMG_0000.JPG.json"
    assert shard_path.exists()
    data = json.loads(shard_path.read_text(encoding="utf-8"))
    assert data["state"]["detections"][0]["action"] == "accepted"  # payload wraps {img_name, state}


def _review_action(client, img_path, det_gt, project_root, **over):
    body = {
        "project_root": str(project_root),
        "image_name": "IMG_0000.JPG",
        "image_path": str(img_path),
        "gt_detect_path": str(det_gt),
        "det_type": "tp",
        "class_id": 0,
        "gt_type": "box",
        "gt_idx": 0,
        "pred_type": "box",
        "pred_idx": 0,
        "bbox": [40.0, 32.0, 60.0, 48.0],
        "action": "accepted",
    }
    body.update(over)
    return client.post("/api/review/action", json=body)


def test_review_accept_fp_adds_prediction_to_gt(client, dataset_root, tmp_path) -> None:
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    det_gt = tmp_path / "gt_detect.json"
    _write_gt_detect(det_gt, [], keep_empty=True)  # start with a confirmed negative (empty GT)
    pred_det = tmp_path / "pred_detect.json"
    _write_pred_detect(pred_det, [(40, 32, 60, 48, 0, 0.9)])
    project_root = tmp_path / "proj"
    (project_root / ".tcip" / "state").mkdir(parents=True)

    resp = _review_action(
        client, img_path, det_gt, project_root,
        pred_detect_path=str(pred_det), det_type="fp", action="accepted",
    )
    assert resp.status_code == 200
    assert resp.json()["annotation_status"] == "partial"  # GT now has the promoted box
    boxes, _ = read_detect(str(det_gt))
    assert len(boxes) == 1 and boxes[0].class_id == 0


def test_review_reject_deletes_reviewed_gt(client, dataset_root, tmp_path) -> None:
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    det_gt = tmp_path / "gt_detect.json"
    _write_gt_detect(det_gt, [(40, 32, 60, 48, 0)])  # one GT box (a missed FN)
    project_root = tmp_path / "proj"
    (project_root / ".tcip" / "state").mkdir(parents=True)

    resp = _review_action(client, img_path, det_gt, project_root, det_type="fn", action="rejected")
    assert resp.status_code == 200
    # Emptying GT does not auto-confirm a negative (that needs an explicit Complete) — it reads as
    # needing review. The label file is kept (empty-objects negative), not deleted.
    assert resp.json()["annotation_status"] == "unannotated"
    assert det_gt.is_file()
    boxes, _ = read_detect(str(det_gt))
    assert boxes == []


def test_review_accept_tp_keeps_gt_untouched(client, dataset_root, tmp_path) -> None:
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    det_gt = tmp_path / "gt_detect.json"
    _write_gt_detect(det_gt, [(40, 32, 60, 48, 0)])
    before = det_gt.read_text()
    project_root = tmp_path / "proj"
    (project_root / ".tcip" / "state").mkdir(parents=True)

    resp = _review_action(client, img_path, det_gt, project_root, det_type="tp", action="accepted")
    assert resp.status_code == 200
    assert resp.json()["annotation_status"] is None  # GT unchanged → no status update
    assert det_gt.read_text() == before


def test_review_edit_writes_edited_box_as_gt(client, dataset_root, tmp_path) -> None:
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    det_gt = tmp_path / "gt_detect.json"
    _write_gt_detect(det_gt, [(40, 32, 60, 48, 0)])
    project_root = tmp_path / "proj"
    (project_root / ".tcip" / "state").mkdir(parents=True)

    resp = _review_action(
        client, img_path, det_gt, project_root,
        det_type="tp", action="edited", edited_box=[10.0, 10.0, 30.0, 30.0],
    )
    assert resp.status_code == 200
    assert resp.json()["annotation_status"] == "partial"
    boxes, _ = read_detect(str(det_gt))
    assert len(boxes) == 1  # replaced the matched GT, still one box
    b = boxes[0]
    assert (b.x1, b.y1, b.x2, b.y2) == (10.0, 10.0, 30.0, 30.0)  # the edited geometry


def test_review_edited_detection_stays_reviewed_after_reload(client, dataset_root, tmp_path) -> None:
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    det_gt = tmp_path / "gt_detect.json"
    _write_gt_detect(det_gt, [(40, 32, 60, 48, 0)])  # one GT box, no predictions → an FN
    project_root = tmp_path / "proj"
    (project_root / ".tcip" / "state").mkdir(parents=True)

    resp = _review_action(
        client, img_path, det_gt, project_root,
        det_type="fn", pred_type=None, pred_idx=None,
        action="edited", edited_box=[10.0, 10.0, 30.0, 30.0],
    )
    assert resp.status_code == 200

    # The next reload rebuilds the FN from the edited GT file — the verdict must still be
    # recognized, i.e. the entry is keyed to the post-edit geometry, not the pre-edit one.
    m = client.post(
        "/api/review/matches",
        json={
            "project_root": str(project_root),
            "image_name": "IMG_0000.JPG",
            "image_path": str(img_path),
            "gt_detect_path": str(det_gt),
        },
    ).json()
    assert len(m["detections"]) == 1
    assert m["detections"][0]["det_type"] == "fn"
    assert m["detections"][0]["reviewed"] is True
    assert m["detections"][0]["reviewed_action"] == "edited"


def test_review_gt_write_without_path_is_rejected(client, dataset_root, tmp_path) -> None:
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    pred_det = tmp_path / "pred_detect.json"
    _write_pred_detect(pred_det, [(40, 32, 60, 48, 0, 0.9)])
    project_root = tmp_path / "proj"
    (project_root / ".tcip" / "state").mkdir(parents=True)

    # Accepting an FP writes detect GT; with no detect path configured the route must
    # refuse loudly rather than report ok while writing nothing.
    resp = _review_action(
        client, img_path, "unused", project_root,
        gt_detect_path=None, pred_detect_path=str(pred_det), det_type="fp", action="accepted",
    )
    assert resp.status_code == 400
    assert "no detect annotations path" in resp.json()["detail"]
    # The refused verdict must not have been recorded as reviewed.
    shard_path = project_root / ".tcip" / "state" / "review" / "IMG_0000.JPG.json"
    if shard_path.exists():
        data = json.loads(shard_path.read_text(encoding="utf-8"))
        assert not data.get("state", {}).get("detections")


def test_review_action_auto_completes_and_audits(
    client: TestClient, dataset_root: Path, tmp_path: Path
) -> None:
    # A single detection on the image: reviewing it flips the image to 'completed'
    # (the only GUI path to that status) and leaves an audit-trail entry.
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    det_gt = tmp_path / "gt.json"
    _write_gt_detect(det_gt, [(40, 32, 60, 48, 0)])
    pred = tmp_path / "pred.json"
    _write_pred_detect(pred, [(40, 32, 60, 48, 0, 0.9)])  # one matching prediction -> one TP
    project_root = tmp_path / "proj"
    (project_root / ".tcip" / "state").mkdir(parents=True)

    resp = client.post(
        "/api/review/action",
        json={
            "project_root": str(project_root),
            "image_name": "IMG_0000.JPG",
            "image_path": str(img_path),
            "gt_detect_path": str(det_gt),
            "pred_detect_path": str(pred),
            "det_type": "tp", "class_id": 0, "conf": 0.9, "iou": 0.95,
            "gt_type": "box", "gt_idx": 0, "pred_type": "box", "pred_idx": 0,
            "bbox": [40.0, 32.0, 60.0, 48.0], "action": "accepted",
            "iou_threshold": 0.3, "conf_threshold": 0.1,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["image_status"] == "completed"
    assert "gui_review_action" in (project_root / ".tcip" / "audit.jsonl").read_text()


def test_review_mark_complete_and_audits(client: TestClient, tmp_path: Path) -> None:
    project_root = tmp_path / "proj"
    (project_root / ".tcip" / "state").mkdir(parents=True)

    resp = client.post(
        "/api/review/mark_complete",
        json={"project_root": str(project_root), "image_name": "IMG_9.JPG"},
    )
    assert resp.status_code == 200
    assert resp.json()["image_status"] == "completed"

    status = client.get(
        "/api/review/image_status",
        params={"project_root": str(project_root), "image_name": "IMG_9.JPG"},
    )
    assert status.json()["status"] == "completed"
    assert "gui_review_mark_complete" in (project_root / ".tcip" / "audit.jsonl").read_text()


def test_review_action_records_real_class_name_and_reviewer(
    client: TestClient, dataset_root: Path, tmp_path: Path
) -> None:
    # The engine resolves the class name from the reviewed label's own subject registry in the
    # dataset — so it records "catkin", not the "class_{id}" placeholder.
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    det_gt = dataset_root / "annotations" / "catkin" / "2-11-26" / "detect" / "IMG_0000.json"
    det_gt.parent.mkdir(parents=True, exist_ok=True)
    _write_gt_detect(det_gt, [(40, 32, 60, 48, 0)])
    pred = tmp_path / "pred.json"
    _write_pred_detect(pred, [(40, 32, 60, 48, 0, 0.9)])
    (dataset_root / "classes").mkdir(exist_ok=True)
    (dataset_root / "classes" / "catkin.json").write_text(
        json.dumps({"0": {"name": "catkin", "color": "#FF0000"}}))
    project_root = tmp_path / "proj"
    state = project_root / ".tcip" / "state"
    state.mkdir(parents=True)

    resp = client.post(
        "/api/review/action",
        json={
            "project_root": str(project_root),
            "image_name": "IMG_0000.JPG",
            "image_path": str(img_path),
            "gt_detect_path": str(det_gt),
            "pred_detect_path": str(pred),
            "det_type": "tp", "class_id": 0, "conf": 0.9, "iou": 0.95,
            "gt_type": "box", "gt_idx": 0, "pred_type": "box", "pred_idx": 0,
            "bbox": [40.0, 32.0, 60.0, 48.0], "action": "accepted",
            "iou_threshold": 0.3, "conf_threshold": 0.1,
        },
    )
    assert resp.status_code == 200
    entry = json.loads((state / "review" / "IMG_0000.JPG.json").read_text())["state"]["detections"][0]
    assert entry["class_name"] == "catkin"  # real name, not "class_0"
    assert entry["reviewed_by"]  # non-empty reviewer


def test_inference_launch_refuses_overwrite_into_verdicted_bucket(
    client: TestClient, tmp_path: Path, monkeypatch
) -> None:
    from tcip_annotation.review_engine import ReviewContext, ReviewDetection, ReviewEngine

    project_root = tmp_path / "proj"
    (project_root / ".tcip" / "state").mkdir(parents=True)
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(project_root))

    images = tmp_path / "imgs"
    images.mkdir()
    Image.new("RGB", (100, 100), (110, 110, 110)).save(images / "img.png")
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")

    out = tmp_path / "preds"
    out.mkdir()
    (out / "img.json").write_text(
        json.dumps({"image": "img", "width": 100, "height": 100, "objects": []})
    )
    engine = ReviewEngine(project_root / ".tcip" / "state")
    ctx = ReviewContext(img_name="img.png", img_width=100, img_height=100,
                        pred_boxes=[PredBBox(10.0, 10.0, 30.0, 30.0, 0, confidence=0.9)])
    det = ReviewDetection(det_type="fp", class_id=0, conf=0.9, iou=None, gt_type=None, gt_idx=None,
                          pred_type="box", pred_idx=0, bbox=(10.0, 10.0, 30.0, 30.0))
    engine.record_detection_action(det, ctx, action="accepted")

    # overwrite=True into a bucket that has a verdict is a 409 (no job is launched).
    resp = client.post("/api/inference/launch", json={
        "checkpoint_path": str(ckpt), "images_dir": str(images),
        "output_dir": str(out), "overwrite": True,
    })
    assert resp.status_code == 409
    assert "verdict" in resp.json()["detail"].lower()


# ── /api/state ───────────────────────────────────────────────────────────


def test_state_snapshot_available(client: TestClient) -> None:
    resp = client.get("/api/state")
    body = resp.json()
    # Minimal shape sanity
    assert "active_tab" in body
    assert "view" in body
    assert "dataset" in body


# ── /api/fs (folder browser) ───────────────────────────────────────────────


def test_fs_list_directories(client: TestClient, tmp_path: Path) -> None:
    (tmp_path / "alpha").mkdir()
    (tmp_path / "beta" / "images").mkdir(parents=True)  # looks like a dataset root
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "afile.txt").write_text("x")
    # Windows system/recovery folders that clutter a picker — filtered out by name.
    (tmp_path / "$RECYCLE.BIN").mkdir()
    (tmp_path / "System Volume Information").mkdir()
    (tmp_path / "FOUND.000").mkdir()

    resp = client.get("/api/fs/list", params={"path": str(tmp_path)})
    assert resp.status_code == 200
    body = resp.json()
    names = {e["name"]: e for e in body["entries"]}
    assert "alpha" in names and "beta" in names
    assert ".hidden" not in names  # hidden dirs skipped
    assert "afile.txt" not in names  # files skipped — directories only
    assert "$RECYCLE.BIN" not in names
    assert "System Volume Information" not in names
    assert "FOUND.000" not in names
    assert names["beta"]["is_dataset_root"] is True
    assert names["alpha"]["is_dataset_root"] is False
    assert body["path"] == str(tmp_path)


def test_fs_list_404_for_non_dir(client: TestClient, tmp_path: Path) -> None:
    f = tmp_path / "x.txt"
    f.write_text("x")
    assert client.get("/api/fs/list", params={"path": str(f)}).status_code == 404


def test_fs_list_confined_by_image_roots(
    client: TestClient, tmp_path: Path, monkeypatch
) -> None:
    allowed = tmp_path / "allowed"
    (allowed / "sub").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setenv("TCIP_IMAGE_ROOTS", str(allowed))

    assert client.get("/api/fs/list", params={"path": str(allowed)}).status_code == 200
    # Browsing outside the configured root is refused.
    assert client.get("/api/fs/list", params={"path": str(outside)}).status_code == 403


# ── Phase 3: native provenance (created_by / accepted_by) ─────────────────


def test_annotate_save_stamps_created_by(client, dataset_root, tmp_path) -> None:
    """A human-drawn box is stamped created_by=user:<gui-user> + created_at."""
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    det_path = tmp_path / "detect" / "IMG_0000.json"
    resp = client.post("/api/annotate/labels", json={
        "image_path": str(img_path), "detect_path": str(det_path),
        "boxes": [{"x1": 50, "y1": 40, "x2": 70, "y2": 60, "class_id": 0}], "polygons": [],
        "user": "zack",
    })
    assert resp.status_code == 200
    obj = json.loads(det_path.read_text())["objects"][0]
    assert obj["created_by"] == "user:zack"
    assert obj["created_at"]


def test_annotate_derived_box_inherits_author(client, dataset_root, tmp_path) -> None:
    """Detect boxes derived from a drawn polygon inherit the polygon's author (not None)."""
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    det_path = tmp_path / "detect" / "IMG_0000.json"
    seg_path = tmp_path / "segment" / "IMG_0000.json"
    resp = client.post("/api/annotate/labels", json={
        "image_path": str(img_path), "detect_path": str(det_path), "segment_path": str(seg_path),
        "boxes": [], "polygons": [{"points": [[10, 10], [30, 10], [30, 30]], "class_id": 0}],
        "user": "emily",
    })
    assert resp.status_code == 200
    assert json.loads(det_path.read_text())["objects"][0]["created_by"] == "user:emily"  # derived
    assert json.loads(seg_path.read_text())["objects"][0]["created_by"] == "user:emily"


def test_annotate_save_falls_back_to_os_user(client, dataset_root, tmp_path) -> None:
    """Omitting user still stamps a user:<...> author (backend OS/env fallback), never bare/None."""
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    det_path = tmp_path / "detect" / "IMG_0000.json"
    resp = client.post("/api/annotate/labels", json={
        "image_path": str(img_path), "detect_path": str(det_path),
        "boxes": [{"x1": 50, "y1": 40, "x2": 70, "y2": 60, "class_id": 0}], "polygons": [],
    })
    assert resp.status_code == 200
    obj = json.loads(det_path.read_text())["objects"][0]
    assert obj["created_by"].startswith("user:")


def test_review_accept_fp_carries_created_by_and_stamps_accepted_by(client, dataset_root, tmp_path) -> None:
    """Accepting an FP prediction into GT carries the prediction's created_by and stamps
    accepted_by=reviewer — the origin travels, and the acceptance is recorded."""
    from tcip_annotation.json_io import write_detect
    from tcip_annotation.state import PredBBox

    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    det_gt = tmp_path / "gt_detect.json"
    _write_gt_detect(det_gt, [], keep_empty=True)  # confirmed negative → the pred shows as FP
    pred_det = tmp_path / "pred_detect.json"
    write_detect(str(pred_det), [PredBBox(
        40, 32, 60, 48, 0, confidence=0.9, created_by="sam", created_at="2026-01-01T00:00:00+00:00")],
        100, 80)
    project_root = tmp_path / "proj"
    (project_root / ".tcip" / "state").mkdir(parents=True)

    resp = _review_action(
        client, img_path, det_gt, project_root,
        pred_detect_path=str(pred_det), det_type="fp", action="accepted", user="zack",
    )
    assert resp.status_code == 200
    obj = json.loads(det_gt.read_text())["objects"][0]
    assert obj["created_by"] == "sam"          # prediction origin carried into GT
    assert obj["accepted_by"] == "user:zack"   # reviewer stamped
    assert obj["accepted_at"]


def test_review_edit_stamps_created_by(client, dataset_root, tmp_path) -> None:
    """A reviewer-drawn edit authors GT with created_by=user:<reviewer>."""
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    det_gt = tmp_path / "gt_detect.json"
    _write_gt_detect(det_gt, [(40, 32, 60, 48, 0)])
    project_root = tmp_path / "proj"
    (project_root / ".tcip" / "state").mkdir(parents=True)

    resp = _review_action(
        client, img_path, det_gt, project_root,
        det_type="tp", action="edited", edited_box=[10.0, 10.0, 30.0, 30.0], user="zack",
    )
    assert resp.status_code == 200
    obj = json.loads(det_gt.read_text())["objects"][0]
    assert obj["created_by"] == "user:zack"


# ── Provenance round-trip fidelity (load → edit → save keeps the original creator) ──


def test_annotate_load_returns_provenance(client, dataset_root, tmp_path) -> None:
    from tcip_annotation.json_io import write_detect
    from tcip_annotation.state import BBox

    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    det = tmp_path / "det.json"
    write_detect(str(det), [BBox(10, 10, 40, 40, 0, created_by="derived:user:zack",
                                 created_at="2026-02-11T00:00:00+00:00",
                                 accepted_by="user:zack")], 100, 80)
    resp = client.get(
        "/api/annotate/labels",
        params={"image_path": str(img_path), "detect_path": str(det)},
    )
    assert resp.status_code == 200
    b = resp.json()["boxes"][0]
    assert b["created_by"] == "derived:user:zack"
    assert b["created_at"] == "2026-02-11T00:00:00+00:00"
    assert b["accepted_by"] == "user:zack"


def test_annotate_resave_preserves_original_creator(client, dataset_root, tmp_path) -> None:
    """A re-save must not wholesale re-stamp loaded shapes to the current annotator — the
    original creator survives (keep-original-creator policy); only NEW shapes get stamped."""
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    det_path = tmp_path / "detect" / "IMG_0000.json"
    resp = client.post("/api/annotate/labels", json={
        "image_path": str(img_path), "detect_path": str(det_path),
        "boxes": [
            {"x1": 10, "y1": 10, "x2": 40, "y2": 40, "class_id": 0,
             "created_by": "derived:user:zack", "created_at": "2026-02-11T00:00:00+00:00",
             "accepted_by": "user:zack"},
            {"x1": 50, "y1": 50, "x2": 70, "y2": 70, "class_id": 0},
        ],
        "polygons": [],
        "user": "emily",
    })
    assert resp.status_code == 200
    objs = json.loads(det_path.read_text())["objects"]
    assert objs[0]["created_by"] == "derived:user:zack"          # original creator kept
    assert objs[0]["created_at"] == "2026-02-11T00:00:00+00:00"  # original timestamp kept
    assert objs[0]["accepted_by"] == "user:zack"                 # acceptance carried
    assert objs[1]["created_by"] == "user:emily"                 # only the new shape is Emily's


def test_annotate_derived_boxes_inherit_polygon_provenance(client, dataset_root, tmp_path) -> None:
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    det_path = tmp_path / "detect" / "IMG_0000.json"
    seg_path = tmp_path / "segment" / "IMG_0000.json"
    resp = client.post("/api/annotate/labels", json={
        "image_path": str(img_path), "detect_path": str(det_path), "segment_path": str(seg_path),
        "boxes": [],
        "polygons": [
            {"points": [[10, 10], [30, 10], [30, 30]], "class_id": 0,
             "created_by": "user:emily", "created_at": "2026-03-02T00:00:00+00:00"},
            {"points": [[50, 50], [70, 50], [70, 70]], "class_id": 0},
        ],
        "user": "zack",
    })
    assert resp.status_code == 200
    det_objs = json.loads(det_path.read_text())["objects"]
    assert det_objs[0]["created_by"] == "user:emily"   # derived box keeps its polygon's author
    assert det_objs[1]["created_by"] == "user:zack"    # new polygon -> stamped -> box inherits


def test_review_class_names_do_not_bleed_across_subjects(
    client: TestClient, dataset_root: Path, tmp_path: Path
) -> None:
    """Class names are resolved per request from the reviewed label's own subject registry, so
    class 0 shows 'catkin' when reviewing catkin and 'bud_mite' when reviewing efb — never one
    subject's name for the other's labels (the bug a project-cached engine would have)."""
    import json

    (dataset_root / "classes").mkdir(exist_ok=True)
    (dataset_root / "classes" / "catkin.json").write_text(
        json.dumps({"0": {"name": "catkin", "color": "#FF0000"}}))
    (dataset_root / "classes" / "efb.json").write_text(
        json.dumps({"0": {"name": "efb_canker", "color": "#00FF00"}}))
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    project_root = tmp_path / "proj"
    (project_root / ".tcip" / "state").mkdir(parents=True)

    def _class_name_for(subject: str) -> str:
        gt = dataset_root / "annotations" / subject / "2-11-26" / "detect" / "IMG_0000.json"
        gt.parent.mkdir(parents=True, exist_ok=True)
        _write_gt_detect(gt, [(40, 32, 60, 48, 0)])
        pred = tmp_path / f"pred_{subject}.json"
        _write_pred_detect(pred, [(40, 32, 60, 48, 0, 0.9)])
        resp = client.post(
            "/api/review/action",
            json={
                "project_root": str(project_root), "image_name": "IMG_0000.JPG",
                "image_path": str(img_path), "gt_detect_path": str(gt),
                "pred_detect_path": str(pred), "det_type": "tp", "class_id": 0, "conf": 0.9,
                "iou": 0.95, "gt_type": "box", "gt_idx": 0, "pred_type": "box", "pred_idx": 0,
                "bbox": [40.0, 32.0, 60.0, 48.0], "action": "accepted",
                "iou_threshold": 0.3, "conf_threshold": 0.1,
            },
        )
        assert resp.status_code == 200
        shard = json.loads((project_root / ".tcip" / "state" / "review" / "IMG_0000.JPG.json").read_text())
        return shard["state"]["detections"][0]["class_name"]

    assert _class_name_for("catkin") == "catkin"
    assert _class_name_for("efb") == "efb_canker"  # same class 0, different subject, different name
