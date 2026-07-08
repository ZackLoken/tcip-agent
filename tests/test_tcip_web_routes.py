"""Integration tests for tcip-web HTTP routes (Slice 1 backend)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tcip_web.app import app
from tcip_web.paths import safe_join


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


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
    assert sorted(body["annotation_types"]) == ["bush", "catkin"]
    assert "baseline" in body["model_names"]


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
            "annotation_type": "catkin",
            "date": "2-11-26",
            "model_name": "baseline",
        },
    )
    assert resp.status_code == 200
    sel = resp.json()["selection"]
    assert sel["annotation_type"] == "catkin"
    assert sel["date"] == "2-11-26"
    assert len(sel["image_list"]) == 3
    assert sel["annotations_detect_dir"].endswith("catkin/2-11-26/detect") or sel[
        "annotations_detect_dir"
    ].endswith("catkin\\2-11-26\\detect")


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
    det_path = tmp_path / "detect" / "IMG_0000.txt"
    seg_path = tmp_path / "segment" / "IMG_0000.txt"

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
    assert len(body["boxes"]) == 1
    assert body["boxes"][0]["class_id"] == 0
    assert len(body["polygons"]) == 1
    assert body["polygons"][0]["class_id"] == 1


def test_annotate_save_empty_preserves_negative(
    client: TestClient, dataset_root: Path, tmp_path: Path
) -> None:
    # Clearing all boxes and saving must keep a 0-byte label file (a confirmed
    # negative), not delete it — empty label files are valid negatives.
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    det_path = tmp_path / "detect" / "IMG_0000.txt"

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
    assert det_path.exists()
    assert det_path.read_text() == ""


def test_annotate_save_label_path_outside_allowed_root_403(
    client: TestClient, dataset_root: Path, tmp_path: Path, monkeypatch
) -> None:
    # With an allow-list configured, a label path outside it must be rejected —
    # write_detect_labels is otherwise an arbitrary file write/delete primitive.
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    monkeypatch.setenv("TCIP_IMAGE_ROOTS", str(dataset_root.resolve()))
    outside = tmp_path / "evil" / "IMG_0000.txt"
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


# ── /api/review ─────────────────────────────────────────────────────────


def test_review_matches_end_to_end(client: TestClient, dataset_root: Path, tmp_path: Path) -> None:
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    det_gt = tmp_path / "gt_detect.txt"
    # Image is 100x80; write one GT covering the center
    det_gt.write_text("0 0.5 0.5 0.2 0.2\n")
    pred_det = tmp_path / "pred_detect.txt"
    pred_det.write_text("0 0.9 0.5 0.5 0.2 0.2\n")  # match
    pred_det.write_text("0 0.9 0.5 0.5 0.2 0.2\n0 0.8 0.8 0.8 0.1 0.1\n")  # add FP
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


def test_review_action_persists(client: TestClient, dataset_root: Path, tmp_path: Path) -> None:
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    det_gt = tmp_path / "gt_detect.txt"
    det_gt.write_text("0 0.5 0.5 0.2 0.2\n")
    pred_det = tmp_path / "pred_detect.txt"
    pred_det.write_text("0 0.9 0.5 0.5 0.2 0.2\n")
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
    # review_stats.json should now exist
    stats_path = project_root / ".tcip" / "state" / "review_stats.json"
    assert stats_path.exists()
    data = json.loads(stats_path.read_text(encoding="utf-8"))
    assert "image" in data
    assert "IMG_0000.JPG" in data["image"]
    assert data["image"]["IMG_0000.JPG"]["detections"][0]["action"] == "accepted"


# ── /api/state ───────────────────────────────────────────────────────────


def test_state_snapshot_available(client: TestClient) -> None:
    resp = client.get("/api/state")
    body = resp.json()
    # Minimal shape sanity
    assert "active_tab" in body
    assert "view" in body
    assert "class_names" in body
