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
    # Per-date maps present for every image date (empty here — the fixture makes trait
    # dirs but no label files).
    assert set(body["traits_by_date"]) == {"2-11-26", "3-2-26"}
    assert body["traits_by_date"]["2-11-26"] == []


def test_dataset_tree_per_date_reflects_actual_labels(client: TestClient, tmp_path: Path) -> None:
    root = tmp_path / "ds"
    (root / "images" / "2026-02-11").mkdir(parents=True)
    (root / "images" / "2026-03-24").mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(root / "images" / "2026-02-11" / "IMG_1.JPG")
    Image.new("RGB", (8, 8)).save(root / "images" / "2026-03-24" / "IMG_2.JPG")
    # catkin labelled + baseline predicted on 02-11; nothing on 03-24.
    det = root / "annotations" / "catkin" / "2026-02-11" / "detect"
    det.mkdir(parents=True)
    (det / "IMG_1.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    pdet = root / "predictions" / "baseline" / "2026-02-11" / "detect"
    pdet.mkdir(parents=True)
    (pdet / "IMG_1.txt").write_text("0 0.9 0.5 0.5 0.1 0.1\n", encoding="utf-8")

    body = client.get("/api/dataset/tree", params={"dataset_root": str(root)}).json()
    assert body["traits_by_date"]["2026-02-11"] == ["catkin"]
    assert body["traits_by_date"]["2026-03-24"] == []
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


def test_dataset_select_advisory_reflects_actual_labels(
    client: TestClient, dataset_root: Path, tmp_path: Path
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    body = {
        "project_root": str(project),
        "dataset_root": str(dataset_root),
        "annotation_type": "catkin",
        "date": "2-11-26",
        "model_name": "baseline",
    }
    # The fixture makes the annotation dirs but no label files → advisory says "starts empty".
    r1 = client.post("/api/dataset/select", json=body).json()
    assert r1["annotations_present"] is False
    assert r1["predictions_present"] is False

    # Drop in a real label + prediction; the advisory flips to present (never rejects either way).
    (dataset_root / "annotations" / "catkin" / "2-11-26" / "detect" / "IMG_0000.txt").write_text(
        "0 0.5 0.5 0.1 0.1\n", encoding="utf-8"
    )
    pdet = dataset_root / "predictions" / "baseline" / "2-11-26" / "detect"
    pdet.mkdir(parents=True, exist_ok=True)
    (pdet / "IMG_0000.txt").write_text("0 0.9 0.5 0.5 0.1 0.1\n", encoding="utf-8")
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
            "annotation_type": "catkin",
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
    det_path = tmp_path / "detect" / "IMG_0000.txt"
    base = _save_box(client, img_path, det_path).json()["base_mtimes"]

    # A concurrent writer changes the file after our client loaded it.
    det_path.write_text("0 0.5 0.5 0.2 0.2\n")
    os.utime(det_path, ns=(base["detect"] + 1_000_000, base["detect"] + 1_000_000))

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
    det_path = tmp_path / "detect" / "IMG_0000.txt"
    base = _save_box(client, img_path, det_path).json()["base_mtimes"]

    # No external change → the current mtime still matches → the save is accepted
    # and returns fresh version tokens.
    resp = _save_box(client, img_path, det_path, base_mtimes=base)
    assert resp.status_code == 200
    assert resp.json()["base_mtimes"]["detect"] is not None


def test_annotate_save_writes_audit_entry(
    client: TestClient, dataset_root: Path, tmp_path: Path
) -> None:
    proj = tmp_path / "proj"
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    det_path = tmp_path / "detect" / "IMG_0000.txt"
    resp = _save_box(client, img_path, det_path, project_root=str(proj))
    assert resp.status_code == 200

    audit = proj / ".tcip" / "audit.jsonl"
    assert audit.exists()
    assert "gui_save_labels" in audit.read_text()


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


def test_review_action_auto_completes_and_audits(
    client: TestClient, dataset_root: Path, tmp_path: Path
) -> None:
    # A single detection on the image: reviewing it flips the image to 'completed'
    # (the only GUI path to that status) and leaves an audit-trail entry.
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    det_gt = tmp_path / "gt.txt"
    det_gt.write_text("0 0.5 0.5 0.2 0.2\n")
    pred = tmp_path / "pred.txt"
    pred.write_text("0 0.9 0.5 0.5 0.2 0.2\n")  # one matching prediction -> one TP
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
    # With classes.json present the engine records the real class name + a reviewer,
    # instead of the "class_{id}" placeholder / empty reviewer the audit flagged.
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    det_gt = tmp_path / "gt.txt"
    det_gt.write_text("0 0.5 0.5 0.2 0.2\n")
    pred = tmp_path / "pred.txt"
    pred.write_text("0 0.9 0.5 0.5 0.2 0.2\n")
    project_root = tmp_path / "proj"
    state = project_root / ".tcip" / "state"
    state.mkdir(parents=True)
    (state / "classes.json").write_text(json.dumps({"0": {"name": "catkin", "color": "#FF0000"}}))

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
    entry = json.loads((state / "review_stats.json").read_text())["image"]["IMG_0000.JPG"][
        "detections"
    ][0]
    assert entry["class_name"] == "catkin"  # real name, not "class_0"
    assert entry["reviewed_by"]  # non-empty reviewer


def test_review_materialize_route(client: TestClient, tmp_path: Path) -> None:
    from PIL import Image

    project_root = tmp_path / "proj"
    state_dir = project_root / ".tcip" / "state"
    state_dir.mkdir(parents=True)
    src = tmp_path / "src"
    src.mkdir()
    for name in ("imgA.png", "imgB.png"):
        Image.new("RGB", (64, 64), (120, 120, 120)).save(src / name)
    (state_dir / "review_stats.json").write_text(
        json.dumps({"image": {
            "imgA.png": {"img_status": "completed", "detections": [
                {"action": "accepted", "class_id": 0,
                 "gt_bbox_norm": [0.5, 0.5, 0.2, 0.2], "pred_bbox_norm": None}]},
            "imgB.png": {"img_status": "completed", "detections": [
                {"action": "rejected", "class_id": 0,
                 "gt_bbox_norm": None, "pred_bbox_norm": [0.8, 0.8, 0.1, 0.1]}]},
        }})
    )
    out = tmp_path / "out"

    resp = client.post(
        "/api/review/materialize",
        json={
            "project_root": str(project_root),
            "source_images_dir": str(src),
            "output_dir": str(out),
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["positive"] == 1 and body["hard_negative"] == 1
    assert (out / "images" / "imgA.png").is_file()
    assert "gui_materialize_review_dataset" in (project_root / ".tcip" / "audit.jsonl").read_text()


def test_review_materialize_missing_state_returns_400(client: TestClient, tmp_path: Path) -> None:
    project_root = tmp_path / "proj"
    (project_root / ".tcip" / "state").mkdir(parents=True)
    src = tmp_path / "src"
    src.mkdir()
    resp = client.post(
        "/api/review/materialize",
        json={
            "project_root": str(project_root),
            "source_images_dir": str(src),
            "output_dir": str(tmp_path / "out"),
        },
    )
    assert resp.status_code == 400


def test_review_queue_launch_validates_inputs(client: TestClient, tmp_path: Path) -> None:
    project_root = tmp_path / "proj"
    project_root.mkdir()
    # A missing checkpoint is a 404 before any work is scheduled.
    resp = client.post(
        "/api/review/queue/launch",
        json={
            "project_root": str(project_root),
            "checkpoint_path": str(tmp_path / "nope.pt"),
            "images_dir": str(tmp_path),
        },
    )
    assert resp.status_code == 404
    # An unknown queue job id is a 404.
    assert client.get("/api/review/queue/does-not-exist").status_code == 404


def test_review_queue_async_flow(client: TestClient, tmp_path: Path, monkeypatch) -> None:
    import time

    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")
    images = tmp_path / "imgs"
    images.mkdir()
    project_root = tmp_path / "proj"
    project_root.mkdir()

    fake = {
        "method": "combined", "task": "detection", "total_candidates": 2,
        "reviewed_skipped": 0, "selected_count": 1,
        "queue": [{"image": str(images / "a.jpg"), "score": 0.9}],
    }
    # Avoid loading torch/a real model — exercise only the async job plumbing.
    monkeypatch.setattr(
        "tcip_mcp.tools.feedback_tools.prioritize_review_queue", lambda **kw: fake
    )

    resp = client.post(
        "/api/review/queue/launch",
        json={
            "project_root": str(project_root),
            "checkpoint_path": str(ckpt),
            "images_dir": str(images),
            "budget": 5,
        },
    )
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    body = {"status": "pending"}
    for _ in range(50):  # worker runs on a thread; fake returns immediately
        body = client.get(f"/api/review/queue/{job_id}").json()
        if body["status"] in ("completed", "failed"):
            break
        time.sleep(0.05)
    assert body["status"] == "completed"
    assert body["result"]["selected_count"] == 1


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
