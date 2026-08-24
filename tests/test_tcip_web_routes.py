"""Integration tests for tcip-web HTTP routes (name-based per-image label schema)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import tcip_store
from tcip_annotation.json_io import read_annotations, write_annotations
from tcip_annotation.state import Annotation, BBox
from tcip_mcp.audit import audit_log_key
from tcip_mcp.class_registry import ClassRegistry, Subject, write_registry
from tcip_web.app import app
from tcip_web.paths import safe_join


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


# ── per-image JSON label fixtures (canonical on-disk format) ─────────────────


def _write_gt(path, boxes, *, w: int = 100, h: int = 80, keep_empty: bool = False,
              subject: str = "catkin") -> None:
    """Author a per-image JSON GT label; each box is pixel-xyxy ``(x1, y1, x2, y2)`` of ``subject``."""
    anns = [Annotation(subject=subject, geometry=BBox(*b)) for b in boxes]
    write_annotations(str(path), anns, w, h, keep_empty=keep_empty)


def _write_pred(path, preds, *, w: int = 100, h: int = 80, subject: str = "catkin") -> None:
    """Author a per-image JSON prediction label; each pred is ``(x1, y1, x2, y2, conf)``."""
    anns = [Annotation(subject=subject, geometry=BBox(p[0], p[1], p[2], p[3]), score=p[4])
            for p in preds]
    write_annotations(str(path), anns, w, h)


def _write_operating_point_sidecar(pred_dir, fields: dict) -> None:
    """Author a bucket's ``operating_point.json`` stamp through the seam, not a bare file write."""
    from tcip_mcp.pipelines.resolution import sidecar_key

    tcip_store.replace(sidecar_key(pred_dir, "operating_point"), fields, expect=tcip_store.Version.ABSENT)


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
    (root / "models" / "baseline").mkdir(parents=True)
    # The dataset's subjects come from its nested registry, not from listing annotations/.
    write_registry(root / "classes.json", ClassRegistry((Subject("catkin"), Subject("bush"))))
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
    # Per-date maps present for every image date (empty here: the registry declares subjects but
    # no label files exist yet).
    assert set(body["subjects_by_date"]) == {"2-11-26", "3-2-26"}
    assert body["subjects_by_date"]["2-11-26"] == []


def test_dataset_tree_per_date_reflects_actual_labels(client: TestClient, tmp_path: Path) -> None:
    root = tmp_path / "ds"
    (root / "images" / "2026-02-11").mkdir(parents=True)
    (root / "images" / "2026-03-24").mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(root / "images" / "2026-02-11" / "IMG_1.JPG")
    Image.new("RGB", (8, 8)).save(root / "images" / "2026-03-24" / "IMG_2.JPG")
    # catkin labelled + baseline predicted on 02-11; nothing on 03-24. One file per image.
    det = root / "annotations" / "2026-02-11"
    det.mkdir(parents=True)
    _write_gt(det / "IMG_1.json", [(1, 1, 3, 3)], w=8, h=8)
    pdet = root / "predictions" / "baseline" / "2026-02-11"
    pdet.mkdir(parents=True)
    _write_pred(pdet / "IMG_1.json", [(1, 1, 3, 3, 0.9)], w=8, h=8)

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
    # One label dir per date now (no subject/task segment).
    assert sel["annotations_dir"].replace("\\", "/").endswith("annotations/2-11-26")


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
    # No label files yet → advisory says "starts empty".
    r1 = client.post("/api/dataset/select", json=body).json()
    assert r1["annotations_present"] is False
    assert r1["predictions_present"] is False

    # Drop in a real label + prediction; the advisory flips to present (never rejects either way).
    ann = dataset_root / "annotations" / "2-11-26"
    ann.mkdir(parents=True, exist_ok=True)
    _write_gt(ann / "IMG_0000.json", [(40, 32, 60, 48)])
    pdet = dataset_root / "predictions" / "baseline" / "2-11-26"
    pdet.mkdir(parents=True, exist_ok=True)
    _write_pred(pdet / "IMG_0000.json", [(40, 32, 60, 48, 0.9)])
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


def test_images_not_found(client: TestClient, tmp_path: Path) -> None:
    resp = client.get("/api/images", params={"path": str(tmp_path / "does_not_exist.jpg")})
    assert resp.status_code == 404


# ── /api/annotate ────────────────────────────────────────────────────────


def test_annotate_load_and_save_roundtrip(client: TestClient, dataset_root: Path, tmp_path: Path) -> None:
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    label_path = tmp_path / "labels" / "IMG_0000.json"

    # Save a box annotation and a polygon annotation into the single per-image file.
    resp = client.post(
        "/api/annotate/labels",
        json={
            "image_path": str(img_path),
            "label_path": str(label_path),
            "annotations": [
                {"subject": "catkin", "bbox": [10, 20, 50, 60]},
                {"subject": "catkin", "points": [[5, 5], [10, 5], [10, 10], [5, 10]]},
            ],
        },
    )
    assert resp.status_code == 200
    assert label_path.exists()

    # Load
    body = client.get(
        "/api/annotate/labels",
        params={"image_path": str(img_path), "label_path": str(label_path)},
    ).json()
    anns = body["annotations"]
    assert len(anns) == 2
    assert all(a["subject"] == "catkin" for a in anns)
    # One carries a box, one a polygon (geometry kinds coexist in one file). The load side reports a
    # polygon as `rings`: a stored shape can be occlusion-split, so it is never flattened to one.
    assert sum("bbox" in a for a in anns) == 1
    assert sum("rings" in a for a in anns) == 1


def test_annotate_save_empty_preserves_negative(
    client: TestClient, dataset_root: Path, tmp_path: Path
) -> None:
    # Clearing all annotations and saving must keep the label file (an {"annotations": []} record),
    # not delete it: it becomes a confirmed negative once the image is explicitly completed.
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    label_path = tmp_path / "labels" / "IMG_0000.json"

    client.post(
        "/api/annotate/labels",
        json={
            "image_path": str(img_path),
            "label_path": str(label_path),
            "annotations": [{"subject": "catkin", "bbox": [10, 20, 50, 60]}],
        },
    )
    assert label_path.exists()

    resp = client.post(
        "/api/annotate/labels",
        json={"image_path": str(img_path), "label_path": str(label_path), "annotations": []},
    )
    assert resp.status_code == 200
    # A present file with no annotations is a confirmed negative (kept, not deleted).
    assert label_path.exists()
    assert read_annotations(str(label_path)) == []


def test_annotate_save_label_path_outside_allowed_root_403(
    client: TestClient, dataset_root: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    # A label path outside every allowed root must be rejected: write_annotations is otherwise
    # an arbitrary file write/delete primitive.
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    outside = tmp_path_factory.mktemp("outside") / "evil" / "IMG_0000.json"
    resp = client.post(
        "/api/annotate/labels",
        json={"image_path": str(img_path), "label_path": str(outside), "annotations": []},
    )
    assert resp.status_code == 403
    assert not outside.exists()


def _save_box(client: TestClient, img_path, label_path, **extra) -> dict:
    resp = client.post(
        "/api/annotate/labels",
        json={
            "image_path": str(img_path),
            "label_path": str(label_path),
            "annotations": [{"subject": "catkin", "bbox": [10, 20, 50, 60]}],
            **extra,
        },
    )
    return resp


def test_annotate_save_refuses_a_token_the_document_has_moved_past(
    client: TestClient, dataset_root: Path, tmp_path: Path
) -> None:
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    label_path = tmp_path / "labels" / "IMG_0000.json"
    base = _save_box(client, img_path, label_path).json()["base_mtime"]

    # A concurrent writer changes the document after our client loaded it.
    label_path.write_text('{"image": "IMG_0000", "width": 100, "height": 80, "annotations": []}')

    resp = client.post(
        "/api/annotate/labels",
        json={
            "image_path": str(img_path),
            "label_path": str(label_path),
            "annotations": [],
            "base_mtime": base,
        },
    )
    assert resp.status_code == 409


def test_annotate_save_with_the_current_token_is_accepted(
    client: TestClient, dataset_root: Path, tmp_path: Path
) -> None:
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    label_path = tmp_path / "labels" / "IMG_0000.json"
    base = _save_box(client, img_path, label_path).json()["base_mtime"]

    # No external change → the token still names the stored document → the save is accepted and
    # returns a fresh one.
    resp = _save_box(client, img_path, label_path, base_mtime=base)
    assert resp.status_code == 200
    token = resp.json()["base_mtime"]
    assert token is not None
    # A string, not a number: the client only ever echoes it back, and a numeric token would be
    # rounded by the browser's JSON parse and mismatch on every save.
    assert isinstance(token, str)


def test_annotate_load_hands_back_a_token_its_own_save_accepts(
    client: TestClient, dataset_root: Path, tmp_path: Path
) -> None:
    """The load and save pair is one compare-and-set over what the client was actually shown."""
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    label_path = tmp_path / "labels" / "IMG_0000.json"
    params = {"image_path": str(img_path), "label_path": str(label_path)}

    loaded = client.get("/api/annotate/labels", params=params).json()
    assert loaded["annotations"] == []
    assert _save_box(client, img_path, label_path,
                     base_mtime=loaded["base_mtime"]).status_code == 200

    # That token said the document did not exist, so replaying it cannot overwrite what the first
    # save created; the token from a fresh load can.
    assert _save_box(client, img_path, label_path,
                     base_mtime=loaded["base_mtime"]).status_code == 409
    reloaded = client.get("/api/annotate/labels", params=params).json()
    assert _save_box(client, img_path, label_path,
                     base_mtime=reloaded["base_mtime"]).status_code == 200


def test_annotate_save_without_a_token_still_writes(
    client: TestClient, dataset_root: Path, tmp_path: Path
) -> None:
    """A caller that supplies no token skips the comparison, exactly as before."""
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    label_path = tmp_path / "labels" / "IMG_0000.json"
    assert _save_box(client, img_path, label_path).status_code == 200
    assert _save_box(client, img_path, label_path).status_code == 200
    assert label_path.is_file()


def test_annotate_save_persists_polygon_as_polygon(client, dataset_root, tmp_path) -> None:
    # A polygon annotation round-trips as a polygon (its points are the source of truth), never
    # collapsed to a box on disk. Image is 100x80.
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    label_path = tmp_path / "labels" / "IMG_0000.json"
    resp = client.post(
        "/api/annotate/labels",
        json={
            "image_path": str(img_path),
            "label_path": str(label_path),
            "annotations": [{"subject": "catkin", "points": [[10, 10], [30, 10], [30, 30], [10, 30]]}],
        },
    )
    assert resp.status_code == 200
    anns = read_annotations(str(label_path))
    assert len(anns) == 1
    from tcip_annotation.state import Polygon
    assert isinstance(anns[0].geometry, Polygon)
    # A hand-drawn contour is the one ring the canvas authored.
    assert anns[0].geometry.rings == [[(10.0, 10.0), (30.0, 10.0), (30.0, 30.0), (10.0, 30.0)]]


def test_annotate_multi_ring_polygon_round_trips_through_the_route(client, dataset_root, tmp_path):
    """An occlusion-split shape loaded onto the canvas and re-saved unedited must keep every ring.

    The save side accepts ``rings`` for exactly this, and the load side reports ``rings`` back, so a
    multi-ring instance_seg shape survives an ordinary open/save with no edits, instead of being
    silently reduced to its first contour.
    """
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    label_path = tmp_path / "labels" / "IMG_0000.json"
    rings = [[[10, 10], [30, 10], [30, 30], [10, 30]], [[60, 10], [80, 10], [80, 30], [60, 30]]]
    resp = client.post("/api/annotate/labels", json={
        "image_path": str(img_path), "label_path": str(label_path),
        "annotations": [{"subject": "catkin", "rings": rings}],
    })
    assert resp.status_code == 200

    stored = read_annotations(str(label_path))
    assert len(stored) == 1  # one instance, not one per contour
    assert stored[0].geometry.rings == [[tuple(map(float, p)) for p in r] for r in rings]

    body = client.get("/api/annotate/labels", params={
        "image_path": str(img_path), "label_path": str(label_path)}).json()
    (ann,) = body["annotations"]
    assert ann["rings"] == rings


def test_annotate_save_prefers_rings_over_points_when_both_are_sent(client, dataset_root, tmp_path):
    """`rings` is the full shape and `points` only ever one contour, so `rings` wins: otherwise a
    client that sends both (a loaded multi-ring shape plus a legacy single-ring mirror) would persist
    the truncated version."""
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    label_path = tmp_path / "labels" / "IMG_0000.json"
    rings = [[[10, 10], [30, 10], [30, 30]], [[60, 10], [80, 10], [80, 30]]]
    resp = client.post("/api/annotate/labels", json={
        "image_path": str(img_path), "label_path": str(label_path),
        "annotations": [{"subject": "catkin", "rings": rings, "points": rings[0]}],
    })
    assert resp.status_code == 200
    (stored,) = read_annotations(str(label_path))
    assert len(stored.geometry.rings) == 2


def test_annotate_save_persists_box_as_box(client, dataset_root, tmp_path) -> None:
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    label_path = tmp_path / "labels" / "IMG_0000.json"
    resp = client.post(
        "/api/annotate/labels",
        json={
            "image_path": str(img_path),
            "label_path": str(label_path),
            "annotations": [{"subject": "catkin", "bbox": [50, 40, 70, 60]}],
        },
    )
    assert resp.status_code == 200
    anns = read_annotations(str(label_path))
    assert len(anns) == 1
    b = anns[0].geometry
    assert (b.x1, b.y1, b.x2, b.y2) == (50.0, 40.0, 70.0, 60.0)  # box, written as drawn


def test_annotate_save_audits_into_the_log_of_the_dataset_it_wrote(
    client: TestClient, dataset_root: Path, tmp_path: Path
) -> None:
    """Labels travel with their dataset, so the trail of a label write is recorded beside them
    and not in the log of whichever project happened to have the dataset open."""
    proj = tmp_path / "proj"
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    label_path = tmp_path / "labels" / "IMG_0000.json"
    resp = _save_box(client, img_path, label_path, project_root=str(proj))
    assert resp.status_code == 200

    assert any(e.get("tool") == "gui_save_labels" for e in _audit_entries(tmp_path))
    assert _audit_entries(proj) == []


# ── /api/review ─────────────────────────────────────────────────────────


def _shard_state(state_dir: Path, img_name: str) -> dict:
    """The one review verdict recorded for ``img_name``, wherever its bucket key put it."""
    from tcip_annotation.review_engine import REVIEW_VERDICTS_STORE

    found = [k for k in tcip_store.keys(REVIEW_VERDICTS_STORE, str(state_dir)) if k.parts[1] == img_name]
    assert len(found) == 1, found
    return tcip_store.read(found[0])["state"]


def _audit_entries(root: Path) -> list[dict]:
    """Every audit entry recorded in the log ``root`` names, through the seam."""
    return list(tcip_store.read_log(audit_log_key(root)).records)


def test_review_matches_end_to_end(client: TestClient, dataset_root: Path, tmp_path: Path) -> None:
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    gt = tmp_path / "gt.json"
    # Image is 100x80; one GT covering the center (pixel xyxy [40,32,60,48]).
    _write_gt(gt, [(40, 32, 60, 48)])
    pred = tmp_path / "pred.json"
    # One prediction matching the GT (TP) + one off-center prediction (FP).
    _write_pred(pred, [(40, 32, 60, 48, 0.9), (75, 60, 85, 68, 0.8)])
    resp = client.post(
        "/api/review/matches",
        json={
            "dataset_root": str(dataset_root),
            "image_name": "IMG_0000.JPG",
            "image_path": str(img_path),
            "gt_path": str(gt),
            "pred_path": str(pred),
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
    # and a third image with no file at all: only the first should surface as a detection stem.
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir(parents=True)
    _write_pred(pred_dir / "IMG_0000.json", [(40, 32, 60, 48, 0.9)])
    write_annotations(str(pred_dir / "IMG_0001.json"), [], 100, 80, keep_empty=True)  # empty negative

    # Give one image a review status so the engine has state to return.
    gt = tmp_path / "gt.json"
    _write_gt(gt, [(40, 32, 60, 48)])
    client.post(
        "/api/review/mark_complete",
        json={"dataset_root": str(dataset_root), "image_name": "IMG_0000.JPG", "gt_path": str(gt)},
    )

    resp = client.get(
        "/api/review/image_statuses",
        params={"dataset_root": str(dataset_root), "pred_dir": str(pred_dir)},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["detection_stems"] == ["IMG_0000"]  # empty + missing files excluded
    assert body["statuses"]["IMG_0000.JPG"] == "completed"


def test_review_action_persists(client: TestClient, dataset_root: Path, tmp_path: Path) -> None:
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    gt = tmp_path / "gt.json"
    _write_gt(gt, [(40, 32, 60, 48)])
    pred = tmp_path / "pred.json"
    _write_pred(pred, [(40, 32, 60, 48, 0.9)])

    resp = client.post(
        "/api/review/action",
        json={
            "dataset_root": str(dataset_root),
            "image_name": "IMG_0000.JPG",
            "image_path": str(img_path),
            "gt_path": str(gt),
            "pred_path": str(pred),
            "det_type": "tp",
            "class_name": "catkin",
            "conf": 0.9,
            "iou": 0.95,
            "gt_idx": 0,
            "pred_idx": 0,
            "bbox": [40.0, 32.0, 60.0, 48.0],
            "action": "accepted",
        },
    )
    assert resp.status_code == 200
    # the image's review verdict should now exist (a shard keyed by image, not one whole-state record)
    state = _shard_state(dataset_root / ".tcip" / "state", "IMG_0000.JPG")
    assert state["detections"][0]["action"] == "accepted"


def test_review_action_resolves_class_id_from_bucket_id_map(
    client: TestClient, dataset_root: Path, tmp_path: Path
) -> None:
    """The verdict entry carries a resolved ``class_id``: the producing bucket's own recorded
    ``id_map`` (``operating_point.json``), read at record time, not the previously-dead
    ``class_id`` field ``review_to_records`` used to default to 0 for every entry."""
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    gt = tmp_path / "gt.json"
    _write_gt(gt, [(40, 32, 60, 48)])
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir(parents=True)
    pred = pred_dir / "pred.json"
    _write_pred(pred, [(40, 32, 60, 48, 0.9)])
    _write_operating_point_sidecar(
        pred_dir, {"checkpoint_sha256": "sha", "experiment_id": None,
                   "id_map": {"dormant": 0, "elongated": 1}})

    resp = _review_action(
        client, img_path, gt, dataset_root,
        pred_path=str(pred), det_type="tp", class_name="elongated", action="accepted",
    )
    assert resp.status_code == 200
    state = _shard_state(dataset_root / ".tcip" / "state", "IMG_0000.JPG")
    assert state["detections"][0]["class_id"] == 1  # resolved via the bucket's id_map


def test_review_action_records_unresolvable_class_id_as_none(
    client: TestClient, dataset_root: Path, tmp_path: Path
) -> None:
    """A verdict class_name the producing bucket's id_map does not recognize (e.g. an attribute-
    scoped bucket handed a GT annotation's raw subject name) records ``class_id: null``, an honest
    unresolved fact, never a guessed 0/1. This is exactly the case a later review-confirmed build
    (``review_calibration.review_to_records``) must refuse on, not silently mis-class."""
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    gt = tmp_path / "gt.json"
    _write_gt(gt, [(40, 32, 60, 48)])
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir(parents=True)
    pred = pred_dir / "pred.json"
    _write_pred(pred, [(40, 32, 60, 48, 0.9)])
    (pred_dir / "operating_point.json").write_text(
        json.dumps({"checkpoint_sha256": "sha", "experiment_id": None,
                    "id_map": {"dormant": 0, "elongated": 1}}),
        encoding="utf-8")

    resp = _review_action(
        client, img_path, gt, dataset_root,
        pred_path=str(pred), det_type="tp", class_name="catkin", action="accepted",
    )
    assert resp.status_code == 200
    state = _shard_state(dataset_root / ".tcip" / "state", "IMG_0000.JPG")
    assert state["detections"][0]["class_id"] is None


def test_review_action_no_sidecar_records_unresolvable_class_id(
    client: TestClient, dataset_root: Path, tmp_path: Path
) -> None:
    """No recorded ``id_map`` at all (no sidecar, or an older bucket) also records
    ``class_id: null``, never silently defaulting to a guessed single class."""
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    gt = tmp_path / "gt.json"
    _write_gt(gt, [(40, 32, 60, 48)])
    pred = tmp_path / "pred.json"
    _write_pred(pred, [(40, 32, 60, 48, 0.9)])

    resp = _review_action(
        client, img_path, gt, dataset_root,
        pred_path=str(pred), det_type="tp", class_name="catkin", action="accepted",
    )
    assert resp.status_code == 200
    state = _shard_state(dataset_root / ".tcip" / "state", "IMG_0000.JPG")
    assert state["detections"][0]["class_id"] is None


def _review_action(client, img_path, gt, dataset_root, **over):
    body = {
        "dataset_root": str(dataset_root),
        "image_name": "IMG_0000.JPG",
        "image_path": str(img_path),
        "gt_path": str(gt),
        "det_type": "tp",
        "class_name": "catkin",
        "gt_idx": 0,
        "pred_idx": 0,
        "bbox": [40.0, 32.0, 60.0, 48.0],
        "action": "accepted",
    }
    body.update(over)
    return client.post("/api/review/action", json=body)


def test_review_accept_fp_adds_prediction_to_gt(client, dataset_root, tmp_path) -> None:
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    gt = tmp_path / "gt.json"
    _write_gt(gt, [], keep_empty=True)  # start with a confirmed negative (empty GT)
    pred = tmp_path / "pred.json"
    _write_pred(pred, [(40, 32, 60, 48, 0.9)])

    resp = _review_action(
        client, img_path, gt, dataset_root,
        pred_path=str(pred), det_type="fp", action="accepted",
    )
    assert resp.status_code == 200
    assert resp.json()["annotation_status"] == "partial"  # GT now has the promoted box
    anns = read_annotations(str(gt))
    assert len(anns) == 1 and anns[0].subject == "catkin"


def test_review_reject_deletes_reviewed_gt(client, dataset_root, tmp_path) -> None:
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    gt = tmp_path / "gt.json"
    _write_gt(gt, [(40, 32, 60, 48)])  # one GT box (a missed FN)

    resp = _review_action(client, img_path, gt, dataset_root,
                          det_type="fn", pred_idx=None, action="rejected")
    assert resp.status_code == 200
    # Emptying GT does not auto-confirm a negative (that needs an explicit Complete): it reads as
    # needing review. The label file is kept (empty record), not deleted.
    assert resp.json()["annotation_status"] == "unannotated"
    assert gt.is_file()
    assert read_annotations(str(gt)) == []


def test_review_accept_tp_keeps_gt_untouched(client, dataset_root, tmp_path) -> None:
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    gt = tmp_path / "gt.json"
    _write_gt(gt, [(40, 32, 60, 48)])
    before = gt.read_text()

    resp = _review_action(client, img_path, gt, dataset_root, det_type="tp", action="accepted")
    assert resp.status_code == 200
    assert resp.json()["annotation_status"] is None  # GT unchanged → no status update
    assert gt.read_text() == before


def test_review_edit_writes_edited_box_as_gt(client, dataset_root, tmp_path) -> None:
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    gt = tmp_path / "gt.json"
    _write_gt(gt, [(40, 32, 60, 48)])

    resp = _review_action(
        client, img_path, gt, dataset_root,
        det_type="tp", action="edited", edited_box=[10.0, 10.0, 30.0, 30.0],
    )
    assert resp.status_code == 200
    assert resp.json()["annotation_status"] == "partial"
    anns = read_annotations(str(gt))
    assert len(anns) == 1  # replaced the matched GT, still one annotation
    b = anns[0].geometry
    assert (b.x1, b.y1, b.x2, b.y2) == (10.0, 10.0, 30.0, 30.0)  # the edited geometry


def test_review_edited_detection_stays_reviewed_after_reload(client, dataset_root, tmp_path) -> None:
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    gt = tmp_path / "gt.json"
    _write_gt(gt, [(40, 32, 60, 48)])  # one GT box, no predictions → an FN

    resp = _review_action(
        client, img_path, gt, dataset_root,
        det_type="fn", pred_idx=None,
        action="edited", edited_box=[10.0, 10.0, 30.0, 30.0],
    )
    assert resp.status_code == 200

    # The next reload rebuilds the FN from the edited GT file: the verdict must still be
    # recognized, i.e. the entry is keyed to the post-edit geometry, not the pre-edit one.
    m = client.post(
        "/api/review/matches",
        json={
            "dataset_root": str(dataset_root),
            "image_name": "IMG_0000.JPG",
            "image_path": str(img_path),
            "gt_path": str(gt),
        },
    ).json()
    assert len(m["detections"]) == 1
    assert m["detections"][0]["det_type"] == "fn"
    assert m["detections"][0]["reviewed"] is True
    assert m["detections"][0]["reviewed_action"] == "edited"


def test_review_gt_write_without_path_is_rejected(client, dataset_root, tmp_path) -> None:
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    pred = tmp_path / "pred.json"
    _write_pred(pred, [(40, 32, 60, 48, 0.9)])

    # Accepting an FP writes GT; with no GT path configured the route must refuse loudly rather
    # than report ok while writing nothing.
    resp = _review_action(
        client, img_path, "unused", dataset_root,
        gt_path=None, pred_path=str(pred), det_type="fp", action="accepted",
    )
    assert resp.status_code == 400
    assert "no annotations path" in resp.json()["detail"]
    # The refused verdict must not have been recorded as reviewed.
    review_dir = dataset_root / ".tcip" / "state" / "review"
    for shard_path in review_dir.rglob("IMG_0000.JPG.json"):
        data = json.loads(shard_path.read_text(encoding="utf-8"))
        assert not data.get("state", {}).get("detections")


def test_review_action_auto_completes_and_audits(
    client: TestClient, dataset_root: Path, tmp_path: Path
) -> None:
    # A single detection on the image: reviewing it flips the image to 'completed'
    # (the only GUI path to that status) and leaves an audit-trail entry.
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    gt = tmp_path / "gt.json"
    _write_gt(gt, [(40, 32, 60, 48)])
    pred = tmp_path / "pred.json"
    _write_pred(pred, [(40, 32, 60, 48, 0.9)])  # one matching prediction -> one TP

    resp = client.post(
        "/api/review/action",
        json={
            "dataset_root": str(dataset_root),
            "image_name": "IMG_0000.JPG",
            "image_path": str(img_path),
            "gt_path": str(gt),
            "pred_path": str(pred),
            "det_type": "tp", "class_name": "catkin", "conf": 0.9, "iou": 0.95,
            "gt_idx": 0, "pred_idx": 0,
            "bbox": [40.0, 32.0, 60.0, 48.0], "action": "accepted",
            "iou_threshold": 0.3, "conf_threshold": 0.1,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["image_status"] == "completed"
    assert any(e.get("tool") == "gui_review_action" for e in _audit_entries(dataset_root))


def test_review_mark_complete_and_audits(client: TestClient, tmp_path: Path) -> None:
    dataset_root = tmp_path / "data"

    resp = client.post(
        "/api/review/mark_complete",
        json={"dataset_root": str(dataset_root), "image_name": "IMG_9.JPG"},
    )
    assert resp.status_code == 200
    assert resp.json()["image_status"] == "completed"

    status = client.get(
        "/api/review/image_status",
        params={"dataset_root": str(dataset_root), "image_name": "IMG_9.JPG"},
    )
    assert status.json()["status"] == "completed"
    assert any(e.get("tool") == "gui_review_mark_complete" for e in _audit_entries(dataset_root))


def test_review_gt_edit_audits_into_the_log_of_the_dataset_it_wrote(
    client: TestClient, dataset_root: Path, tmp_path: Path
) -> None:
    """A review that edits ground truth changes a record that travels with the dataset, so the
    entry belongs beside the labels rather than in the project the breeder is working out of. The
    project root is a genuinely different directory here, so a log written there is a log in the
    wrong place rather than the same file under another name."""
    project_root = tmp_path / "proj"
    project_root.mkdir()
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    label_path = dataset_root / "annotations" / "2-11-26" / "IMG_0000.json"

    resp = client.post("/api/review/save_gt", json={
        "dataset_root": str(dataset_root),
        "image_name": "IMG_0000.JPG", "image_path": str(img_path), "label_path": str(label_path),
        "annotations": [{"subject": "catkin", "bbox": [10.0, 10.0, 30.0, 30.0]}],
    })
    assert resp.status_code == 200

    assert any(e.get("tool") == "gui_review_save_gt" for e in _audit_entries(dataset_root))
    assert _audit_entries(project_root) == []


def test_review_action_records_subject_name_and_reviewer(
    client: TestClient, dataset_root: Path, tmp_path: Path
) -> None:
    # A prediction stores its subject name on disk, so the recorded verdict carries the real name
    # ("catkin") directly: no registry lookup, no "class_{id}" placeholder.
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    gt = tmp_path / "gt.json"
    _write_gt(gt, [(40, 32, 60, 48)])
    pred = tmp_path / "pred.json"
    _write_pred(pred, [(40, 32, 60, 48, 0.9)])
    state = dataset_root / ".tcip" / "state"

    resp = client.post(
        "/api/review/action",
        json={
            "dataset_root": str(dataset_root),
            "image_name": "IMG_0000.JPG",
            "image_path": str(img_path),
            "gt_path": str(gt),
            "pred_path": str(pred),
            "det_type": "tp", "class_name": "catkin", "conf": 0.9, "iou": 0.95,
            "gt_idx": 0, "pred_idx": 0,
            "bbox": [40.0, 32.0, 60.0, 48.0], "action": "accepted",
            "iou_threshold": 0.3, "conf_threshold": 0.1,
        },
    )
    assert resp.status_code == 200
    entry = _shard_state(state, "IMG_0000.JPG")["detections"][0]
    assert entry["class_name"] == "catkin"  # real name, straight from the annotation's subject
    assert entry["reviewed_by"]  # non-empty reviewer


def _verdicted_launch_dataset(tmp_path: Path, monkeypatch) -> tuple[Path, Path, str]:
    """A dataset with one image, one canonical bucket and one verdict in its own verdict store.

    The platform root is pinned to a different, empty root, so a launch door counting verdicts
    there rather than in the dataset's own store would find none.
    """
    from tcip_annotation.review_engine import ReviewContext, ReviewDetection, ReviewEngine
    from tcip_mcp.dataset_layout import image_dir, prediction_dir
    from tcip_mcp.prediction_buckets import bucket_key_of

    platform_root = tmp_path / "platform"
    (platform_root / ".tcip" / "state").mkdir(parents=True)
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(platform_root))

    dataset_root = tmp_path / "data"
    date = "2026-02-11"
    images = image_dir(dataset_root, date)
    images.mkdir(parents=True)
    Image.new("RGB", (100, 100), (110, 110, 110)).save(images / "img.png")
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")

    out = prediction_dir(dataset_root, "baseline", date)
    out.mkdir(parents=True)
    (out / "img.json").write_text(
        json.dumps({"image": "img", "width": 100, "height": 100, "annotations": []})
    )
    engine = ReviewEngine(dataset_root / ".tcip" / "state")
    ctx = ReviewContext(img_name="img.png", img_width=100, img_height=100,
                        preds=[Annotation(subject="catkin", geometry=BBox(10.0, 10.0, 30.0, 30.0),
                                          score=0.9)])
    det = ReviewDetection(det_type="fp", class_name="catkin", conf=0.9, iou=None, gt_idx=None,
                          pred_idx=0, bbox=(10.0, 10.0, 30.0, 30.0))
    engine.record_detection_action(bucket_key_of(out), det, ctx, action="accepted")
    return dataset_root, ckpt, date


def test_inference_launch_refuses_overwrite_into_verdicted_bucket(
    client: TestClient, tmp_path: Path, monkeypatch
) -> None:
    """The launch door counts a bucket's verdicts in the store belonging to the dataset it is
    writing into, so the breeder's recorded verdicts are the ones that freeze it."""
    dataset_root, ckpt, date = _verdicted_launch_dataset(tmp_path, monkeypatch)

    # overwrite=True into a bucket that has a verdict is a 409 (no job is launched).
    resp = client.post("/api/inference/launch", json={
        "checkpoint_path": str(ckpt), "dataset_root": str(dataset_root),
        "model_name": "baseline", "date": date, "overwrite": True,
    })
    assert resp.status_code == 409
    assert "verdict" in resp.json()["detail"].lower()


def test_inference_launch_writes_an_unreviewed_bucket_in_place(
    client: TestClient, tmp_path: Path, monkeypatch
) -> None:
    """The same dataset-scoped guard still admits the ordinary re-run into an unreviewed bucket."""
    from tcip_mcp.dataset_layout import image_dir, prediction_dir
    from tcip_web.routes import inference as inference_routes

    # The bucket resolution under test is the route's own synchronous step; the prediction pass
    # behind it is not what this pins.
    monkeypatch.setattr(inference_routes, "_worker", lambda job: None)

    platform_root = tmp_path / "platform"
    (platform_root / ".tcip" / "state").mkdir(parents=True)
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(platform_root))

    dataset_root = tmp_path / "data"
    date = "2026-02-11"
    images = image_dir(dataset_root, date)
    images.mkdir(parents=True)
    Image.new("RGB", (100, 100), (110, 110, 110)).save(images / "img.png")
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")

    resp = client.post("/api/inference/launch", json={
        "checkpoint_path": str(ckpt), "dataset_root": str(dataset_root),
        "model_name": "baseline", "date": date, "overwrite": True,
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["bucket_redirected"] is False
    assert Path(body["output_dir"]) == prediction_dir(dataset_root, "baseline", date)


def _launch_setup(tmp_path, monkeypatch):
    from tcip_mcp.dataset_layout import image_dir
    from tcip_web.routes import inference as inference_routes

    monkeypatch.setattr(inference_routes, "_worker", lambda job: None)
    platform_root = tmp_path / "platform"
    (platform_root / ".tcip" / "state").mkdir(parents=True)
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(platform_root))

    dataset_root = tmp_path / "data"
    date = "2026-02-11"
    images = image_dir(dataset_root, date)
    images.mkdir(parents=True)
    Image.new("RGB", (100, 100), (110, 110, 110)).save(images / "img.png")
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")
    return str(ckpt), str(dataset_root), date, inference_routes


def test_inference_launch_resolves_explicit_conf_and_max_dets_source_from_the_payload(
    client: TestClient, tmp_path: Path, monkeypatch,
) -> None:
    """A caller-stated conf/max_dets equal to the platform default is stamped 'explicit' on the
    job, the resolution launch_inference threads into raw_operating_point via the worker."""
    from tcip_mcp.pipelines.resolution import DEFAULT_CONF, DEFAULT_MAX_DETS

    ckpt, dataset_root, date, inference_routes = _launch_setup(tmp_path, monkeypatch)

    resp = client.post("/api/inference/launch", json={
        "checkpoint_path": ckpt, "dataset_root": dataset_root, "model_name": "baseline",
        "date": date, "conf": DEFAULT_CONF, "max_dets": DEFAULT_MAX_DETS,
    })
    assert resp.status_code == 200, resp.text
    job = inference_routes._get(resp.json()["job_id"])
    assert job.conf_source == "explicit"
    assert job.max_dets_source == "explicit"
    assert job.conf == DEFAULT_CONF
    assert job.max_dets == DEFAULT_MAX_DETS


def test_inference_launch_defaults_conf_and_max_dets_source_when_omitted(
    client: TestClient, tmp_path: Path, monkeypatch,
) -> None:
    """The rail must admit the ordinary, unstated launch: an omitted conf/max_dets still resolves
    to the platform default and is stamped 'default' on the job, never 'explicit'."""
    from tcip_mcp.pipelines.resolution import DEFAULT_CONF, DEFAULT_MAX_DETS

    ckpt, dataset_root, date, inference_routes = _launch_setup(tmp_path, monkeypatch)

    resp = client.post("/api/inference/launch", json={
        "checkpoint_path": ckpt, "dataset_root": dataset_root, "model_name": "baseline",
        "date": date,
    })
    assert resp.status_code == 200, resp.text
    job = inference_routes._get(resp.json()["job_id"])
    assert job.conf_source == "default"
    assert job.max_dets_source == "default"
    assert job.conf == DEFAULT_CONF
    assert job.max_dets == DEFAULT_MAX_DETS


# ── /api/state ───────────────────────────────────────────────────────────


def test_state_snapshot_available(client: TestClient) -> None:
    resp = client.get("/api/state")
    body = resp.json()
    # Minimal shape sanity
    assert "active_tab" in body
    assert "view" in body
    assert "dataset" in body


def test_state_tab_push_mutates_the_store(client: TestClient) -> None:
    from tcip_web.state import TAB_NAMES

    assert "review" in TAB_NAMES
    resp = client.post("/api/state/tab", json={"active_tab": "review"})
    assert resp.status_code == 200
    assert client.get("/api/state").json()["active_tab"] == "review"
    client.post("/api/state/tab", json={"active_tab": "annotate"})


def test_state_tab_push_rejects_unknown_tabs(client: TestClient) -> None:
    before = client.get("/api/state").json()["active_tab"]
    resp = client.post("/api/state/tab", json={"active_tab": "dashboard"})
    assert resp.status_code == 400
    assert "dashboard" in resp.json()["detail"]
    assert client.get("/api/state").json()["active_tab"] == before


# ── /api/fs (folder browser) ───────────────────────────────────────────────


def test_fs_list_directories(client: TestClient, tmp_path: Path) -> None:
    (tmp_path / "alpha").mkdir()
    (tmp_path / "beta" / "images").mkdir(parents=True)  # looks like a dataset root
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "afile.txt").write_text("x")
    # Windows system/recovery folders that clutter a picker, filtered out by name.
    (tmp_path / "$RECYCLE.BIN").mkdir()
    (tmp_path / "System Volume Information").mkdir()
    (tmp_path / "FOUND.000").mkdir()

    resp = client.get("/api/fs/list", params={"path": str(tmp_path)})
    assert resp.status_code == 200
    body = resp.json()
    names = {e["name"]: e for e in body["entries"]}
    assert "alpha" in names and "beta" in names
    assert ".hidden" not in names  # hidden dirs skipped
    assert "afile.txt" not in names  # files skipped, directories only
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


def test_fs_list_is_unconfined_from_a_local_connection(
    client: TestClient, tmp_path_factory: pytest.TempPathFactory
) -> None:
    # The picker route is unconfined on a connection from this machine (the default TestClient);
    # confinement on a routable arrival is covered by test_web_path_guard_permanent_on.py.
    outside = tmp_path_factory.mktemp("outside")
    (outside / "sub").mkdir()
    assert client.get("/api/fs/list", params={"path": str(outside)}).status_code == 200


# ── native provenance (created_by / accepted_by) ─────────────────


def test_annotate_save_stamps_created_by(client, dataset_root, tmp_path) -> None:
    """A human-drawn box is stamped created_by=user:<gui-user> + created_at."""
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    label_path = tmp_path / "labels" / "IMG_0000.json"
    resp = client.post("/api/annotate/labels", json={
        "image_path": str(img_path), "label_path": str(label_path),
        "annotations": [{"subject": "catkin", "bbox": [50, 40, 70, 60]}],
        "user": "breeder",
    })
    assert resp.status_code == 200
    obj = json.loads(label_path.read_text())["annotations"][0]
    assert obj["created_by"] == "user:breeder"
    assert obj["created_at"]


def test_annotate_save_polygon_stamps_author(client, dataset_root, tmp_path) -> None:
    """A human-drawn polygon is stamped created_by=user:<gui-user> (not None)."""
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    label_path = tmp_path / "labels" / "IMG_0000.json"
    resp = client.post("/api/annotate/labels", json={
        "image_path": str(img_path), "label_path": str(label_path),
        "annotations": [{"subject": "catkin", "points": [[10, 10], [30, 10], [30, 30]]}],
        "user": "emily",
    })
    assert resp.status_code == 200
    assert json.loads(label_path.read_text())["annotations"][0]["created_by"] == "user:emily"


def test_annotate_save_falls_back_to_os_user(client, dataset_root, tmp_path) -> None:
    """Omitting user still stamps a user:<...> author (backend OS/env fallback), never bare/None."""
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    label_path = tmp_path / "labels" / "IMG_0000.json"
    resp = client.post("/api/annotate/labels", json={
        "image_path": str(img_path), "label_path": str(label_path),
        "annotations": [{"subject": "catkin", "bbox": [50, 40, 70, 60]}],
    })
    assert resp.status_code == 200
    obj = json.loads(label_path.read_text())["annotations"][0]
    assert obj["created_by"].startswith("user:")


def test_review_accept_fp_carries_created_by_and_stamps_accepted_by(client, dataset_root, tmp_path) -> None:
    """Accepting an FP prediction into GT carries the prediction's created_by and stamps
    accepted_by=reviewer: the origin travels, and the acceptance is recorded."""
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    gt = tmp_path / "gt.json"
    _write_gt(gt, [], keep_empty=True)  # confirmed negative → the pred shows as FP
    pred = tmp_path / "pred.json"
    write_annotations(str(pred), [Annotation(
        subject="catkin", geometry=BBox(40, 32, 60, 48), score=0.9,
        created_by="sam", created_at="2026-01-01T00:00:00+00:00")], 100, 80)

    resp = _review_action(
        client, img_path, gt, dataset_root,
        pred_path=str(pred), det_type="fp", action="accepted", user="breeder",
    )
    assert resp.status_code == 200
    obj = json.loads(gt.read_text())["annotations"][0]
    assert obj["created_by"] == "sam"          # prediction origin carried into GT
    assert obj["accepted_by"] == "user:breeder"   # reviewer stamped
    assert obj["accepted_at"]


def test_review_edit_stamps_created_by(client, dataset_root, tmp_path) -> None:
    """A reviewer-drawn edit authors GT with created_by=user:<reviewer>."""
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    gt = tmp_path / "gt.json"
    _write_gt(gt, [(40, 32, 60, 48)])

    resp = _review_action(
        client, img_path, gt, dataset_root,
        det_type="tp", action="edited", edited_box=[10.0, 10.0, 30.0, 30.0], user="breeder",
    )
    assert resp.status_code == 200
    obj = json.loads(gt.read_text())["annotations"][0]
    assert obj["created_by"] == "user:breeder"


# ── Provenance round-trip fidelity (load → edit → save keeps the original creator) ──


def test_annotate_load_returns_provenance(client, dataset_root, tmp_path) -> None:
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    label_path = tmp_path / "det.json"
    write_annotations(str(label_path), [Annotation(
        subject="catkin", geometry=BBox(10, 10, 40, 40), created_by="derived:user:breeder",
        created_at="2026-02-11T00:00:00+00:00", accepted_by="user:breeder")], 100, 80)
    resp = client.get(
        "/api/annotate/labels",
        params={"image_path": str(img_path), "label_path": str(label_path)},
    )
    assert resp.status_code == 200
    a = resp.json()["annotations"][0]
    assert a["created_by"] == "derived:user:breeder"
    assert a["created_at"] == "2026-02-11T00:00:00+00:00"
    assert a["accepted_by"] == "user:breeder"


def test_annotate_resave_preserves_original_creator(client, dataset_root, tmp_path) -> None:
    """A re-save must not wholesale re-stamp loaded shapes to the current annotator: the
    original creator survives (keep-original-creator policy); only new shapes get stamped."""
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    label_path = tmp_path / "labels" / "IMG_0000.json"
    resp = client.post("/api/annotate/labels", json={
        "image_path": str(img_path), "label_path": str(label_path),
        "annotations": [
            {"subject": "catkin", "bbox": [10, 10, 40, 40],
             "created_by": "derived:user:breeder", "created_at": "2026-02-11T00:00:00+00:00",
             "accepted_by": "user:breeder"},
            {"subject": "catkin", "bbox": [50, 50, 70, 70]},
        ],
        "user": "emily",
    })
    assert resp.status_code == 200
    objs = json.loads(label_path.read_text())["annotations"]
    assert objs[0]["created_by"] == "derived:user:breeder"          # original creator kept
    assert objs[0]["created_at"] == "2026-02-11T00:00:00+00:00"  # original timestamp kept
    assert objs[0]["accepted_by"] == "user:breeder"                 # acceptance carried
    assert objs[1]["created_by"] == "user:emily"                 # only the new shape is Emily's


def test_annotate_polygons_keep_and_stamp_provenance(client, dataset_root, tmp_path) -> None:
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    label_path = tmp_path / "labels" / "IMG_0000.json"
    resp = client.post("/api/annotate/labels", json={
        "image_path": str(img_path), "label_path": str(label_path),
        "annotations": [
            {"subject": "catkin", "points": [[10, 10], [30, 10], [30, 30]],
             "created_by": "user:emily", "created_at": "2026-03-02T00:00:00+00:00"},
            {"subject": "catkin", "points": [[50, 50], [70, 50], [70, 70]]},
        ],
        "user": "breeder",
    })
    assert resp.status_code == 200
    objs = json.loads(label_path.read_text())["annotations"]
    assert objs[0]["created_by"] == "user:emily"   # round-tripped shape keeps its author
    assert objs[1]["created_by"] == "user:breeder"    # new polygon -> stamped to the current annotator


def test_review_subject_names_flow_from_annotations(
    client: TestClient, dataset_root: Path, tmp_path: Path
) -> None:
    """The recorded class name is the annotation's own subject: reviewing a catkin records
    'catkin', reviewing an efb records 'efb'; the name rides on the label, so one subject's name
    can never bleed onto another's (the bug a project-cached numeric-id engine would have)."""
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"

    def _class_name_for(subject: str) -> str:
        gt = tmp_path / f"gt_{subject}.json"
        _write_gt(gt, [(40, 32, 60, 48)], subject=subject)
        pred = tmp_path / f"pred_{subject}.json"
        _write_pred(pred, [(40, 32, 60, 48, 0.9)], subject=subject)
        resp = client.post(
            "/api/review/action",
            json={
                "dataset_root": str(dataset_root), "image_name": "IMG_0000.JPG",
                "image_path": str(img_path), "gt_path": str(gt),
                "pred_path": str(pred), "det_type": "tp", "class_name": subject, "conf": 0.9,
                "iou": 0.95, "gt_idx": 0, "pred_idx": 0,
                "bbox": [40.0, 32.0, 60.0, 48.0], "action": "accepted",
                "iou_threshold": 0.3, "conf_threshold": 0.1,
            },
        )
        assert resp.status_code == 200
        state = _shard_state(dataset_root / ".tcip" / "state", "IMG_0000.JPG")
        return state["detections"][0]["class_name"]

    assert _class_name_for("catkin") == "catkin"
    assert _class_name_for("efb") == "efb"  # different subject, its own name, no bleed
