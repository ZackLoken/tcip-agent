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
              subject: str = "bud") -> None:
    """Author a per-image JSON GT label; each box is pixel-xyxy ``(x1, y1, x2, y2)`` of ``subject``."""
    anns = [Annotation(subject=subject, geometry=BBox(*b)) for b in boxes]
    write_annotations(str(path), anns, w, h, keep_empty=keep_empty)


def _write_pred(path, preds, *, w: int = 100, h: int = 80, subject: str = "bud") -> None:
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
    (root / "predictions" / "baseline").mkdir(parents=True)
    # The dataset's subjects come from its nested registry, not from listing annotations/.
    write_registry(root / "classes.json", ClassRegistry((Subject("bud"), Subject("bush"))))
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
    assert sorted(body["subjects"]) == ["bud", "bush"]
    assert "baseline" in body["model_names"]
    # Per-date maps present for every image date (empty here: the registry declares subjects but
    # no label files exist yet).
    assert set(body["subjects_by_date"]) == {"2-11-26", "3-2-26"}
    assert body["subjects_by_date"]["2-11-26"] == []
    assert body["label_problem"] is None


def test_dataset_tree_reports_a_label_problem_and_keeps_listing_other_dates(
    client: TestClient, dataset_root: Path,
) -> None:
    """A corrupt label on one date costs that date's own subject list, never the whole tree: the
    other date's dates_with_images/subjects_by_date entries are unaffected."""
    bad = dataset_root / "annotations" / "2-11-26"
    bad.mkdir(parents=True)
    (bad / "IMG_0000.json").write_text("not json {][", encoding="utf-8")

    resp = client.get("/api/dataset/tree", params={"dataset_root": str(dataset_root)})
    assert resp.status_code == 200
    body = resp.json()
    assert "2-11-26" in body["dates_with_images"] and "3-2-26" in body["dates_with_images"]
    assert body["subjects_by_date"]["2-11-26"] == []
    assert body["subjects_by_date"]["3-2-26"] == []
    assert body["label_problem"] is not None
    assert str(bad / "IMG_0000.json") in body["label_problem"]


def test_dataset_tree_label_problem_is_not_stale_after_an_in_place_edit(
    client: TestClient, dataset_root: Path,
) -> None:
    """A label edited in place leaves its directory's own mtime untouched on most filesystems, so
    label_problem must never answer from a cached tree built before the edit."""
    ann = dataset_root / "annotations" / "2-11-26"
    ann.mkdir(parents=True)
    _write_gt(ann / "IMG_0000.json", [(1, 1, 3, 3)])

    first = client.get("/api/dataset/tree", params={"dataset_root": str(dataset_root)}).json()
    assert first["label_problem"] is None

    (ann / "IMG_0000.json").write_text("not json {][", encoding="utf-8")

    second = client.get("/api/dataset/tree", params={"dataset_root": str(dataset_root)}).json()
    assert second["label_problem"] is not None
    assert str(ann / "IMG_0000.json") in second["label_problem"]


def test_dataset_tree_per_date_reflects_actual_labels(client: TestClient, tmp_path: Path) -> None:
    root = tmp_path / "ds"
    (root / "images" / "2026-02-11").mkdir(parents=True)
    (root / "images" / "2026-03-24").mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(root / "images" / "2026-02-11" / "IMG_1.JPG")
    Image.new("RGB", (8, 8)).save(root / "images" / "2026-03-24" / "IMG_2.JPG")
    # bud labelled + baseline predicted on 02-11; nothing on 03-24. One file per image.
    det = root / "annotations" / "2026-02-11"
    det.mkdir(parents=True)
    _write_gt(det / "IMG_1.json", [(1, 1, 3, 3)], w=8, h=8)
    pdet = root / "predictions" / "baseline" / "2026-02-11"
    pdet.mkdir(parents=True)
    _write_pred(pdet / "IMG_1.json", [(1, 1, 3, 3, 0.9)], w=8, h=8)

    body = client.get("/api/dataset/tree", params={"dataset_root": str(root)}).json()
    assert body["subjects_by_date"]["2026-02-11"] == ["bud"]
    assert body["subjects_by_date"]["2026-03-24"] == []
    assert body["models_by_date"]["2026-02-11"] == ["baseline"]
    assert body["models_by_date"]["2026-03-24"] == []


def test_the_label_memo_serves_the_tree_the_registry_and_the_review_scan_alike(
    client: TestClient, dataset_root: Path, monkeypatch,
) -> None:
    """The dataset tree, the class registry's draft scan and the review batch all parse the
    same date's label files; each file's parse is paid once, not once per route."""
    import tcip_annotation.json_io as json_io

    ann = dataset_root / "annotations" / "2-11-26"
    ann.mkdir(parents=True)
    for i in range(5):
        _write_gt(ann / f"IMG_{i:04d}.json", [(1, 1, 3, 3)])

    calls = []
    real_build = json_io._annotations_of

    def _counting_build(data):
        calls.append(data)
        return real_build(data)

    monkeypatch.setattr(json_io, "_annotations_of", _counting_build)

    client.get("/api/dataset/tree", params={"dataset_root": str(dataset_root)})
    client.get(
        "/api/classes/load",
        params={"project_root": str(dataset_root), "annotations_dir": str(ann)},
    )
    client.get(
        "/api/review/image_statuses",
        params={"dataset_root": str(dataset_root), "gt_dir": str(ann)},
    )

    assert len(calls) == 5, "each of the 5 label files must be parsed exactly once, not per route"


def test_dataset_select_carries_a_label_problem_with_no_subject_named(
    client: TestClient, dataset_root: Path, tmp_path: Path,
) -> None:
    """A corrupt label makes the date's own subject list empty, which is exactly the date a
    subject-less default-open selects; the advisory must name the problem even then, not only
    when a subject happens to be named."""
    project = tmp_path / "proj"
    project.mkdir()
    ann = dataset_root / "annotations" / "2-11-26"
    ann.mkdir(parents=True)
    (ann / "IMG_0000.json").write_text("not json {][", encoding="utf-8")

    resp = client.post(
        "/api/dataset/select",
        json={
            "project_root": str(project), "dataset_root": str(dataset_root),
            "subject": None, "date": "2-11-26",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["annotations_present"] is False
    assert body["label_problem"] is not None
    assert str(ann / "IMG_0000.json") in body["label_problem"]


def test_dataset_select_populates_state(client: TestClient, dataset_root: Path, tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    resp = client.post(
        "/api/dataset/select",
        json={
            "project_root": str(project),
            "dataset_root": str(dataset_root),
            "subject": "bud",
            "date": "2-11-26",
            "model_name": "baseline",
        },
    )
    assert resp.status_code == 200
    sel = resp.json()["selection"]
    assert sel["subject"] == "bud"
    assert sel["date"] == "2-11-26"
    assert len(sel["image_list"]) == 3
    assert sel["image_list"][0].startswith("IMG_")
    # One label dir per date now (no subject/task segment).
    assert sel["annotations_dir"].replace("\\", "/").endswith("annotations/2-11-26")


def test_dataset_select_returns_400_for_a_stem_collision(
    client: TestClient, dataset_root: Path, tmp_path: Path
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    Image.new("RGB", (100, 80)).save(dataset_root / "images" / "2-11-26" / "IMG_0000.PNG")

    resp = client.post(
        "/api/dataset/select",
        json={
            "project_root": str(project),
            "dataset_root": str(dataset_root),
            "subject": "bud",
            "date": "2-11-26",
            "model_name": "baseline",
        },
    )
    assert resp.status_code == 400


def test_dataset_select_advisory_reflects_actual_labels(
    client: TestClient, dataset_root: Path, tmp_path: Path
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    body = {
        "project_root": str(project),
        "dataset_root": str(dataset_root),
        "subject": "bud",
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


def test_dataset_select_still_selects_over_an_unreadable_label(
    client: TestClient, dataset_root: Path, tmp_path: Path
) -> None:
    """The advisory check never blocks a selection: an unreadable label reads as advisory-absent
    rather than refusing the select outright."""
    project = tmp_path / "proj"
    project.mkdir()
    ann = dataset_root / "annotations" / "2-11-26"
    ann.mkdir(parents=True, exist_ok=True)
    (ann / "IMG_0000.json").write_text("not json {][", encoding="utf-8")

    resp = client.post(
        "/api/dataset/select",
        json={
            "project_root": str(project), "dataset_root": str(dataset_root),
            "subject": "bud", "date": "2-11-26", "model_name": "baseline",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["annotations_present"] is False
    assert body["label_problem"] is not None
    assert str(ann / "IMG_0000.json") in body["label_problem"]


def test_dataset_select_rejects_a_project_root_outside_the_allowed_roots(
    client: TestClient, dataset_root: Path, tmp_path_factory,
) -> None:
    outside = tmp_path_factory.mktemp("outside")
    resp = client.post(
        "/api/dataset/select",
        json={"project_root": str(outside), "dataset_root": str(dataset_root)},
    )
    assert resp.status_code == 403


def test_dataset_select_answers_the_binding_generation(
    client: TestClient, dataset_root: Path, tmp_path: Path,
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    resp = client.post(
        "/api/dataset/select",
        json={"project_root": str(project), "dataset_root": str(dataset_root)},
    )
    assert resp.status_code == 200
    assert isinstance(resp.json()["generation"], int)


def test_dataset_select_generation_bumps_only_when_the_root_changes(
    client: TestClient, dataset_root: Path, tmp_path: Path,
) -> None:
    """Generation stability: re-selecting the same root and navigating dates/subjects on it
    bump nothing; a different root bumps."""
    project = tmp_path / "proj"
    project.mkdir()
    other = tmp_path / "other"
    other.mkdir()

    g1 = client.post(
        "/api/dataset/select",
        json={"project_root": str(project), "dataset_root": str(dataset_root), "date": "2-11-26"},
    ).json()["generation"]

    same_root_navigated = client.post(
        "/api/dataset/select",
        json={
            "project_root": str(project), "dataset_root": str(dataset_root),
            "date": "3-2-26", "subject": "bud",
        },
    ).json()["generation"]
    assert same_root_navigated == g1

    same_root_reselected = client.post(
        "/api/dataset/select",
        json={"project_root": str(project), "dataset_root": str(dataset_root), "date": "2-11-26"},
    ).json()["generation"]
    assert same_root_reselected == g1

    different_root = client.post(
        "/api/dataset/select",
        json={"project_root": str(other), "dataset_root": str(dataset_root), "date": "2-11-26"},
    ).json()["generation"]
    assert different_root == g1 + 1


def test_dataset_select_answers_service_unavailable_on_a_busy_binding(
    client: TestClient, dataset_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The binding write states what it does on StoreBusy: 503, nothing else adopted."""
    import tcip_store as ts
    from tcip_web.routes import dataset as dataset_route
    from tcip_web.state import store

    def _busy(*_a, **_kw):
        key = dataset_route.canvas_open_binding_key()
        raise ts.StoreBusy((key,), key, 5.0)

    monkeypatch.setattr(dataset_route.ts, "transaction", _busy)
    project = tmp_path / "proj"
    project.mkdir()
    generation_before = store.binding_generation
    dataset_root_before = store.state.dataset.dataset_root
    resp = client.post(
        "/api/dataset/select",
        json={"project_root": str(project), "dataset_root": str(dataset_root)},
    )
    assert resp.status_code == 503
    assert store.project_root is None
    assert store.binding_generation == generation_before  # nothing was adopted
    assert store.state.dataset.dataset_root == dataset_root_before


def test_dataset_select_never_pairs_the_old_dataset_with_the_new_generation_on_a_mid_select_failure(
    client: TestClient, dataset_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The in-memory generation moves only immediately before the mutate it names: an
    exception raised while gathering the new selection (after the binding record has already
    moved to the new root) must leave the in-memory generation still naming whatever the
    in-memory dataset names, never a mismatched pair a connect-time replay could deliver."""
    from tcip_web.routes import dataset as dataset_route
    from tcip_web.state import store

    project = tmp_path / "proj"
    project.mkdir()
    first = client.post(
        "/api/dataset/select",
        json={"project_root": str(project), "dataset_root": str(dataset_root)},
    ).json()

    other_project = tmp_path / "other_proj"
    other_project.mkdir()

    def _boom(*_a, **_kw):
        raise OSError("simulated directory listing failure")

    monkeypatch.setattr(dataset_route, "list_logical_images", _boom)
    with pytest.raises(OSError):
        client.post(
            "/api/dataset/select",
            json={
                "project_root": str(other_project), "dataset_root": str(dataset_root),
                "date": "2-11-26",
            },
        )

    assert store.binding_generation == first["generation"]
    assert store.state.dataset.project_root == str(project)


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
            "subject": "bud",
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


def test_images_not_found(client: TestClient, tmp_path: Path) -> None:
    resp = client.get("/api/images", params={"path": str(tmp_path / "does_not_exist.jpg")})
    assert resp.status_code == 404


def test_images_serve_returns_400_for_a_stem_collision(client: TestClient, dataset_root: Path) -> None:
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    Image.new("RGB", (100, 80)).save(dataset_root / "images" / "2-11-26" / "IMG_0000.PNG")

    resp = client.get("/api/images", params={"path": str(img_path)})
    assert resp.status_code == 400
    assert "IMG_0000.JPG" in resp.text and "IMG_0000.PNG" in resp.text


def test_images_bands_returns_400_for_a_stem_collision(client: TestClient, dataset_root: Path) -> None:
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    Image.new("RGB", (100, 80)).save(dataset_root / "images" / "2-11-26" / "IMG_0000.PNG")

    resp = client.get("/api/images/bands", params={"path": str(img_path)})
    assert resp.status_code == 400


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
                {"subject": "bud", "bbox": [10, 20, 50, 60]},
                {"subject": "bud", "points": [[5, 5], [10, 5], [10, 10], [5, 10]]},
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
    assert all(a["subject"] == "bud" for a in anns)
    # One carries a box, one a polygon (geometry kinds coexist in one file). The load side reports a
    # polygon as `rings`: a stored shape can be occlusion-split, so it is never flattened to one.
    assert sum("bbox" in a for a in anns) == 1
    assert sum("rings" in a for a in anns) == 1


def test_annotate_load_returns_400_for_a_stem_collision(
    client: TestClient, dataset_root: Path, tmp_path: Path,
) -> None:
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    Image.new("RGB", (100, 80)).save(dataset_root / "images" / "2-11-26" / "IMG_0000.PNG")
    label_path = tmp_path / "labels" / "IMG_0000.json"

    resp = client.get(
        "/api/annotate/labels",
        params={"image_path": str(img_path), "label_path": str(label_path)},
    )
    assert resp.status_code == 400


def test_annotate_load_refuses_an_unreadable_label(
    client: TestClient, dataset_root: Path, tmp_path: Path,
) -> None:
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    label_path = tmp_path / "labels" / "IMG_0000.json"
    label_path.parent.mkdir(parents=True)
    label_path.write_text("not json {][", encoding="utf-8")

    resp = client.get(
        "/api/annotate/labels",
        params={"image_path": str(img_path), "label_path": str(label_path)},
    )
    assert resp.status_code == 400
    assert str(label_path) in resp.json()["detail"]


def test_annotate_load_authorship_person_tool_and_unattributed(
    client: TestClient, dataset_root: Path, tmp_path: Path,
) -> None:
    """The load route's authorship field classifies each record: a person's own created_by reads
    person, a bare producer with no accepted_by reads tool, and no created_by at all reads
    unattributed. Built through the platform's own writer, never hand-written JSON."""
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    label_path = tmp_path / "labels" / "IMG_0000.json"
    anns = [
        Annotation(subject="bush", geometry=BBox(1, 1, 10, 10), created_by="user:breeder"),
        Annotation(subject="bush", geometry=BBox(11, 11, 20, 20), created_by="sam"),
        Annotation(subject="bush", geometry=BBox(21, 21, 30, 30)),
    ]
    write_annotations(str(label_path), anns, 100, 80)

    body = client.get(
        "/api/annotate/labels",
        params={"image_path": str(img_path), "label_path": str(label_path)},
    ).json()
    by_bbox = {tuple(a["bbox"]): a["authorship"] for a in body["annotations"]}
    assert by_bbox[(1.0, 1.0, 10.0, 10.0)] == "person"
    assert by_bbox[(11.0, 11.0, 20.0, 20.0)] == "tool"
    assert by_bbox[(21.0, 21.0, 30.0, 30.0)] == "unattributed"


def test_annotate_load_authorship_agrees_with_is_unadjudicated_agent_authorship(
    client: TestClient, dataset_root: Path, tmp_path: Path,
) -> None:
    """authorship_of and is_unadjudicated_agent_authorship classify every shape the same way: one
    predicate, never two spellings of the agent-authorship rule."""
    from tcip_annotation.json_io import is_unadjudicated_agent_authorship

    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    label_path = tmp_path / "labels" / "IMG_0000.json"
    anns = [
        Annotation(subject="bush", geometry=BBox(1, 1, 10, 10), created_by="user:breeder"),
        Annotation(subject="bush", geometry=BBox(11, 11, 20, 20), created_by="sam"),
        Annotation(subject="bush", geometry=BBox(21, 21, 30, 30)),
        Annotation(subject="bush", geometry=BBox(31, 31, 40, 40),
                  created_by="model:m1", accepted_by="user:breeder"),
    ]
    write_annotations(str(label_path), anns, 100, 80)

    body = client.get(
        "/api/annotate/labels",
        params={"image_path": str(img_path), "label_path": str(label_path)},
    ).json()
    loaded = read_annotations(str(label_path))
    assert len(body["annotations"]) == len(loaded)
    for a_dict, a in zip(body["annotations"], loaded):
        assert (a_dict["authorship"] == "tool") == is_unadjudicated_agent_authorship(a)


def test_annotate_load_authorship_tool_accepted_through_review(
    client: TestClient, dataset_root: Path, tmp_path: Path,
) -> None:
    """A model's own prediction, once a reviewer accepts it into ground truth, reads
    tool_accepted: its created_by travels into GT and accepted_by is the reviewer's sign-off, so
    it is no longer an unadjudicated tool call but it is still not the reviewer's own hand."""
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    gt = tmp_path / "gt.json"
    write_annotations(str(gt), [], 100, 80, keep_empty=True)
    pred = tmp_path / "pred.json"
    write_annotations(
        str(pred),
        [Annotation(subject="bush", geometry=BBox(40, 32, 60, 48), score=0.9,
                   created_by="model:m1")],
        100, 80,
    )

    resp = _review_action(
        client, img_path, gt, dataset_root,
        pred_path=str(pred), det_type="fp", action="accepted",
    )
    assert resp.status_code == 200

    body = client.get(
        "/api/annotate/labels",
        params={"image_path": str(img_path), "label_path": str(gt)},
    ).json()
    assert len(body["annotations"]) == 1
    assert body["annotations"][0]["created_by"] == "model:m1"
    assert body["annotations"][0]["authorship"] == "tool_accepted"


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
            "annotations": [{"subject": "bud", "bbox": [10, 20, 50, 60]}],
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
            "annotations": [{"subject": "bud", "bbox": [10, 20, 50, 60]}],
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
            "annotations": [{"subject": "bud", "points": [[10, 10], [30, 10], [30, 30], [10, 30]]}],
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
        "annotations": [{"subject": "bud", "rings": rings}],
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
        "annotations": [{"subject": "bud", "rings": rings, "points": rings[0]}],
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
            "annotations": [{"subject": "bud", "bbox": [50, 40, 70, 60]}],
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


def test_review_matches_returns_400_for_a_stem_collision(
    client: TestClient, dataset_root: Path, tmp_path: Path,
) -> None:
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    Image.new("RGB", (100, 80)).save(dataset_root / "images" / "2-11-26" / "IMG_0000.PNG")
    gt = tmp_path / "gt.json"
    _write_gt(gt, [(40, 32, 60, 48)])
    pred = tmp_path / "pred.json"
    _write_pred(pred, [(40, 32, 60, 48, 0.9)])

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
    assert resp.status_code == 400


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
    assert body["unreadable"] == []


def test_review_image_statuses_batch_reports_an_unreadable_prediction(
    client: TestClient, dataset_root: Path, tmp_path: Path,
) -> None:
    """A corrupt prediction document costs its own stem, never the whole batch: the good stem
    still surfaces as a detection, and the bad one is named by its document's path in unreadable
    instead of silently reading as nothing to review."""
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir(parents=True)
    _write_pred(pred_dir / "IMG_0000.json", [(40, 32, 60, 48, 0.9)])
    bad = pred_dir / "IMG_0002.json"
    bad.write_text("not json {][", encoding="utf-8")

    resp = client.get(
        "/api/review/image_statuses",
        params={"dataset_root": str(dataset_root), "pred_dir": str(pred_dir)},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["detection_stems"] == ["IMG_0000"]
    assert body["unreadable"] == [str(bad)]


def test_review_image_statuses_checks_a_prediction_even_when_gt_already_has_objects(
    client: TestClient, dataset_root: Path, tmp_path: Path,
) -> None:
    """A stem the GT directory already resolved to "has objects" must still have its own
    prediction document opened: an unreadable prediction must not go unnoticed just because
    another directory already answered for the same stem."""
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "predictions"
    gt_dir.mkdir(parents=True)
    pred_dir.mkdir(parents=True)
    _write_gt(gt_dir / "IMG_0000.json", [(40, 32, 60, 48)])
    bad = pred_dir / "IMG_0000.json"
    bad.write_text("not json {][", encoding="utf-8")

    resp = client.get(
        "/api/review/image_statuses",
        params={"dataset_root": str(dataset_root), "gt_dir": str(gt_dir), "pred_dir": str(pred_dir)},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["unreadable"] == [str(bad)]


def test_review_image_statuses_admits_a_bucket_holding_sidecars(
    client: TestClient, dataset_root: Path, tmp_path: Path,
) -> None:
    """A prediction bucket's own provenance stamps are not label documents and must never be
    read as one, or enumerated as an image with nothing to review."""
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir(parents=True)
    _write_pred(pred_dir / "IMG_0000.json", [(40, 32, 60, 48, 0.9)])
    (pred_dir / "operating_point.json").write_text('{"conf": {"value": 0.5}}', encoding="utf-8")

    resp = client.get(
        "/api/review/image_statuses",
        params={"dataset_root": str(dataset_root), "pred_dir": str(pred_dir)},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["detection_stems"] == ["IMG_0000"]
    assert body["unreadable"] == []


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
            "class_name": "bud",
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
    write_annotations(str(pred), [Annotation(subject="bud", geometry=BBox(40, 32, 60, 48),
                                              score=0.9, attributes={"phenology_stage": "open"})],
                       100, 80)
    _write_operating_point_sidecar(
        pred_dir, {"checkpoint_sha256": "sha", "experiment_id": None,
                   "subject": "bud", "attribute": "phenology_stage",
                   "id_map": {"closed": 0, "open": 1}})

    resp = _review_action(
        client, img_path, gt, dataset_root,
        pred_path=str(pred), det_type="tp", class_name="open", action="accepted",
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
    write_annotations(str(pred), [Annotation(subject="bud", geometry=BBox(40, 32, 60, 48),
                                              score=0.9, attributes={"phenology_stage": "closed"})],
                       100, 80)
    _write_operating_point_sidecar(
        pred_dir, {"checkpoint_sha256": "sha", "experiment_id": None,
                   "subject": "bud", "attribute": "phenology_stage",
                   "id_map": {"closed": 0, "open": 1}})

    resp = _review_action(
        client, img_path, gt, dataset_root,
        pred_path=str(pred), det_type="tp", class_name="bud", action="accepted",
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
        pred_path=str(pred), det_type="tp", class_name="bud", action="accepted",
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
        "class_name": "bud",
        "gt_idx": 0,
        "pred_idx": 0,
        "bbox": [40.0, 32.0, 60.0, 48.0],
        "action": "accepted",
    }
    body.update(over)
    return client.post("/api/review/action", json=body)


def test_review_action_swept_records_verdict_without_mutating_gt(
    client: TestClient, dataset_root: Path, tmp_path: Path
) -> None:
    """A sweep attestation ("checked this image for missed objects, found none") records a
    verdict entry but writes nothing to ground truth: no geometry, no gt/pred index."""
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    gt = tmp_path / "gt.json"
    _write_gt(gt, [(40, 32, 60, 48)])
    pred = tmp_path / "pred.json"
    _write_pred(pred, [(40, 32, 60, 48, 0.9)])

    resp = _review_action(
        client, img_path, gt, dataset_root,
        pred_path=str(pred), det_type="sweep", class_name="", gt_idx=None, pred_idx=None,
        bbox=[0.0, 0.0, 100.0, 80.0], action="swept",
    )
    assert resp.status_code == 200
    state = _shard_state(dataset_root / ".tcip" / "state", "IMG_0000.JPG")
    assert state["detections"][0]["action"] == "swept"
    assert state["detections"][0]["gt_bbox_norm"] is None
    assert state["detections"][0]["pred_bbox_norm"] is None
    assert len(read_annotations(str(gt))) == 1  # unchanged from the pristine GT


def test_review_action_refuses_an_action_outside_the_declared_vocabulary(
    client: TestClient, dataset_root: Path, tmp_path: Path
) -> None:
    """An action the vocabulary doesn't declare is refused at the route, before anything is
    recorded or written."""
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    gt = tmp_path / "gt.json"
    _write_gt(gt, [(40, 32, 60, 48)])
    pred = tmp_path / "pred.json"
    _write_pred(pred, [(40, 32, 60, 48, 0.9)])

    resp = _review_action(
        client, img_path, gt, dataset_root,
        pred_path=str(pred), det_type="tp", class_name="bud", action="approved",
    )
    assert resp.status_code == 422
    assert not (dataset_root / ".tcip" / "state").exists()
    assert len(read_annotations(str(gt))) == 1  # unchanged


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
    assert len(anns) == 1 and anns[0].subject == "bud"


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
    """A single detection on the image: reviewing it flips the image to 'completed' (the only
    GUI path to that status) and leaves an audit-trail entry. That entry belongs beside the
    labels rather than in the project the breeder is working out of; the project root here is a
    genuinely different directory, so a log written there is a log in the wrong place rather
    than the same file under another name."""
    project_root = tmp_path / "proj"
    project_root.mkdir()
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
            "det_type": "tp", "class_name": "bud", "conf": 0.9, "iou": 0.95,
            "gt_idx": 0, "pred_idx": 0,
            "bbox": [40.0, 32.0, 60.0, 48.0], "action": "accepted",
            "iou_threshold": 0.3, "conf_threshold": 0.1,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["image_status"] == "completed"
    assert any(e.get("tool") == "gui_review_action" for e in _audit_entries(dataset_root))
    assert _audit_entries(project_root) == []


def test_review_mark_complete_and_audits(client: TestClient, tmp_path: Path) -> None:
    dataset_root = tmp_path / "data"

    resp = client.post(
        "/api/review/mark_complete",
        json={"dataset_root": str(dataset_root), "image_name": "IMG_9.JPG"},
    )
    assert resp.status_code == 200
    assert resp.json()["image_status"] == "completed"

    status = client.get(
        "/api/review/image_statuses",
        params={"dataset_root": str(dataset_root)},
    )
    assert status.json()["statuses"]["IMG_9.JPG"] == "completed"
    assert any(e.get("tool") == "gui_review_mark_complete" for e in _audit_entries(dataset_root))


def test_review_mark_complete_refuses_an_unreadable_gt(
    client: TestClient, dataset_root: Path, tmp_path: Path,
) -> None:
    gt = tmp_path / "gt.json"
    gt.write_text("not json {][", encoding="utf-8")

    resp = client.post(
        "/api/review/mark_complete",
        json={
            "dataset_root": str(dataset_root), "image_name": "IMG_0000.JPG",
            "gt_path": str(gt), "subject": "bud",
        },
    )
    assert resp.status_code == 400
    assert str(gt) in resp.json()["detail"]


def test_review_mark_complete_refusal_persists_nothing(
    client: TestClient, dataset_root: Path, tmp_path: Path,
) -> None:
    """A 400 on the unreadable GT read must leave the image exactly as it was: no review mark, no
    audit line, because a claim derived from a document nobody can read is a claim about nothing."""
    gt = tmp_path / "gt.json"
    gt.write_text("not json {][", encoding="utf-8")

    before = client.get(
        "/api/review/image_statuses",
        params={"dataset_root": str(dataset_root)},
    ).json()
    assert before["statuses"].get("IMG_0000.JPG", "not_started") == "not_started"

    resp = client.post(
        "/api/review/mark_complete",
        json={
            "dataset_root": str(dataset_root), "image_name": "IMG_0000.JPG",
            "gt_path": str(gt), "subject": "bud",
        },
    )
    assert resp.status_code == 400

    after = client.get(
        "/api/review/image_statuses",
        params={"dataset_root": str(dataset_root)},
    ).json()
    assert after["statuses"].get("IMG_0000.JPG", "not_started") == "not_started"
    assert _audit_entries(dataset_root) == []


def test_review_mark_complete_refuses_an_unreadable_prediction(
    client: TestClient, dataset_root: Path, tmp_path: Path,
) -> None:
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir(parents=True)
    (pred_dir / "IMG_0000.json").write_text("not json {][", encoding="utf-8")

    resp = client.post(
        "/api/review/mark_complete",
        json={
            "dataset_root": str(dataset_root), "image_name": "IMG_0000.JPG",
            "pred_dir": str(pred_dir),
        },
    )
    assert resp.status_code == 400


def test_review_action_records_subject_name_and_reviewer(
    client: TestClient, dataset_root: Path, tmp_path: Path
) -> None:
    # A prediction stores its subject name on disk, so the recorded verdict carries the real name
    # ("bud") directly: no registry lookup, no "class_{id}" placeholder.
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
            "det_type": "tp", "class_name": "bud", "conf": 0.9, "iou": 0.95,
            "gt_idx": 0, "pred_idx": 0,
            "bbox": [40.0, 32.0, 60.0, 48.0], "action": "accepted",
            "iou_threshold": 0.3, "conf_threshold": 0.1,
        },
    )
    assert resp.status_code == 200
    entry = _shard_state(state, "IMG_0000.JPG")["detections"][0]
    assert entry["class_name"] == "bud"  # real name, straight from the annotation's subject
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
    monkeypatch.setenv("TCIP_STATE_ROOT", str(platform_root))

    dataset_root = tmp_path / "data"
    date = "2026-02-11"
    images = image_dir(dataset_root, date)
    images.mkdir(parents=True)
    Image.new("RGB", (100, 100), (110, 110, 110)).save(images / "img.png")
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")

    out = prediction_dir(dataset_root, "baseline", date)
    out.mkdir(parents=True)
    write_annotations(out / "img.json", [], img_w=100, img_h=100, keep_empty=True)
    engine = ReviewEngine(dataset_root / ".tcip" / "state")
    ctx = ReviewContext(img_name="img.png", img_width=100, img_height=100,
                        preds=[Annotation(subject="bud", geometry=BBox(10.0, 10.0, 30.0, 30.0),
                                          score=0.9)])
    det = ReviewDetection(det_type="fp", class_name="bud", conf=0.9, iou=None, gt_idx=None,
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
    monkeypatch.setenv("TCIP_STATE_ROOT", str(platform_root))

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
    monkeypatch.setenv("TCIP_STATE_ROOT", str(platform_root))

    dataset_root = tmp_path / "data"
    date = "2026-02-11"
    images = image_dir(dataset_root, date)
    images.mkdir(parents=True)
    Image.new("RGB", (100, 100), (110, 110, 110)).save(images / "img.png")
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")
    return str(ckpt), str(dataset_root), date, inference_routes


def test_inference_launch_refuses_a_bucket_that_already_holds_a_document(
    client: TestClient, tmp_path: Path, monkeypatch,
) -> None:
    """A guard: at b2ac13e1 the route resolves with no refuse_documents keyword and admits a
    second publish beside a prior run's own documents; this launch must be refused by name."""
    from tcip_mcp.dataset_layout import prediction_dir
    from tcip_mcp.prediction_buckets import BucketHoldsDocuments

    ckpt, dataset_root, date, inference_routes = _launch_setup(tmp_path, monkeypatch)
    bucket = prediction_dir(Path(dataset_root), "baseline", date)
    bucket.mkdir(parents=True, exist_ok=True)
    doc_path = bucket / "img.json"
    write_annotations(doc_path, [], img_w=100, img_h=100, keep_empty=True)
    before_bytes = doc_path.read_bytes()
    before_jobs = inference_routes._list_jobs()
    expected_message = str(BucketHoldsDocuments("baseline", 1, "baseline@r2"))

    resp = client.post("/api/inference/launch", json={
        "checkpoint_path": ckpt, "dataset_root": dataset_root,
        "model_name": "baseline", "date": date,
    })

    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"] == {
        "kind": "bucket_holds_documents",
        "message": expected_message,
        "date": date,
        "requested_model_name": "baseline",
        "requested_output_dir": str(bucket),
        "document_stem_count": 1,
        "suggested_model_name": "baseline@r2",
        "suggested_output_dir": str(prediction_dir(Path(dataset_root), "baseline@r2", date)),
    }
    assert inference_routes._list_jobs() == before_jobs
    assert doc_path.read_bytes() == before_bytes
    assert _audit_entries(Path(dataset_root)) == []


def test_inference_launch_admits_the_suggested_fresh_bucket(
    client: TestClient, tmp_path: Path, monkeypatch,
) -> None:
    """Admits valid work: the bucket a document refusal suggests is itself free, and posting the
    identical launch with model_name set to it writes in place rather than refusing again."""
    from tcip_mcp.dataset_layout import prediction_dir

    ckpt, dataset_root, date, _inference_routes = _launch_setup(tmp_path, monkeypatch)
    bucket = prediction_dir(Path(dataset_root), "baseline", date)
    bucket.mkdir(parents=True, exist_ok=True)
    write_annotations(bucket / "img.json", [], img_w=100, img_h=100, keep_empty=True)

    resp = client.post("/api/inference/launch", json={
        "checkpoint_path": ckpt, "dataset_root": dataset_root,
        "model_name": "baseline@r2", "date": date,
    })

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["bucket_redirected"] is False
    assert Path(body["output_dir"]) == prediction_dir(Path(dataset_root), "baseline@r2", date)


def test_inference_launch_redirects_past_a_verdict_though_the_bucket_also_holds_a_document(
    client: TestClient, tmp_path: Path, monkeypatch,
) -> None:
    """Coverage of the verdict-first order: a bucket carrying both a verdict and a document still
    redirects rather than refusing on the document (the fixture leaves the real worker running,
    so nothing about a prediction pass is asserted here, only the route's own synchronous step)."""
    dataset_root, ckpt, date = _verdicted_launch_dataset(tmp_path, monkeypatch)

    resp = client.post("/api/inference/launch", json={
        "checkpoint_path": str(ckpt), "dataset_root": str(dataset_root),
        "model_name": "baseline", "date": date,
    })

    assert resp.status_code == 200, resp.text
    assert resp.json()["bucket_redirected"] is True


def test_inference_launch_admits_a_bucket_holding_only_a_stamp(
    client: TestClient, tmp_path: Path, monkeypatch,
) -> None:
    """Coverage of the document predicate's own edge: a bucket carrying an earlier run's stamp
    and no prediction document still runs in place (a cancel before the first image is the tab's
    own producer of this state); the worker is stubbed, so the stamp's own bytes are not asserted
    here."""
    from tcip_mcp.dataset_layout import image_dir, prediction_dir
    from tcip_mcp.pipelines.resolution import operating_point_stamp, write_sidecar

    ckpt, dataset_root, date, _inference_routes = _launch_setup(tmp_path, monkeypatch)
    bucket = prediction_dir(Path(dataset_root), "baseline", date)
    stamp = operating_point_stamp(
        {"conf": {"value": 0.5}},
        validated=False,
        validated_by=None,
        tile_size_validated=None,
        shippable_issues=[],
        id_map=None,
        subject="bud",
        attribute=None,
        trait=None,
        dataset_hash="abc123",
        checkpoint="baseline",
        checkpoint_sha256="f" * 64,
        experiment_id="exp_001",
        images_dir=str(image_dir(Path(dataset_root), date)),
        raster_path=None,
        produced_at="2026-01-01T00:00:00+00:00",
    )
    write_sidecar(bucket, stamp)

    resp = client.post("/api/inference/launch", json={
        "checkpoint_path": ckpt, "dataset_root": dataset_root,
        "model_name": "baseline", "date": date,
    })

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["bucket_redirected"] is False
    assert Path(body["output_dir"]) == bucket


def test_inference_launch_refuses_by_name_with_no_suggestion_when_every_variant_is_taken(
    client: TestClient, tmp_path: Path, monkeypatch,
) -> None:
    """A guard: at b2ac13e1 the route never passes refuse_documents, so three document-holding
    directories raise nothing and the launch writes in place; with the keyword on, an exhausted
    search at a lowered ceiling refuses by name with no suggestion."""
    import functools

    from tcip_mcp.dataset_layout import prediction_dir
    from tcip_mcp.prediction_buckets import resolve_writable_bucket

    ckpt, dataset_root, date, _inference_routes = _launch_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "tcip_mcp.prediction_buckets.resolve_writable_bucket",
        functools.partial(resolve_writable_bucket, max_variants=3),
    )
    for name in ("baseline", "baseline@r2", "baseline@r3"):
        bucket = prediction_dir(Path(dataset_root), name, date)
        bucket.mkdir(parents=True, exist_ok=True)
        write_annotations(bucket / "img.json", [], img_w=100, img_h=100, keep_empty=True)

    resp = client.post("/api/inference/launch", json={
        "checkpoint_path": ckpt, "dataset_root": dataset_root,
        "model_name": "baseline", "date": date,
    })

    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert detail["kind"] == "bucket_holds_documents"
    assert detail["suggested_model_name"] is None
    assert detail["suggested_output_dir"] is None


def test_inference_launch_refuses_a_second_launch_while_the_first_still_writes(
    client: TestClient, tmp_path: Path, monkeypatch,
) -> None:
    """A guard: at b2ac13e1 the route has no in-flight check, so a second launch of a job still
    writing resolves a fresh job over the same images rather than being refused by that job's
    own identity; admitted again once the first job is terminal."""
    import threading
    import time

    ckpt, dataset_root, date, inference_routes = _launch_setup(tmp_path, monkeypatch)
    event = threading.Event()

    def _wait_worker(job) -> None:
        job.status = "running"
        event.wait(timeout=5)
        job.status = "completed"

    monkeypatch.setattr(inference_routes, "_worker", _wait_worker)

    first = client.post("/api/inference/launch", json={
        "checkpoint_path": ckpt, "dataset_root": dataset_root,
        "model_name": "baseline", "date": date,
    })
    assert first.status_code == 200, first.text
    job_id = first.json()["job_id"]

    second = client.post("/api/inference/launch", json={
        "checkpoint_path": ckpt, "dataset_root": dataset_root,
        "model_name": "baseline", "date": date,
    })
    assert second.status_code == 409, second.text
    detail = second.json()["detail"]
    assert detail == {
        "kind": "bucket_in_flight",
        "message": detail["message"],
        "date": date,
        "requested_output_dir": first.json()["output_dir"],
        "job_id": job_id,
    }
    assert isinstance(detail["message"], str) and detail["message"]

    event.set()
    job = inference_routes._get(job_id)
    for _ in range(100):
        if job.status not in ("pending", "running"):
            break
        time.sleep(0.05)
    assert job.status == "completed"

    third = client.post("/api/inference/launch", json={
        "checkpoint_path": ckpt, "dataset_root": dataset_root,
        "model_name": "baseline", "date": date,
    })
    assert third.status_code == 200, third.text


def test_inference_launch_refuses_by_the_requested_name_though_the_redirected_bucket_moved(
    client: TestClient, tmp_path: Path, monkeypatch,
) -> None:
    """A guard: keyed on the requested (dataset_root, model_name, date) rather than the resolved
    path, so a second launch of a verdicted model and date whose first job redirected to @r2 and
    already wrote a document there is refused naming that job, never resolved past it (which the
    resolver would otherwise do, landing on @r3) into a second concurrent job."""
    import threading
    import time

    from tcip_web.routes import inference as inference_routes

    dataset_root, ckpt, date = _verdicted_launch_dataset(tmp_path, monkeypatch)
    event = threading.Event()

    def _wait_worker(job) -> None:
        job.status = "running"
        event.wait(timeout=5)
        job.status = "completed"

    monkeypatch.setattr(inference_routes, "_worker", _wait_worker)

    first = client.post("/api/inference/launch", json={
        "checkpoint_path": str(ckpt), "dataset_root": str(dataset_root),
        "model_name": "baseline", "date": date,
    })
    assert first.status_code == 200, first.text
    assert first.json()["bucket_redirected"] is True
    redirected_dir = Path(first.json()["output_dir"])
    write_annotations(redirected_dir / "img2.json", [], img_w=100, img_h=100, keep_empty=True)

    second = client.post("/api/inference/launch", json={
        "checkpoint_path": str(ckpt), "dataset_root": str(dataset_root),
        "model_name": "baseline", "date": date,
    })
    assert second.status_code == 409, second.text
    detail = second.json()["detail"]
    assert detail["kind"] == "bucket_in_flight"
    assert detail["job_id"] == first.json()["job_id"]

    event.set()
    job = inference_routes._get(first.json()["job_id"])
    for _ in range(100):
        if job.status not in ("pending", "running"):
            break
        time.sleep(0.05)


def test_inference_launch_resolves_explicit_conf_and_max_dets_source_from_the_payload(
    client: TestClient, tmp_path: Path, monkeypatch,
) -> None:
    """A caller-stated conf/max_dets equal to the platform default is recorded as stated on the
    job, which launch_inference threads into raw_operating_point via the worker to stamp
    'explicit'."""
    from tcip_mcp.pipelines.resolution import DEFAULT_CONF, DEFAULT_MAX_DETS

    ckpt, dataset_root, date, inference_routes = _launch_setup(tmp_path, monkeypatch)

    resp = client.post("/api/inference/launch", json={
        "checkpoint_path": ckpt, "dataset_root": dataset_root, "model_name": "baseline",
        "date": date, "conf": DEFAULT_CONF, "max_dets": DEFAULT_MAX_DETS,
    })
    assert resp.status_code == 200, resp.text
    job = inference_routes._get(resp.json()["job_id"])
    assert job.conf_stated is True
    assert job.max_dets_stated is True
    assert job.conf == DEFAULT_CONF
    assert job.max_dets == DEFAULT_MAX_DETS


def test_inference_launch_defaults_conf_and_max_dets_source_when_omitted(
    client: TestClient, tmp_path: Path, monkeypatch,
) -> None:
    """The rail must admit the ordinary, unstated launch: an omitted conf/max_dets still resolves
    to the platform default and is recorded as unstated on the job, never as stated."""
    from tcip_mcp.pipelines.resolution import DEFAULT_CONF, DEFAULT_MAX_DETS

    ckpt, dataset_root, date, inference_routes = _launch_setup(tmp_path, monkeypatch)

    resp = client.post("/api/inference/launch", json={
        "checkpoint_path": ckpt, "dataset_root": dataset_root, "model_name": "baseline",
        "date": date,
    })
    assert resp.status_code == 200, resp.text
    job = inference_routes._get(resp.json()["job_id"])
    assert job.conf_stated is False
    assert job.max_dets_stated is False
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


def test_state_socket_broadcasts_a_mutation_while_open(client: TestClient) -> None:
    """A mutation made while ``/ws/state`` is connected pushes the same envelope shape the
    connect-time replay sends: the broadcast and the replay build it from different inputs
    (a subscriber payload versus the store's own snapshot and version), and only this exercises
    the broadcast path."""
    with client.websocket_connect("ws://127.0.0.1/ws/state") as ws:
        replay = ws.receive_json()
        assert replay["type"] == "state_snapshot"
        client.post("/api/state/tab", json={"active_tab": "training"})
        pushed = ws.receive_json()
    assert pushed["type"] == "state_snapshot"
    assert pushed["state"]["active_tab"] == "training"
    assert pushed["version"] == replay["version"] + 1
    client.post("/api/state/tab", json={"active_tab": "annotate"})


def test_dataset_state_route_is_retired(client: TestClient) -> None:
    """``/api/dataset/state`` duplicated ``/api/state`` over the same singleton snapshot and had
    no browser caller; ``/api/state`` is the one route left over it."""
    assert client.get("/api/dataset/state").status_code == 404


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
        "annotations": [{"subject": "bud", "bbox": [50, 40, 70, 60]}],
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
        "annotations": [{"subject": "bud", "points": [[10, 10], [30, 10], [30, 30]]}],
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
        "annotations": [{"subject": "bud", "bbox": [50, 40, 70, 60]}],
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
        subject="bud", geometry=BBox(40, 32, 60, 48), score=0.9,
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
        subject="bud", geometry=BBox(10, 10, 40, 40), created_by="derived:user:breeder",
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
            {"subject": "bud", "bbox": [10, 10, 40, 40],
             "created_by": "derived:user:breeder", "created_at": "2026-02-11T00:00:00+00:00",
             "accepted_by": "user:breeder"},
            {"subject": "bud", "bbox": [50, 50, 70, 70]},
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
            {"subject": "bud", "points": [[10, 10], [30, 10], [30, 30]],
             "created_by": "user:emily", "created_at": "2026-03-02T00:00:00+00:00"},
            {"subject": "bud", "points": [[50, 50], [70, 50], [70, 70]]},
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
    """The recorded class name is the annotation's own subject: reviewing a bud records
    'bud', reviewing an efb records 'efb'; the name rides on the label, so one subject's name
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

    assert _class_name_for("bud") == "bud"
    assert _class_name_for("efb") == "efb"  # different subject, its own name, no bleed
