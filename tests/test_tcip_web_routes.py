"""Integration tests for tcip-web HTTP routes (name-based per-image label schema)."""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import tcip_store
from tcip_annotation.json_io import read_label_document
from tcip_annotation.state import Annotation, BBox, Polygon
from tcip_mcp.pipelines.proposal import Proposal
from tcip_mcp.subject_registry import Subject, SubjectRegistry
from tcip_web import jobstore
from tcip_web.paths import safe_join
from tests import REFUSED_NAMES
from tests._audit_fixtures import audit_rows
from tests._producer_fixtures import (
    image_label_key,
    label_image,
    registry_over,
)
from tests._verified_checkpoint_fixtures import SAMPLE_DETECTOR_PASS

if TYPE_CHECKING:
    import threading
    from types import ModuleType

    from httpx import Response
    from pydantic import JsonValue

    from tcip_mcp.buckets import Bucket
    from tcip_web.routes.inference import InferenceJob
    from tcip_web.state import OpenProject

# ── per-image JSON label fixtures (canonical on-disk format) ─────────────────


def _write_gt(
    image: Path,
    boxes: list[tuple[float, float, float, float]],
    *,
    w: int = 100,
    h: int = 80,
    keep_empty: bool = False,
    subject: str = "bud",
) -> None:
    """Save the label document of ``image`` holding ``boxes`` of ``subject``.

    Each box is pixel-xyxy ``(x1, y1, x2, y2)``.
    """
    anns = [Annotation(subject=subject, geometry=BBox(*b)) for b in boxes]
    label_image(image, anns, w, h, keep_empty=keep_empty)


def _stored(image: Path) -> list[Annotation]:
    """Return the annotations of ``image``'s label document as stored."""
    return read_label_document(image_label_key(image)).annotations


def _unreadable(image: Path) -> None:
    """Write a label document for ``image`` whose stored bytes no longer decode."""
    from tests._record_damage_fixtures import damage_record

    _write_gt(image, [(1, 1, 3, 3)])
    damage_record(image_label_key(image), b"not json {][")


def _write_pred(
    dataset_root: Path,
    preds: dict[str, list[tuple]],
    *,
    w: int = 100,
    h: int = 80,
    subject: str = "bud",
    name: str = "baseline",
) -> Bucket:
    """Publish and return the bucket ``<name>/2-11-26`` of the project holding ``dataset_root``.

    It holds one document per image name of ``preds`` under ``images/2-11-26``, each entry
    ``(x1, y1, x2, y2, conf)`` a box labeled 1, scoped to ``subject``.
    """
    pytest.importorskip("torch")
    from tests._chain_fixtures import published

    return published(
        dataset_root.parent,
        f"{name}/2-11-26",
        [
            {
                "image": str(dataset_root / "images" / "2-11-26" / image),
                "width": w,
                "height": h,
                "boxes": [list(p[:4]) for p in boxes],
                "scores": [p[4] for p in boxes],
                "labels": [1] * len(boxes),
                "cap": len(boxes) + 1,
            }
            for image, boxes in preds.items()
        ],
        scope={"subject": subject},
    )


def _published_bucket(
    project: Path, name: str, image: Path, labels: list[int], *, scope: dict
) -> str:
    """Publish the bucket ``name`` under ``scope`` holding ``image``'s document; return its name.

    The document holds one box at ``(40, 32, 60, 48)`` per entry of ``labels``, labeled with
    that entry.
    """
    pytest.importorskip("torch")
    from tests._chain_fixtures import published

    published(
        project,
        name,
        [
            {
                "image": str(image),
                "width": 100,
                "height": 80,
                "boxes": [[40.0, 32.0, 60.0, 48.0]] * len(labels),
                "scores": [0.9] * len(labels),
                "labels": labels,
                "cap": len(labels) + 1,
            }
        ],
        scope=scope,
    )
    return name


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
    # The dataset's subjects come from its nested registry, not from its label documents.
    registry_over(
        root, SubjectRegistry((Subject("bud", ""), Subject("bush", "")))
    )
    # Add some images
    for i in range(3):
        img = Image.new("RGB", (100, 80), color=(128, 128, 128))
        img.save(root / "images" / "2-11-26" / f"IMG_{i:04d}.JPG")
    return root


def test_dataset_tree(opened_client: TestClient, dataset_root: Path) -> None:
    resp = opened_client.get(
        "/api/dataset/tree", params={"dataset_root": str(dataset_root)}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "2-11-26" in body["dates_with_images"]
    assert "3-2-26" in body["dates_with_images"]
    assert sorted(body["subjects"]) == ["bud", "bush"]
    # No bucket is published, so no date lists one.
    assert body["buckets_by_date"] == {"2-11-26": [], "3-2-26": []}
    # Per-date maps present for every image date (empty here: the registry declares subjects but
    # no label document exists yet).
    assert set(body["subjects_by_date"]) == {"2-11-26", "3-2-26"}
    assert body["subjects_by_date"]["2-11-26"] == []
    assert body["label_problem"] is None


def test_dataset_tree_reports_a_label_problem_and_keeps_listing_other_dates(
    opened_client: TestClient,
    dataset_root: Path,
) -> None:
    """The dataset tree names a corrupt label in ``label_problem`` and still lists both dates."""
    _unreadable(dataset_root / "images" / "2-11-26" / "IMG_0000.JPG")

    resp = opened_client.get(
        "/api/dataset/tree", params={"dataset_root": str(dataset_root)}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert (
        "2-11-26" in body["dates_with_images"]
        and "3-2-26" in body["dates_with_images"]
    )
    assert body["subjects_by_date"]["2-11-26"] == []
    assert body["subjects_by_date"]["3-2-26"] == []
    assert body["label_problem"] is not None
    assert "IMG_0000" in body["label_problem"]


def test_dataset_tree_label_problem_is_not_stale_after_an_edit(
    opened_client: TestClient,
    dataset_root: Path,
) -> None:
    """The tree's label_problem names a document damaged after the tree was first answered."""
    image = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    _write_gt(image, [(1, 1, 3, 3)])

    first = opened_client.get(
        "/api/dataset/tree", params={"dataset_root": str(dataset_root)}
    ).json()
    assert first["label_problem"] is None

    _unreadable(image)

    second = opened_client.get(
        "/api/dataset/tree", params={"dataset_root": str(dataset_root)}
    ).json()
    assert second["label_problem"] is not None
    assert "IMG_0000" in second["label_problem"]


def test_dataset_tree_per_date_reflects_actual_labels(
    opened_client: TestClient,
    tmp_path: Path,
    opened_project: Path,
) -> None:
    root = tmp_path / "ds"
    (root / "images" / "2026-02-11").mkdir(parents=True)
    (root / "images" / "2026-03-24").mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(root / "images" / "2026-02-11" / "IMG_1.JPG")
    Image.new("RGB", (8, 8)).save(root / "images" / "2026-03-24" / "IMG_2.JPG")
    # bud labeled + a bucket published over 02-11; nothing on 03-24.
    _write_gt(
        root / "images" / "2026-02-11" / "IMG_1.JPG", [(1, 1, 3, 3)], w=8, h=8
    )
    name = _published_bucket(
        opened_project,
        "baseline/2026-02-11",
        root / "images" / "2026-02-11" / "IMG_1.JPG",
        [1],
        scope={"subject": "bud"},
    )

    body = opened_client.get(
        "/api/dataset/tree", params={"dataset_root": str(root)}
    ).json()
    assert body["subjects_by_date"]["2026-02-11"] == ["bud"]
    assert body["subjects_by_date"]["2026-03-24"] == []
    assert body["buckets_by_date"] == {"2026-02-11": [name], "2026-03-24": []}


def test_dataset_select_carries_a_label_problem_with_no_subject_named(
    opened_client: TestClient,
    dataset_root: Path,
    opened_project: Path,
) -> None:
    """A subject-less selection carries the label problem of a corrupt label.

    A corrupt label makes the date's own subject list empty; the selection naming no subject
    still names the problem document.
    """
    _unreadable(dataset_root / "images" / "2-11-26" / "IMG_0000.JPG")

    resp = opened_client.post(
        "/api/dataset/select",
        json={
            "dataset_root": str(dataset_root),
            "subject": None,
            "date": "2-11-26",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["annotations_present"] is False
    assert body["label_problem"] is not None
    assert "IMG_0000" in body["label_problem"]


def test_dataset_select_populates_state(
    opened_client: TestClient,
    dataset_root: Path,
    opened_project: Path,
) -> None:
    resp = opened_client.post(
        "/api/dataset/select",
        json={
            "dataset_root": str(dataset_root),
            "subject": "bud",
            "date": "2-11-26",
        },
    )
    assert resp.status_code == 200
    sel = resp.json()["selection"]
    assert sel["subject"] == "bud"
    assert sel["date"] == "2-11-26"
    assert len(sel["image_list"]) == 3
    assert sel["image_list"][0].startswith("IMG_")
    assert sel["images_dir"].replace("\\", "/").endswith("images/2-11-26")


def test_dataset_select_returns_400_for_a_stem_collision(
    opened_client: TestClient, dataset_root: Path, opened_project: Path
) -> None:
    Image.new("RGB", (100, 80)).save(
        dataset_root / "images" / "2-11-26" / "IMG_0000.PNG"
    )

    resp = opened_client.post(
        "/api/dataset/select",
        json={
            "dataset_root": str(dataset_root),
            "subject": "bud",
            "date": "2-11-26",
        },
    )
    assert resp.status_code == 400


def test_dataset_select_advisory_reflects_actual_labels(
    opened_client: TestClient, dataset_root: Path, opened_project: Path
) -> None:
    body = {
        "dataset_root": str(dataset_root),
        "subject": "bud",
        "date": "2-11-26",
    }
    bucket = "baseline/2-11-26"
    # No label documents and no bucket yet → advisory says "starts empty"; a bucket name no
    # bucket is published under is refused rather than read as one holding nothing.
    r1 = opened_client.post("/api/dataset/select", json=body).json()
    assert r1["annotations_present"] is False
    assert r1["predictions_present"] is False
    refused = opened_client.post(
        "/api/dataset/select", json={**body, "bucket": bucket}
    )
    assert (
        refused.status_code == 400 and "no bucket" in refused.json()["detail"]
    )
    body["bucket"] = bucket

    # Save a real label and publish a bucket; the advisory flips to present (never rejects
    # either way).
    image = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    _write_gt(image, [(40, 32, 60, 48)])
    _published_bucket(
        opened_project, bucket, image, [1], scope={"subject": "bud"}
    )
    r2 = opened_client.post("/api/dataset/select", json=body).json()
    assert r2["annotations_present"] is True
    assert r2["predictions_present"] is True


def test_dataset_select_still_selects_over_an_unreadable_label(
    opened_client: TestClient, dataset_root: Path, opened_project: Path
) -> None:
    """A selection over an unreadable label still selects.

    The selection answers ok with no annotations present and names the label in
    ``label_problem``.
    """
    _unreadable(dataset_root / "images" / "2-11-26" / "IMG_0000.JPG")

    resp = opened_client.post(
        "/api/dataset/select",
        json={
            "dataset_root": str(dataset_root),
            "subject": "bud",
            "date": "2-11-26",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["annotations_present"] is False
    assert body["label_problem"] is not None
    assert "IMG_0000" in body["label_problem"]


def test_an_undated_captures_subjects_are_listed_by_every_door_alike(
    opened_client: TestClient,
    tmp_path: Path,
    opened_project: Path,
) -> None:
    """The tree, subjects route and core query list an undated capture's subjects.

    Each lists ``bud``; the selection of ``bud`` reports annotations present and lists the
    capture's image.
    """
    from tcip_mcp.dataset_layout import UNDATED_BUCKET, capture_subjects

    root = tmp_path / "undated_ds"
    (root / "images" / UNDATED_BUCKET).mkdir(parents=True)
    image = root / "images" / UNDATED_BUCKET / "IMG_F.JPG"
    Image.new("RGB", (100, 80)).save(image)
    _write_gt(image, [(40, 32, 60, 48)])

    tree = opened_client.get(
        "/api/dataset/tree", params={"dataset_root": str(root)}
    ).json()
    selected = opened_client.post(
        "/api/dataset/select",
        json={
            "dataset_root": str(root),
            "subject": "bud",
            "date": UNDATED_BUCKET,
        },
    ).json()
    loaded = opened_client.get(
        "/api/subjects/load",
        params={"dataset_root": str(root), "date": UNDATED_BUCKET},
    ).json()

    assert tree["subjects_by_date"] == {UNDATED_BUCKET: ["bud"]}, tree
    assert selected["annotations_present"] is True, selected
    assert selected["selection"]["image_list"] == ["IMG_F.JPG"], selected
    assert loaded["discovered"] == ["bud"], loaded
    assert capture_subjects(root, UNDATED_BUCKET) == (["bud"], [])


def test_a_selection_states_its_capture(
    opened_client: TestClient, tmp_path: Path, opened_project: Path
) -> None:
    root = tmp_path / "no_capture_named"
    (root / "images").mkdir(parents=True)

    resp = opened_client.post(
        "/api/dataset/select",
        json={"dataset_root": str(root), "subject": "bud"},
    )

    assert resp.status_code == 422, resp.text


def test_dataset_select_rejects_a_dataset_root_outside_the_allowed_roots(
    opened_client: TestClient,
    opened_project: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    outside = tmp_path_factory.mktemp("outside")
    resp = opened_client.post(
        "/api/dataset/select",
        json={"dataset_root": str(outside), "date": "2-11-26"},
    )
    assert resp.status_code == 403


def test_dataset_select_refuses_while_no_project_is_open(
    client: TestClient,
    dataset_root: Path,
) -> None:
    resp = client.post(
        "/api/dataset/select",
        json={"dataset_root": str(dataset_root), "date": "2-11-26"},
    )
    assert resp.status_code == 409


def test_dataset_nav_persists_current_index(
    opened_client: TestClient, dataset_root: Path, opened_project: Path
) -> None:
    opened_client.post(
        "/api/dataset/select",
        json={
            "dataset_root": str(dataset_root),
            "subject": "bud",
            "date": "2-11-26",
        },
    )
    # A valid position is accepted and shows up in the live GuiState the agent reads.
    ok = opened_client.post(
        "/api/dataset/nav", json={"current_image_index": 2}
    )
    assert ok.status_code == 200
    assert ok.json()["current_image_index"] == 2
    assert (
        opened_client.get("/api/state").json()["dataset"][
            "current_image_index"
        ]
        == 2
    )
    # Out of range (3 images → valid 0..2) is rejected, not silently clamped.
    assert (
        opened_client.post(
            "/api/dataset/nav", json={"current_image_index": 9}
        ).status_code
        == 400
    )


# ── /api/images ──────────────────────────────────────────────────────────


def _view(path: Path) -> dict:
    """Return an image request for ``path`` at a 3840x2160 display."""
    return {"path": str(path), "display_pixels": 3840 * 2160}


def test_images_serve(opened_client: TestClient, dataset_root: Path) -> None:
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    resp = opened_client.get("/api/images", params=_view(img_path))
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    im = Image.open(io.BytesIO(resp.content))
    assert im.size == (100, 80)


def test_images_downsample(
    opened_client: TestClient, dataset_root: Path
) -> None:
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    resp = opened_client.get(
        "/api/images", params={"path": str(img_path), "display_pixels": 2000}
    )
    im = Image.open(io.BytesIO(resp.content))
    assert im.size == (50, 40)


def test_images_etag_revalidation(
    opened_client: TestClient, dataset_root: Path
) -> None:
    # First fetch carries an ETag + Cache-Control; re-requesting with If-None-Match gets a
    # cheap 304 (no re-decode/re-encode), and the ETag varies with the render params.
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    first = opened_client.get("/api/images", params=_view(img_path))
    etag = first.headers.get("etag")
    assert etag and "cache-control" in first.headers

    again = opened_client.get(
        "/api/images", params=_view(img_path), headers={"If-None-Match": etag}
    )
    assert again.status_code == 304
    assert again.content == b""

    # A smaller display is a different variant -> different ETag -> full 200.
    variant = opened_client.get(
        "/api/images",
        params={"path": str(img_path), "display_pixels": 2000},
        headers={"If-None-Match": etag},
    )
    assert variant.status_code == 200
    assert variant.headers["etag"] != etag


def test_images_not_found(opened_client: TestClient, tmp_path: Path) -> None:
    resp = opened_client.get(
        "/api/images", params=_view(tmp_path / "does_not_exist.jpg")
    )
    assert resp.status_code == 404


def test_images_serve_returns_400_for_a_stem_collision(
    opened_client: TestClient, dataset_root: Path
) -> None:
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    Image.new("RGB", (100, 80)).save(
        dataset_root / "images" / "2-11-26" / "IMG_0000.PNG"
    )

    resp = opened_client.get("/api/images", params=_view(img_path))
    assert resp.status_code == 400
    assert "IMG_0000.JPG" in resp.text and "IMG_0000.PNG" in resp.text


def test_images_bands_returns_400_for_a_stem_collision(
    opened_client: TestClient, dataset_root: Path
) -> None:
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    Image.new("RGB", (100, 80)).save(
        dataset_root / "images" / "2-11-26" / "IMG_0000.PNG"
    )

    resp = opened_client.get(
        "/api/images/bands", params={"path": str(img_path)}
    )
    assert resp.status_code == 400


# ── /api/annotate ────────────────────────────────────────────────────────


def test_annotate_load_and_save_roundtrip(
    opened_client: TestClient, dataset_root: Path, tmp_path: Path
) -> None:
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"

    # Save a box annotation and a polygon annotation into the single per-image document.
    resp = opened_client.post(
        "/api/annotate/labels",
        json={
            "image_path": str(img_path),
            "annotations": [
                {"subject": "bud", "bbox": [10, 20, 50, 60]},
                {
                    "subject": "bud",
                    "points": [[5, 5], [10, 5], [10, 10], [5, 10]],
                },
            ],
            "user": "breeder",
        },
    )
    assert resp.status_code == 200
    assert len(_stored(img_path)) == 2

    # Load
    body = opened_client.get(
        "/api/annotate/labels",
        params={"image_path": str(img_path)},
    ).json()
    anns = body["annotations"]
    assert len(anns) == 2
    assert all(a["subject"] == "bud" for a in anns)
    # One carries a box, one a polygon (geometry kinds coexist in one file). The load side reports a
    # polygon as `rings`: a stored shape can be occlusion-split, so it is never flattened to one.
    assert sum("bbox" in a for a in anns) == 1
    assert sum("rings" in a for a in anns) == 1


def test_an_image_under_an_additive_image_root_loads_and_saves(
    opened_client: TestClient,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """An image under an additive image root loads and saves by its own label key.

    With the image's root configured as an additional image root, a save of one ``bud`` box
    answers 200, and both the load and the stored document hold it.
    """
    from tcip_web.state import store

    extra = tmp_path_factory.mktemp("additive") / "archive"
    image = extra / "images" / "2026-03-04" / "IMG_X.JPG"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (100, 80)).save(image)
    store.configure(store.workspace, (extra.resolve(),))

    saved = opened_client.post(
        "/api/annotate/labels",
        json={
            "image_path": str(image),
            "annotations": [{"subject": "bud", "bbox": [10, 20, 50, 60]}],
            "user": "breeder",
        },
    )
    loaded = opened_client.get(
        "/api/annotate/labels", params={"image_path": str(image)}
    )

    assert saved.status_code == 200, saved.text[:300]
    assert loaded.status_code == 200, loaded.text[:300]
    assert [a["subject"] for a in loaded.json()["annotations"]] == ["bud"]
    assert [a.subject for a in _stored(image)] == ["bud"]


def test_annotate_load_returns_400_for_a_stem_collision(
    opened_client: TestClient,
    dataset_root: Path,
    tmp_path: Path,
) -> None:
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    Image.new("RGB", (100, 80)).save(
        dataset_root / "images" / "2-11-26" / "IMG_0000.PNG"
    )

    resp = opened_client.get(
        "/api/annotate/labels",
        params={"image_path": str(img_path)},
    )
    assert resp.status_code == 400


def test_annotate_load_refuses_an_unreadable_label(
    opened_client: TestClient,
    dataset_root: Path,
    tmp_path: Path,
) -> None:
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    _unreadable(img_path)

    resp = opened_client.get(
        "/api/annotate/labels",
        params={"image_path": str(img_path)},
    )
    assert resp.status_code == 400
    assert "IMG_0000" in resp.json()["detail"]


def test_annotate_load_authorship_person_tool_and_unattributed(
    opened_client: TestClient,
    dataset_root: Path,
    tmp_path: Path,
) -> None:
    """The load route's authorship field classifies each record.

    A person's own created_by reads person, a bare producer with no accepted_by reads tool, and
    no created_by at all reads unattributed. The records are written through ``label_image``,
    the platform's own writer.
    """
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    anns = [
        Annotation(
            subject="bush",
            geometry=BBox(1, 1, 10, 10),
            created_by="user:breeder",
        ),
        Annotation(
            subject="bush", geometry=BBox(11, 11, 20, 20), created_by="sam"
        ),
        Annotation(subject="bush", geometry=BBox(21, 21, 30, 30)),
    ]
    label_image(img_path, anns, 100, 80)

    body = opened_client.get(
        "/api/annotate/labels",
        params={"image_path": str(img_path)},
    ).json()
    by_bbox = {tuple(a["bbox"]): a["authorship"] for a in body["annotations"]}
    assert by_bbox[(1.0, 1.0, 10.0, 10.0)] == "person"
    assert by_bbox[(11.0, 11.0, 20.0, 20.0)] == "tool"
    assert by_bbox[(21.0, 21.0, 30.0, 30.0)] == "unattributed"


def test_annotate_load_authorship_agrees_with_is_unadjudicated_agent_authorship(
    opened_client: TestClient,
    dataset_root: Path,
    tmp_path: Path,
) -> None:
    """The load route's authorship and ``is_unadjudicated_agent_authorship`` agree on each shape.

    For each of four writer-produced shapes, the load answers ``tool`` exactly where
    is_unadjudicated_agent_authorship answers true.
    """
    from tcip_annotation.json_io import is_unadjudicated_agent_authorship

    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    anns = [
        Annotation(
            subject="bush",
            geometry=BBox(1, 1, 10, 10),
            created_by="user:breeder",
        ),
        Annotation(
            subject="bush", geometry=BBox(11, 11, 20, 20), created_by="sam"
        ),
        Annotation(subject="bush", geometry=BBox(21, 21, 30, 30)),
        Annotation(
            subject="bush",
            geometry=BBox(31, 31, 40, 40),
            created_by="model:m1",
            accepted_by="user:breeder",
        ),
    ]
    label_image(img_path, anns, 100, 80)

    body = opened_client.get(
        "/api/annotate/labels",
        params={"image_path": str(img_path)},
    ).json()
    loaded = _stored(img_path)
    assert len(body["annotations"]) == len(loaded)
    for a_dict, a in zip(body["annotations"], loaded):
        assert (
            a_dict["authorship"] == "tool"
        ) == is_unadjudicated_agent_authorship(a)


def test_annotate_load_authorship_tool_accepted_through_the_editor(
    opened_client: TestClient,
    dataset_root: Path,
    tmp_path: Path,
) -> None:
    """A model's proposal a person accepts through the editor reads tool_accepted.

    Its created_by travels into GT and accepted_by is the person's sign-off: an adjudicated
    tool proposal, not the person's own hand.
    """
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    _write_gt(img_path, [], keep_empty=True)
    bucket = _write_pred(
        dataset_root, {img_path.name: [(40, 32, 60, 48, 0.9)]}, subject="bush"
    )
    produced_by = (
        read_label_document(bucket.document_key(img_path.stem))
        .annotations[0]
        .created_by
    )
    assert str(produced_by).startswith("model:")

    resp = opened_client.post(
        "/api/annotate/labels",
        json={
            "image_path": str(img_path),
            "annotations": [],
            "user": "breeder",
            "bucket": bucket.name,
            "accept": [0],
        },
    )
    assert resp.status_code == 200, resp.text

    body = opened_client.get(
        "/api/annotate/labels",
        params={"image_path": str(img_path)},
    ).json()
    assert len(body["annotations"]) == 1
    assert body["annotations"][0]["created_by"] == produced_by
    assert body["annotations"][0]["authorship"] == "tool_accepted"


def test_annotate_save_empty_preserves_negative(
    opened_client: TestClient, dataset_root: Path, tmp_path: Path
) -> None:
    # Clearing all annotations and saving must keep the label document (an {"annotations": []}
    # record), not delete it: it becomes a confirmed negative once the image is completed.
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"

    opened_client.post(
        "/api/annotate/labels",
        json={
            "image_path": str(img_path),
            "annotations": [{"subject": "bud", "bbox": [10, 20, 50, 60]}],
            "user": "breeder",
        },
    )
    assert len(_stored(img_path)) == 1

    resp = opened_client.post(
        "/api/annotate/labels",
        json={
            "image_path": str(img_path),
            "annotations": [],
            "user": "breeder",
        },
    )
    assert resp.status_code == 200
    assert tcip_store.exists(image_label_key(img_path))
    assert _stored(img_path) == []


def test_annotate_save_of_an_image_outside_allowed_root_403(
    opened_client: TestClient, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """An image outside every allowed root is refused, and no document is written for it."""
    outside = (
        tmp_path_factory.mktemp("outside")
        / "images"
        / "2-11-26"
        / "IMG_0000.JPG"
    )
    outside.parent.mkdir(parents=True)
    Image.new("RGB", (100, 80)).save(outside)
    resp = opened_client.post(
        "/api/annotate/labels",
        json={
            "image_path": str(outside),
            "annotations": [],
            "user": "breeder",
        },
    )
    assert resp.status_code == 403
    assert not tcip_store.exists(image_label_key(outside))


def _save_box(
    opened_client: TestClient, img_path: Path, **extra: object
) -> Response:
    resp = opened_client.post(
        "/api/annotate/labels",
        json={
            "image_path": str(img_path),
            "annotations": [{"subject": "bud", "bbox": [10, 20, 50, 60]}],
            "user": "breeder",
            **extra,
        },
    )
    return resp


def test_annotate_save_refuses_a_token_the_document_has_moved_past(
    opened_client: TestClient, dataset_root: Path, tmp_path: Path
) -> None:
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    base = _save_box(opened_client, img_path).json()["base_mtime"]

    # A concurrent writer changes the document after our client loaded it.
    _write_gt(img_path, [], keep_empty=True)

    resp = opened_client.post(
        "/api/annotate/labels",
        json={
            "image_path": str(img_path),
            "annotations": [],
            "base_mtime": base,
            "user": "breeder",
        },
    )
    assert resp.status_code == 409
    assert resp.json()["detail"]["kind"] == "version_conflict"


def test_annotate_save_with_the_current_token_is_accepted(
    opened_client: TestClient, dataset_root: Path, tmp_path: Path
) -> None:
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    base = _save_box(opened_client, img_path).json()["base_mtime"]

    # No external change → the token still names the stored document → the save is accepted and
    # returns a fresh one.
    resp = _save_box(opened_client, img_path, base_mtime=base)
    assert resp.status_code == 200
    token = resp.json()["base_mtime"]
    assert token is not None
    # A string, not a number: the client only ever echoes it back, and a numeric token would be
    # rounded by the browser's JSON parse and mismatch on every save.
    assert isinstance(token, str)


def test_annotate_load_hands_back_a_token_its_own_save_accepts(
    opened_client: TestClient, dataset_root: Path, tmp_path: Path
) -> None:
    """A save accepts the token its load handed back, and a replay of that token is refused.

    The load and save pair is one compare-and-set over what the client was shown: a fresh
    load's token saves again.
    """
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    params = {"image_path": str(img_path)}

    loaded = opened_client.get("/api/annotate/labels", params=params).json()
    assert loaded["annotations"] == []
    assert (
        _save_box(
            opened_client, img_path, base_mtime=loaded["base_mtime"]
        ).status_code
        == 200
    )

    # That token said the document did not exist, so replaying it cannot overwrite what the first
    # save created; the token from a fresh load can.
    assert (
        _save_box(
            opened_client, img_path, base_mtime=loaded["base_mtime"]
        ).status_code
        == 409
    )
    reloaded = opened_client.get("/api/annotate/labels", params=params).json()
    assert (
        _save_box(
            opened_client, img_path, base_mtime=reloaded["base_mtime"]
        ).status_code
        == 200
    )


def test_annotate_save_without_a_token_still_writes(
    opened_client: TestClient, dataset_root: Path, tmp_path: Path
) -> None:
    """A caller that supplies no token skips the comparison and its write still lands."""
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    assert _save_box(opened_client, img_path).status_code == 200
    assert _save_box(opened_client, img_path).status_code == 200
    assert len(_stored(img_path)) == 1


def test_annotate_save_persists_polygon_as_polygon(
    opened_client: TestClient, dataset_root: Path, tmp_path: Path
) -> None:
    # A polygon annotation round-trips as a polygon (its points are the source of truth), never
    # collapsed to a box on disk. Image is 100x80.
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    resp = opened_client.post(
        "/api/annotate/labels",
        json={
            "image_path": str(img_path),
            "annotations": [
                {
                    "subject": "bud",
                    "points": [[10, 10], [30, 10], [30, 30], [10, 30]],
                }
            ],
            "user": "breeder",
        },
    )
    assert resp.status_code == 200
    anns = _stored(img_path)
    assert len(anns) == 1
    assert isinstance(anns[0].geometry, Polygon)
    # A hand-drawn contour is the one ring the canvas authored.
    assert anns[0].geometry.rings == [
        [(10.0, 10.0), (30.0, 10.0), (30.0, 30.0), (10.0, 30.0)]
    ]


def test_annotate_multi_ring_polygon_round_trips_through_the_route(
    opened_client: TestClient, dataset_root: Path, tmp_path: Path
) -> None:
    """An occlusion-split shape re-saved through the route keeps both its rings.

    The save stores one instance with both rings, and the load reports those ``rings`` back.
    """
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    rings = [
        [[10, 10], [30, 10], [30, 30], [10, 30]],
        [[60, 10], [80, 10], [80, 30], [60, 30]],
    ]
    resp = opened_client.post(
        "/api/annotate/labels",
        json={
            "image_path": str(img_path),
            "annotations": [{"subject": "bud", "rings": rings}],
            "user": "breeder",
        },
    )
    assert resp.status_code == 200

    stored = _stored(img_path)
    assert len(stored) == 1  # one instance, not one per contour
    assert stored[0].geometry.rings == [
        [tuple(map(float, p)) for p in r] for r in rings
    ]

    body = opened_client.get(
        "/api/annotate/labels", params={"image_path": str(img_path)}
    ).json()
    (ann,) = body["annotations"]
    assert ann["rings"] == rings


def test_annotate_save_prefers_rings_over_points_when_both_are_sent(
    opened_client: TestClient, dataset_root: Path, tmp_path: Path
) -> None:
    """A save that sends both `rings` and `points` stores the `rings`.

    A two-ring shape sent with its first ring as `points` is stored with both rings.
    """
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    rings = [[[10, 10], [30, 10], [30, 30]], [[60, 10], [80, 10], [80, 30]]]
    resp = opened_client.post(
        "/api/annotate/labels",
        json={
            "image_path": str(img_path),
            "annotations": [
                {"subject": "bud", "rings": rings, "points": rings[0]}
            ],
            "user": "breeder",
        },
    )
    assert resp.status_code == 200
    (stored,) = _stored(img_path)
    assert len(stored.geometry.rings) == 2


def test_annotate_save_persists_box_as_box(
    opened_client: TestClient, dataset_root: Path, tmp_path: Path
) -> None:
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    resp = opened_client.post(
        "/api/annotate/labels",
        json={
            "image_path": str(img_path),
            "annotations": [{"subject": "bud", "bbox": [50, 40, 70, 60]}],
            "user": "breeder",
        },
    )
    assert resp.status_code == 200
    anns = _stored(img_path)
    assert len(anns) == 1
    b = anns[0].geometry
    assert (b.x1, b.y1, b.x2, b.y2) == (
        50.0,
        40.0,
        70.0,
        60.0,
    )  # box, written as drawn


def test_annotate_save_audits_into_the_log_of_the_dataset_it_wrote(
    opened_client: TestClient,
    opened: OpenProject,
    dataset_root: Path,
    tmp_path: Path,
) -> None:
    """A label save audits into the log of the dataset it wrote.

    One save through the route and one through the tool leave one ``save_label_document`` line
    apiece in the dataset's log, with the same arguments but for the version, and none in the
    project's log.
    """
    from tcip_mcp.tools.annotation_tools import save_annotations
    from tests._web_fixtures import bound_to

    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    before = len(audit_rows(dataset_root))
    resp = _save_box(opened_client, img_path)
    assert resp.status_code == 200
    answer = save_annotations(
        bound_to(opened),
        str(img_path),
        annotations=[{"subject": "bud", "bbox": [1, 1, 5, 5]}],
        actor=None,
    )
    assert "error" not in answer, answer

    lines = audit_rows(dataset_root)[before:]
    assert [line["tool"] for line in lines] == ["save_label_document"] * 2
    assert [
        {k: v for k, v in line["arguments"].items() if k != "version"}
        for line in lines
    ] == [
        {
            "capture": "2-11-26",
            "stem": "IMG_0000",
            "n_annotations": 1,
            "accepted": [],
            "rejected": [],
            "confirmed": [],
            "complete": {},
            "flagged": [],
            "resolved": [],
        }
    ] * 2
    assert not any(
        e.get("tool") == "save_label_document" for e in audit_rows(tmp_path)
    )


def test_a_document_the_bucket_record_does_not_name_is_no_prediction_of_it(
    opened_client: TestClient,
    dataset_root: Path,
) -> None:
    """An image the bucket's record does not name has no document key in it; a named one has."""
    bucket = _write_pred(
        dataset_root, {"IMG_0000.JPG": [(40, 32, 60, 48, 0.9)]}
    )

    assert bucket.document_key("IMG_0000") is not None
    assert bucket.document_key("IMG_0001") is None


# ── /api/annotate/propose ────────────────────────────────────────────────


class _PromptedEngine:
    """An engine answering one ringless candidate and one with rings, keeping each prompt."""

    def __init__(self) -> None:
        self.prompts: list[dict[str, JsonValue]] = []

    def propose(self, image_path: str, **params: JsonValue) -> Proposal:
        self.prompts.append(params)
        return Proposal(
            [
                {
                    "candidate_id": 0,
                    "bbox": [10.0, 10.0, 30.0, 30.0],
                    "area": 400.0,
                    "rings": None,
                    "score": 0.8,
                },
                {
                    "candidate_id": 1,
                    "bbox": [40.0, 20.0, 60.0, 50.0],
                    "area": 600.0,
                    "rings": [
                        [
                            (40.0, 20.0),
                            (60.0, 20.0),
                            (60.0, 50.0),
                            (40.0, 50.0),
                        ]
                    ],
                    "score": 0.7,
                },
            ],
            None,
        )


@pytest.fixture
def engines(monkeypatch: pytest.MonkeyPatch) -> dict[str, _PromptedEngine]:
    """Register a :class:`_PromptedEngine` under each shipped engine's name, for this test only.

    The registry is a copy of the shipped one for the test's duration, so the shipped engines are
    back once it ends; the stubs are keyed by the name each stands in for.
    """
    from tcip_mcp.pipelines import proposal
    from tcip_mcp.pipelines.proposal_engines import (
        PromptableSegmenter,
        TextPromptedDetector,
    )

    monkeypatch.setattr(proposal, "_ENGINES", dict(proposal._ENGINES))
    stubs = {
        name: _PromptedEngine()
        for name in (PromptableSegmenter.name, TextPromptedDetector.name)
    }
    for name, stub in stubs.items():
        proposal.register_proposal_engine(name, stub)
    return stubs


_PHRASE: dict[str, JsonValue] = {"phrase": "open bud"}
"""A phrase, which the find door takes and the prompt door refuses."""


def _image_of(dataset_root: Path) -> Path:
    return dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"


def _propose(
    client: TestClient, image: Path, prompt: dict[str, JsonValue]
) -> Response:
    return client.post(
        "/api/annotate/propose",
        json={
            "image_path": str(image),
            "subject": "bud",
            "user": "breeder",
            "prompt": prompt,
        },
    )


def test_placed_points_and_drawn_boxes_prompt_the_segmenter_together_and_stage_every_candidate(
    opened_client: TestClient,
    dataset_root: Path,
    engines: dict[str, _PromptedEngine],
) -> None:
    """The points and boxes of one prompt reach the segmenter as one call, staged as one bucket.

    The segmenter is asked for its one best mask per point and per box; the answered bucket holds
    a box and a polygon of the subject, and the proposals route serves them by index.
    """
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.pipelines.proposal_engines import (
        PromptableSegmenter,
        SegmenterParams,
        TextPromptedDetector,
    )

    image = _image_of(dataset_root)
    resp = _propose(
        opened_client,
        image,
        {"points": [[20, 20], [30, 30]], "boxes": [[10, 10, 50, 40]]},
    )

    assert resp.status_code == 200, resp.text
    answer = resp.json()
    assert answer["proposal_count"] == 2
    (prompt,) = engines[PromptableSegmenter.name].prompts
    assert SegmenterParams.model_validate(prompt) == SegmenterParams(
        points=[(20.0, 20.0), (30.0, 30.0)],
        boxes=[(10.0, 10.0, 50.0, 40.0)],
        multimask_output=False,
    )
    assert engines[TextPromptedDetector.name].prompts == []
    bucket = read_bucket(dataset_root, answer["bucket"])
    key = bucket.document_key(image.stem)
    assert key is not None
    staged = read_label_document(key).annotations
    assert [a.subject for a in staged] == ["bud", "bud"]
    assert [type(a.geometry) for a in staged] == [BBox, Polygon]
    served = opened_client.get(
        "/api/annotate/proposals",
        params={"image_path": str(image), "bucket": answer["bucket"]},
    )
    assert served.status_code == 200, served.text
    assert [p["index"] for p in served.json()["proposals"]] == [0, 1]


def test_states_read_each_image_of_the_selected_capture_for_one_subject(
    opened_client: TestClient, dataset_root: Path
) -> None:
    """A labeled image reads partial, a marked one complete, the untouched one unannotated."""
    images = dataset_root / "images" / "2-11-26"
    _write_gt(images / "IMG_0000.JPG", [(10, 10, 30, 30)])
    opened_client.post(
        "/api/dataset/select",
        json={
            "dataset_root": str(dataset_root),
            "subject": "bud",
            "date": "2-11-26",
        },
    )
    marked = opened_client.post(
        "/api/annotate/labels",
        json={
            "image_path": str(images / "IMG_0001.JPG"),
            "annotations": [{"subject": "bud", "bbox": [10, 20, 50, 60]}],
            "complete": {"bud": True},
            "user": "breeder",
        },
    )
    assert marked.status_code == 200, marked.text[:300]

    states = opened_client.get(
        "/api/annotate/states", params={"subject": "bud"}
    )

    assert states.status_code == 200, states.text
    assert states.json()["states"] == {
        "IMG_0000.JPG": "partial",
        "IMG_0001.JPG": "complete",
        "IMG_0002.JPG": "unannotated",
    }


def test_stagings_list_each_staged_bucket_of_an_image_and_none_of_another(
    opened_client: TestClient,
    dataset_root: Path,
    engines: dict[str, _PromptedEngine],
) -> None:
    """Two prompts on one image stage two buckets, both listed for it and neither for the next."""
    image = _image_of(dataset_root)
    first = _propose(opened_client, image, {"points": [[20, 20]]}).json()
    second = _propose(
        opened_client, image, {"boxes": [[10, 10, 50, 40]]}
    ).json()
    other = dataset_root / "images" / "2-11-26" / "IMG_0001.JPG"

    listed = opened_client.get(
        "/api/annotate/stagings", params={"image_path": str(image)}
    )
    none = opened_client.get(
        "/api/annotate/stagings", params={"image_path": str(other)}
    )

    assert listed.status_code == 200, listed.text
    assert sorted(listed.json()["buckets"]) == sorted(
        [first["bucket"], second["bucket"]]
    )
    assert none.json()["buckets"] == []


def test_a_bucket_holding_no_document_for_an_image_serves_no_proposals(
    opened_client: TestClient,
    dataset_root: Path,
    engines: dict[str, _PromptedEngine],
) -> None:
    """A prompt's bucket belongs to one image; another image of the date serves an empty list."""
    staged = _propose(
        opened_client, _image_of(dataset_root), {"points": [[20, 20]]}
    ).json()
    other = dataset_root / "images" / "2-11-26" / "IMG_0001.JPG"

    served = opened_client.get(
        "/api/annotate/proposals",
        params={"image_path": str(other), "bucket": staged["bucket"]},
    )

    assert served.status_code == 200, served.text
    assert served.json()["proposals"] == []


def test_a_prompt_records_the_person_who_sent_it_on_both_audit_lines(
    opened_client: TestClient,
    dataset_root: Path,
    engines: dict[str, _PromptedEngine],
) -> None:
    """The proposal's and the publication's audit lines both name the person the prompt names."""
    from tcip_mcp.audit import ACTOR_KEY

    resp = _propose(
        opened_client, _image_of(dataset_root), {"points": [[20, 20]]}
    )

    assert resp.status_code == 200, resp.text
    actors = {
        tool: [row.get(ACTOR_KEY) for row in audit_rows(dataset_root, tool)]
        for tool in ("propose_annotations", "prediction_bucket_published")
    }
    assert actors == {
        "propose_annotations": ["user:breeder"],
        "prediction_bucket_published": ["user:breeder"],
    }


def _find(client: TestClient, image: Path, phrase: str) -> Response:
    return client.post(
        "/api/annotate/find",
        json={"image_path": str(image), "phrase": phrase, "user": "breeder"},
    )


def test_find_answers_the_detectors_boxes_under_its_libraries_defaults_and_stages_nothing(
    opened_client: TestClient,
    dataset_root: Path,
    engines: dict[str, _PromptedEngine],
) -> None:
    """The phrase reaches the detector with its libraries' defaults; its boxes come back unstaged.

    The boxes are prompts for the person to send on, so no bucket is published by a find.
    """
    from tcip_mcp.buckets import buckets_under
    from tcip_mcp.pipelines.proposal_engines import (
        DetectorParams,
        PromptableSegmenter,
        TextPromptedDetector,
        detector_defaults,
    )

    resp = _find(opened_client, _image_of(dataset_root), "open bud")

    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "boxes": [[10.0, 10.0, 30.0, 30.0], [40.0, 20.0, 60.0, 50.0]]
    }
    (prompt,) = engines[TextPromptedDetector.name].prompts
    assert DetectorParams.model_validate(prompt) == DetectorParams(
        phrase="open bud", **detector_defaults()
    )
    assert engines[PromptableSegmenter.name].prompts == []
    assert buckets_under(dataset_root) == []


def test_a_phrase_sent_as_a_prompt_is_refused_by_the_payload(
    opened_client: TestClient,
    dataset_root: Path,
    engines: dict[str, _PromptedEngine],
) -> None:
    resp = _propose(opened_client, _image_of(dataset_root), _PHRASE)

    assert resp.status_code == 422
    assert "phrase" in resp.text
    assert all(stub.prompts == [] for stub in engines.values())


def test_a_prompt_naming_no_subject_or_two_kinds_is_refused_by_the_payload(
    opened_client: TestClient,
    dataset_root: Path,
    engines: dict[str, _PromptedEngine],
) -> None:
    """A prompt naming no subject or no user, or holding a point and a phrase, answers 422.

    None of them runs an engine.
    """
    image = str(_image_of(dataset_root))
    whole = {
        "image_path": image,
        "subject": "bud",
        "user": "breeder",
        "prompt": {"points": [[20, 20]]},
    }

    unnamed, anonymous = (
        opened_client.post(
            "/api/annotate/propose",
            json={k: v for k, v in whole.items() if k != unstated},
        )
        for unstated in ("subject", "user")
    )
    both = _propose(
        opened_client, Path(image), {"points": [[20, 20]], **_PHRASE}
    )

    assert [r.status_code for r in (unnamed, anonymous, both)] == [422] * 3
    assert "user" in anonymous.text
    assert all(stub.prompts == [] for stub in engines.values())


def test_a_second_prompt_on_one_image_stages_a_second_bucket_beside_the_first(
    opened_client: TestClient,
    dataset_root: Path,
    engines: dict[str, _PromptedEngine],
) -> None:
    """A second prompt on an image stages a bucket of its own; the first's document is unchanged."""
    from tcip_mcp.buckets import read_bucket

    image = _image_of(dataset_root)
    first = _propose(opened_client, image, {"points": [[20, 20]]}).json()
    key = read_bucket(dataset_root, first["bucket"]).document_key(image.stem)
    assert key is not None
    written = tcip_store.read_versioned(key).version

    second = _propose(opened_client, image, {"boxes": [[10, 10, 50, 40]]})

    assert second.status_code == 200, second.text
    assert second.json()["bucket"] != first["bucket"]
    assert read_bucket(dataset_root, second.json()["bucket"]).documents
    assert tcip_store.read_versioned(key).version == written


class _PromptPlacedEngine:
    """An engine answering one ringless 10 px box whose left edge is the prompt's x."""

    def propose(self, image_path: str, **params: JsonValue) -> Proposal:
        from tcip_mcp.pipelines.proposal_engines import SegmenterParams

        stated = SegmenterParams.model_validate(params)
        x = (stated.points or [(b[0], b[1]) for b in stated.boxes])[0][0]
        return Proposal(
            [
                {
                    "candidate_id": 0,
                    "bbox": [x, 10.0, x + 10.0, 20.0],
                    "area": 100.0,
                    "rings": None,
                    "score": 0.9,
                }
            ],
            None,
        )


def test_a_prompt_overtaken_by_another_stages_its_own_run(
    opened_client: TestClient,
    dataset_root: Path,
    engines: dict[str, _PromptedEngine],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A prompt whose run is overtaken before it stages publishes its own run's geometry.

    A second prompt on the same image runs whole between the first prompt's two doors; each
    answered bucket holds the box at its own prompt's x.
    """
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.pipelines.proposal import register_proposal_engine
    from tcip_mcp.pipelines.proposal_engines import PromptableSegmenter
    from tcip_mcp.tools import proposal_tools
    from tcip_web.routes import annotate

    register_proposal_engine(PromptableSegmenter.name, _PromptPlacedEngine())
    image = _image_of(dataset_root)
    real_stage = proposal_tools.stage_proposals
    overtaking: list[annotate.ProposeAnswer] = []
    overtaken: list[bool] = []

    def stage_after_another_prompt(
        project: Path,
        image_path: str,
        *,
        assignments: proposal_tools.StagedAssignments,
        actor: str | None,
    ) -> dict:
        if not overtaken:
            overtaken.append(True)
            overtaking.append(
                annotate.propose(
                    annotate.ProposePayload.model_validate(
                        {
                            "image_path": str(image),
                            "subject": "bud",
                            "user": "breeder",
                            "prompt": {"points": [[60, 15]]},
                        }
                    )
                )
            )
        return real_stage(
            project, image_path, assignments=assignments, actor=actor
        )

    monkeypatch.setattr(
        proposal_tools, "stage_proposals", stage_after_another_prompt
    )
    first = _propose(opened_client, image, {"points": [[40, 15]]})

    assert first.status_code == 200, first.text

    def staged_x(bucket: str | None) -> float:
        assert bucket is not None
        key = read_bucket(dataset_root, bucket).document_key(image.stem)
        assert key is not None
        (staged,) = read_label_document(key).annotations
        assert isinstance(staged.geometry, BBox)
        return staged.geometry.x1

    assert staged_x(first.json()["bucket"]) == pytest.approx(40.0)
    assert staged_x(overtaking[0].bucket) == pytest.approx(60.0)


def test_an_engine_proposing_nothing_answers_no_bucket(
    opened_client: TestClient,
    dataset_root: Path,
    engines: dict[str, _PromptedEngine],
) -> None:
    from tcip_mcp.buckets import buckets_under
    from tcip_mcp.pipelines.proposal import register_proposal_engine
    from tcip_mcp.pipelines.proposal_engines import PromptableSegmenter

    class Empty:
        def propose(self, image_path: str, **params: JsonValue) -> Proposal:
            return Proposal([], None)

    register_proposal_engine(PromptableSegmenter.name, Empty())
    resp = _propose(
        opened_client, _image_of(dataset_root), {"points": [[5, 5]]}
    )

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"bucket": None, "proposal_count": 0}
    assert buckets_under(dataset_root) == []


def test_a_doors_error_answers_400_carrying_its_text(
    opened_client: TestClient,
    dataset_root: Path,
    engines: dict[str, _PromptedEngine],
) -> None:
    from tcip_mcp.pipelines.proposal import register_proposal_engine
    from tcip_mcp.pipelines.proposal_engines import PromptableSegmenter

    class Unloadable:
        def propose(self, image_path: str, **params: JsonValue) -> Proposal:
            raise FileNotFoundError("no checkpoint at the stated name")

    register_proposal_engine(PromptableSegmenter.name, Unloadable())
    resp = _propose(
        opened_client, _image_of(dataset_root), {"points": [[5, 5]]}
    )

    assert resp.status_code == 400
    assert "no checkpoint at the stated name" in resp.json()["detail"]


def _launch_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[str, str, str, ModuleType]:
    from tcip_mcp.dataset_layout import image_dir
    from tcip_web.routes import inference as inference_routes

    monkeypatch.setattr(inference_routes, "_worker", lambda job: None)

    dataset_root = tmp_path / "data"
    date = "2026-02-11"
    images = image_dir(dataset_root, date)
    images.mkdir(parents=True)
    Image.new("RGB", (100, 100), (110, 110, 110)).save(images / "img.png")
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")
    return str(ckpt), str(dataset_root), date, inference_routes


def _held_worker(
    monkeypatch: pytest.MonkeyPatch, inference_routes: ModuleType
) -> threading.Event:
    """Replace the inference worker with one that holds its job running; return the release.

    The job's status reads ``running`` until the returned event is set or five seconds pass,
    then ``completed``.
    """
    import threading

    event = threading.Event()

    def held(job: InferenceJob) -> None:
        job.status = "running"
        event.wait(timeout=5)
        job.status = "completed"

    monkeypatch.setattr(inference_routes, "_worker", held)
    return event


BUCKET = "baseline/2026-02-11"
"""The bucket every launch here names."""


def test_a_launch_into_a_bucket_that_exists_fails_its_job_and_writes_nothing(
    opened_client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A launch into a bucket that exists fails its job and leaves the bucket's version.

    The job ends failed saying a bucket is published once, the bucket's record keeps its
    version, and the dataset's log gains no line.
    """
    import tcip_store
    from tcip_mcp.dataset_layout import bucket_key
    from tcip_web.routes import inference
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    real_worker = inference._worker
    _ckpt, dataset_root, date, inference_routes = _launch_setup(
        tmp_path, monkeypatch
    )
    monkeypatch.setattr(inference_routes, "_worker", real_worker)
    ckpt = registered_checkpoint(tmp_path)
    _published_bucket(
        tmp_path,
        BUCKET,
        Path(dataset_root) / "images" / date / "img.png",
        [1],
        scope={"subject": "bud"},
    )
    before = tcip_store.read_versioned(
        bucket_key(dataset_root, BUCKET)
    ).version
    logged = len(audit_rows(Path(dataset_root)))

    resp = opened_client.post(
        "/api/inference/launch",
        json={
            "user": "tester",
            "checkpoint_path": ckpt,
            "dataset_root": dataset_root,
            "date": date,
            "bucket": BUCKET,
            "stated": {"tile": False, **SAMPLE_DETECTOR_PASS},
        },
    )

    assert resp.status_code == 200, resp.text
    job = inference_routes._get(resp.json()["job_id"])
    job.thread.join(60)
    assert job.status == "failed"
    assert "a bucket is published once" in job.error
    assert (
        tcip_store.read_versioned(bucket_key(dataset_root, BUCKET)).version
        == before
    )
    assert len(audit_rows(Path(dataset_root))) == logged


def test_inference_launch_leaves_the_bucket_for_its_publication_to_create(
    opened_client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tcip_store
    from tcip_mcp.dataset_layout import bucket_key

    ckpt, dataset_root, date, _inference_routes = _launch_setup(
        tmp_path, monkeypatch
    )

    resp = opened_client.post(
        "/api/inference/launch",
        json={
            "user": "tester",
            "checkpoint_path": ckpt,
            "dataset_root": dataset_root,
            "date": date,
            "bucket": BUCKET,
            "stated": {},
        },
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["bucket"] == BUCKET
    assert not tcip_store.exists(bucket_key(dataset_root, BUCKET))


def test_inference_launch_refuses_a_second_launch_while_the_first_still_writes(
    opened_client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second launch while the first job still writes is refused naming that job.

    The refusal names the first job's id, the date and the bucket; after the first job
    completes, a third launch is admitted.
    """
    import time

    ckpt, dataset_root, date, inference_routes = _launch_setup(
        tmp_path, monkeypatch
    )
    event = _held_worker(monkeypatch, inference_routes)

    first = opened_client.post(
        "/api/inference/launch",
        json={
            "user": "tester",
            "checkpoint_path": ckpt,
            "dataset_root": dataset_root,
            "date": date,
            "bucket": BUCKET,
            "stated": {},
        },
    )
    assert first.status_code == 200, first.text
    job_id = first.json()["job_id"]

    second = opened_client.post(
        "/api/inference/launch",
        json={
            "user": "tester",
            "checkpoint_path": ckpt,
            "dataset_root": dataset_root,
            "date": date,
            "bucket": BUCKET,
            "stated": {},
        },
    )
    assert second.status_code == 409, second.text
    detail = second.json()["detail"]
    assert detail == {
        "kind": "bucket_exists",
        "message": detail["message"],
        "date": date,
        "requested_bucket": BUCKET,
        "job_id": job_id,
    }

    event.set()
    job = inference_routes._get(job_id)
    for _ in range(100):
        if not jobstore.live(job):
            break
        time.sleep(0.05)
    assert job.status == "completed"

    third = opened_client.post(
        "/api/inference/launch",
        json={
            "user": "tester",
            "checkpoint_path": ckpt,
            "dataset_root": dataset_root,
            "date": date,
            "bucket": BUCKET,
            "stated": {},
        },
    )
    assert third.status_code == 200, third.text


def test_inference_launch_in_flight_check_resolves_a_differently_spelled_dataset_root(
    opened_client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The in-flight launch check finds the live job by resolved directory identity.

    A trailing-separator spelling of the same dataset root still names the first job.
    """
    import os
    import time

    ckpt, dataset_root, date, inference_routes = _launch_setup(
        tmp_path, monkeypatch
    )
    event = _held_worker(monkeypatch, inference_routes)

    first = opened_client.post(
        "/api/inference/launch",
        json={
            "user": "tester",
            "checkpoint_path": ckpt,
            "dataset_root": dataset_root,
            "date": date,
            "bucket": BUCKET,
            "stated": {},
        },
    )
    assert first.status_code == 200, first.text
    job_id = first.json()["job_id"]

    second = opened_client.post(
        "/api/inference/launch",
        json={
            "user": "tester",
            "checkpoint_path": ckpt,
            "dataset_root": dataset_root + os.sep,
            "date": date,
            "bucket": BUCKET,
            "stated": {},
        },
    )
    assert second.status_code == 409, second.text
    detail = second.json()["detail"]
    assert detail["kind"] == "bucket_exists"
    assert detail["job_id"] == job_id

    event.set()
    job = inference_routes._get(job_id)
    for _ in range(100):
        if not jobstore.live(job):
            break
        time.sleep(0.05)
    assert job.status == "completed"


def test_inference_launch_resolves_explicit_conf_source_from_the_payload(
    opened_client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller-stated conf travels on the launched job as stated."""
    from tests._verified_checkpoint_fixtures import SAMPLE_CONF

    ckpt, dataset_root, date, inference_routes = _launch_setup(
        tmp_path, monkeypatch
    )

    resp = opened_client.post(
        "/api/inference/launch",
        json={
            "user": "tester",
            "checkpoint_path": ckpt,
            "dataset_root": dataset_root,
            "date": date,
            "bucket": BUCKET,
            "stated": {
                "conf": SAMPLE_CONF,
                "rationale": "a sample reason the test states",
            },
        },
    )
    assert resp.status_code == 200, resp.text
    job = inference_routes._get(resp.json()["job_id"])
    assert job.stated.conf == SAMPLE_CONF


def test_inference_launch_carries_an_omitted_conf_as_unstated(
    opened_client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An omitted conf travels on the launched job as unstated."""
    ckpt, dataset_root, date, inference_routes = _launch_setup(
        tmp_path, monkeypatch
    )

    resp = opened_client.post(
        "/api/inference/launch",
        json={
            "user": "tester",
            "checkpoint_path": ckpt,
            "dataset_root": dataset_root,
            "date": date,
            "bucket": BUCKET,
            "stated": {},
        },
    )
    assert resp.status_code == 200, resp.text
    job = inference_routes._get(resp.json()["job_id"])
    assert job.stated.conf is None


# ── /api/state ───────────────────────────────────────────────────────────


def test_state_snapshot_available(opened_client: TestClient) -> None:
    resp = opened_client.get("/api/state")
    body = resp.json()
    # Minimal shape sanity
    assert "active_tab" in body
    assert "view" in body
    assert "dataset" in body


def test_state_tab_push_mutates_the_store(opened_client: TestClient) -> None:
    from tcip_mcp.web_client import TAB_NAMES

    assert "results" in TAB_NAMES
    resp = opened_client.post("/api/state/tab", json={"active_tab": "results"})
    assert resp.status_code == 200
    assert opened_client.get("/api/state").json()["active_tab"] == "results"
    opened_client.post("/api/state/tab", json={"active_tab": "annotate"})


def test_state_tab_push_rejects_unknown_tabs(
    opened_client: TestClient,
) -> None:
    before = opened_client.get("/api/state").json()["active_tab"]
    for retired in ("dashboard", "review"):
        resp = opened_client.post(
            "/api/state/tab", json={"active_tab": retired}
        )
        assert resp.status_code == 400
        assert retired in resp.json()["detail"]
    assert opened_client.get("/api/state").json()["active_tab"] == before


def test_state_socket_broadcasts_a_mutation_while_open(
    opened_client: TestClient,
) -> None:
    """A mutation made while ``/ws/state`` is connected pushes the replay's envelope shape.

    After a tab mutation the socket pushes a ``state_snapshot`` naming the new tab, its version
    one past the connect-time replay's.
    """
    with opened_client.websocket_connect("ws://127.0.0.1/ws/state") as ws:
        replay = ws.receive_json()
        assert replay["type"] == "state_snapshot"
        opened_client.post("/api/state/tab", json={"active_tab": "training"})
        pushed = ws.receive_json()
    assert pushed["type"] == "state_snapshot"
    assert pushed["state"]["active_tab"] == "training"
    assert pushed["version"] == replay["version"] + 1
    opened_client.post("/api/state/tab", json={"active_tab": "annotate"})


def test_dataset_state_route_is_retired(opened_client: TestClient) -> None:
    """``/api/dataset/state`` is not registered."""
    assert opened_client.get("/api/dataset/state").status_code == 404


# ── /api/fs (folder browser) ───────────────────────────────────────────────


def test_fs_list_directories(
    opened_client: TestClient, tmp_path: Path
) -> None:
    (tmp_path / "alpha").mkdir()
    (tmp_path / "beta" / "images").mkdir(
        parents=True
    )  # looks like a dataset root
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "afile.txt").write_text("x")
    # Windows system/recovery folders that clutter a picker, filtered out by name.
    (tmp_path / "$RECYCLE.BIN").mkdir()
    (tmp_path / "System Volume Information").mkdir()
    (tmp_path / "FOUND.000").mkdir()

    resp = opened_client.get("/api/fs/list", params={"path": str(tmp_path)})
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


def test_fs_list_404_for_non_dir(
    opened_client: TestClient, tmp_path: Path
) -> None:
    f = tmp_path / "x.txt"
    f.write_text("x")
    assert (
        opened_client.get("/api/fs/list", params={"path": str(f)}).status_code
        == 404
    )


def test_fs_list_is_unconfined_from_a_local_connection(
    opened_client: TestClient, tmp_path_factory: pytest.TempPathFactory
) -> None:
    # The picker route is unconfined on a connection from this machine (the default TestClient);
    # confinement on a routable arrival is covered by test_web_path_guard_permanent_on.py.
    outside = tmp_path_factory.mktemp("outside")
    (outside / "sub").mkdir()
    assert (
        opened_client.get(
            "/api/fs/list", params={"path": str(outside)}
        ).status_code
        == 200
    )


# ── native provenance (created_by / accepted_by) ─────────────────


def test_annotate_save_stamps_created_by(
    opened_client: TestClient, dataset_root: Path, tmp_path: Path
) -> None:
    """A human-drawn box is stamped created_by=user:<gui-user> + created_at."""
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    resp = opened_client.post(
        "/api/annotate/labels",
        json={
            "image_path": str(img_path),
            "annotations": [{"subject": "bud", "bbox": [50, 40, 70, 60]}],
            "user": "breeder",
        },
    )
    assert resp.status_code == 200
    obj = _stored(img_path)[0]
    assert obj.created_by == "user:breeder"
    assert obj.created_at


def test_annotate_save_polygon_stamps_author(
    opened_client: TestClient, dataset_root: Path, tmp_path: Path
) -> None:
    """A human-drawn polygon is stamped created_by=user:<gui-user> (not None)."""
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    resp = opened_client.post(
        "/api/annotate/labels",
        json={
            "image_path": str(img_path),
            "annotations": [
                {"subject": "bud", "points": [[10, 10], [30, 10], [30, 30]]}
            ],
            "user": "emily",
        },
    )
    assert resp.status_code == 200
    assert _stored(img_path)[0].created_by == "user:emily"


def test_annotate_save_naming_no_one_writes_nothing(
    opened_client: TestClient, dataset_root: Path
) -> None:
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    for name in REFUSED_NAMES:
        resp = opened_client.post(
            "/api/annotate/labels",
            json={
                "image_path": str(img_path),
                "annotations": [{"subject": "bud", "bbox": [50, 40, 70, 60]}],
                "user": name,
            },
        )
        assert resp.status_code == 400, (name, resp.text)
    assert not tcip_store.exists(image_label_key(img_path))


# ── Provenance round-trip fidelity (load → edit → save keeps the original creator) ──


def test_annotate_load_returns_provenance(
    opened_client: TestClient, dataset_root: Path, tmp_path: Path
) -> None:
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    label_image(
        img_path,
        [
            Annotation(
                subject="bud",
                geometry=BBox(10, 10, 40, 40),
                created_by="derived:user:breeder",
                created_at="2026-02-11T00:00:00+00:00",
                accepted_by="user:breeder",
            )
        ],
        100,
        80,
    )
    resp = opened_client.get(
        "/api/annotate/labels",
        params={"image_path": str(img_path)},
    )
    assert resp.status_code == 200
    a = resp.json()["annotations"][0]
    assert a["created_by"] == "derived:user:breeder"
    assert a["created_at"] == "2026-02-11T00:00:00+00:00"
    assert a["accepted_by"] == "user:breeder"


def test_annotate_resave_preserves_original_creator(
    opened_client: TestClient, dataset_root: Path, tmp_path: Path
) -> None:
    """A re-save keeps each loaded shape's original creator and stamps only new shapes.

    The loaded shape keeps its creator, creation time and acceptance; the added shape is
    created by the saving annotator.
    """
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    label_image(
        img_path,
        [
            Annotation(
                subject="bud",
                geometry=BBox(10, 10, 40, 40),
                created_by="derived:user:breeder",
                created_at="2026-02-11T00:00:00+00:00",
                accepted_by="user:breeder",
            )
        ],
        100,
        80,
    )
    loaded = opened_client.get(
        "/api/annotate/labels", params={"image_path": str(img_path)}
    ).json()["annotations"]
    resp = opened_client.post(
        "/api/annotate/labels",
        json={
            "image_path": str(img_path),
            "annotations": [
                *loaded,
                {"subject": "bud", "bbox": [50, 50, 70, 70]},
            ],
            "user": "emily",
        },
    )
    assert resp.status_code == 200
    objs = _stored(img_path)
    assert (
        objs[0].created_by == "derived:user:breeder"
    )  # original creator kept
    assert (
        objs[0].created_at == "2026-02-11T00:00:00+00:00"
    )  # original timestamp kept
    assert objs[0].accepted_by == "user:breeder"  # acceptance carried
    assert objs[1].created_by == "user:emily"  # only the new shape is Emily's


def test_annotate_resave_keeps_the_crowd_flag(
    opened_client: TestClient, dataset_root: Path, tmp_path: Path
) -> None:
    """A crowd region the load route hands the canvas comes back on save as a crowd region.

    The flag round-trips through the payload, and the shape without one saves without it.
    """
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    label_image(
        img_path,
        [
            Annotation(
                subject="bud", geometry=BBox(10, 10, 40, 40), iscrowd=True
            ),
            Annotation(subject="bud", geometry=BBox(50, 50, 70, 70)),
        ],
        100,
        100,
    )

    load = opened_client.get(
        "/api/annotate/labels", params={"image_path": str(img_path)}
    )
    loaded = load.json()
    assert [a["iscrowd"] for a in loaded["annotations"]] == [True, False]

    resp = opened_client.post(
        "/api/annotate/labels",
        json={
            "image_path": str(img_path),
            "annotations": loaded["annotations"],
            "base_mtime": loaded["base_mtime"],
            "user": "emily",
        },
    )
    assert resp.status_code == 200, resp.text
    assert [a.iscrowd for a in _stored(img_path)] == [True, False]


@pytest.mark.parametrize(
    "flag, status, crowd",
    [
        ("yes", 400, None),
        ("false", 400, None),
        ("0", 400, None),
        (2, 400, None),
        (None, 200, False),
        (True, 200, True),
        (1, 200, True),
    ],
    ids=["yes", "false_string", "zero_string", "two", "null", "true", "one"],
)
def test_annotate_save_reads_the_crowd_flag_through_the_decoders_check(
    opened_client: TestClient,
    dataset_root: Path,
    tmp_path: Path,
    flag: str | int | bool | None,
    status: int,
    crowd: bool | None,
) -> None:
    """The save route refuses a crowd flag that is a string or 2, and stores the others.

    A refused flag writes no label document; a null stores as no crowd, and ``True`` or ``1``
    as a crowd region.
    """
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    resp = opened_client.post(
        "/api/annotate/labels",
        json={
            "image_path": str(img_path),
            "annotations": [
                {"subject": "bud", "bbox": [10, 10, 40, 40], "iscrowd": flag}
            ],
            "user": "breeder",
        },
    )
    assert resp.status_code == status, resp.text
    if status == 400:
        assert "iscrowd" in resp.json()["detail"]
        assert not tcip_store.exists(image_label_key(img_path))
    else:
        assert [a.iscrowd for a in _stored(img_path)] == [crowd]


def test_the_mcp_read_and_the_web_load_project_an_annotation_alike(
    opened_client: TestClient, dataset_root: Path, tmp_path: Path
) -> None:
    """The MCP read and the web load project a writer-produced annotation alike.

    For a crowd box and an attributed polygon, the web load's records equal the MCP read's but
    for ``authorship`` and the document ``index``, which read ``person`` then ``unattributed``
    and 0 then 1.
    """
    from tcip_mcp.tools.annotation_tools import read_annotations as mcp_read

    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    label_image(
        img_path,
        [
            Annotation(
                subject="bud",
                geometry=BBox(10, 10, 40, 40),
                iscrowd=True,
                created_by="user:breeder",
                created_at="2026-02-11T00:00:00+00:00",
            ),
            Annotation(
                subject="bud",
                geometry=Polygon([[(50.0, 50.0), (70.0, 50.0), (70.0, 70.0)]]),
                attributes={"stage": "open"},
            ),
        ],
        100,
        100,
    )

    web = opened_client.get(
        "/api/annotate/labels", params={"image_path": str(img_path)}
    ).json()["annotations"]
    mcp = mcp_read(tmp_path, str(img_path))["labels"]["annotations"]

    editor_only = {"authorship", "index"}
    assert [
        {k: v for k, v in a.items() if k not in editor_only} for a in web
    ] == mcp
    assert [a["authorship"] for a in web] == ["person", "unattributed"]
    assert [a["index"] for a in web] == [0, 1]
    assert mcp[0]["iscrowd"] is True and mcp[1]["iscrowd"] is False


def test_annotate_polygons_keep_and_stamp_provenance(
    opened_client: TestClient, dataset_root: Path, tmp_path: Path
) -> None:
    img_path = dataset_root / "images" / "2-11-26" / "IMG_0000.JPG"
    label_image(
        img_path,
        [
            Annotation(
                subject="bud",
                geometry=Polygon([[(10.0, 10.0), (30.0, 10.0), (30.0, 30.0)]]),
                created_by="user:emily",
                created_at="2026-03-02T00:00:00+00:00",
            )
        ],
        100,
        80,
    )
    loaded = opened_client.get(
        "/api/annotate/labels", params={"image_path": str(img_path)}
    ).json()["annotations"]
    resp = opened_client.post(
        "/api/annotate/labels",
        json={
            "image_path": str(img_path),
            "annotations": [
                *loaded,
                {"subject": "bud", "points": [[50, 50], [70, 50], [70, 70]]},
            ],
            "user": "breeder",
        },
    )
    assert resp.status_code == 200
    objs = _stored(img_path)
    assert (
        objs[0].created_by == "user:emily"
    )  # round-tripped shape keeps its author
    # new polygon -> stamped to the current annotator
    assert objs[1].created_by == "user:breeder"
