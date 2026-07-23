"""Tests for the class-registry / per-image-status routes."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tcip_annotation.json_io import write_annotations
from tcip_annotation.state import Annotation, BBox
from tcip_web.app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _catkin(x1, y1, x2, y2, *, subject: str = "catkin") -> Annotation:
    return Annotation(subject=subject, geometry=BBox(x1, y1, x2, y2))


def test_load_empty_registry(client: TestClient, tmp_path: Path) -> None:
    resp = client.get("/api/classes/load", params={"project_root": str(tmp_path)})
    assert resp.status_code == 200
    assert resp.json() == {"subjects": {}}


def test_save_then_load_round_trip(client: TestClient, tmp_path: Path) -> None:
    # The registry is one nested classes.json in the DATASET; it travels with the image set.
    save = client.post(
        "/api/classes/save",
        json={
            "project_root": str(tmp_path),
            "dataset_root": str(tmp_path),
            "subjects": {
                "catkin": {
                    "description": "a hazelnut catkin",
                    "attributes": {
                        "elongation": {"type": "categorical", "values": ["dormant", "elongated"]}
                    },
                },
                "bush": {"description": "one hazelnut bush crown"},
            },
        },
    )
    assert save.status_code == 200
    assert save.json()["n_subjects"] == 2

    load = client.get(
        "/api/classes/load",
        params={"project_root": str(tmp_path), "dataset_root": str(tmp_path)},
    ).json()
    subjects = load["subjects"]
    assert set(subjects) == {"catkin", "bush"}
    assert subjects["catkin"]["attributes"]["elongation"]["values"] == ["dormant", "elongated"]
    # Lands in the dataset as one nested classes.json (no per-subject files, no numeric ids).
    on_disk = json.loads((tmp_path / "classes.json").read_text())
    assert set(on_disk) == {"catkin", "bush"}


def test_save_refuses_malformed_registry(client: TestClient, tmp_path: Path) -> None:
    """A malformed registry (here: an attribute with no ``values``) is refused, not silently
    written — a bad registry would assign ids over garbage."""
    r = client.post(
        "/api/classes/save",
        json={"project_root": str(tmp_path), "dataset_root": str(tmp_path),
              "subjects": {"catkin": {"attributes": {"elongation": {"type": "categorical"}}}}},
    )
    assert r.status_code == 400


def test_registry_holds_multiple_subjects(client: TestClient, tmp_path: Path) -> None:
    # catkin and bush each name their own object; the one nested registry keeps them distinct.
    save = client.post(
        "/api/classes/save",
        json={
            "project_root": str(tmp_path),
            "dataset_root": str(tmp_path),
            "subjects": {
                "catkin": {"description": "a catkin"},
                "bush": {"description": "a bush"},
            },
        },
    )
    assert save.status_code == 200

    load = client.get(
        "/api/classes/load",
        params={"project_root": str(tmp_path), "dataset_root": str(tmp_path)},
    ).json()
    assert load["subjects"]["catkin"]["description"] == "a catkin"
    assert load["subjects"]["bush"]["description"] == "a bush"
    assert (tmp_path / "classes.json").is_file()


def test_dataset_root_derived_from_annotation_dir(client: TestClient, tmp_path: Path) -> None:
    """A caller that passes only an annotations dir still lands the registry in the dataset."""
    ann = tmp_path / "annotations" / "2026-03-02"
    ann.mkdir(parents=True)
    r = client.post(
        "/api/classes/save",
        json={"project_root": str(tmp_path), "annotations_dir": str(ann),
              "subjects": {"catkin": {"description": "a catkin"}}},
    )
    assert r.status_code == 200
    assert (tmp_path / "classes.json").is_file()


def test_load_derives_subjects_from_labels_when_registry_absent(
    client: TestClient, tmp_path: Path
) -> None:
    # No saved registry, but labels exist → derive a provisional (detection-only) registry from the
    # subjects present, so the canvas never loads empty.
    ann = tmp_path / "annotations" / "d"
    ann.mkdir(parents=True)
    write_annotations(
        str(ann / "IMG_A.json"),
        [_catkin(50, 50, 60, 60), _catkin(20, 20, 30, 30, subject="bush")],
        100, 100,
    )
    load = client.get(
        "/api/classes/load",
        params={"project_root": str(tmp_path), "annotations_dir": str(ann)},
    ).json()
    assert set(load["subjects"]) == {"bush", "catkin"}


def test_image_status_round_trip(client: TestClient, tmp_path: Path) -> None:
    client.post(
        "/api/classes/image_status",
        json={"project_root": str(tmp_path), "image_name": "IMG_0001.JPG",
              "status": "complete", "subject": "catkin"},
    )
    resp = client.get(
        "/api/classes/image_status", params={"project_root": str(tmp_path), "subject": "catkin"}
    )
    body = resp.json()
    assert body["statuses"]["IMG_0001.JPG"] == "complete"


def test_image_status_rejects_invalid(client: TestClient, tmp_path: Path) -> None:
    resp = client.post(
        "/api/classes/image_status",
        json={"project_root": str(tmp_path), "image_name": "x", "status": "bogus"},
    )
    assert resp.status_code == 400


def test_derive_statuses_negatives_are_intentional(client: TestClient, tmp_path: Path) -> None:
    # A negative is intentional (completed + empty), never inferred from an empty file alone —
    # so an accidental empty file, or flipping through an image, doesn't become a training negative.
    ann = tmp_path / "annotations"
    ann.mkdir()
    write_annotations(str(ann / "IMG_A.json"), [_catkin(50, 50, 60, 60)], 100, 100)  # objects, not completed
    write_annotations(str(ann / "IMG_B.json"), [], 100, 100, keep_empty=True)  # present {"annotations": []}
    write_annotations(str(ann / "IMG_E.json"), [_catkin(50, 50, 60, 60)], 100, 100)  # objects, completed
    # IMG_C.json missing; IMG_D completed but has no file (empty)

    resp = client.post(
        "/api/classes/image_status/derive",
        json={
            "project_root": str(tmp_path),
            "annotations_dir": str(ann),
            "subject": "catkin",
            "image_list": ["IMG_A.JPG", "IMG_B.JPG", "IMG_C.JPG", "IMG_D.JPG", "IMG_E.JPG"],
            "complete_override": ["IMG_D.JPG", "IMG_E.JPG"],
        },
    )
    assert resp.json()["statuses"] == {
        "IMG_A.JPG": "partial",  # has objects, not completed
        "IMG_B.JPG": "unannotated",  # empty file alone is not a negative — needs review
        "IMG_C.JPG": "unannotated",  # no file
        "IMG_D.JPG": "negative",  # completed + empty → confirmed negative (intentional)
        "IMG_E.JPG": "complete",  # completed + has objects
    }


def test_derive_statuses_cache_invalidates_on_label_write(client: TestClient, tmp_path: Path) -> None:
    # The per-file label-JSON memo is keyed on mtime_ns — a write after a prior derive() call
    # must not serve the stale (pre-write) parse of the same path.
    ann = tmp_path / "annotations"
    ann.mkdir()
    label = ann / "IMG_A.json"
    write_annotations(str(label), [], 100, 100, keep_empty=True)
    os.utime(label, (1_000_000, 1_000_000))

    req = {
        "project_root": str(tmp_path),
        "annotations_dir": str(ann),
        "subject": "catkin",
        "image_list": ["IMG_A.JPG"],
        "complete_override": [],
    }
    first = client.post("/api/classes/image_status/derive", json=req).json()
    assert first["statuses"]["IMG_A.JPG"] == "unannotated"

    write_annotations(str(label), [_catkin(50, 50, 60, 60)], 100, 100)
    os.utime(label, (2_000_000, 2_000_000))  # force a distinct mtime_ns from the first write

    second = client.post("/api/classes/image_status/derive", json=req).json()
    assert second["statuses"]["IMG_A.JPG"] == "partial"


def test_load_derived_registry_cache_invalidates_on_label_write(
    client: TestClient, tmp_path: Path
) -> None:
    # Same memo, exercised through load_classes' label-derived subject list.
    ann = tmp_path / "annotations" / "d"
    ann.mkdir(parents=True)
    label = ann / "IMG_A.json"
    write_annotations(str(label), [_catkin(50, 50, 60, 60)], 100, 100)
    os.utime(label, (1_000_000, 1_000_000))

    params = {"project_root": str(tmp_path), "annotations_dir": str(ann)}
    first = client.get("/api/classes/load", params=params).json()
    assert set(first["subjects"]) == {"catkin"}

    write_annotations(
        str(label), [_catkin(50, 50, 60, 60), _catkin(20, 20, 30, 30, subject="bush")], 100, 100
    )
    os.utime(label, (2_000_000, 2_000_000))

    second = client.get("/api/classes/load", params=params).json()
    assert set(second["subjects"]) == {"bush", "catkin"}


def test_image_status_bulk(client: TestClient, tmp_path: Path) -> None:
    resp = client.post(
        "/api/classes/image_status/bulk",
        json={
            "project_root": str(tmp_path),
            "subject": "catkin",
            "statuses": {
                "A.JPG": "complete",
                "B.JPG": "partial",
                "C.JPG": "invalid_ignored",
            },
        },
    )
    assert resp.json()["n"] == 3
    loaded = client.get(
        "/api/classes/image_status", params={"project_root": str(tmp_path), "subject": "catkin"}
    ).json()
    assert loaded["statuses"]["A.JPG"] == "complete"
    assert loaded["statuses"]["B.JPG"] == "partial"
    assert "C.JPG" not in loaded["statuses"]  # invalid skipped
