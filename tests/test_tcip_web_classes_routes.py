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
    # Confirmations are dataset-native (K13.5 slice 4): a write must locate dataset_root.
    post = client.post(
        "/api/classes/image_status",
        json={"project_root": str(tmp_path), "dataset_root": str(tmp_path),
              "image_name": "IMG_0001.JPG", "status": "complete", "subject": "catkin"},
    )
    assert post.status_code == 200
    resp = client.get(
        "/api/classes/image_status",
        params={"project_root": str(tmp_path), "dataset_root": str(tmp_path), "subject": "catkin"},
    )
    body = resp.json()
    assert body["statuses"]["IMG_0001.JPG"] == "complete"
    assert (tmp_path / ".tcip" / "state" / "image_status.json").is_file()


def test_image_status_rejects_invalid(client: TestClient, tmp_path: Path) -> None:
    resp = client.post(
        "/api/classes/image_status",
        json={"project_root": str(tmp_path), "dataset_root": str(tmp_path),
              "image_name": "x", "status": "bogus"},
    )
    assert resp.status_code == 400


def test_image_status_write_refuses_without_a_locatable_dataset(
    client: TestClient, tmp_path: Path
) -> None:
    """A write that can't locate dataset_root must fail loudly, not silently write nowhere anyone
    reads (mirrors save_classes' same refusal)."""
    resp = client.post(
        "/api/classes/image_status",
        json={"project_root": str(tmp_path), "image_name": "IMG_0001.JPG",
              "status": "complete", "subject": "catkin"},
    )
    assert resp.status_code == 400


def test_image_status_lands_at_dataset_root_not_an_unrelated_project(
    client: TestClient, tmp_path: Path
) -> None:
    """A project referencing a dataset elsewhere must confirm negatives INTO the dataset, not into
    the project's own private state — the cross-root case K13.5 slice 4 exists to fix."""
    project_root = tmp_path / "project"
    dataset_root = tmp_path / "shared_dataset"
    project_root.mkdir()
    dataset_root.mkdir()

    client.post(
        "/api/classes/image_status",
        json={"project_root": str(project_root), "dataset_root": str(dataset_root),
              "image_name": "IMG_0001.JPG", "status": "negative", "subject": "catkin"},
    )
    assert (dataset_root / ".tcip" / "state" / "image_status.json").is_file()
    assert not (project_root / ".tcip" / "state" / "image_status.json").is_file()


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


def test_save_classes_confines_dataset_root_to_allowed_roots(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    monkeypatch.setenv("TCIP_IMAGE_ROOTS", str(allowed))
    outside = tmp_path / "outside"
    outside.mkdir()
    resp = client.post(
        "/api/classes/save",
        json={"project_root": str(outside), "dataset_root": str(outside),
              "subjects": {"catkin": {"description": "a catkin"}}},
    )
    assert resp.status_code == 403


def test_image_status_confines_dataset_root_to_allowed_roots(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    monkeypatch.setenv("TCIP_IMAGE_ROOTS", str(allowed))
    outside = tmp_path / "outside"
    outside.mkdir()
    resp = client.post(
        "/api/classes/image_status",
        json={"project_root": str(outside), "dataset_root": str(outside),
              "image_name": "IMG_0001.JPG", "status": "complete", "subject": "catkin"},
    )
    assert resp.status_code == 403


def test_image_status_bulk_confines_dataset_root_to_allowed_roots(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    monkeypatch.setenv("TCIP_IMAGE_ROOTS", str(allowed))
    outside = tmp_path / "outside"
    outside.mkdir()
    resp = client.post(
        "/api/classes/image_status/bulk",
        json={"project_root": str(outside), "dataset_root": str(outside),
              "subject": "catkin", "statuses": {"A.JPG": "complete"}},
    )
    assert resp.status_code == 403


def test_get_image_status_confines_dataset_root_to_allowed_roots(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    monkeypatch.setenv("TCIP_IMAGE_ROOTS", str(allowed))
    outside = tmp_path / "outside"
    outside.mkdir()
    resp = client.get(
        "/api/classes/image_status",
        params={"project_root": str(outside), "dataset_root": str(outside), "subject": "catkin"},
    )
    assert resp.status_code == 403


def test_derive_image_status_confines_annotations_dir_to_allowed_roots(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    monkeypatch.setenv("TCIP_IMAGE_ROOTS", str(allowed))
    outside = tmp_path / "outside" / "annotations"
    outside.mkdir(parents=True)
    resp = client.post(
        "/api/classes/image_status/derive",
        json={"project_root": str(tmp_path), "annotations_dir": str(outside),
              "subject": "catkin", "image_list": ["IMG_A.JPG"]},
    )
    assert resp.status_code == 403


def test_unrestricted_deployment_is_unaffected_by_the_confinement_guard(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TCIP_IMAGE_ROOTS unset (the default) must stay a no-op — the rail must admit valid work,
    not only reject invalid work."""
    monkeypatch.delenv("TCIP_IMAGE_ROOTS", raising=False)
    resp = client.post(
        "/api/classes/image_status",
        json={"project_root": str(tmp_path), "dataset_root": str(tmp_path),
              "image_name": "IMG_0001.JPG", "status": "complete", "subject": "catkin"},
    )
    assert resp.status_code == 200


def _read_audit_entries(dataset_root: Path) -> list[dict]:
    path = dataset_root / ".tcip" / "audit.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_set_image_status_writes_a_dataset_scoped_audit_entry(
    client: TestClient, tmp_path: Path
) -> None:
    # image_status.json is dataset-native (K13.5 slice 4), so its audit trail is colocated with the
    # dataset rather than guessed into a project's own audit.jsonl — distinct roots here so the
    # assertion actually pins which root the entry followed, not just that one was written.
    project_root = tmp_path / "project"
    dataset_root = tmp_path / "shared_dataset"
    project_root.mkdir()
    dataset_root.mkdir()
    client.post(
        "/api/classes/image_status",
        json={"project_root": str(project_root), "dataset_root": str(dataset_root),
              "image_name": "IMG_0001.JPG", "status": "complete", "subject": "catkin"},
    )
    entries = _read_audit_entries(dataset_root)
    assert len(entries) == 1
    assert entries[0]["tool"] == "gui_set_image_status"
    assert entries[0]["source"] == "gui"
    assert entries[0]["arguments"]["image_name"] == "IMG_0001.JPG"
    assert entries[0]["arguments"]["status"] == "complete"
    assert _read_audit_entries(project_root) == []  # not the project's own audit.jsonl


def test_set_image_status_bulk_audit_entry_records_only_what_was_applied(
    client: TestClient, tmp_path: Path
) -> None:
    project_root = tmp_path / "project"
    dataset_root = tmp_path / "shared_dataset"
    project_root.mkdir()
    dataset_root.mkdir()
    client.post(
        "/api/classes/image_status/bulk",
        json={
            "project_root": str(project_root), "dataset_root": str(dataset_root), "subject": "catkin",
            "statuses": {"A.JPG": "complete", "B.JPG": "invalid_ignored"},
        },
    )
    entries = _read_audit_entries(dataset_root)
    assert len(entries) == 1
    assert entries[0]["tool"] == "gui_set_image_status_bulk"
    # B.JPG's status was invalid and never written — the audit entry must not claim it was.
    assert entries[0]["arguments"]["statuses"] == {"A.JPG": "complete"}
    assert _read_audit_entries(project_root) == []


def test_save_classes_writes_a_dataset_scoped_audit_entry(client: TestClient, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    dataset_root = tmp_path / "shared_dataset"
    project_root.mkdir()
    dataset_root.mkdir()
    client.post(
        "/api/classes/save",
        json={"project_root": str(project_root), "dataset_root": str(dataset_root),
              "subjects": {"catkin": {"description": "a catkin"}}},
    )
    entries = _read_audit_entries(dataset_root)
    assert len(entries) == 1
    assert entries[0]["tool"] == "gui_save_classes"
    assert entries[0]["arguments"]["n_subjects"] == 1
    assert _read_audit_entries(project_root) == []


def test_image_status_bulk_writes_no_audit_entry_when_every_status_is_invalid(
    client: TestClient, tmp_path: Path
) -> None:
    """No real write happened, so no audit entry should claim one did."""
    client.post(
        "/api/classes/image_status/bulk",
        json={"project_root": str(tmp_path), "dataset_root": str(tmp_path), "subject": "catkin",
              "statuses": {"A.JPG": "invalid_ignored"}},
    )
    assert _read_audit_entries(tmp_path) == []


def test_image_status_bulk(client: TestClient, tmp_path: Path) -> None:
    resp = client.post(
        "/api/classes/image_status/bulk",
        json={
            "project_root": str(tmp_path),
            "dataset_root": str(tmp_path),
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
        "/api/classes/image_status",
        params={"project_root": str(tmp_path), "dataset_root": str(tmp_path), "subject": "catkin"},
    ).json()
    assert loaded["statuses"]["A.JPG"] == "complete"
    assert loaded["statuses"]["B.JPG"] == "partial"
    assert "C.JPG" not in loaded["statuses"]  # invalid skipped
