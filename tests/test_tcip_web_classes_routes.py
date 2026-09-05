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
    return TestClient(app, base_url="http://127.0.0.1")


def _bud(x1, y1, x2, y2, *, subject: str = "bud") -> Annotation:
    return Annotation(subject=subject, geometry=BBox(x1, y1, x2, y2))


def _status_store_exists(dataset_root: Path) -> bool:
    """Whether that root holds a confirmed-negative store, asked of whichever backend is bound."""
    import tcip_store as ts
    from tcip_mcp.dataset_layout import image_status_key

    return ts.exists(image_status_key(dataset_root))


def test_load_empty_registry(client: TestClient, tmp_path: Path) -> None:
    resp = client.get("/api/classes/load", params={"project_root": str(tmp_path)})
    assert resp.status_code == 200
    assert resp.json() == {"subjects": {}, "version": None, "unreadable": []}


def test_save_then_load_round_trip(client: TestClient, tmp_path: Path) -> None:
    # The registry is one nested classes.json in the DATASET; it travels with the image set.
    save = client.post(
        "/api/classes/save",
        json={
            "project_root": str(tmp_path),
            "dataset_root": str(tmp_path),
            "subjects": {
                "bud": {
                    "description": "a currant bud",
                    "attributes": {
                        "opening": {"type": "categorical", "values": ["closed", "open"]}
                    },
                },
                "bush": {"description": "one currant bush crown"},
            },
            "version": None,
        },
    )
    assert save.status_code == 200
    assert save.json()["n_subjects"] == 2

    load = client.get(
        "/api/classes/load",
        params={"project_root": str(tmp_path), "dataset_root": str(tmp_path)},
    ).json()
    subjects = load["subjects"]
    assert set(subjects) == {"bud", "bush"}
    assert subjects["bud"]["attributes"]["opening"]["values"] == ["closed", "open"]
    # Lands in the dataset as one nested classes.json (no per-subject files, no numeric ids).
    on_disk = json.loads((tmp_path / "classes.json").read_text())
    assert set(on_disk) == {"bud", "bush"}


def test_save_refuses_an_empty_registry(client: TestClient, tmp_path: Path) -> None:
    """A registry write states subjects; it never clears them, at either door."""
    r = client.post(
        "/api/classes/save",
        json={"project_root": str(tmp_path), "dataset_root": str(tmp_path), "subjects": {},
              "version": None},
    )
    assert r.status_code == 400
    assert not (tmp_path / "classes.json").exists()


def test_save_refuses_dropping_a_declared_subject(client: TestClient, tmp_path: Path) -> None:
    """A stale browser posting a subset of the stored registry is refused by name, not silently
    written: the additive-only toolbar makes a drop arriving here a sign of staleness."""
    first = client.post(
        "/api/classes/save",
        json={"project_root": str(tmp_path), "dataset_root": str(tmp_path),
              "subjects": {"leaf": {"description": "one leaf"}, "bush": {"description": "b"}},
              "version": None},
    )
    assert first.status_code == 200

    dropped = client.post(
        "/api/classes/save",
        json={"project_root": str(tmp_path), "dataset_root": str(tmp_path),
              "subjects": {"bush": {"description": "b"}}, "version": first.json()["version"]},
    )
    assert dropped.status_code == 400
    assert "leaf" in dropped.text
    on_disk = json.loads((tmp_path / "classes.json").read_text())
    assert "leaf" in on_disk  # the refused write never landed


def test_load_returns_the_version_and_save_round_trips_it(
    client: TestClient, tmp_path: Path
) -> None:
    """The toolbar carries the version it loaded back into its next save, the shape that lets a
    stale browser be told apart from a caller with nothing to assert."""
    save = client.post(
        "/api/classes/save",
        json={"project_root": str(tmp_path), "dataset_root": str(tmp_path),
              "subjects": {"leaf": {"description": "one leaf"}}, "version": None},
    )
    assert save.status_code == 200

    load = client.get(
        "/api/classes/load",
        params={"project_root": str(tmp_path), "dataset_root": str(tmp_path)},
    ).json()
    assert load["version"]

    grown = client.post(
        "/api/classes/save",
        json={"project_root": str(tmp_path), "dataset_root": str(tmp_path),
              "subjects": {"leaf": {"description": "one leaf"}, "bush": {}},
              "version": load["version"]},
    )
    assert grown.status_code == 200
    assert grown.json()["version"]


def test_save_refuses_a_stale_version(client: TestClient, tmp_path: Path) -> None:
    """A save carrying a version the store has moved past since is refused with 409, naming the
    conflict, rather than silently overwriting a registry the browser never saw."""
    client.post(
        "/api/classes/save",
        json={"project_root": str(tmp_path), "dataset_root": str(tmp_path),
              "subjects": {"leaf": {"description": "one leaf"}}, "version": None},
    )
    stale_load = client.get(
        "/api/classes/load",
        params={"project_root": str(tmp_path), "dataset_root": str(tmp_path)},
    ).json()
    client.post(
        "/api/classes/save",
        json={"project_root": str(tmp_path), "dataset_root": str(tmp_path),
              "subjects": {"leaf": {"description": "one leaf"}, "bush": {}},
              "version": stale_load["version"]},
    )

    conflicted = client.post(
        "/api/classes/save",
        json={"project_root": str(tmp_path), "dataset_root": str(tmp_path),
              "subjects": {"leaf": {"description": "one leaf"}, "bush": {}, "tip": {}},
              "version": stale_load["version"]},
    )
    assert conflicted.status_code == 409


def test_save_with_a_null_version_refuses_over_a_registry_written_meanwhile(
    client: TestClient, tmp_path: Path
) -> None:
    """A null version names an absent registry, not an unconditional write: a browser that never
    loaded a registry still refuses when an agent has written one in the meantime."""
    from tcip_mcp.class_registry import read_registry
    from tcip_mcp.tools.annotation_tools import write_class_map

    result = write_class_map(str(tmp_path), {"leaf": {"description": "written by the agent"}})
    assert "error" not in result

    # Additive (keeps "leaf"), so only the version check can refuse this, not the by-name drop rail.
    resp = client.post(
        "/api/classes/save",
        json={"project_root": str(tmp_path), "dataset_root": str(tmp_path),
              "subjects": {"leaf": {"description": "written by the agent"},
                           "bush": {"description": "written by the browser"}},
              "version": None},
    )
    assert resp.status_code == 409
    assert read_registry(tmp_path / "classes.json").subject("bush") is None


def test_save_with_a_null_version_succeeds_over_a_still_absent_registry(
    client: TestClient, tmp_path: Path
) -> None:
    """A null version asserts the registry is still absent; over an actually absent one the write
    lands rather than being treated as unconditional."""
    resp = client.post(
        "/api/classes/save",
        json={"project_root": str(tmp_path), "dataset_root": str(tmp_path),
              "subjects": {"bush": {"description": "first write"}}, "version": None},
    )
    assert resp.status_code == 200


def test_save_refuses_malformed_registry(client: TestClient, tmp_path: Path) -> None:
    """A malformed registry (here: an attribute with no ``values``) is refused, not silently
    written: a bad registry would assign ids over garbage."""
    r = client.post(
        "/api/classes/save",
        json={"project_root": str(tmp_path), "dataset_root": str(tmp_path),
              "subjects": {"bud": {"attributes": {"opening": {"type": "categorical"}}}},
              "version": None},
    )
    assert r.status_code == 400


def test_registry_holds_multiple_subjects(client: TestClient, tmp_path: Path) -> None:
    # bud and bush each name their own object; the one nested registry keeps them distinct.
    save = client.post(
        "/api/classes/save",
        json={
            "project_root": str(tmp_path),
            "dataset_root": str(tmp_path),
            "subjects": {
                "bud": {"description": "a bud"},
                "bush": {"description": "a bush"},
            },
            "version": None,
        },
    )
    assert save.status_code == 200

    load = client.get(
        "/api/classes/load",
        params={"project_root": str(tmp_path), "dataset_root": str(tmp_path)},
    ).json()
    assert load["subjects"]["bud"]["description"] == "a bud"
    assert load["subjects"]["bush"]["description"] == "a bush"
    assert (tmp_path / "classes.json").is_file()


def test_dataset_root_derived_from_annotation_dir(client: TestClient, tmp_path: Path) -> None:
    """A caller that passes only an annotations dir still lands the registry in the dataset."""
    ann = tmp_path / "annotations" / "2026-03-02"
    ann.mkdir(parents=True)
    r = client.post(
        "/api/classes/save",
        json={"project_root": str(tmp_path), "annotations_dir": str(ann),
              "subjects": {"bud": {"description": "a bud"}}, "version": None},
    )
    assert r.status_code == 200
    assert (tmp_path / "classes.json").is_file()


def test_load_derives_subjects_from_labels_when_registry_absent(
    client: TestClient, tmp_path: Path
) -> None:
    # No saved registry, but labels exist -> derive a draft (detection-only) registry from the
    # subjects present, so the canvas never loads empty.
    ann = tmp_path / "annotations" / "d"
    ann.mkdir(parents=True)
    write_annotations(
        str(ann / "IMG_A.json"),
        [_bud(50, 50, 60, 60), _bud(20, 20, 30, 30, subject="bush")],
        100, 100,
    )
    load = client.get(
        "/api/classes/load",
        params={"project_root": str(tmp_path), "annotations_dir": str(ann)},
    ).json()
    assert set(load["subjects"]) == {"bush", "bud"}
    assert load["unreadable"] == []


def test_load_reports_an_unreadable_label_and_still_derives_the_rest(
    client: TestClient, tmp_path: Path
) -> None:
    """One corrupt label file costs its own name, never the whole draft-registry scan."""
    ann = tmp_path / "annotations" / "d"
    ann.mkdir(parents=True)
    write_annotations(str(ann / "IMG_A.json"), [_bud(50, 50, 60, 60)], 100, 100)
    (ann / "IMG_B.json").write_text("not json {][", encoding="utf-8")

    load = client.get(
        "/api/classes/load",
        params={"project_root": str(tmp_path), "annotations_dir": str(ann)},
    ).json()
    assert set(load["subjects"]) == {"bud"}
    assert load["unreadable"] == [str(ann / "IMG_B.json")]


def test_load_reports_an_unreadable_label_beside_a_saved_registry(
    client: TestClient, tmp_path: Path
) -> None:
    """A saved classes.json answers the subject list, but a corrupt label file under
    annotations_dir is still worth surfacing: the registry load must not stop scanning for
    unreadable documents just because a registry was found."""
    ann = tmp_path / "annotations" / "d"
    ann.mkdir(parents=True)
    write_annotations(str(ann / "IMG_A.json"), [_bud(50, 50, 60, 60)], 100, 100)
    (ann / "IMG_B.json").write_text("not json {][", encoding="utf-8")

    save = client.post(
        "/api/classes/save",
        json={"project_root": str(tmp_path), "dataset_root": str(tmp_path),
              "subjects": {"bud": {"description": "a bud"}}, "version": None},
    )
    assert save.status_code == 200

    load = client.get(
        "/api/classes/load",
        params={"project_root": str(tmp_path), "dataset_root": str(tmp_path),
                "annotations_dir": str(ann)},
    ).json()
    assert set(load["subjects"]) == {"bud"}
    assert load["unreadable"] == [str(ann / "IMG_B.json")]


def test_load_reports_the_guards_resolved_path_not_the_clients_spelling(
    client: TestClient, tmp_path: Path
) -> None:
    """A dotdot segment names the same directory once resolved; the unreadable path reported is
    the guard's resolved path, never the client's own unnormalized string."""
    ann = tmp_path / "annotations" / "d"
    ann.mkdir(parents=True)
    (ann / "IMG_B.json").write_text("not json {][", encoding="utf-8")
    (tmp_path / "annotations" / "sibling").mkdir()
    raw = str(tmp_path / "annotations" / "sibling" / ".." / "d")
    assert raw != str(ann)

    load = client.get(
        "/api/classes/load",
        params={"project_root": str(tmp_path), "annotations_dir": raw},
    ).json()
    assert load["unreadable"] == [str(ann.resolve() / "IMG_B.json")]


def test_cached_label_annotations_raises_on_a_read_failure_other_than_absence(
    tmp_path: Path, monkeypatch
) -> None:
    """Only a missing file derives an empty status; any other OSError reading its bytes (a
    permission error on a present file) is a read failure, not a fact about absence."""
    from tcip_web.label_annotations_cache import cached_label_annotations

    label = tmp_path / "a.json"
    write_annotations(str(label), [_bud(1, 1, 2, 2)], 10, 10)

    real_read_bytes = Path.read_bytes

    def _denied(self, *args, **kwargs):
        if self == label:
            raise PermissionError(f"denied: {self}")
        return real_read_bytes(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", _denied)

    from tcip_annotation.json_io import UnreadableLabelDocument

    with pytest.raises(UnreadableLabelDocument):
        cached_label_annotations(label)


def test_cached_label_annotations_detects_an_edit_that_lands_on_the_same_mtime(
    tmp_path: Path,
) -> None:
    """Two writes close enough together can land on the identical filesystem timestamp; a memo
    keyed on mtime alone would then serve the first write's parse for the second. Forcing that
    exact collision here (rather than hoping two real writes race into it) makes the guard
    reproducible."""
    from tcip_web.label_annotations_cache import cached_label_annotations

    label = tmp_path / "a.json"
    write_annotations(str(label), [_bud(1, 1, 2, 2)], 10, 10)
    os.utime(label, (1_000_000, 1_000_000))
    first = cached_label_annotations(label)
    assert len(first) == 1

    label.write_text("not json {][", encoding="utf-8")
    os.utime(label, (1_000_000, 1_000_000))  # identical mtime, different size

    from tcip_annotation.json_io import UnreadableLabelDocument

    with pytest.raises(UnreadableLabelDocument):
        cached_label_annotations(label)


def test_cached_label_annotations_detects_a_same_size_edit_that_lands_on_the_same_mtime(
    tmp_path: Path,
) -> None:
    """A same-size in-place edit (one subject renamed to an equal-length name) forced onto the
    same mtime as the write it replaces cannot hide behind an (mtime, size) fingerprint; the memo
    answers the edit's own content."""
    from tcip_web.label_annotations_cache import cached_label_annotations

    label = tmp_path / "a.json"
    write_annotations(str(label), [_bud(1, 1, 2, 2, subject="bud")], 10, 10)
    os.utime(label, (1_000_000, 1_000_000))
    first = cached_label_annotations(label)
    assert [a.subject for a in first] == ["bud"]

    write_annotations(str(label), [_bud(1, 1, 2, 2, subject="leafxx")], 10, 10)
    os.utime(label, (1_000_000, 1_000_000))  # identical mtime and byte count, different content

    second = cached_label_annotations(label)
    assert [a.subject for a in second] == ["leafxx"]


def test_cached_label_annotations_hands_out_the_same_records_on_a_hit(tmp_path: Path) -> None:
    """Every caller reading one path under one digest shares the same tuple of records; a memo
    hit is not a fresh parse."""
    from tcip_web.label_annotations_cache import cached_label_annotations

    label = tmp_path / "a.json"
    write_annotations(str(label), [_bud(1, 1, 2, 2)], 10, 10)

    first = cached_label_annotations(label)
    second = cached_label_annotations(label)
    assert first is second


def test_image_status_round_trip(client: TestClient, tmp_path: Path) -> None:
    # Confirmations are dataset-native: a write must locate dataset_root.
    post = client.post(
        "/api/classes/image_status",
        json={"project_root": str(tmp_path), "dataset_root": str(tmp_path),
              "image_name": "IMG_0001.JPG", "status": "complete", "subject": "bud"},
    )
    assert post.status_code == 200
    resp = client.get(
        "/api/classes/image_status",
        params={"project_root": str(tmp_path), "dataset_root": str(tmp_path), "subject": "bud"},
    )
    body = resp.json()
    assert body["statuses"]["IMG_0001.JPG"] == "complete"
    assert _status_store_exists(tmp_path)


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
              "status": "complete", "subject": "bud"},
    )
    assert resp.status_code == 400


def test_image_status_write_refuses_without_a_subject(
    client: TestClient, tmp_path: Path
) -> None:
    """A write with no subject must fail loudly rather than land in the "" bucket, which
    ``get_image_status`` never returns anything meaningful for."""
    resp = client.post(
        "/api/classes/image_status",
        json={"project_root": str(tmp_path), "dataset_root": str(tmp_path),
              "image_name": "IMG_0001.JPG", "status": "complete"},
    )
    assert resp.status_code == 400


def test_image_status_bulk_write_refuses_without_a_subject(
    client: TestClient, tmp_path: Path
) -> None:
    """A bulk write with no subject must fail loudly rather than land in the "" bucket, which
    ``get_image_status`` never returns anything meaningful for."""
    resp = client.post(
        "/api/classes/image_status/bulk",
        json={"project_root": str(tmp_path), "dataset_root": str(tmp_path),
              "statuses": {"IMG_0001.JPG": "complete"}},
    )
    assert resp.status_code == 400


def test_image_status_lands_at_dataset_root_not_an_unrelated_project(
    client: TestClient, tmp_path: Path
) -> None:
    """A project referencing a dataset elsewhere must confirm negatives into the dataset, not into
    the project's own private state."""
    project_root = tmp_path / "project"
    dataset_root = tmp_path / "shared_dataset"
    project_root.mkdir()
    dataset_root.mkdir()

    client.post(
        "/api/classes/image_status",
        json={"project_root": str(project_root), "dataset_root": str(dataset_root),
              "image_name": "IMG_0001.JPG", "status": "negative", "subject": "bud"},
    )
    assert _status_store_exists(dataset_root)
    assert not _status_store_exists(project_root)


def test_derive_statuses_negatives_are_intentional(client: TestClient, tmp_path: Path) -> None:
    # A negative is intentional (completed + empty), never inferred from an empty file alone:
    # so an accidental empty file, or flipping through an image, doesn't become a training negative.
    ann = tmp_path / "annotations"
    ann.mkdir()
    write_annotations(str(ann / "IMG_A.json"), [_bud(50, 50, 60, 60)], 100, 100)  # objects, not completed
    write_annotations(str(ann / "IMG_B.json"), [], 100, 100, keep_empty=True)  # present {"annotations": []}
    write_annotations(str(ann / "IMG_E.json"), [_bud(50, 50, 60, 60)], 100, 100)  # objects, completed
    # IMG_C.json missing; IMG_D completed but has no file (empty)

    resp = client.post(
        "/api/classes/image_status/derive",
        json={
            "project_root": str(tmp_path),
            "annotations_dir": str(ann),
            "subject": "bud",
            "image_list": ["IMG_A.JPG", "IMG_B.JPG", "IMG_C.JPG", "IMG_D.JPG", "IMG_E.JPG"],
            "complete_override": ["IMG_D.JPG", "IMG_E.JPG"],
        },
    )
    assert resp.json()["statuses"] == {
        "IMG_A.JPG": "partial",  # has objects, not completed
        "IMG_B.JPG": "unannotated",  # empty file alone is not a negative, needs review
        "IMG_C.JPG": "unannotated",  # no file
        "IMG_D.JPG": "negative",  # completed + empty → confirmed negative (intentional)
        "IMG_E.JPG": "complete",  # completed + has objects
    }


def test_derive_statuses_reports_an_unreadable_label_and_still_derives_the_rest(
    client: TestClient, tmp_path: Path
) -> None:
    ann = tmp_path / "annotations"
    ann.mkdir()
    write_annotations(str(ann / "IMG_A.json"), [_bud(50, 50, 60, 60)], 100, 100)
    (ann / "IMG_B.json").write_text("not json {][", encoding="utf-8")

    resp = client.post(
        "/api/classes/image_status/derive",
        json={
            "project_root": str(tmp_path), "annotations_dir": str(ann), "subject": "bud",
            "image_list": ["IMG_A.JPG", "IMG_B.JPG"], "complete_override": [],
        },
    )
    body = resp.json()
    assert body["statuses"] == {"IMG_A.JPG": "partial"}
    assert body["unreadable"] == [str(ann / "IMG_B.json")]


def test_derive_statuses_cache_invalidates_on_label_write(client: TestClient, tmp_path: Path) -> None:
    # The per-file label-JSON memo is keyed on mtime_ns: a write after a prior derive() call
    # must not serve the stale (pre-write) parse of the same path.
    ann = tmp_path / "annotations"
    ann.mkdir()
    label = ann / "IMG_A.json"
    write_annotations(str(label), [], 100, 100, keep_empty=True)
    os.utime(label, (1_000_000, 1_000_000))

    req = {
        "project_root": str(tmp_path),
        "annotations_dir": str(ann),
        "subject": "bud",
        "image_list": ["IMG_A.JPG"],
        "complete_override": [],
    }
    first = client.post("/api/classes/image_status/derive", json=req).json()
    assert first["statuses"]["IMG_A.JPG"] == "unannotated"

    write_annotations(str(label), [_bud(50, 50, 60, 60)], 100, 100)
    os.utime(label, (2_000_000, 2_000_000))  # force a distinct mtime_ns from the first write

    second = client.post("/api/classes/image_status/derive", json=req).json()
    assert second["statuses"]["IMG_A.JPG"] == "partial"


def test_load_derives_subjects_excludes_a_bucket_sidecar(
    client: TestClient, tmp_path: Path
) -> None:
    """A bucket's own provenance stamp is not a per-image label: it must not seed the draft
    registry, and it is not reported under unreadable either, since it was never meant to be read
    as one."""
    from tcip_mcp.pipelines.resolution import write_sidecar

    ann = tmp_path / "annotations" / "d"
    ann.mkdir(parents=True)
    write_annotations(str(ann / "IMG_A.json"), [_bud(50, 50, 60, 60)], 100, 100)
    write_sidecar(ann, {"checkpoint_sha256": "sha", "experiment_id": None,
                       "subject": "bud", "attribute": None})

    load = client.get(
        "/api/classes/load",
        params={"project_root": str(tmp_path), "annotations_dir": str(ann)},
    ).json()
    assert set(load["subjects"]) == {"bud"}
    assert load["unreadable"] == []


def test_load_derived_registry_cache_invalidates_on_label_write(
    client: TestClient, tmp_path: Path
) -> None:
    # Same memo, exercised through load_classes' label-derived subject list.
    ann = tmp_path / "annotations" / "d"
    ann.mkdir(parents=True)
    label = ann / "IMG_A.json"
    write_annotations(str(label), [_bud(50, 50, 60, 60)], 100, 100)
    os.utime(label, (1_000_000, 1_000_000))

    params = {"project_root": str(tmp_path), "annotations_dir": str(ann)}
    first = client.get("/api/classes/load", params=params).json()
    assert set(first["subjects"]) == {"bud"}

    write_annotations(
        str(label), [_bud(50, 50, 60, 60), _bud(20, 20, 30, 30, subject="bush")], 100, 100
    )
    os.utime(label, (2_000_000, 2_000_000))

    second = client.get("/api/classes/load", params=params).json()
    assert set(second["subjects"]) == {"bush", "bud"}


def test_save_classes_confines_dataset_root_to_allowed_roots(
    client: TestClient, tmp_path_factory: pytest.TempPathFactory
) -> None:
    outside = tmp_path_factory.mktemp("outside")
    resp = client.post(
        "/api/classes/save",
        json={"project_root": str(outside), "dataset_root": str(outside),
              "subjects": {"bud": {"description": "a bud"}}, "version": None},
    )
    assert resp.status_code == 403


def test_image_status_confines_dataset_root_to_allowed_roots(
    client: TestClient, tmp_path_factory: pytest.TempPathFactory
) -> None:
    outside = tmp_path_factory.mktemp("outside")
    resp = client.post(
        "/api/classes/image_status",
        json={"project_root": str(outside), "dataset_root": str(outside),
              "image_name": "IMG_0001.JPG", "status": "complete", "subject": "bud"},
    )
    assert resp.status_code == 403


def test_image_status_bulk_confines_dataset_root_to_allowed_roots(
    client: TestClient, tmp_path_factory: pytest.TempPathFactory
) -> None:
    outside = tmp_path_factory.mktemp("outside")
    resp = client.post(
        "/api/classes/image_status/bulk",
        json={"project_root": str(outside), "dataset_root": str(outside),
              "subject": "bud", "statuses": {"A.JPG": "complete"}},
    )
    assert resp.status_code == 403


def test_get_image_status_confines_dataset_root_to_allowed_roots(
    client: TestClient, tmp_path_factory: pytest.TempPathFactory
) -> None:
    outside = tmp_path_factory.mktemp("outside")
    resp = client.get(
        "/api/classes/image_status",
        params={"project_root": str(outside), "dataset_root": str(outside), "subject": "bud"},
    )
    assert resp.status_code == 403


def test_derive_image_status_confines_annotations_dir_to_allowed_roots(
    client: TestClient, tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    outside = tmp_path_factory.mktemp("outside") / "annotations"
    outside.mkdir()
    resp = client.post(
        "/api/classes/image_status/derive",
        json={"project_root": str(tmp_path), "annotations_dir": str(outside),
              "subject": "bud", "image_list": ["IMG_A.JPG"]},
    )
    assert resp.status_code == 403


def test_load_classes_confines_annotations_dir_before_scanning_it(
    client: TestClient, tmp_path: Path, tmp_path_factory: pytest.TempPathFactory, monkeypatch,
) -> None:
    """The guard runs before the scan: a refused annotations_dir is never parsed, so a request
    naming a dataset_root outside the allowed roots costs nothing beyond the 403 it returns."""
    import tcip_annotation.json_io as json_io

    outside = tmp_path_factory.mktemp("outside")
    ann = outside / "annotations"
    ann.mkdir()
    write_annotations(str(ann / "SECRET.json"), [_bud(1, 1, 2, 2, subject="leaked")], 10, 10)

    def _must_not_be_called(path):
        raise AssertionError(f"the annotations dir must be guarded before any file is read: {path}")

    monkeypatch.setattr(json_io, "read_annotations", _must_not_be_called)

    resp = client.get(
        "/api/classes/load",
        params={"project_root": str(tmp_path), "dataset_root": str(outside),
                "annotations_dir": str(ann)},
    )
    assert resp.status_code == 403


def test_a_dataset_root_inside_the_workspace_clears_the_confinement_guard(
    client: TestClient, tmp_path: Path
) -> None:
    """The rail must admit valid work, not only reject invalid work: with no additive
    TCIP_IMAGE_ROOTS set, a dataset root under the workspace is still admitted."""
    resp = client.post(
        "/api/classes/image_status",
        json={"project_root": str(tmp_path), "dataset_root": str(tmp_path),
              "image_name": "IMG_0001.JPG", "status": "complete", "subject": "bud"},
    )
    assert resp.status_code == 200


def _read_audit_entries(dataset_root: Path) -> list[dict]:
    """That root's audit trail, read through the seam so the claim holds on either backend."""
    import tcip_store as ts
    from tcip_mcp.audit import audit_log_key

    return list(ts.read_log(audit_log_key(dataset_root)).records)


def test_set_image_status_writes_a_dataset_scoped_audit_entry(
    client: TestClient, tmp_path: Path
) -> None:
    # image_status.json is dataset-native, so its audit trail is colocated with the dataset rather
    # than guessed into a project's own audit.jsonl: distinct roots here so the assertion actually
    # pins which root the entry followed, not just that one was written.
    project_root = tmp_path / "project"
    dataset_root = tmp_path / "shared_dataset"
    project_root.mkdir()
    dataset_root.mkdir()
    client.post(
        "/api/classes/image_status",
        json={"project_root": str(project_root), "dataset_root": str(dataset_root),
              "image_name": "IMG_0001.JPG", "status": "complete", "subject": "bud"},
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
            "project_root": str(project_root), "dataset_root": str(dataset_root), "subject": "bud",
            "statuses": {"A.JPG": "complete", "B.JPG": "invalid_ignored"},
        },
    )
    entries = _read_audit_entries(dataset_root)
    assert len(entries) == 1
    assert entries[0]["tool"] == "gui_set_image_status_bulk"
    # B.JPG's status was invalid and never written: the audit entry must not claim it was.
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
              "subjects": {"bud": {"description": "a bud"}}, "version": None},
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
        json={"project_root": str(tmp_path), "dataset_root": str(tmp_path), "subject": "bud",
              "statuses": {"A.JPG": "invalid_ignored"}},
    )
    assert _read_audit_entries(tmp_path) == []


def test_image_status_bulk(client: TestClient, tmp_path: Path) -> None:
    resp = client.post(
        "/api/classes/image_status/bulk",
        json={
            "project_root": str(tmp_path),
            "dataset_root": str(tmp_path),
            "subject": "bud",
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
        params={"project_root": str(tmp_path), "dataset_root": str(tmp_path), "subject": "bud"},
    ).json()
    assert loaded["statuses"]["A.JPG"] == "complete"
    assert loaded["statuses"]["B.JPG"] == "partial"
    assert "C.JPG" not in loaded["statuses"]  # invalid skipped
