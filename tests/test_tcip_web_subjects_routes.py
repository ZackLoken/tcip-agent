"""Tests for the subject-registry routes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tcip_annotation.state import Annotation
from tests._audit_fixtures import AUDIT_ENTRY_KEYS, audit_rows
from tests._producer_fixtures import box_annotation, image_label_key, label_image

DATE = "2026-03-02"


def _label(root: Path, stem: str, annotations: list[Annotation]) -> None:
    """``annotations`` saved as the label document of image ``stem`` of capture :data:`DATE`."""
    label_image(root / "images" / DATE / f"{stem}.jpg", annotations, 100, 100)


def _unreadable(root: Path, stem: str) -> None:
    """A label document of image ``stem`` of capture :data:`DATE` whose bytes no longer
    decode."""
    from tests._record_damage_fixtures import damage_record

    _label(root, stem, [box_annotation(1, 1, 2, 2)])
    damage_record(image_label_key(root / "images" / DATE / f"{stem}.jpg"), b"not json {][")


def test_a_missing_registry_answers_discovery_never_a_registry(
    client: TestClient, tmp_path: Path
) -> None:
    """With no registry stored, the load answers no registry and the names the labels hold as
    discovery, never a registry made of those names."""
    _label(tmp_path, "IMG_A", [box_annotation(50, 50, 60, 60)])

    resp = client.get("/api/subjects/load",
                      params={"dataset_root": str(tmp_path), "date": "2026-04-01"})
    assert resp.status_code == 200
    assert resp.json() == {"subjects": None, "discovered": [], "version": None, "unreadable": []}
    found = client.get("/api/subjects/load", params={
        "dataset_root": str(tmp_path), "date": DATE}).json()
    assert found == {"subjects": None, "discovered": ["bud"], "version": None, "unreadable": []}


def test_save_then_load_round_trip(client: TestClient, tmp_path: Path) -> None:
    # The registry is one nested subjects.json in the DATASET; it travels with the image set.
    save = client.post(
        "/api/subjects/save",
        json={
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
            "user": "tester",
        },
    )
    assert save.status_code == 200
    assert save.json()["n_subjects"] == 2

    load = client.get(
        "/api/subjects/load",
        params={"dataset_root": str(tmp_path), "date": DATE},
    ).json()
    subjects = load["subjects"]
    assert set(subjects) == {"bud", "bush"}
    assert subjects["bud"]["attributes"]["opening"]["values"] == ["closed", "open"]
    # Lands in the dataset as one nested subjects.json (no per-subject files, no numeric ids).
    on_disk = json.loads((tmp_path / "subjects.json").read_text())
    assert set(on_disk) == {"bud", "bush"}


def test_save_refuses_an_empty_registry(client: TestClient, tmp_path: Path) -> None:
    """A registry write states subjects; it never clears them, at either door."""
    r = client.post(
        "/api/subjects/save",
        json={"dataset_root": str(tmp_path), "subjects": {},
              "version": None, "user": "tester"},
    )
    assert r.status_code == 400
    assert not (tmp_path / "subjects.json").exists()


def test_save_refuses_dropping_a_declared_subject(client: TestClient, tmp_path: Path) -> None:
    """A stale browser posting a subset of the stored registry is refused by name, not silently
    written: the additive-only toolbar makes a drop arriving here a sign of staleness."""
    first = client.post(
        "/api/subjects/save",
        json={"dataset_root": str(tmp_path),
              "subjects": {"leaf": {"description": "one leaf"}, "bush": {"description": "b"}},
              "version": None, "user": "tester"},
    )
    assert first.status_code == 200

    dropped = client.post(
        "/api/subjects/save",
        json={"dataset_root": str(tmp_path),
              "subjects": {"bush": {"description": "b"}}, "version": first.json()["version"], "user": "tester"},
    )
    assert dropped.status_code == 400
    assert "leaf" in dropped.text
    on_disk = json.loads((tmp_path / "subjects.json").read_text())
    assert "leaf" in on_disk  # the refused write never landed


def test_load_returns_the_version_and_save_round_trips_it(
    client: TestClient, tmp_path: Path
) -> None:
    """The toolbar carries the version it loaded back into its next save, the shape that lets a
    stale browser be told apart from a caller with nothing to assert."""
    save = client.post(
        "/api/subjects/save",
        json={"dataset_root": str(tmp_path),
              "subjects": {"leaf": {"description": "one leaf"}}, "version": None, "user": "tester"},
    )
    assert save.status_code == 200

    load = client.get(
        "/api/subjects/load",
        params={"dataset_root": str(tmp_path), "date": DATE},
    ).json()
    assert load["version"]

    grown = client.post(
        "/api/subjects/save",
        json={"dataset_root": str(tmp_path),
              "subjects": {"leaf": {"description": "one leaf"}, "bush": {}},
              "version": load["version"], "user": "tester"},
    )
    assert grown.status_code == 200
    assert grown.json()["version"]


def test_save_refuses_a_stale_version(client: TestClient, tmp_path: Path) -> None:
    """A save carrying a version the store has moved past since is refused with 409, naming the
    conflict, rather than silently overwriting a registry the browser never saw."""
    client.post(
        "/api/subjects/save",
        json={"dataset_root": str(tmp_path),
              "subjects": {"leaf": {"description": "one leaf"}}, "version": None, "user": "tester"},
    )
    stale_load = client.get(
        "/api/subjects/load",
        params={"dataset_root": str(tmp_path), "date": DATE},
    ).json()
    client.post(
        "/api/subjects/save",
        json={"dataset_root": str(tmp_path),
              "subjects": {"leaf": {"description": "one leaf"}, "bush": {}},
              "version": stale_load["version"], "user": "tester"},
    )

    conflicted = client.post(
        "/api/subjects/save",
        json={"dataset_root": str(tmp_path),
              "subjects": {"leaf": {"description": "one leaf"}, "bush": {}, "tip": {}},
              "version": stale_load["version"], "user": "tester"},
    )
    assert conflicted.status_code == 409


def test_save_with_a_null_version_refuses_over_a_registry_written_meanwhile(
    client: TestClient, tmp_path: Path
) -> None:
    """A null version names an absent registry, not an unconditional write: a browser that never
    loaded a registry still refuses when an agent has written one in the meantime."""
    from tcip_mcp.subject_registry import read_registry
    from tcip_mcp.tools.annotation_tools import write_subject_registry

    result = write_subject_registry(tmp_path, str(tmp_path),
                                    {"leaf": {"description": "written by the agent"}})
    assert "error" not in result

    # Additive (keeps "leaf"), so only the version check can refuse this, not the by-name drop rail.
    resp = client.post(
        "/api/subjects/save",
        json={"dataset_root": str(tmp_path),
              "subjects": {"leaf": {"description": "written by the agent"},
                           "bush": {"description": "written by the browser"}},
              "version": None, "user": "tester"},
    )
    assert resp.status_code == 409
    assert read_registry(tmp_path).subject("bush") is None


def test_save_with_a_null_version_succeeds_over_a_still_absent_registry(
    client: TestClient, tmp_path: Path
) -> None:
    """A null version asserts the registry is still absent; over an actually absent one the write
    lands rather than being treated as unconditional."""
    resp = client.post(
        "/api/subjects/save",
        json={"dataset_root": str(tmp_path),
              "subjects": {"bush": {"description": "first write"}}, "version": None, "user": "tester"},
    )
    assert resp.status_code == 200


def test_save_refuses_a_same_values_attribute_type_flip(client: TestClient, tmp_path: Path) -> None:
    """The route passes neither allow_removals nor allow_type_changes, so a type flip has no door
    here at all, in either direction: only a values-only growth or a same-type re-save lands."""
    first = client.post(
        "/api/subjects/save",
        json={"dataset_root": str(tmp_path),
              "subjects": {"bud": {"attributes": {
                  "opening": {"type": "categorical", "values": ["closed", "open"]}}}},
              "version": None, "user": "tester"},
    )
    assert first.status_code == 200

    flipped = client.post(
        "/api/subjects/save",
        json={"dataset_root": str(tmp_path),
              "subjects": {"bud": {"attributes": {
                  "opening": {"type": "ordinal", "values": ["closed", "open"]}}}},
              "version": first.json()["version"], "user": "tester"},
    )
    assert flipped.status_code == 400
    assert "bud.opening" in flipped.text
    assert "categorical" in flipped.text
    assert "ordinal" in flipped.text


def test_save_refuses_the_reverse_attribute_type_flip_too(
    client: TestClient, tmp_path: Path
) -> None:
    first = client.post(
        "/api/subjects/save",
        json={"dataset_root": str(tmp_path),
              "subjects": {"bud": {"attributes": {
                  "opening": {"type": "ordinal", "values": ["closed", "open"]}}}},
              "version": None, "user": "tester"},
    )
    assert first.status_code == 200

    flipped = client.post(
        "/api/subjects/save",
        json={"dataset_root": str(tmp_path),
              "subjects": {"bud": {"attributes": {
                  "opening": {"type": "categorical", "values": ["closed", "open"]}}}},
              "version": first.json()["version"], "user": "tester"},
    )
    assert flipped.status_code == 400
    assert "bud.opening" in flipped.text


def test_save_refuses_malformed_registry(client: TestClient, tmp_path: Path) -> None:
    """A malformed registry (here: an attribute with no ``values``) is refused, not silently
    written: a bad registry would assign ids over garbage."""
    r = client.post(
        "/api/subjects/save",
        json={"dataset_root": str(tmp_path),
              "subjects": {"bud": {"attributes": {"opening": {"type": "categorical"}}}},
              "version": None, "user": "tester"},
    )
    assert r.status_code == 400


def test_registry_holds_multiple_subjects(client: TestClient, tmp_path: Path) -> None:
    # bud and bush each name their own object; the one nested registry keeps them distinct.
    save = client.post(
        "/api/subjects/save",
        json={
            "dataset_root": str(tmp_path),
            "subjects": {
                "bud": {"description": "a bud"},
                "bush": {"description": "a bush"},
            },
            "version": None,
            "user": "tester",
        },
    )
    assert save.status_code == 200

    load = client.get(
        "/api/subjects/load",
        params={"dataset_root": str(tmp_path), "date": DATE},
    ).json()
    assert load["subjects"]["bud"]["description"] == "a bud"
    assert load["subjects"]["bush"]["description"] == "a bush"
    assert (tmp_path / "subjects.json").is_file()


def test_load_derives_subjects_from_labels_when_registry_absent(
    client: TestClient, tmp_path: Path
) -> None:
    # No saved registry, but labels exist: the subjects present are discovered, sorted.
    _label(tmp_path, "IMG_A", [box_annotation(50, 50, 60, 60), box_annotation(20, 20, 30, 30, subject="bush")])
    load = client.get(
        "/api/subjects/load",
        params={"dataset_root": str(tmp_path), "date": DATE},
    ).json()
    assert load["discovered"] == ["bud", "bush"]
    assert load["unreadable"] == []


def test_load_reports_an_unreadable_label_and_still_derives_the_rest(
    client: TestClient, tmp_path: Path
) -> None:
    """One corrupt label document costs its own name, never the whole discovery scan."""
    _label(tmp_path, "IMG_A", [box_annotation(50, 50, 60, 60)])
    _unreadable(tmp_path, "IMG_B")

    load = client.get(
        "/api/subjects/load",
        params={"dataset_root": str(tmp_path), "date": DATE},
    ).json()
    assert load["discovered"] == ["bud"]
    assert load["unreadable"] == ["IMG_B"]


def test_load_reports_an_unreadable_label_beside_a_saved_registry(
    client: TestClient, tmp_path: Path
) -> None:
    """A saved subjects.json answers the subject list, but a corrupt label document of the
    capture is still worth surfacing: the registry load must not stop scanning for unreadable
    documents just because a registry was found."""
    _label(tmp_path, "IMG_A", [box_annotation(50, 50, 60, 60)])
    _unreadable(tmp_path, "IMG_B")

    save = client.post(
        "/api/subjects/save",
        json={"dataset_root": str(tmp_path),
              "subjects": {"bud": {"description": "a bud"}}, "version": None, "user": "tester"},
    )
    assert save.status_code == 200

    load = client.get(
        "/api/subjects/load",
        params={"dataset_root": str(tmp_path), "date": DATE},
    ).json()
    assert set(load["subjects"]) == {"bud"}
    assert load["unreadable"] == ["IMG_B"]


def test_load_derived_subjects_follow_a_label_write(
    client: TestClient, tmp_path: Path
) -> None:
    """Discovery answers the documents as they are stored now, never a scan from before a
    write."""
    _label(tmp_path, "IMG_A", [box_annotation(50, 50, 60, 60)])

    params = {"dataset_root": str(tmp_path), "date": DATE}
    first = client.get("/api/subjects/load", params=params).json()
    assert first["discovered"] == ["bud"]

    _label(tmp_path, "IMG_A", [box_annotation(50, 50, 60, 60), box_annotation(20, 20, 30, 30, subject="bush")])

    second = client.get("/api/subjects/load", params=params).json()
    assert second["discovered"] == ["bud", "bush"]


def test_save_subjects_confines_dataset_root_to_allowed_roots(
    client: TestClient, tmp_path_factory: pytest.TempPathFactory
) -> None:
    outside = tmp_path_factory.mktemp("outside")
    resp = client.post(
        "/api/subjects/save",
        json={"dataset_root": str(outside),
              "subjects": {"bud": {"description": "a bud"}}, "version": None, "user": "tester"},
    )
    assert resp.status_code == 403


def test_load_subjects_confines_the_dataset_root_before_scanning_it(
    client: TestClient, tmp_path_factory: pytest.TempPathFactory, monkeypatch,
) -> None:
    """The guard runs before the scan: a refused dataset root's documents are never read, so a
    request naming one outside the allowed roots costs nothing beyond the 403 it returns."""
    import tcip_annotation.json_io as json_io

    outside = tmp_path_factory.mktemp("outside")
    _label(outside, "SECRET", [box_annotation(1, 1, 2, 2, subject="leaked")])

    def _must_not_be_called(key):
        raise AssertionError(f"the dataset root must be guarded before any document is read: {key}")

    monkeypatch.setattr(json_io, "read_label_document", _must_not_be_called)

    resp = client.get(
        "/api/subjects/load",
        params={"dataset_root": str(outside), "date": DATE},
    )
    assert resp.status_code == 403


def test_a_dataset_root_inside_the_workspace_clears_the_confinement_guard(
    client: TestClient, tmp_path: Path
) -> None:
    """With no additive TCIP_IMAGE_ROOTS set, a dataset root under the workspace is admitted."""
    resp = client.post(
        "/api/subjects/save",
        json={"dataset_root": str(tmp_path),
              "subjects": {"bud": {"description": "a bud"}}, "version": None, "user": "tester"},
    )
    assert resp.status_code == 200


def test_a_registry_save_leaves_one_library_line_through_either_door(
    client: TestClient, opened_project: Path
) -> None:
    """The route and the tool both save through ``replace_registry``, which writes the act's one
    line into the dataset's own log: the same registry saved through each leaves exactly one line
    apiece, the same facts recorded, and nothing in the project's log."""
    from tcip_mcp.tools.annotation_tools import write_subject_registry

    subjects = {"bud": {"description": "a bud"}}
    through_route, through_tool = opened_project / "route_dataset", opened_project / "tool_dataset"
    through_route.mkdir()
    through_tool.mkdir()
    resp = client.post("/api/subjects/save", json={
        "dataset_root": str(through_route), "subjects": subjects, "version": None, "user": "tester"})
    assert resp.status_code == 200, resp.text
    assert "error" not in write_subject_registry(opened_project, str(through_tool), subjects)

    (route_line,), (tool_line,) = (audit_rows(through_route),
                                   audit_rows(through_tool))
    assert route_line["tool"] == tool_line["tool"] == "replace_registry"

    def facts(line: dict) -> dict:
        return {k: v for k, v in line["arguments"].items()
                if k not in ("subjects_path", "version")}

    assert facts(route_line) == facts(tool_line)
    assert route_line["arguments"]["version"] == resp.json()["version"]
    assert route_line["actor"] == "user:tester" and "actor" not in tool_line
    assert set(route_line) - {"actor"} <= AUDIT_ENTRY_KEYS and set(tool_line) <= AUDIT_ENTRY_KEYS
    assert "tester" not in json.dumps({k: v for k, v in route_line.items() if k != "actor"})
    assert not [e for e in audit_rows(opened_project) if e["tool"] == "replace_registry"]


def test_a_registry_save_answers_and_audits_the_location_it_wrote(
    client: TestClient, opened_project: Path
) -> None:
    """The save is keyed by the dataset root it names: the answer, the audit line and the stored
    registry all name the one location the write landed at."""
    from tcip_mcp.dataset_layout import subjects_path
    from tcip_mcp.subject_registry import read_registry

    dataset_root = opened_project / "named_dataset"
    dataset_root.mkdir()
    resp = client.post("/api/subjects/save", json={
        "dataset_root": str(dataset_root), "subjects": {"bud": {}}, "version": None, "user": "tester"})
    assert resp.status_code == 200, resp.text

    written = str(subjects_path(dataset_root))
    (line,) = audit_rows(dataset_root)
    assert resp.json()["subjects_path"] == line["arguments"]["subjects_path"] == written
    assert read_registry(dataset_root).subject("bud") is not None


def test_a_registry_load_answers_content_and_version_from_one_read(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A save landing just after the load reads the registry cannot pair the old content with the
    new version: the load's version is the one its content was read at, so posting it back
    refuses as stale."""
    import tcip_store

    from tcip_mcp.tools.annotation_tools import write_subject_registry

    assert "error" not in write_subject_registry(tmp_path, str(tmp_path), {"bud": {}})
    real = tcip_store.read_blob_versioned
    interleaved: list[dict] = []

    def read_then_save(key, *args, **kwargs):
        answer = real(key, *args, **kwargs)
        if not interleaved:
            interleaved.append({})
            interleaved[0] = write_subject_registry(tmp_path, str(tmp_path),
                                                    {"bud": {}, "bush": {}})
        return answer

    monkeypatch.setattr(tcip_store, "read_blob_versioned", read_then_save)
    load = client.get("/api/subjects/load",
                      params={"dataset_root": str(tmp_path), "date": DATE}).json()
    monkeypatch.setattr(tcip_store, "read_blob_versioned", real)

    assert "error" not in interleaved[0]
    assert set(load["subjects"]) == {"bud"}
    stale = client.post("/api/subjects/save", json={
        "dataset_root": str(tmp_path), "subjects": {"bud": {}, "bush": {}, "tip": {}},
        "version": load["version"], "user": "tester"})
    assert stale.status_code == 409


# ── A committed write whose audit line could not be appended answers 409, not 200 ─────────


class _AppendRefused(RuntimeError):
    """Stands in for whatever stops a real append: a busy lock, a refused root, a bad key."""


def _refuse_append(*args: object, **kwargs: object) -> None:
    raise _AppendRefused("the audit log could not be appended to")


def test_save_subjects_answers_409_with_the_committed_body_on_a_lost_audit_line(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The registry write already committed; a lost audit line answers the gap, not a 200. An
    identical resave changes nothing but the version token, so the refused-append pass's
    ``committed`` is compared field by field (version excepted) against the first pass's real
    200 body."""
    import tcip_mcp.audit as audit_module

    healthy = client.post(
        "/api/subjects/save",
        json={"dataset_root": str(tmp_path),
              "subjects": {"bud": {"description": "a bud"}}, "version": None, "user": "tester"},
    )
    assert healthy.status_code == 200, healthy.text
    healthy_body = healthy.json()

    monkeypatch.setattr(audit_module, "append", _refuse_append)
    resp = client.post(
        "/api/subjects/save",
        json={"dataset_root": str(tmp_path),
              "subjects": {"bud": {"description": "a bud"}}, "version": healthy_body["version"], "user": "tester"},
    )
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["error"] == "audit_entry_not_written"
    committed = detail["committed"]
    assert committed["status"] == "ok"
    assert committed["n_subjects"] == 1
    assert committed["subjects_path"] == str(tmp_path / "subjects.json")
    assert {k: v for k, v in committed.items() if k != "version"} == {
        k: v for k, v in healthy_body.items() if k != "version"
    }
    on_disk = json.loads((tmp_path / "subjects.json").read_text())
    assert set(on_disk) == {"bud"}
