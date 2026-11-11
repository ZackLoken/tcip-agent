"""Tests for the workspace project front-door routes."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from tcip_mcp.dataset_layout import image_dir
from tests import REFUSED_NAMES
from tests._busy_store_fixtures import (
    GAVE_UP_WITHIN_S,
    SHORT_WAIT_S,
    held_by_another_connection,
    locked_by_another_holder,
    seconds_given_up_after,
    waiting,
)
from tests._web_fixtures import INVARIANCES

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tcip_store import RefusedRead
    from tcip_web.state import OpenProject


@pytest.fixture
def workspace_dir(tmp_path: Path) -> Path:
    """Configure and return a fresh workspace the backend is started with."""
    from tcip_web.state import store

    ws = (tmp_path / "ws").resolve()
    ws.mkdir()
    store.configure(ws, ())
    return ws


def _make_project(
    ws: Path,
    name: str,
    *,
    dates: Sequence[str] = (),
    subjects: Sequence[str] = (),
) -> OpenProject:
    """Make a workspace project through ``tests._web_fixtures.named_project``.

    Its display name is ``name``, and it holds one image per date and the named subjects.
    """
    from tests._producer_fixtures import write_image
    from tests._web_fixtures import named_project

    made = named_project(ws / name, name)
    proj = made.root
    for d in dates:
        write_image(image_dir(proj, d) / "img.png", (8, 8))
    if subjects:
        from tcip_mcp.subject_registry import Subject, SubjectRegistry
        from tests._producer_fixtures import registry_over

        registry_over(
            proj,
            SubjectRegistry(tuple(Subject(s, "") for s in sorted(subjects))),
        )
    return made


def _listed(client: TestClient) -> dict[str, dict]:
    return {
        p["record"]["display_name"]: p
        for p in client.get("/api/projects").json()["projects"]
    }


def test_list_projects_lists_workspace_projects(
    client: TestClient, workspace_dir: Path
) -> None:
    _make_project(
        workspace_dir,
        "currant_bud_valley-farm",
        dates=["2026-02-11"],
        subjects=["bud", "bush"],
    )
    _make_project(workspace_dir, "chestnut_burr_site-b", dates=["2026-03-01"])

    resp = client.get("/api/projects")
    assert resp.status_code == 200
    body = resp.json()
    assert body["workspace"] == str(workspace_dir.resolve())
    names = {p["record"]["display_name"] for p in body["projects"]}
    assert names == {"currant_bud_valley-farm", "chestnut_burr_site-b"}

    hz = _listed(client)["currant_bud_valley-farm"]
    assert hz["tree"]["captures"] == [
        {"name": "2026-02-11", "date": "2026-02-11"}
    ]
    assert hz["tree"]["subjects"] == ["bud", "bush"]  # sorted
    assert hz["tree"]["buckets_by_date"] == {
        "2026-02-11": []
    }  # no bucket is published
    assert hz["image_count"] == 1


def test_projects_report_per_date_subject_model_availability(
    client: TestClient, workspace_dir: Path
) -> None:
    # bud labeled on 02-11 (+ a bucket published over it); bush labeled on 03-02;
    # 03-24 has images but nothing labeled. One name-based label document per image.
    pytest.importorskip("torch")
    from tcip_annotation.state import Annotation, BBox
    from tests._chain_fixtures import detected, detection_pass, published
    from tests._producer_fixtures import label_image

    proj = _make_project(
        workspace_dir,
        "currant_bud_valley-farm",
        dates=["2026-02-11", "2026-03-02", "2026-03-24"],
        subjects=["bud", "bush"],
    ).root
    label_image(
        image_dir(proj, "2026-02-11") / "img.png",
        [Annotation(subject="bud", geometry=BBox(1, 1, 7, 7))],
        8,
        8,
    )
    label_image(
        image_dir(proj, "2026-03-02") / "img.png",
        [Annotation(subject="bush", geometry=BBox(2, 2, 6, 6))],
        8,
        8,
    )
    published(
        proj,
        "baseline/2026-02-11",
        [
            detection_pass(
                image_dir(proj, "2026-02-11") / "img.png",
                (8, 8),
                [detected((1.0, 1.0, 7.0, 7.0), 0.9)],
                cap=2,
            )
        ],
        scope={"subject": "bud"},
    )

    hz = _listed(client)["currant_bud_valley-farm"]["tree"]
    # Flat lists still list everything present anywhere.
    assert hz["subjects"] == ["bud", "bush"]
    # Per-date maps reflect where labels/predictions actually are.
    assert hz["subjects_by_date"]["2026-02-11"] == ["bud"]
    assert hz["subjects_by_date"]["2026-03-02"] == ["bush"]
    assert hz["subjects_by_date"]["2026-03-24"] == []  # images but no labels
    assert hz["buckets_by_date"] == {
        "2026-02-11": ["baseline/2026-02-11"],
        "2026-03-02": [],
        "2026-03-24": [],
    }
    assert hz["label_problem"] is None


def test_projects_report_a_label_problem_and_still_list(
    client: TestClient, workspace_dir: Path
) -> None:
    """A corrupt label under one project is reported without failing the whole listing.

    The project still lists, its other dates are unaffected, and the document is named.
    """
    from tcip_annotation.state import Annotation, BBox
    from tests._producer_fixtures import image_label_key, label_image
    from tests._record_damage_fixtures import damage_record

    proj = _make_project(
        workspace_dir,
        "currant_bud_valley-farm",
        dates=["2026-02-11", "2026-03-02"],
        subjects=["bud"],
    ).root
    bad = image_dir(proj, "2026-02-11") / "img.png"
    label_image(
        bad, [Annotation(subject="bud", geometry=BBox(1, 1, 7, 7))], 8, 8
    )
    damage_record(image_label_key(bad), b"not json {][")

    resp = client.get("/api/projects")
    assert resp.status_code == 200
    hz = _listed(client)["currant_bud_valley-farm"]["tree"]
    assert hz["subjects_by_date"]["2026-02-11"] == []
    assert hz["subjects_by_date"]["2026-03-02"] == []
    assert hz["label_problem"] is not None
    assert "img" in hz["label_problem"]


def test_list_ignores_dirs_without_tcip(
    client: TestClient, workspace_dir: Path
) -> None:
    _make_project(workspace_dir, "real_project_site", dates=["2026-02-11"])
    (workspace_dir / "not_a_project").mkdir()  # no .tcip/

    assert set(_listed(client)) == {"real_project_site"}


def _ingest_one_image(project: Path, scratch: Path, date: str) -> None:
    """Ingest one image stating the capture ``date`` into ``project`` through ``ingest_images``.

    The source image is written under ``scratch``.
    """
    from datetime import datetime

    from PIL import Image

    from tcip_mcp.tools.ingest_tools import IngestManifest, ingest_images
    from tests._image_fixtures import capture_exif

    source = scratch / "raw"
    source.mkdir()
    Image.new("RGB", (8, 8)).save(
        source / "img.png",
        exif=capture_exif(datetime.fromisoformat(date).replace(hour=9)),
    )
    ingested = ingest_images(project, str(source), actor=None)
    assert isinstance(ingested, IngestManifest), ingested
    assert ingested.copied == 1, ingested


def _assert_listed_unread_for_its_wait(
    listing: dict, held: Path, took_s: float
) -> None:
    """Assert ``listing`` lists the project ``held`` as unread with the store's busy sentence.

    The entry names the directory as ``root`` and carries no tree and no image count; the
    sentence's wait lies inside the band the store contract test holds a short wait to, and the
    request ``took_s`` seconds, less than a second store read on the held root would have added:
    each waits at least ``SHORT_WAIT_S`` again.
    """
    (entry,) = (
        p for p in listing["projects"] if p["root"] == str(held.resolve())
    )
    assert entry["kind"] == "unread", entry
    assert "tree" not in entry and "image_count" not in entry, entry
    sentence = entry["problem"]["refusal"]["message"]
    assert "was busy" in sentence, sentence
    waited_s = seconds_given_up_after(sentence)
    assert SHORT_WAIT_S <= waited_s < GAVE_UP_WITHIN_S, sentence
    assert took_s < waited_s + SHORT_WAIT_S, (took_s, sentence)


def test_list_over_a_held_store_lists_it_unread_and_the_other_project_read(
    client: TestClient, workspace_dir: Path
) -> None:
    """A project whose store is held past a short wait lists as unread beside a readable one.

    The answer is 200, the free project listed read with its record and tree, and the held one
    unread with its directory and the store's busy sentence, after one lock wait.
    """
    import tcip_store

    held = _make_project(workspace_dir, "held_project").root
    free = _make_project(workspace_dir, "free_project", dates=["2026-02-11"])
    tcip_store.release_root(held)

    with waiting(SHORT_WAIT_S), held_by_another_connection(held):
        started = time.monotonic()
        resp = client.get("/api/projects")
        took_s = time.monotonic() - started

    assert resp.status_code == 200, resp.text
    listing = resp.json()
    _assert_listed_unread_for_its_wait(listing, held, took_s)
    (read,) = (p for p in listing["projects"] if p["kind"] == "read")
    assert read["record"]["id"] == free.id
    assert read["root"] == str(free.root.resolve())
    assert read["tree"]["captures"] == [
        {"name": "2026-02-11", "date": "2026-02-11"}
    ]


def test_list_over_a_held_store_holding_an_image_lists_it_unread_then_read_once_released(
    client: TestClient, workspace_dir: Path, tmp_path: Path
) -> None:
    """A held project holding a capture lists unread as one holding none does, and reads after.

    Its capture is ingested before the hold, so the label documents' key listing is the first
    store read a tree would make; once the hold is released the listing reads it with its
    capture and its image.
    """
    import tcip_store

    held = _make_project(workspace_dir, "held_project").root
    _ingest_one_image(held, tmp_path, "2026-02-11")
    _make_project(workspace_dir, "free_project")
    tcip_store.release_root(held)

    with waiting(SHORT_WAIT_S):
        with held_by_another_connection(held):
            started = time.monotonic()
            resp = client.get("/api/projects")
            took_s = time.monotonic() - started
        after = client.get("/api/projects")

    assert resp.status_code == 200, resp.text
    _assert_listed_unread_for_its_wait(resp.json(), held, took_s)
    assert after.status_code == 200, after.text
    (read,) = (
        p for p in after.json()["projects"] if p["root"] == str(held.resolve())
    )
    assert read["kind"] == "read"
    assert read["tree"]["captures"] == [
        {"name": "2026-02-11", "date": "2026-02-11"}
    ]
    assert read["image_count"] == 1


def test_list_over_a_held_store_of_the_open_project_names_it_open_and_lists_it_unread(
    client: TestClient, workspace_dir: Path
) -> None:
    """The open project, held past a short wait, lists unread; the listing's open_id names it."""
    import tcip_store

    made = _make_project(workspace_dir, "held_project")
    _make_project(workspace_dir, "free_project")
    opened = client.post(
        "/api/projects/open", json={"id": made.id, "user": "grower"}
    )
    assert opened.status_code == 200, opened.text
    tcip_store.release_root(made.root)

    with waiting(SHORT_WAIT_S), held_by_another_connection(made.root):
        started = time.monotonic()
        resp = client.get("/api/projects")
        took_s = time.monotonic() - started

    assert resp.status_code == 200, resp.text
    assert resp.json()["open_id"] == made.id
    _assert_listed_unread_for_its_wait(resp.json(), made.root, took_s)


def test_a_listed_projects_tree_is_the_dataset_trees_without_its_directory(
    client: TestClient, workspace_dir: Path
) -> None:
    """The listing's tree and GET /api/dataset/tree agree; the latter adds the root."""
    _make_project(
        workspace_dir,
        "currant_bud_valley-farm",
        dates=["2026-02-11"],
        subjects=["bud"],
    )

    (listed,) = client.get("/api/projects").json()["projects"]
    served = client.get(
        "/api/dataset/tree", params={"dataset_root": listed["root"]}
    )

    assert served.status_code == 200, served.text
    assert served.json() == {**listed["tree"], "dataset_root": listed["root"]}


def test_list_sorted_by_modified_desc(
    client: TestClient, workspace_dir: Path
) -> None:
    older = _make_project(
        workspace_dir, "old_project_site", dates=["2026-02-11"]
    ).root
    newer = _make_project(
        workspace_dir, "new_project_site", dates=["2026-03-01"]
    ).root
    os.utime(older, (1_000_000, 1_000_000))
    os.utime(newer, (2_000_000, 2_000_000))

    order = [
        p["record"]["display_name"]
        for p in client.get("/api/projects").json()["projects"]
    ]
    assert order == ["new_project_site", "old_project_site"]


def test_opening_a_project_by_id_marks_it_open_in_the_list(
    client: TestClient, workspace_dir: Path
) -> None:
    made = _make_project(
        workspace_dir, "currant_bud_valley-farm", dates=["2026-02-11"]
    )
    listed = _listed(client)["currant_bud_valley-farm"]
    assert client.get("/api/projects").json()["open_id"] is None
    assert "is_open" not in listed

    resp = client.post(
        "/api/projects/open", json={"id": made.id, "user": "grower"}
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "project": {"id": made.id, "path": str(made.root.resolve())},
    }

    assert client.get("/api/projects").json()["open_id"] == made.id


def test_the_open_answer_names_the_project_as_the_state_snapshot_names_it(
    client: TestClient, workspace_dir: Path
) -> None:
    """The open route's project and the state snapshot's project, read after it, are equal."""
    made = _make_project(workspace_dir, "hill_block")

    answer = client.post(
        "/api/projects/open", json={"id": made.id, "user": "grower"}
    )
    assert answer.status_code == 200, answer.text
    with client.websocket_connect("ws://127.0.0.1/ws/state") as ws:
        snapshot = ws.receive_json()

    assert answer.json()["project"] == snapshot["project"]


def test_opening_a_project_whose_gui_state_its_model_refuses_answers_the_models_issues(
    client: TestClient, workspace_dir: Path
) -> None:
    """Opening a project whose persisted GUI state its model refuses answers 409 with why.

    The detail names the snapshot and the project it belongs to beside the model's issues, each
    by its place and message, and carries no library documentation link; nothing is opened.
    """
    import tcip_store
    from tcip_mcp.web_client import gui_snapshot_key

    made = _make_project(workspace_dir, "hill_block")
    tcip_store.replace(
        gui_snapshot_key(made.root),
        {"active_tab": "not-a-real-tab"},
        expect=tcip_store.Version.ABSENT,
    )

    resp = client.post(
        "/api/projects/open", json={"id": made.id, "user": "grower"}
    )

    assert resp.status_code == 409
    assert "errors.pydantic.dev" not in resp.text
    detail = resp.json()["detail"]
    assert "GUI snapshot" in detail["where"]
    assert str(made.root) in detail["where"]
    refusal = detail["refusal"]
    assert refusal["kind"] == "invalid"
    assert ["active_tab"] in [issue["loc"] for issue in refusal["issues"]]
    assert client.get("/api/projects").json()["open_id"] is None


def test_opening_an_id_no_project_holds_is_404(
    client: TestClient, workspace_dir: Path
) -> None:
    _make_project(workspace_dir, "currant_bud_valley-farm")
    resp = client.post(
        "/api/projects/open", json={"id": "000000000000", "user": "grower"}
    )
    assert resp.status_code == 404
    assert "000000000000" in resp.json()["detail"]
    assert client.get("/api/projects").json()["open_id"] is None


def test_opening_a_project_whose_store_is_held_answers_409_not_404(
    client: TestClient, workspace_dir: Path
) -> None:
    held = _make_project(workspace_dir, "locked_orchard")

    with locked_by_another_holder(held.root):
        resp = client.post(
            "/api/projects/open", json={"id": held.id, "user": "grower"}
        )

    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert "was busy" in detail, detail
    assert "gave up after" in detail, detail
    assert "no project in the workspace has id" not in detail, detail
    assert client.get("/api/projects").json()["open_id"] is None


def test_a_held_project_does_not_stop_another_from_opening(
    client: TestClient, workspace_dir: Path
) -> None:
    held = _make_project(workspace_dir, "locked_orchard")
    free = _make_project(workspace_dir, "free_orchard")

    with locked_by_another_holder(held.root):
        opened = client.post(
            "/api/projects/open", json={"id": free.id, "user": "grower"}
        )
        unknown = client.post(
            "/api/projects/open", json={"id": "0" * 12, "user": "grower"}
        )

    assert opened.status_code == 200, opened.text
    assert opened.json()["project"]["id"] == free.id
    assert unknown.status_code == 409, unknown.text
    detail = unknown.json()["detail"]
    assert str(held.root) in detail, detail
    assert str(free.root) not in detail, detail


def test_a_held_project_opens_once_the_hold_is_released(
    client: TestClient, workspace_dir: Path
) -> None:
    held = _make_project(workspace_dir, "locked_orchard")
    with locked_by_another_holder(held.root):
        pass

    resp = client.post(
        "/api/projects/open", json={"id": held.id, "user": "grower"}
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["project"]["id"] == held.id


def test_opening_a_project_naming_no_one_is_refused_and_opens_nothing(
    client: TestClient, workspace_dir: Path
) -> None:
    made = _make_project(
        workspace_dir, "currant_bud_valley-farm", dates=["2026-02-11"]
    )

    for stated in REFUSED_NAMES:
        resp = client.post(
            "/api/projects/open", json={"id": made.id, "user": stated}
        )
        assert resp.status_code == 400
        assert client.get("/api/projects").json()["open_id"] is None


def test_list_reports_the_record_across_four_project_states(
    client: TestClient, workspace_dir: Path
) -> None:
    """The listing reports each project's record or why it is absent.

    A project with a record, one with ``.tcip`` and nothing else, one whose record is
    undecodable, and one whose record decodes to something else: all four list, each with its
    record or the problem naming why it is absent. The recordless one gets no database
    published under it by the listing.
    """
    import tcip_store
    from tcip_mcp.project_record import project_record_key
    from tcip_store.file_backend import database_file
    from tcip_web.routes.projects import ReadProject, UnreadProject
    from tests._record_damage_fixtures import damage_record
    from tests._web_fixtures import INVARIANCES

    recorded = _make_project(workspace_dir, "currant_bud_recorded").root

    recordless = workspace_dir / "currant_bud_recordless"
    (recordless / ".tcip").mkdir(parents=True)

    undecodable = _make_project(workspace_dir, "currant_bud_undecodable").root
    damage_record(project_record_key(str(undecodable)), b"{not valid json")

    invalid = _make_project(workspace_dir, "currant_bud_invalid").root
    key = project_record_key(str(invalid))
    current = tcip_store.read_versioned(key).version
    tcip_store.replace(key, {"not_site": "x"}, expect=current)

    by_path = {
        p["root"]: p for p in client.get("/api/projects").json()["projects"]
    }
    assert set(by_path) == {
        str(p.resolve()) for p in (recorded, recordless, undecodable, invalid)
    }

    good = ReadProject.model_validate(by_path[str(recorded.resolve())]).record
    assert (good.display_name, good.site) == (
        "currant_bud_recorded",
        "north orchard",
    )
    assert len(good.id) == 12
    assert good.invariances == INVARIANCES

    def problem(project: Path) -> RefusedRead:
        return UnreadProject.model_validate(
            by_path[str(project.resolve())]
        ).problem

    assert "initialize_project" in str(problem(recordless))
    assert not database_file(str(recordless)).is_file()
    assert "does not decode" in str(problem(undecodable))
    assert problem(invalid).refusal.kind == "invalid"
    assert str(invalid) in problem(invalid).where


def test_the_listing_reads_a_stated_rotation_angle_back_as_degrees(
    client: TestClient, workspace_dir: Path
) -> None:
    """A project created stating its rotation invariance as 5 degrees lists it as 5.0."""
    from tcip_mcp.project_record import CaptureInvariances
    from tcip_mcp.tools.project_tools import initialize_project

    project = workspace_dir / "currant_bud_rotation"
    answers = INVARIANCES.model_dump() | {"rotation": 5}
    created = initialize_project(
        str(project),
        "currant_bud_rotation",
        "north orchard",
        CaptureInvariances.model_validate(answers),
        actor=None,
    )
    assert "error" not in created, created

    (listed,) = client.get("/api/projects").json()["projects"]

    rotation = listed["record"]["invariances"]["rotation"]
    assert isinstance(rotation, float) and rotation == 5.0, listed


def _correction(project_id: str, user: str = "tester") -> dict:
    """Return an invariances correction request built by the route's own model.

    The correction states no vertical flip, and rotation up to 5 degrees.
    """
    from tcip_mcp.project_record import CaptureInvariances
    from tcip_web.routes.projects import InvariancesRequest

    corrected = CaptureInvariances.model_validate(
        INVARIANCES.model_dump() | {"vertical_flip": False, "rotation": 5.0}
    )
    return InvariancesRequest(
        id=project_id, invariances=corrected, user=user
    ).model_dump()


def test_correcting_the_invariances_answers_the_typed_correction_and_the_record_rereads_it(
    client: TestClient, workspace_dir: Path
) -> None:
    from tcip_mcp.identity import actor
    from tcip_mcp.project_record import read_record
    from tests._audit_fixtures import audit_rows

    made = _make_project(workspace_dir, "valley_block")
    body = _correction(made.id)

    resp = client.post("/api/projects/invariances", json=body)

    assert resp.status_code == 200, resp.text
    answer = resp.json()
    assert answer["record"]["invariances"] == body["invariances"]
    assert answer["previous"]["invariances"] == INVARIANCES.model_dump()
    assert answer["record"]["id"] == answer["previous"]["id"] == made.id
    written = body["invariances"]
    assert read_record(made.root).invariances.model_dump() == written
    listed = _listed(client)["valley_block"]["record"]
    assert listed["invariances"] == written
    (line,) = audit_rows(made.root, "project_record_corrected")
    assert line["actor"] == actor("tester")


def test_an_invariances_correction_for_an_id_no_project_holds_is_404(
    client: TestClient, workspace_dir: Path
) -> None:
    _make_project(workspace_dir, "valley_block")

    resp = client.post("/api/projects/invariances", json=_correction("0" * 12))

    assert resp.status_code == 404


@pytest.mark.parametrize(
    ("change", "status"),
    [
        ({"user": ""}, 400),
        ({"invariances": {"horizontal_flip": False}}, 422),
        ({"invariances": INVARIANCES.model_dump() | {"rotation": 181.0}}, 422),
    ],
    ids=["naming-no-one", "half-filled", "out-of-range"],
)
def test_a_refused_invariances_correction_writes_nothing(
    client: TestClient, workspace_dir: Path, change: dict, status: int
) -> None:
    from tcip_mcp.project_record import read_record
    from tests._audit_fixtures import audit_rows

    made = _make_project(workspace_dir, "valley_block")
    before, lines = read_record(made.root), audit_rows(made.root)

    resp = client.post(
        "/api/projects/invariances", json=_correction(made.id) | change
    )

    assert resp.status_code == status, resp.text
    assert read_record(made.root) == before
    assert audit_rows(made.root) == lines
