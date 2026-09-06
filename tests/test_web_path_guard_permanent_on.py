"""The web layer's path guard with no env var set: derived allow-set, identity containment, and the
project-scoped Results doors.

Every refusal here is paired with the legitimate call the same guard must still admit. The
``tmp_path`` fixture is a project inside its own workspace (``<workspace>/project``), so a project
built there is admitted by the derived rule; ``outside`` is a directory pytest creates beside that
workspace, which no rule admits.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import tcip_store
from tcip_mcp.audit import audit_log_key
from tcip_mcp.tools.project_tools import upsert_dataset
from tcip_web.app import app
from tcip_web.paths import assert_path_allowed
from tcip_web.state import store

from tests._operationalization_fixtures import seed_confirmed_crossing, write_spec
from tests._trait_fixtures import BUD_OPENING
from tests.test_results_mapping_summary_and_audit_anchoring import _capture_fixture
from tests.test_tcip_web_results_routes import _phenology_fixture


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


@pytest.fixture
def outside(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A directory beside the test's workspace that no allow-set rule admits."""
    return tmp_path_factory.mktemp("outside")


@pytest.fixture
def closed_project():
    """Leave no project open after the test, whatever it opened."""
    yield
    store.close_project()


def _open(client: TestClient, project_root: Path, dataset_root: Path | None = None) -> None:
    resp = client.post("/api/dataset/select", json={
        "project_root": str(project_root), "dataset_root": str(dataset_root or project_root)})
    assert resp.status_code == 200, resp.text


def _project(path: Path) -> Path:
    (path / ".tcip").mkdir(parents=True, exist_ok=True)
    return path


def _image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8)).save(path)
    return path


def _link_to(link: Path, target: Path) -> None:
    """A directory symlink from ``link`` to ``target``, skipping the test where this machine
    cannot make one rather than failing on an environment limitation."""
    target.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks not available on this machine: {exc}")


# ── the derived allow-set ──────────────────────────────────────────────────


def test_the_workspace_is_admitted_and_a_sibling_outside_it_is_refused_with_no_env_var(
    tmp_path: Path, outside: Path,
) -> None:
    from tcip_web.paths import allowed_roots

    inside = tmp_path / "images" / "a.jpg"
    inside.parent.mkdir(parents=True)
    inside.write_bytes(b"x")
    assert assert_path_allowed(str(inside)) == inside.resolve()
    assert allowed_roots()[0] == tmp_path.parent.resolve()
    with pytest.raises(ValueError, match="outside the allowed roots"):
        assert_path_allowed(str(outside / "leak.jpg"))


def test_a_workspace_project_reached_through_a_link_is_admitted_as_itself(
    tmp_path: Path, outside: Path,
) -> None:
    """A project the workspace lists through a junction or symlink resolves elsewhere and must
    still be admitted, or the front door would list a project no route can open."""
    real = _project(outside / "linked-project")
    link = tmp_path.parent / "linked"
    try:
        link.symlink_to(real, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks not available on this machine: {exc}")
    assert assert_path_allowed(str(link / "images")) == (real / "images").resolve()


def test_a_platform_state_root_outside_any_workspace_is_not_admitted_on_its_own(
    tmp_path: Path, outside: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The development pin (the repo root) is the server's own state, not breeder data."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(outside))
    (outside / ".tcip").mkdir()
    with pytest.raises(ValueError, match="outside the allowed roots"):
        assert_path_allowed(str(outside / ".tcip" / "audit.jsonl"))


def test_a_dataset_registered_to_a_workspace_project_is_admitted_wherever_it_lives(
    tmp_path: Path, outside: Path,
) -> None:
    project = _project(tmp_path)
    external = outside / "field-data"
    (external / "images").mkdir(parents=True)
    with pytest.raises(ValueError):
        assert_path_allowed(str(external / "images"))
    upsert_dataset(project, {"id": "ds-1", "path": str(external), "crop": "currant",
                             "fingerprint": "v1:f"})
    assert assert_path_allowed(str(external / "images")) == (external / "images").resolve()


def test_a_dataset_registered_as_the_projects_own_tree_contributes_no_relative_root(
    tmp_path: Path,
) -> None:
    """register_dataset stores "." for a project's own dataset; allowed_roots must resolve that
    entry against the project root rather than pass the bare "." into the allow-set, where a
    same-file comparison against an unresolved "." would silently admit whatever directory the
    server process happens to be running from."""
    from tcip_mcp.tools.project_tools import register_dataset
    from tcip_web.paths import allowed_roots

    project = _project(tmp_path)
    registered = register_dataset(str(project), crop="currant", project_root=str(project))
    assert "error" not in registered

    roots = allowed_roots()

    assert all(root.is_absolute() for root in roots)
    assert project.resolve() in roots


def test_an_imports_staging_tree_is_never_admitted_even_under_the_workspace(
    tmp_path: Path,
) -> None:
    """The import door stages a half-extracted project under ``<workspace>/.imports/<uuid>/``,
    which sits under the workspace, an allowed root; a route resolving into it while the import
    is in flight must still refuse, or a guarded route could read a project's confirmed
    negatives before the accounting and adoption steps have judged them."""
    workspace = tmp_path.parent
    staged = workspace / ".imports" / "run-1" / "images" / "a.jpg"
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b"x")

    with pytest.raises(ValueError, match="outside the allowed roots"):
        assert_path_allowed(str(staged))


def test_image_roots_stay_additive_on_top_of_the_derived_set(
    tmp_path: Path, outside: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    extra = outside / "archive"
    extra.mkdir()
    monkeypatch.setenv("TCIP_IMAGE_ROOTS", str(extra))
    assert assert_path_allowed(str(extra / "scan.tif")) == (extra / "scan.tif").resolve()
    inside = tmp_path / "still-admitted"
    inside.mkdir()
    assert assert_path_allowed(str(inside)) == inside.resolve()


def test_a_path_that_does_not_exist_yet_is_judged_by_its_nearest_existing_ancestor(
    tmp_path: Path, outside: Path,
) -> None:
    to_write = tmp_path / "proj" / "results_export" / "x.csv"
    assert assert_path_allowed(str(to_write)) == to_write.resolve()
    with pytest.raises(ValueError):
        assert_path_allowed(str(outside / "results_export" / "x.csv"))


@pytest.mark.skipif(os.name != "nt", reason="case-insensitive spellings are a Windows path shape")
def test_containment_is_by_identity_so_a_case_variant_spelling_is_the_same_directory(
    tmp_path: Path,
) -> None:
    inside = tmp_path / "Proj" / "images" / "a.jpg"
    inside.parent.mkdir(parents=True)
    inside.write_bytes(b"x")
    variant = Path(str(inside).upper())
    assert assert_path_allowed(str(variant)).samefile(inside)


def test_a_registry_that_will_not_decode_raises_rather_than_admitting_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tcip_mcp.tools import project_tools
    from tcip_web.paths import allowed_roots

    _project(tmp_path)

    def broken(_root):
        raise tcip_store.DecodeError("registry bytes are not a document")

    monkeypatch.setattr(project_tools, "read_datasets", broken)
    with pytest.raises(RuntimeError, match="dataset registry of project"):
        allowed_roots()


def test_a_registry_carrying_a_bare_fingerprint_raises_naming_the_one_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ValueError read_datasets raises for a bare pre-prefix fingerprint is wrapped the
    same way DecodeError is, naming the one project whose registry carries it, rather than
    propagating as a plain ValueError a caller's own ``except ValueError`` would misread as a
    403 against every project in the workspace."""
    from tcip_mcp.tools import project_tools
    from tcip_web.paths import allowed_roots

    ws = tmp_path.parent
    _project(tmp_path)
    bad = _project(ws / "bad_project")
    real_read_datasets = project_tools.read_datasets

    def flaky(root):
        if Path(root).resolve() == bad.resolve():
            raise ValueError(
                f"dataset registry entry 'x' under {root} carries a fingerprint 'deadbeef' "
                "that names no formula version; re-register it through register_dataset")
        return real_read_datasets(root)

    monkeypatch.setattr(project_tools, "read_datasets", flaky)
    with pytest.raises(RuntimeError, match=f"dataset registry of project.*{bad.name}"):
        allowed_roots()


# ── dataset, sessions, review, annotate: the choke points ─────────────────


def test_dataset_routes_refuse_an_outside_root_and_serve_an_inside_one(
    client: TestClient, tmp_path: Path, outside: Path, closed_project,
) -> None:
    inside = tmp_path / "proj"
    _image(inside / "images" / "2026-02-11" / "a.jpg")
    _image(outside / "images" / "2026-02-11" / "a.jpg")

    assert client.get("/api/dataset/tree", params={"dataset_root": str(outside)}).status_code == 403
    assert client.post("/api/dataset/select", json={
        "project_root": str(outside), "dataset_root": str(outside)}).status_code == 403
    assert store.project_root is None

    assert client.get("/api/dataset/tree", params={"dataset_root": str(inside)}).status_code == 200
    selected = client.post("/api/dataset/select", json={
        "project_root": str(inside), "dataset_root": str(inside), "date": "2026-02-11"})
    assert selected.status_code == 200
    assert selected.json()["selection"]["image_list"] == ["a.jpg"]
    _open(client, inside)
    assert store.project_root == inside.resolve()


def test_an_annotations_link_inside_an_allowed_root_loads_in_both_routes(
    client: TestClient, tmp_path: Path, closed_project,
) -> None:
    """The class registry route and the dataset tree's per-date scan read one directory under
    one guard: a symlink whose target genuinely sits inside the allow-set is admitted by both."""
    from tcip_annotation.json_io import write_annotations
    from tcip_annotation.state import Annotation, BBox
    from tcip_web.routes.dataset import _subjects_by_date

    project = _project(tmp_path)
    date = "2026-02-11"
    real_annotations = tmp_path.parent / "nas" / "annotations_store" / date
    real_annotations.mkdir(parents=True)
    write_annotations(str(real_annotations / "IMG_0001.json"),
                      [Annotation(subject="bud", geometry=BBox(1, 1, 5, 5))], 10, 10)
    ann_dir = project / "annotations"
    ann_dir.mkdir()
    _link_to(ann_dir / date, real_annotations)

    by_date, problem = _subjects_by_date(project, [date])
    assert by_date[date] == ["bud"]
    assert problem is None

    load = client.get("/api/classes/load", params={
        "project_root": str(project), "dataset_root": str(project),
        "annotations_dir": str(ann_dir / date)})
    assert load.status_code == 200
    assert set(load.json()["subjects"]) == {"bud"}

    select = client.post("/api/dataset/select", json={
        "project_root": str(project), "dataset_root": str(project),
        "subject": "bud", "date": date})
    assert select.status_code == 200
    assert select.json()["annotations_present"] is True
    assert select.json()["label_problem"] is None


def test_an_annotations_link_outside_every_allowed_root_is_refused_by_both_routes(
    client: TestClient, tmp_path: Path, outside: Path, closed_project,
) -> None:
    """The same directory 403s the class registry route and is reported as this date's problem
    by the dataset tree, rather than the tree quietly listing what the registry route refuses."""
    from tcip_annotation.json_io import write_annotations
    from tcip_annotation.state import Annotation, BBox
    from tcip_web.routes.dataset import _subjects_by_date

    project = _project(tmp_path)
    date = "2026-02-11"
    real_annotations = outside / "nas" / "annotations_store" / date
    real_annotations.mkdir(parents=True)
    write_annotations(str(real_annotations / "IMG_0001.json"),
                      [Annotation(subject="bud", geometry=BBox(1, 1, 5, 5))], 10, 10)
    ann_dir = project / "annotations"
    ann_dir.mkdir()
    _link_to(ann_dir / date, real_annotations)

    by_date, problem = _subjects_by_date(project, [date])
    assert by_date[date] == []
    assert problem is not None and "outside the allowed roots" in problem

    resp = client.get("/api/classes/load", params={
        "project_root": str(project), "dataset_root": str(project),
        "annotations_dir": str(ann_dir / date)})
    assert resp.status_code == 403

    # The selection door is advisory and never rejects, so it reports the same refusal as the
    # date's label problem rather than scanning what the other two routes refuse.
    select = client.post("/api/dataset/select", json={
        "project_root": str(project), "dataset_root": str(project),
        "subject": "bud", "date": date})
    assert select.status_code == 200
    assert select.json()["annotations_present"] is False
    assert "outside the allowed roots" in (select.json()["label_problem"] or "")


def test_the_state_store_persists_under_the_guarded_root_not_the_snapshot_it_loaded(
    tmp_path: Path, outside: Path,
) -> None:
    """A gui.json edited on disk to name another project must not redirect the next flush."""
    from tcip_mcp.web_client import gui_snapshot_key
    from tcip_web.state import DatasetSelection, GuiState, StateStore

    inside = _project(tmp_path / "proj")
    tcip_store.replace(gui_snapshot_key(str(inside)),
                       GuiState(dataset=DatasetSelection(project_root=str(outside))).model_dump(mode="json"))
    s = StateStore()
    assert s.open_project(inside.resolve()) is True
    assert s.state.dataset.project_root == str(outside)
    s._flush_sync()
    assert tcip_store.read(gui_snapshot_key(str(inside)), default=None) is not None
    assert tcip_store.read(gui_snapshot_key(str(outside)), default=None) is None
    assert not (outside / ".tcip").exists()


def test_session_routes_confine_the_project_root_and_the_dataset_root_they_record(
    client: TestClient, tmp_path: Path, outside: Path,
) -> None:
    inside = _project(tmp_path / "proj")
    assert client.post("/api/sessions/start", json={"project_root": str(outside)}).status_code == 403
    assert client.get("/api/sessions/load", params={"project_root": str(outside)}).status_code == 403
    assert client.post("/api/sessions/image_event", json={
        "project_root": str(inside), "image_name": "a.jpg", "final_annotation_count": 1,
        "dataset_root": str(outside)}).status_code == 403
    assert not (outside / ".tcip").exists()

    assert client.post("/api/sessions/start", json={"project_root": str(inside)}).status_code == 200
    ok = client.post("/api/sessions/image_event", json={
        "project_root": str(inside), "image_name": "a.jpg", "final_annotation_count": 1,
        "session_seconds_delta": 2.0, "dataset_root": str(inside)})
    assert ok.status_code == 200
    loaded = client.get("/api/sessions/load", params={"project_root": str(inside)})
    assert loaded.status_code == 200
    assert loaded.json()["sessions"][0]["images"]["a.jpg"]["dataset_root"] == str(inside.resolve())


def test_review_routes_confine_the_dataset_root_and_the_label_files_they_read(
    client: TestClient, tmp_path: Path, outside: Path,
) -> None:
    inside = _project(tmp_path / "proj")
    image = _image(inside / "images" / "2026-02-11" / "a.jpg")
    assert client.get("/api/review/image_statuses", params={
        "dataset_root": str(outside)}).status_code == 403
    assert client.post("/api/review/matches", json={
        "dataset_root": str(inside), "image_name": "a.jpg", "image_path": str(image),
        "gt_path": str(outside / "a.json")}).status_code == 403
    assert not (outside / ".tcip").exists()

    assert client.get("/api/review/image_statuses", params={
        "dataset_root": str(inside)}).status_code == 200
    assert client.post("/api/review/matches", json={
        "dataset_root": str(inside), "image_name": "a.jpg", "image_path": str(image)}).status_code == 200


def test_a_label_write_is_refused_before_it_happens_when_its_dataset_root_is_outside(
    client: TestClient, tmp_path: Path, outside: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The audit scope is derived and guarded before the label is written, so a refused write
    leaves no label behind. The image is admitted through the additive roots so that only the
    label's own dataset root is what refuses."""
    image = _image(outside / "dataset" / "images" / "2026-02-11" / "a.jpg")
    label = outside / "dataset" / "annotations" / "2026-02-11" / "a.json"
    monkeypatch.setenv("TCIP_IMAGE_ROOTS", str(outside / "dataset" / "images"))
    resp = client.post("/api/annotate/labels", json={
        "image_path": str(image), "label_path": str(label), "annotations": []})
    assert resp.status_code == 403
    assert not label.exists()

    inside_image = _image(tmp_path / "proj" / "images" / "2026-02-11" / "a.jpg")
    inside_label = tmp_path / "proj" / "annotations" / "2026-02-11" / "a.json"
    ok = client.post("/api/annotate/labels", json={
        "image_path": str(inside_image), "label_path": str(inside_label), "annotations": []})
    assert ok.status_code == 200, ok.text
    assert inside_label.exists()


# ── the Results doors belong to the open project ──────────────────────────


@pytest.mark.usefixtures("seed_bud_operationalization", "closed_project")
def test_a_results_door_refuses_until_a_project_is_open_and_then_serves_its_own_evidence(
    client: TestClient, tmp_path: Path,
) -> None:
    body = _phenology_fixture(tmp_path, validated=True, fractions=(0.75, 1.0), detections=4)
    store.close_project()
    refused = client.post("/api/results/phenology_measurement", json=body)
    assert refused.status_code == 409
    assert "open a project" in refused.json()["detail"]

    _open(client, tmp_path, tmp_path / "ds")
    assert client.post("/api/results/phenology_measurement", json=body).status_code == 200


@pytest.mark.usefixtures("seed_bud_operationalization", "closed_project")
def test_a_delivery_cannot_name_one_project_while_another_is_open(
    client: TestClient, tmp_path: Path,
) -> None:
    body = _phenology_fixture(tmp_path, validated=True, fractions=(0.75, 1.0), detections=4)
    other = _project(tmp_path / "other")
    _open(client, other)
    resp = client.post("/api/results/phenology_measurement", json=body)
    assert resp.status_code == 403
    assert "not the open project" in resp.json()["detail"]
    assert not (other / "results_export").exists()


@pytest.mark.usefixtures("seed_bud_operationalization", "closed_project")
def test_a_delivery_from_another_projects_evidence_is_refused_by_name(
    client: TestClient, tmp_path: Path,
) -> None:
    """Project B, open and fully set up, is handed project A's mapping and predictions: both inside
    the managed allow-set, neither belonging to B. No export and no audit line lands in B."""
    body = _phenology_fixture(tmp_path, validated=True, fractions=(0.75, 1.0), detections=4)
    b = _project(tmp_path / "b")
    write_spec(b, BUD_OPENING)
    seed_confirmed_crossing(b, BUD_OPENING.name, measured_subject="bud")
    _open(client, b)

    diverted = {**body, "project_root": str(b)}
    resp = client.post("/api/results/phenology_measurement", json=diverted)
    assert resp.status_code == 403
    assert "does not belong to project" in resp.json()["detail"]
    resp = client.post("/api/results/export_csv",
                       json={**diverted, "payload": "milestones", "filename": "x.csv"})
    assert resp.status_code == 403
    assert not (b / "results_export").exists()
    assert not any(r["tool"] == "results.export_csv"
                   for r in tcip_store.read_log(audit_log_key(b)).records)


@pytest.mark.usefixtures("seed_bud_operationalization", "closed_project")
def test_a_delivery_from_a_dataset_registered_to_the_open_project_is_admitted(
    client: TestClient, tmp_path: Path, outside: Path,
) -> None:
    """Evidence living outside the workspace entirely, registered to the project, passes the
    belonging rail. The evidence gate then judges the relocated buckets on its own terms, so the
    proof here is that neither the belonging refusal nor the allow-set refusal answers."""
    import shutil

    body = _phenology_fixture(tmp_path, validated=True, fractions=(0.75, 1.0), detections=4)
    copied = outside / "ds"
    shutil.copytree(tmp_path / "ds", copied)
    relocated = {**body, "predictions_by_date": {
        d: str(copied / Path(p).relative_to(tmp_path / "ds"))
        for d, p in body["predictions_by_date"].items()}}
    _open(client, tmp_path)
    refused = client.post("/api/results/phenology_measurement", json=relocated)
    assert refused.status_code == 403
    assert "does not belong to project" in refused.json()["detail"]

    upsert_dataset(tmp_path, {"id": "ds-1", "path": str(copied), "crop": "currant",
                              "fingerprint": "v1:f"})
    resp = client.post("/api/results/phenology_measurement", json=relocated)
    assert resp.status_code not in (403, 409), resp.text


@pytest.mark.usefixtures("seed_bud_operationalization", "closed_project")
def test_a_mapping_build_writes_and_audits_under_the_open_project_only(
    client: TestClient, tmp_path: Path, outside: Path,
) -> None:
    payload = _capture_fixture(tmp_path)
    _open(client, tmp_path, tmp_path / "images")

    # The payload carries no path: a persist_path pointed outside the project names nothing
    # this door reads, so the build still lands under the open project, addressed by name.
    elsewhere = outside / "plant_mapping.json"
    resp = client.post("/api/results/plant_mapping/build",
                       json={**payload, "persist_path": str(elsewhere)})
    assert resp.status_code == 200, resp.text
    assert not elsewhere.exists()

    foreign_images = _image(outside / "images" / "2026-02-11" / "z.jpg").parent.parent
    resp = client.post("/api/results/plant_mapping/build",
                       json={**payload, "images_root": str(foreign_images)})
    assert resp.status_code == 403
    assert "does not belong to project" in resp.json()["detail"]

    # The breeder's plant-location file is reference data picked from wherever they keep it.
    moved_csv = outside / "plots.csv"
    moved_csv.write_text(Path(payload["csv_path"]).read_text(encoding="utf-8"), encoding="utf-8")
    from tests._binding_fixtures import register_plant_registry_for

    moved_registry = register_plant_registry_for([moved_csv], name="moved-plots")
    ok = client.post("/api/results/plant_mapping/build",
                     json={**payload, "plant_registry": moved_registry})
    assert ok.status_code == 200, ok.text
    built = [r for r in tcip_store.read_log(audit_log_key(tmp_path)).records
             if r["tool"] == "gui_build_plant_mapping"]
    assert len(built) == 2
    assert built[-1]["arguments"]["name"] == payload["name"]
    from tcip_mcp.pipelines.postprocessing import plant_mapping

    build = plant_mapping.load_mapping(tmp_path, payload["name"])
    assert build is not None
    assert set(build.assignments.keys()) == {"2026-02-11", "2026-02-25"}


# ── the picker: unconfined from this machine, confined from the network ───


def test_the_picker_browses_the_whole_machine_from_a_local_connection(
    client: TestClient, outside: Path,
) -> None:
    (outside / "somewhere").mkdir()
    resp = client.get("/api/fs/list", params={"path": str(outside)})
    assert resp.status_code == 200
    assert "somewhere" in {e["name"] for e in resp.json()["entries"]}


def test_the_picker_is_confined_to_the_allowed_roots_on_a_routable_connection(
    tmp_path: Path, outside: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A connection accepted on a routable address (the TestClient base URL sets the ASGI server
    address), served under the operator's exposure opt-in, lists only the allowed roots and
    refuses anything outside them."""
    monkeypatch.setenv("TCIP_WEB_ALLOW_INSECURE", "1")
    (outside / "somewhere").mkdir()
    workspace = tmp_path.parent
    lan = TestClient(app, base_url="http://192.168.1.23:8765")
    refused = lan.get("/api/fs/list", params={"path": str(outside)})
    assert refused.status_code == 403
    top = lan.get("/api/fs/list")
    assert top.status_code == 200
    assert str(workspace.resolve()) in [e["path"] for e in top.json()["entries"]]
    assert str(outside.resolve()) not in [e["path"] for e in top.json()["entries"]]
    inside = lan.get("/api/fs/list", params={"path": str(workspace)})
    assert inside.status_code == 200
    assert inside.json()["parent"] is None
