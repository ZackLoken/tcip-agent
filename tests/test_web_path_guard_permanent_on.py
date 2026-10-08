"""The web layer's path guard with no env var set: derived allow-set, identity containment, and the
project-scoped Results doors.

Every refusal here is paired with the legitimate call the same guard must still admit. The
``tmp_path`` fixture is a project inside its own workspace (``<workspace>/project``), so a project
built there is admitted by the derived rule; ``outside`` is a directory pytest creates beside that
workspace, which no rule admits.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

import tcip_store
from tcip_mcp.audit import audit_log_key
from tcip_mcp.tools.project_tools import upsert_dataset
from tcip_web.paths import assert_path_allowed
from tcip_web.state import store

from tests._producer_fixtures import write_image
from tests._trait_fixtures import propose, seed_confirmed_crossing
from tests._trait_fixtures import BUD_OPENING
from tests._web_fixtures import new_project, open_new_project
from tests.test_results_mapping_summary_and_audit_anchoring import _capture_fixture

if TYPE_CHECKING:
    from tests._chain_fixtures import Series


@pytest.fixture
def outside(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A directory beside the test's workspace that no allow-set rule admits."""
    return tmp_path_factory.mktemp("outside")


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
    real = new_project(outside / "linked-project").root
    link = tmp_path.parent / "linked"
    try:
        link.symlink_to(real, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks not available on this machine: {exc}")
    assert assert_path_allowed(str(link / "images")) == (real / "images").resolve()


def test_a_dataset_registered_to_a_workspace_project_is_admitted_wherever_it_lives(
    tmp_path: Path, outside: Path,
) -> None:
    project = new_project(tmp_path).root
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

    project = new_project(tmp_path).root
    registered = register_dataset(project, str(project), "currant")
    assert "error" not in registered

    roots = allowed_roots()

    assert all(root.is_absolute() for root in roots)
    assert project.resolve() in roots


def test_an_imports_staging_tree_is_never_admitted_even_under_the_workspace(
    tmp_path: Path,
) -> None:
    """The import door stages a half-extracted project under ``<workspace>/.imports/<uuid>/``,
    which sits under the workspace, an allowed root; a route resolving into it while the import
    is in flight still refuses."""
    workspace = tmp_path.parent
    staged = workspace / ".imports" / "run-1" / "images" / "a.jpg"
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b"x")

    with pytest.raises(ValueError, match="outside the allowed roots"):
        assert_path_allowed(str(staged))


def test_image_roots_stay_additive_on_top_of_the_derived_set(
    tmp_path: Path, outside: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tcip_web.state import store

    extra = outside / "archive"
    extra.mkdir()
    store.configure(store.workspace, (extra.resolve(),))
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

    new_project(tmp_path)

    def broken(_root):
        raise tcip_store.DecodeError("registry bytes are not a document")

    monkeypatch.setattr(project_tools, "read_datasets", broken)
    with pytest.raises(RuntimeError, match="dataset registry of project"):
        allowed_roots()


# ── dataset, sessions, review, annotate: the choke points ─────────────────


def test_dataset_routes_refuse_an_outside_root_and_serve_an_inside_one(
    client: TestClient, tmp_path: Path, outside: Path,
) -> None:
    open_new_project(tmp_path)
    inside = tmp_path / "proj"
    write_image(inside / "images" / "2026-02-11" / "a.jpg", (8, 8))
    write_image(outside / "images" / "2026-02-11" / "a.jpg", (8, 8))

    assert client.get("/api/dataset/tree", params={"dataset_root": str(outside)}).status_code == 403
    assert client.post("/api/dataset/select", json={
        "dataset_root": str(outside), "date": "2026-02-11"}).status_code == 403
    assert store.state.dataset.dataset_root is None

    assert client.get("/api/dataset/tree", params={"dataset_root": str(inside)}).status_code == 200
    selected = client.post("/api/dataset/select", json={
        "dataset_root": str(inside), "date": "2026-02-11"})
    assert selected.status_code == 200
    assert selected.json()["selection"]["image_list"] == ["a.jpg"]


def test_the_proposals_route_confines_the_image_whose_bucket_it_reads(
    client: TestClient, tmp_path: Path, outside: Path,
) -> None:
    from tcip_mcp.tools.proposal_tools import stage_proposals

    inside = open_new_project(tmp_path / "proj").root
    image = write_image(inside / "images" / "2026-02-11" / "a.jpg", (8, 8))
    staged = stage_proposals(inside, str(image), model_name="sketch", boxes=[
        {"subject": "bud", "conf": 0.9, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}])
    assert "error" not in staged, staged
    stranger = write_image(outside / "images" / "2026-02-11" / "a.jpg", (8, 8))

    assert client.get("/api/annotate/proposals", params={
        "image_path": str(stranger), "bucket": staged["bucket"]}).status_code == 403
    assert not (outside / ".tcip").exists()

    resp = client.get("/api/annotate/proposals", params={
        "image_path": str(image), "bucket": staged["bucket"]})
    assert resp.status_code == 200, resp.text
    assert len(resp.json()["proposals"]) == 1


# ── the Results doors belong to the open project ──────────────────────────


def _series(project: Path) -> Series:
    """An assessed attributed series under ``project`` (``_chain_fixtures.attributed_series``),
    left open in the web backend."""
    pytest.importorskip("torch")
    from tests._chain_fixtures import attributed_series

    return attributed_series(project, fractions=(0.0, 1.0))


def test_a_results_door_refuses_until_a_project_is_open_and_then_serves_its_own_evidence(
    client: TestClient, tmp_path: Path,
) -> None:
    series = _series(tmp_path)
    body = series.body()
    asyncio.run(store.close_project(series.project.id))
    refused = client.post("/api/results/phenology_measurement", json=body)
    assert refused.status_code == 409
    assert "no project is open" in refused.json()["detail"]

    asyncio.run(store.open_project(series.project))
    assert client.post("/api/results/phenology_measurement", json=body).status_code == 200


def test_a_delivery_from_another_projects_evidence_is_refused_by_name(
    client: TestClient, tmp_path: Path,
) -> None:
    """Project B, open and fully set up, is handed project A's mapping and predictions: both inside
    the managed allow-set, neither belonging to B. No export and no audit line lands in B."""
    body = _series(tmp_path).body()
    b = new_project(tmp_path.parent / "b").root
    propose(b, BUD_OPENING)
    seed_confirmed_crossing(b, BUD_OPENING.name, measured_subject="bud")
    open_new_project(b)

    resp = client.post("/api/results/phenology_measurement", json=body)
    assert resp.status_code == 403
    assert "does not belong to project" in resp.json()["detail"]
    resp = client.post("/api/results/export_csv",
                       json={**body, "payload": "milestones", "filename": "x.csv",
                             "user": "tester"})
    assert resp.status_code == 403
    assert not (b / "results_export").exists()
    assert not any(r["tool"] in ("results.export_csv", "delivery_event")
                   for r in tcip_store.read_log(audit_log_key(b)).records)


def test_a_delivery_from_a_dataset_registered_to_the_open_project_is_admitted(
    client: TestClient, tmp_path: Path, outside: Path,
) -> None:
    """Evidence living outside the workspace entirely, registered to the project, passes the
    belonging rail. The measurement then judges the relocated buckets on its own terms, so the
    proof here is that neither the belonging refusal nor the allow-set refusal answers."""
    import shutil

    body = _series(tmp_path).body()
    copied = outside / "ds"
    tcip_store.release_root(body["dataset_root"])
    shutil.copytree(body["dataset_root"], copied)
    relocated = {**body, "dataset_root": str(copied)}
    refused = client.post("/api/results/phenology_measurement", json=relocated)
    assert refused.status_code == 403
    assert "does not belong to project" in refused.json()["detail"]

    upsert_dataset(tmp_path, {"id": "ds-1", "path": str(copied)})
    resp = client.post("/api/results/phenology_measurement", json=relocated)
    assert resp.status_code not in (403, 409), resp.text


def test_a_mapping_build_writes_and_audits_under_the_open_project_only(
    client: TestClient, tmp_path: Path, outside: Path,
) -> None:
    payload = _capture_fixture(tmp_path)
    open_new_project(tmp_path)

    # The payload carries no path: a persist_path pointed outside the project names nothing
    # this door reads, so the build still lands under the open project, addressed by name.
    elsewhere = outside / "plant_mapping.json"
    resp = client.post("/api/results/plant_mapping/build",
                       json={**payload, "persist_path": str(elsewhere)})
    assert resp.status_code == 200, resp.text
    assert not elsewhere.exists()

    foreign_images = write_image(outside / "images" / "2026-02-11" / "z.jpg", (8, 8)).parent.parent
    resp = client.post("/api/results/plant_mapping/build",
                       json={**payload, "images_root": str(foreign_images)})
    assert resp.status_code == 403
    assert "does not belong to project" in resp.json()["detail"]

    # The breeder's plant-location file is reference data picked from wherever they keep it.
    moved_csv = outside / "plots.csv"
    moved_csv.write_text(Path(payload["csv_path"]).read_text(encoding="utf-8"), encoding="utf-8")
    from tests._mapping_fixtures import register_plant_registry_for

    moved_registry = register_plant_registry_for(tmp_path, [moved_csv], name="moved-plots")
    ok = client.post("/api/results/plant_mapping/build",
                     json={**payload, "plant_registry": moved_registry})
    assert ok.status_code == 200, ok.text
    built = [r for r in tcip_store.read_log(audit_log_key(tmp_path)).records
             if r["tool"] == "plant_mapping_built"]
    assert len(built) == 2
    assert built[-1]["arguments"]["name"] == payload["name"]
    from tcip_mcp.pipelines.postprocessing import plant_mapping

    build = plant_mapping.load_mapping(tmp_path, payload["name"])
    assert build is not None
    assert set(build.assignments.keys()) == {"2026-02-11", "2026-02-25"}


# ── the picker: the whole machine, the only arrival the backend serves ───


def test_the_picker_browses_the_whole_machine_from_a_local_connection(
    client: TestClient, outside: Path,
) -> None:
    (outside / "somewhere").mkdir()
    resp = client.get("/api/fs/list", params={"path": str(outside)})
    assert resp.status_code == 200
    assert "somewhere" in {e["name"] for e in resp.json()["entries"]}
