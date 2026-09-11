"""delete_stray_state_file deletes exactly the files stray_state.stray_state_files and the
doctor's check_stray_state_files agree are strays under a project's .tcip/state, and refuses by
name over everything else the accounting classifies (a claimed store's own file, a blob, the
storage backend's own bookkeeping, the state root's own database home, a directory, a link or
junction, a traversal attempt, and a state root whose accounting itself refuses).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tcip_mcp.cli import doctor
from tcip_mcp.stray_state import stray_state_file_refusal, stray_state_files
from tcip_mcp.tools.meta_tools import read_audit_log
from tcip_mcp.tools.phenology_tools import register_plant_registry
from tcip_mcp.tools.project_tools import delete_stray_state_file, initialize_project


def _project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "proj"
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "unused_workspace"))
    monkeypatch.setenv("TCIP_STATE_ROOT", str(root))
    res = initialize_project(str(root), site="orchard block")
    assert "error" not in res, res
    return root


def _state(root: Path) -> Path:
    state = root / ".tcip" / "state"
    state.mkdir(parents=True, exist_ok=True)
    return state


# ── admits-valid-work ─────────────────────────────────────────────────────


def test_admits_valid_work_deletes_each_stray_and_the_doctor_stops_naming_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _project(tmp_path, monkeypatch)
    state = _state(root)
    notes = state / "notes.json"
    notes.write_text("{}", encoding="utf-8")
    probe_dir = state / "probe"
    probe_dir.mkdir()
    output = probe_dir / "output.txt"
    output.write_text("x", encoding="utf-8")

    assert set(stray_state_files(root)) == {notes, output}

    findings: list = []
    doctor.check_stray_state_files(root, findings)
    assert any("notes.json" in msg for _, msg in findings)
    assert any(str(Path("probe") / "output.txt") in msg for _, msg in findings)

    for relative_path, path in (("notes.json", notes), (str(Path("probe") / "output.txt"), output)):
        size_before = path.stat().st_size
        res = delete_stray_state_file(
            project_root=str(root), relative_path=relative_path, reason="clearing a leftover file")
        assert "error" not in res, res
        assert res["relative_path"] == relative_path
        assert res["size_bytes"] == size_before
        assert res["reason"] == "clearing a leftover file"
        assert not path.exists()

    log = read_audit_log(scope=str(root), tool="delete_stray_state_file")
    assert log["count"] == 2
    assert all(e["arguments"]["reason"] == "clearing a leftover file" for e in log["entries"])

    findings_after: list = []
    doctor.check_stray_state_files(root, findings_after)
    assert findings_after == []
    assert stray_state_files(root) == ()


# ── refusals, each by name ────────────────────────────────────────────────


def test_refuses_a_plant_registry_file_naming_the_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _project(tmp_path, monkeypatch)
    csv_path = tmp_path / "plants.csv"
    csv_path.write_text(
        "plot_name,accession_name,WGS84_centroid_x,WGS84_centroid_y\n"
        "P1,acc-9,-90.058,43.197\n",
        encoding="utf-8",
    )
    reg = register_plant_registry(name="reg", csv_paths=[str(csv_path)], crop="black locust",
                                  site="block")
    assert "error" not in reg, reg

    from tcip_store.binding import is_database_backend

    relative_path = str(Path("plant_registries") / "reg.json")
    target_on_disk = root / ".tcip" / "state" / relative_path
    if is_database_backend():
        assert not target_on_disk.exists()
    else:
        assert target_on_disk.is_file()

    res = delete_stray_state_file(project_root=str(root), relative_path=relative_path, reason="x")
    assert "error" in res
    if is_database_backend():
        assert "does not exist" in res["error"]
    else:
        assert "plant_registries" in res["error"]


def test_refuses_image_status_json_naming_the_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tcip_store.binding import is_database_backend

    if is_database_backend():
        pytest.skip("image_status.json is a store's own document, read raw only under the file "
                    "backend; the database-backend case is covered by the plant-registries test")

    from tcip_mcp.dataset_layout import replace_image_status_store

    root = _project(tmp_path, monkeypatch)
    replace_image_status_store(root, {"bud/2026-03-04": {"a_1.jpg": {
        "status": "negative", "recorded_by": "user:breeder", "recorded_at": "2026-03-04T00:00:00Z"}}})

    res = delete_stray_state_file(
        project_root=str(root), relative_path="image_status.json", reason="x")
    assert "error" in res
    assert "image_status" in res["error"]


def test_refuses_the_state_roots_own_database_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _project(tmp_path, monkeypatch)
    state = _state(root)
    (state / ".tcip").mkdir(exist_ok=True)
    (state / ".tcip" / "store.db").write_bytes(b"not a real database, just bytes")

    res = delete_stray_state_file(
        project_root=str(root), relative_path=str(Path(".tcip") / "store.db"), reason="x")

    assert "error" in res
    assert "database home" in res["error"]


def test_refuses_a_lock_file_as_bookkeeping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _project(tmp_path, monkeypatch)
    state = _state(root)
    (state / "something.lock").write_text("x", encoding="utf-8")

    res = delete_stray_state_file(
        project_root=str(root), relative_path="something.lock", reason="x")

    assert "error" in res
    assert "backend's own artifact" in res["error"]


@pytest.mark.parametrize("relative_path", [
    "../project.json", str(Path("..") / "hpo" / "s" / "result.json"),
])
def test_refuses_a_relative_traversal_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative_path: str,
) -> None:
    root = _project(tmp_path, monkeypatch)
    _state(root)

    res = delete_stray_state_file(
        project_root=str(root), relative_path=relative_path, reason="x")

    assert "error" in res
    assert "carrying a .. segment" in res["error"]


def test_refuses_an_absolute_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _project(tmp_path, monkeypatch)
    _state(root)

    absolute = str((tmp_path / "elsewhere.json").resolve())
    res = delete_stray_state_file(project_root=str(root), relative_path=absolute, reason="x")

    assert "error" in res


def test_refuses_a_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _project(tmp_path, monkeypatch)
    state = _state(root)
    (state / "adir").mkdir()

    res = delete_stray_state_file(project_root=str(root), relative_path="adir", reason="x")

    assert "error" in res
    assert "directory" in res["error"]


def test_refuses_an_absent_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _project(tmp_path, monkeypatch)
    _state(root)

    res = delete_stray_state_file(
        project_root=str(root), relative_path="does-not-exist.json", reason="x")

    assert "error" in res
    assert "does not exist" in res["error"]


def test_refuses_an_empty_reason(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _project(tmp_path, monkeypatch)
    state = _state(root)
    (state / "notes.json").write_text("{}", encoding="utf-8")

    res = delete_stray_state_file(project_root=str(root), relative_path="notes.json", reason="  ")

    assert "error" in res
    assert "non-empty reason" in res["error"]
    assert (state / "notes.json").is_file()


def test_a_state_root_the_accounting_refuses_is_an_error_dict_and_a_warn_finding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An AnchorMisplaced split_manifest.json at the tree root, the same misplacement
    bundle's own tests provoke, is caught and answered rather than left to escape as a
    traceback: an error dict from the tool, a warn finding from the doctor."""
    root = _project(tmp_path, monkeypatch)
    state = _state(root)
    (state / "notes.json").write_text("{}", encoding="utf-8")
    (root / "split_manifest.json").write_text("{}", encoding="utf-8")

    res = delete_stray_state_file(project_root=str(root), relative_path="notes.json", reason="x")
    assert "error" in res
    assert "accounting refused" in res["error"]

    findings: list = []
    doctor.check_stray_state_files(root, findings)
    assert findings and findings[0][0] == "warn"
    assert "accounting refused" in findings[0][1]


def test_a_file_two_stores_claim_equally_refuses_naming_a_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No shipped claim table produces a real cross-root collision today (confirmed: fiona --
    no, rather ``CrossRootCollision`` has no producing test anywhere in the tree); this builds
    one the way ``bundle``'s own ``BundleAccounting``/``AdoptionPlan``/``PlanEntry`` dataclasses
    let a caller construct one directly, the shape the accounting itself would report if two
    stores' templates both matched one file (coverage over the ``collisions`` sentence in
    ``stray_state``'s own docstring)."""
    from tcip_mcp.tools import bundle as bundle_module
    from tcip_store.adoption import AdoptionPlan, PlanEntry

    root = _project(tmp_path, monkeypatch)
    state = _state(root)
    colliding = state / "colliding.json"
    colliding.write_text("{}", encoding="utf-8")

    real = bundle_module.account_for(root)
    plan_a = AdoptionPlan(root=str(state), layout="state",
                          entries=(PlanEntry(store="store_a", parts=(), path=colliding),),
                          claimed=(colliding,))
    plan_b = AdoptionPlan(root=str(state), layout="state",
                          entries=(PlanEntry(store="store_b", parts=(), path=colliding),),
                          claimed=(colliding,))
    synthetic = bundle_module.BundleAccounting(
        tree=real.tree, derived=real.derived, plans=(plan_a, plan_b), blobs=(), bookkeeping=(),
        unaccounted=(), collisions=(colliding,), registered_checkpoints=frozenset(),
    )
    monkeypatch.setattr(bundle_module, "account_for", lambda tree: synthetic)

    target, refusal = stray_state_file_refusal(str(root), "colliding.json")

    assert refusal is not None
    assert "store_a" in refusal
    assert target == colliding
