"""delete_stray_state_file deletes exactly the files stray_state.stray_state_files and the
doctor's check_stray_state_files agree are strays under a project's .tcip/state, and refuses by
name over everything else the accounting classifies: a claimed store's own file, a registered
checkpoint read as a blob, the storage backend's own bookkeeping, the state root's own database
home, a directory, a link or junction on any segment, a traversal attempt, a Windows
drive-relative path naming another drive, and a state root whose accounting itself refuses. A
project_root spelled in another case reaches the same verdicts.
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
    assert any(".tcip/state/probe/output.txt" in msg for _, msg in findings)

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
    """An AnchorMisplaced selection.json at the tree root, the same misplacement
    bundle's own tests provoke, is caught and answered rather than left to escape as a
    traceback: an error dict from the tool, a warn finding from the doctor."""
    root = _project(tmp_path, monkeypatch)
    state = _state(root)
    (state / "notes.json").write_text("{}", encoding="utf-8")
    (root / "selection.json").write_text("{}", encoding="utf-8")

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
    """coverage of the ordering the collisions sentence in stray_state's own docstring states: a
    path more than one derived root's plan claims refuses under whichever store the walk reaches
    first. No shipped claim table produces a cross-root collision, so the accounting is built here
    the way bundle's own BundleAccounting, AdoptionPlan and PlanEntry dataclasses let a caller
    construct one, which is the shape account_for itself would report."""
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


# ── the branches the predicate's own order reaches last ───────────────────


def test_refuses_a_registered_checkpoint_under_the_state_root_as_a_blob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """coverage of the blob branch, the one class reachable under .tcip/state: a model registry
    entry may name a checkpoint anywhere, so a .pt registered from under the state root is a
    recognized blob there and is refused as one rather than read as an unclaimed stray."""
    from tcip_mcp.tools.model_tools import register_model

    root = _project(tmp_path, monkeypatch)
    state = _state(root)
    checkpoint = state / "weights.pt"
    checkpoint.write_bytes(b"not a real checkpoint")

    res = register_model(name="m1", checkpoint_path=str(checkpoint),
                         config={"arch": "probe"}, project_path=str(root))
    assert "error" not in res, res

    assert checkpoint not in stray_state_files(root)
    target, refusal = stray_state_file_refusal(str(root), "weights.pt")

    assert refusal is not None and "blob" in refusal
    assert target == checkpoint
    assert checkpoint.is_file()


def test_refuses_a_link_and_leaves_the_file_it_points_at_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """guard. A link under the state root is refused by name, and the file it points at is not
    touched: the door acts on the path the caller named and the audit line records, so following
    the link would delete something neither of them names."""
    import os

    root = _project(tmp_path, monkeypatch)
    state = _state(root)
    real = state / "real.json"
    real.write_text("{}", encoding="utf-8")
    link = state / "link.json"
    try:
        os.symlink(real, link)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"this host does not allow creating a symlink: {exc}")

    target, refusal = stray_state_file_refusal(str(root), "link.json")
    assert refusal is not None and "link or junction" in refusal
    assert target == link

    res = delete_stray_state_file(project_root=str(root), relative_path="link.json", reason="x")
    assert "error" in res and "link or junction" in res["error"]
    assert real.is_file(), "the link's target must survive a refused delete"
    assert os.path.lexists(link)


@pytest.mark.skipif("sys.platform != 'win32'")
def test_refuses_a_drive_relative_path_naming_another_drive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """coverage. A Windows drive-relative path naming a different drive is neither absolute by
    is_external_form's grammar nor carries a .. segment, but joining it replaces the state root's
    own drive, so it normalizes outside the state root and is refused there. This is the one
    input that reaches that refusal, which would otherwise read as handling for a case nothing
    can produce."""
    root = _project(tmp_path, monkeypatch)
    other_drive = "Z:" if str(root)[:1].upper() != "Z" else "Y:"

    _target, refusal = stray_state_file_refusal(str(root), f"{other_drive}notes.json")

    assert refusal is not None and "outside the state root" in refusal


@pytest.mark.skipif("sys.platform != 'win32'")
def test_a_differently_cased_project_root_reaches_the_same_verdicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """coverage of the normalization rule the module docstring states: every membership test
    compares normcase, so a project_root spelled in another case classifies a stray and a claimed
    file exactly as the canonical spelling does."""
    root = _project(tmp_path, monkeypatch)
    state = _state(root)
    (state / "notes.json").write_text("{}", encoding="utf-8")
    shouted = str(root).upper()

    canonical_target, canonical = stray_state_file_refusal(str(root), "notes.json")
    shouted_target, shouted_refusal = stray_state_file_refusal(shouted, "notes.json")

    assert canonical is None and shouted_refusal is None
    assert str(shouted_target).lower() == str(canonical_target).lower()
    assert len(stray_state_files(shouted)) == len(stray_state_files(root))


def test_the_doctor_names_a_stray_under_a_relative_project_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """guard. The doctor's finding is built from the state root the listing itself resolved, never
    from the spelling the operator typed, so `tcip doctor .` or a root reached through a link
    reports the stray instead of raising out of a read-only diagnostic."""
    root = _project(tmp_path, monkeypatch)
    state = _state(root)
    (state / "notes.json").write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    findings: list = []
    doctor.check_stray_state_files(Path("proj"), findings)

    assert findings and findings[0][0] == "info"
    assert ".tcip/state/notes.json" in findings[0][1]
