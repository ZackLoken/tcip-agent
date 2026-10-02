"""What an archive must refuse to bundle, and what a restored bundle still carries.

An archive is a file bundle, so under a database backend it exports every database under the
tree to its files before scanning them, rather than restoring a project whose stored records are
simply absent because nobody exported first. Both doors are checked here: before the copy, and
again after it, so the bundle reported is the bundle verified.
"""

from __future__ import annotations

import zipfile
from contextlib import contextmanager
from pathlib import Path

import tcip_store as ts
from tcip_mcp.tools import project_tools
from tcip_mcp.tools.project_tools import archive_project, import_project
from tcip_mcp.web_client import gui_snapshot_key
from tcip_store.file_backend import FileBackend
from tcip_store.sqlite_backend import SqliteBackend

_RECORD = {"active_tab": "annotate", "active_subject": "ü"}
"""One record a root's database holds, which its files hold only once exported."""


@contextmanager
def bound(backend):
    """Bind one backend for a block, since these cases write as rows and archive as files.

    The suite's own backend is put back on the way out rather than dropped, since a process with
    nothing bound is the state the seam refuses outright.
    """
    from tcip_store.store import _backend

    previous = _backend()
    ts.bind(backend)
    try:
        yield backend
    finally:
        ts.bind(previous)
        backend.close()


def export_files(root) -> None:
    """Write this root's rows back out as the files every reader outside the seam sees."""
    from tcip_store.export import export_root

    export_root(str(root), report=lambda line: None)


def _project(tmp_path: Path) -> Path:
    """A dataset root with one image, one empty label, and the registry that decodes it."""
    from tcip_mcp.subject_registry import SubjectRegistry, Subject
    from tests._producer_fixtures import registry_over

    root = tmp_path / "project"
    (root / "images" / "2026-03-04").mkdir(parents=True)
    (root / "images" / "2026-03-04" / "a_1.jpg").write_bytes(b"\xff\xd8\xff")
    (root / "annotations" / "2026-03-04").mkdir(parents=True)
    (root / "annotations" / "2026-03-04" / "a_1.json").write_text(
        '{"annotations": []}', encoding="utf-8"
    )
    registry_over(root, SubjectRegistry(subjects=(Subject(name="bud"),)))
    return root


def test_a_project_whose_state_is_not_yet_in_its_files_still_archives(tmp_path):
    """The record lives in a row no operator has exported yet; the door exports it itself before
    bundling, so a project the doors create archives without that step run by hand and the record
    is not simply absent from the restored files."""
    with bound(SqliteBackend()):
        root = _project(tmp_path)
        ts.replace(gui_snapshot_key(root), _RECORD, expect=ts.Version.ABSENT)
        result = archive_project(root, str(tmp_path / "bundle.zip"))

    assert "error" not in result
    with zipfile.ZipFile(str(tmp_path / "bundle.zip")) as zf:
        names = zf.namelist()
        record_name = next(name for name in names if name.endswith("state/gui.json"))
        content = zf.read(record_name)
    assert b"active_subject" in content


def test_a_project_whose_files_are_current_archives(tmp_path):
    """The partner of the refusal: the gate is about state the files are missing, and an
    exported project is not missing any."""
    with bound(SqliteBackend()):
        root = _project(tmp_path)
        ts.replace(gui_snapshot_key(root), _RECORD, expect=ts.Version.ABSENT)
        export_files(root)
        result = archive_project(root, str(tmp_path / "bundle.zip"))

    assert "error" not in result
    assert (tmp_path / "bundle.zip").is_file()


def test_a_project_written_during_the_copy_takes_its_own_output_back(tmp_path, monkeypatch):
    """The counters are re-read after the copy, so the bundle reported is the bundle verified
    rather than a mix of before and after."""
    with bound(SqliteBackend()) as backend:
        root = _project(tmp_path)
        ts.replace(gui_snapshot_key(root), _RECORD, expect=ts.Version.ABSENT)
        export_files(root)
        original = project_tools._database_counters
        seen: list[int] = []

        def moving(target: Path):
            counters = original(target)
            seen.append(len(seen))
            if len(seen) > 1:
                current = ts.read_versioned(gui_snapshot_key(root))
                ts.replace(gui_snapshot_key(root), {"active_tab": "results"},
                           expect=current.version)
                return original(target)
            return counters

        monkeypatch.setattr(project_tools, "_database_counters", moving)
        result = archive_project(root, str(tmp_path / "bundle.zip"))
        del backend

    assert "error" in result
    assert "changed while it was being archived" in result["error"]
    assert not (tmp_path / "bundle.zip").exists()


def test_a_file_that_is_not_a_database_refuses_the_archive_rather_than_tracebacking(tmp_path):
    """Whether the files hold this project's state is the question the archive gate answers, and
    a database it cannot open leaves that unanswered. The tool reports it; a driver error
    escaping untyped would come out of the MCP door as a traceback instead."""
    root = _project(tmp_path)
    (root / ".tcip").mkdir(parents=True, exist_ok=True)
    (root / ".tcip" / "store.db").write_bytes(b"not a database, just some bytes\n")

    result = archive_project(root, str(tmp_path / "bundle.zip"))

    assert "error" in result
    assert "not a SQLite database" in result["error"]
    assert not (tmp_path / "bundle.zip").exists()


def test_no_database_file_travels_in_the_bundle(tmp_path):
    """A bundle is a file layout: carrying the database too would give the restored project two
    authorities and no way to tell which one a reader is looking at."""
    with bound(SqliteBackend()):
        root = _project(tmp_path)
        ts.replace(gui_snapshot_key(root), _RECORD, expect=ts.Version.ABSENT)
        export_files(root)
        archive_project(root, str(tmp_path / "bundle.zip"))

    with zipfile.ZipFile(str(tmp_path / "bundle.zip")) as zf:
        names = zf.namelist()

    assert not [name for name in names if "store.db" in name]


def test_a_restored_project_conformed_to_a_database_still_holds_its_records(
    tmp_path, monkeypatch,
):
    """The whole round trip the gate exists for: rows out to files, files into a bundle, bundle
    into a fresh directory, and that directory adopted back into a database with the record
    still saying what it said.
    """
    from tcip_store.adoption import adopt_root
    from tcip_store.layout_claims import ROOT

    with bound(SqliteBackend()):
        root = _project(tmp_path)
        ts.replace(gui_snapshot_key(root), _RECORD, expect=ts.Version.ABSENT)
        export_files(root)
        archive_project(root, str(tmp_path / "bundle.zip"))

    restored = tmp_path / "restored"
    with bound(FileBackend()):
        assert "error" not in import_project(str(tmp_path / "bundle.zip"), str(restored))
        assert ts.read(gui_snapshot_key(restored)) == _RECORD

    adopt_root(str(restored), ROOT, report=lambda line: None)

    with bound(SqliteBackend()):
        assert ts.read(gui_snapshot_key(restored)) == _RECORD


def test_a_render_cache_and_hash_cache_are_not_bundled_and_are_counted(tmp_path):
    """The narrowed bundle: a render-cache sidecar and image_hash_cache.json carry into the
    archive none of them; the left-behind count covers the whole tree, an unbundled stray outside
    .tcip included. A sweep's directory travels whole, Ray's own experiment store in it
    included (Ray is the real producer there; this suite's faked tune_search never runs it, so
    the file is hand-placed)."""
    root = _project(tmp_path)
    (root / ".tcip" / "cache" / "img").mkdir(parents=True)
    (root / ".tcip" / "cache" / "img" / "abc123.jpg").write_bytes(b"\xff\xd8\xff")
    (root / ".tcip" / "hpo" / "study1").mkdir(parents=True)
    (root / ".tcip" / "hpo" / "study1" / "experiment_state-2026-01-01_00-00-00.json").write_text(
        "{}", encoding="utf-8")
    (root / ".tcip" / "state").mkdir(parents=True, exist_ok=True)
    (root / ".tcip" / "state" / "image_hash_cache.json").write_text("{}", encoding="utf-8")
    (root / "an_unbundled_stray.txt").write_text("stray", encoding="utf-8")

    result = archive_project(root, str(tmp_path / "bundle.zip"))

    assert "error" not in result
    with zipfile.ZipFile(str(tmp_path / "bundle.zip")) as zf:
        names = zf.namelist()
    assert not any("cache" in name for name in names)
    assert any("experiment_state" in name for name in names)
    assert not any("an_unbundled_stray" in name for name in names)
    assert result["left_behind"]["unaccounted"] >= 3


def test_the_tcip_bundle_carries_a_retrospective(tmp_path):
    """A retrospective is prose the platform writes and every reader of it reads it as a file,
    so a bundle that drops it drops the project's own account of itself."""
    from tcip_mcp.tools.meta_tools import write_retrospective

    root = _project(tmp_path)
    write_retrospective(root, "session", task="a task", worked="what happened", did_not_work="")

    archive_project(root, str(tmp_path / "bundle.zip"))

    with zipfile.ZipFile(str(tmp_path / "bundle.zip")) as zf:
        names = zf.namelist()

    assert any(name.endswith("session.md") for name in names)
