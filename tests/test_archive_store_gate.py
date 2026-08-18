"""What an archive must refuse to bundle, and what a restored bundle still carries.

An archive is a file bundle, so under a database backend it ships the last export's files and
nothing else. A bundle taken while the database holds writes the files do not would restore a
project whose confirmed negatives are simply absent, and absence is what an unannotated image
looks like to a trainer. Both doors are checked here: before the copy, and again after it, so
the bundle reported is the bundle verified.
"""

from __future__ import annotations

import zipfile
from contextlib import contextmanager
from pathlib import Path

import tcip_store as ts
from tcip_mcp import dataset_layout
from tcip_mcp.tools import project_tools
from tcip_mcp.tools.project_tools import archive_project, import_project
from tcip_store.file_backend import FileBackend
from tcip_store.sqlite_backend import SqliteBackend

_NEGATIVE = {"catkin/2026-03-04": {"a_1.jpg": {"status": "negative", "by": "user:ü"}}}
"""One confirmed negative: an image a human marked done with nothing on it."""


@contextmanager
def bound(backend):
    """Bind one backend for a block, since these cases write as rows and archive as files."""
    ts.bind(backend)
    try:
        yield backend
    finally:
        ts.unbind()
        backend.close()


def export_files(root) -> None:
    """Write this root's rows back out as the files every reader outside the seam sees."""
    from tcip_store.export import export_root

    export_root(str(root), report=lambda line: None)


def _project(tmp_path: Path) -> Path:
    """A dataset root with one image, one empty label, and the registry that decodes it."""
    from tcip_mcp import class_registry
    from tcip_mcp.class_registry import ClassRegistry, Subject

    root = tmp_path / "project"
    (root / "images" / "2026-03-04").mkdir(parents=True)
    (root / "images" / "2026-03-04" / "a_1.jpg").write_bytes(b"\xff\xd8\xff")
    (root / "annotations" / "2026-03-04").mkdir(parents=True)
    (root / "annotations" / "2026-03-04" / "a_1.json").write_text(
        '{"annotations": []}', encoding="utf-8"
    )
    class_registry.write_registry(
        root / "classes.json", ClassRegistry(subjects=(Subject(name="catkin"),))
    )
    return root


def test_a_project_whose_state_is_not_in_its_files_refuses_to_archive(tmp_path):
    """The confirmed negative lives in a row the bundle would not carry, so the archive would
    restore an annotated project with none of its human decisions in it."""
    root = _project(tmp_path)
    with bound(SqliteBackend()):
        ts.replace(dataset_layout.image_status_key(root), _NEGATIVE, expect=ts.Version.ABSENT)

    result = archive_project(str(root), str(tmp_path / "bundle.zip"))

    assert "error" in result
    assert "scripts/export_store.py" in result["error"]
    assert "image_status" in result["error"]
    assert not (tmp_path / "bundle.zip").exists()


def test_a_project_whose_files_are_current_archives(tmp_path):
    """The partner of the refusal: the gate is about state the files are missing, and an
    exported project is not missing any."""
    root = _project(tmp_path)
    with bound(SqliteBackend()):
        ts.replace(dataset_layout.image_status_key(root), _NEGATIVE, expect=ts.Version.ABSENT)
        export_files(root)

    result = archive_project(str(root), str(tmp_path / "bundle.zip"))

    assert "error" not in result
    assert (tmp_path / "bundle.zip").is_file()


def test_a_project_written_during_the_copy_takes_its_own_output_back(tmp_path, monkeypatch):
    """The counters are re-read after the copy, so the bundle reported is the bundle verified
    rather than a mix of before and after."""
    root = _project(tmp_path)
    with bound(SqliteBackend()) as backend:
        ts.replace(dataset_layout.image_status_key(root), _NEGATIVE, expect=ts.Version.ABSENT)
        export_files(root)
        original = project_tools._database_counters
        seen: list[int] = []

        def moving(target: Path):
            counters = original(target)
            seen.append(len(seen))
            if len(seen) > 1:
                current = ts.read_versioned(dataset_layout.image_status_key(root))
                ts.replace(
                    dataset_layout.image_status_key(root),
                    {"catkin/2026-03-04": {"a_1.jpg": {"status": "complete", "by": "user:ü"}}},
                    expect=current.version,
                )
                return original(target)
            return counters

        monkeypatch.setattr(project_tools, "_database_counters", moving)
        result = archive_project(str(root), str(tmp_path / "bundle.zip"))
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

    result = archive_project(str(root), str(tmp_path / "bundle.zip"))

    assert "error" in result
    assert "not a SQLite database" in result["error"]
    assert not (tmp_path / "bundle.zip").exists()


def test_no_database_file_travels_in_the_bundle(tmp_path):
    """A bundle is a file layout: carrying the database too would give the restored project two
    authorities and no way to tell which one a reader is looking at."""
    root = _project(tmp_path)
    with bound(SqliteBackend()):
        ts.replace(dataset_layout.image_status_key(root), _NEGATIVE, expect=ts.Version.ABSENT)
        export_files(root)
    archive_project(str(root), str(tmp_path / "bundle.zip"))

    with zipfile.ZipFile(str(tmp_path / "bundle.zip")) as zf:
        names = zf.namelist()

    assert not [name for name in names if "store.db" in name]


def test_a_restored_project_conformed_to_a_database_still_holds_its_confirmed_negatives(
    tmp_path,
):
    """The whole round trip the gate exists for: rows out to files, files into a bundle, bundle
    into a fresh directory, and that directory adopted back into a database with the human's
    negative still saying negative."""
    from tcip_store.adoption import adopt_root
    from tcip_store.layout_claims import ROOT

    root = _project(tmp_path)
    with bound(SqliteBackend()):
        ts.replace(dataset_layout.image_status_key(root), _NEGATIVE, expect=ts.Version.ABSENT)
        export_files(root)
    archive_project(str(root), str(tmp_path / "bundle.zip"))

    restored = tmp_path / "restored"
    with bound(FileBackend()):
        assert "error" not in import_project(str(tmp_path / "bundle.zip"), str(restored))
        assert ts.read(dataset_layout.image_status_key(restored)) == _NEGATIVE

    adopt_root(str(restored), ROOT, report=lambda line: None)

    with bound(SqliteBackend()):
        assert ts.read(dataset_layout.image_status_key(restored)) == _NEGATIVE


def test_the_tcip_bundle_carries_a_retrospective(tmp_path):
    """A retrospective is prose the platform writes and every reader of it reads it as a file,
    so a bundle that drops it drops the project's own account of itself."""
    root = _project(tmp_path)
    (root / ".tcip" / "retrospectives").mkdir(parents=True)
    (root / ".tcip" / "retrospectives" / "session.md").write_text("what happened", encoding="utf-8")

    archive_project(str(root), str(tmp_path / "bundle.zip"))

    with zipfile.ZipFile(str(tmp_path / "bundle.zip")) as zf:
        names = zf.namelist()

    assert any(name.endswith("session.md") for name in names)
