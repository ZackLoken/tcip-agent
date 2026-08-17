"""The doctor reads state as files, so it has to say when the files are behind the database.

Four of its checks read raw bytes off disk. Under a database backend those bytes are whatever
the last export wrote, so a check that ran anyway would report a clean project from an earlier
session's state. Zero findings and "this could not be checked" are different answers, and the
whole point of running the doctor at session start is that it never quietly gives the first
one for the second.
"""

from __future__ import annotations

from contextlib import contextmanager

import tcip_store as ts
from scripts import doctor
from tcip_mcp import dataset_layout
from tcip_store.file_backend import FileBackend
from tcip_store.sqlite_backend import SqliteBackend


@contextmanager
def bound(backend):
    """Bind one backend for a block: these cases write as rows and read as files."""
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


def _project(tmp_path):
    """A dataset root with one image and one label, the shape the doctor's checks expect."""
    root = tmp_path / "project"
    (root / "images" / "2026-03-04").mkdir(parents=True)
    (root / "images" / "2026-03-04" / "a_1.jpg").write_bytes(b"\xff\xd8\xff")
    (root / "annotations" / "2026-03-04").mkdir(parents=True)
    (root / "annotations" / "2026-03-04" / "a_1.json").write_text(
        '{"annotations": []}', encoding="utf-8"
    )
    return root


def test_a_project_with_no_store_database_is_gated_on_nothing(tmp_path):
    """Most projects are on the file layout, where every check reads the authority directly."""
    root = _project(tmp_path)

    assert doctor.staleness_findings(root) == {}


def test_a_status_store_written_and_not_exported_makes_the_negatives_check_invalid(tmp_path):
    """The confirmed-negative check reads image_status off disk. Behind the database, the file
    it reads is the state of the last export, and an image marked negative since then reads as
    unconfirmed."""
    root = _project(tmp_path)
    with bound(SqliteBackend()):
        ts.replace(
            dataset_layout.image_status_key(root),
            {"catkin/2026-03-04": {"a_1.jpg": {"status": "negative", "by": "user:ü"}}},
            expect=ts.Version.ABSENT,
        )

    invalid = doctor.staleness_findings(root)

    assert "check_negatives" in invalid
    assert "image_status" in invalid["check_negatives"]


def test_the_gate_clears_once_the_files_have_been_written_out(tmp_path):
    """The partner of the refusal: a check the export has caught up with runs normally, or the
    doctor would be permanently invalid on every project that uses a database."""
    root = _project(tmp_path)
    with bound(SqliteBackend()):
        ts.replace(
            dataset_layout.image_status_key(root),
            {"catkin/2026-03-04": {"a_1.jpg": {"status": "negative", "by": "user:ü"}}},
            expect=ts.Version.ABSENT,
        )
        export_files(root)

    assert doctor.staleness_findings(root) == {}


def test_a_store_no_check_reads_never_gates_a_check(tmp_path):
    """The gate is exactly what the checks read: a live-state store the doctor never opens
    cannot make a check it does not feed report anything."""
    root = _project(tmp_path)
    with bound(SqliteBackend()):
        ts.replace(
            dataset_layout.view_coverage_key(root),
            {"catkin/2026-03-04": {}},
            expect=ts.Version.ABSENT,
        )

    assert doctor.staleness_findings(root) == {}


def test_a_stale_check_is_reported_as_an_error_naming_the_export_script(tmp_path, capsys):
    """What an operator actually sees: the doctor's own output says the check did not run and
    what to do about it, rather than printing a clean line for state it never read."""
    root = _project(tmp_path)
    with bound(SqliteBackend()):
        ts.replace(
            dataset_layout.image_status_key(root),
            {"catkin/2026-03-04": {"a_1.jpg": {"status": "negative", "by": "user:ü"}}},
            expect=ts.Version.ABSENT,
        )

    with bound(FileBackend()):
        findings: list[tuple[str, str]] = []
        invalid = doctor.staleness_findings(root)
        for check in (doctor.check_negatives,):
            reason = invalid.get(check.__name__)
            if reason:
                findings.append(("error", reason))
            else:
                check(root, findings)

    assert findings and findings[0][0] == "error"
    assert "image_status" in findings[0][1]


def test_a_file_that_is_not_a_database_makes_the_check_invalid_rather_than_tracebacking(
    tmp_path, monkeypatch, capsys
):
    """The doctor's contract is an exit code, and a check whose database will not open has to
    come out as that check being invalid. A driver error escaping untyped would leave the run
    with no findings, no exit code and a traceback."""
    root = _project(tmp_path)
    (root / ".tcip").mkdir(parents=True, exist_ok=True)
    (root / ".tcip" / "store.db").write_bytes(b"not a database, just some bytes\n")

    invalid = doctor.staleness_findings(root)
    monkeypatch.setattr("sys.argv", ["doctor.py", str(root)])
    code = doctor.main()
    printed = capsys.readouterr().out

    assert "not a SQLite database" in invalid["check_negatives"]
    assert code == 2
    assert "check_negatives" in printed and "invalid, not clean" in printed


def test_the_doctor_run_reports_the_invalid_check_and_exits_nonzero(tmp_path, monkeypatch, capsys):
    """End to end through the script the operating posture tells everyone to run."""
    root = _project(tmp_path)
    with bound(SqliteBackend()):
        ts.replace(
            dataset_layout.image_status_key(root),
            {"catkin/2026-03-04": {"a_1.jpg": {"status": "negative", "by": "user:ü"}}},
            expect=ts.Version.ABSENT,
        )
    monkeypatch.setattr("sys.argv", ["doctor.py", str(root)])

    code = doctor.main()
    printed = capsys.readouterr().out

    assert code == 2
    assert "check_negatives" in printed
    assert "scripts/export_store.py" in printed
    assert "invalid, not clean" in printed
