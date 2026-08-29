"""Tests for the per-project status pointer (tcip_mcp.project_status)."""

from __future__ import annotations

import os
from pathlib import Path

import tcip_store
from tcip_mcp.project_status import (
    PROJECT_STATUS_STORE,
    project_status_key,
    project_status_path,
    read_project_status,
    record_distillation,
    record_report,
    record_retrospective,
)
from tcip_store.binding import BACKEND_ENV, DEFAULT_BACKEND, FILE_BACKEND, SQLITE_BACKEND


def _damage_project_status(root: Path, data: bytes) -> None:
    """Put ``data`` behind the project status record, wherever the bound backend keeps it.

    A record must already exist at the key (``record_report`` seeds one); this corrupts the bytes
    behind it in place, on the same path the bound backend actually reads, so the case is genuine
    on both backends rather than reporting absence on one and corruption on the other.
    """
    from tcip_store.store import _backend

    key = project_status_key(root)
    name = os.environ.get(BACKEND_ENV) or DEFAULT_BACKEND
    if name == FILE_BACKEND:
        backend = _backend()
        backend.path_for(key).write_bytes(data)
        return
    if name != SQLITE_BACKEND:
        raise ValueError(f"no bytes-corruption path for backend {name!r}")
    import sqlite3

    from tcip_store.sqlite_backend import database_path, encode_parts

    conn = sqlite3.connect(str(database_path(str(key.root))), isolation_level=None)
    try:
        conn.execute(
            "update records set value = ? where store = ? and parts = ?",
            (data, key.store, encode_parts(key.parts)),
        )
    finally:
        conn.close()


def test_project_status_path(tmp_path: Path):
    assert project_status_path(tmp_path) == tmp_path / ".tcip" / "state" / "project_status.json"


def test_read_project_status_absent_returns_empty(tmp_path: Path):
    assert read_project_status(tmp_path) == {}


def test_read_project_status_corrupt_json_is_flagged(tmp_path: Path):
    record_report(tmp_path)  # seed a real record so a damaged one has somewhere to overwrite
    _damage_project_status(tmp_path, b"{not valid json")
    assert read_project_status(tmp_path) == {"_corrupt": True}


def test_read_project_status_non_dict_shape_is_flagged(tmp_path: Path):
    # Valid JSON, but not a dict: same shape guard as dataset_layout.normalize_status_store.
    tcip_store.replace(project_status_key(tmp_path), [1, 2, 3], expect=tcip_store.Version.ABSENT)
    assert read_project_status(tmp_path) == {"_corrupt": True}


def test_read_project_status_version_refused_is_flagged_distinctly_from_corrupt(tmp_path: Path):
    record_report(tmp_path)  # seed a real record so a poisoned one has somewhere to overwrite
    poisoned = tcip_store.get_descriptor(PROJECT_STATUS_STORE).codec.encode(
        {"reports_since_last_retrospective": 1, "schema_version": 99})
    _damage_project_status(tmp_path, poisoned)
    assert read_project_status(tmp_path) == {"_version_refused": True}


def test_record_functions_are_best_effort_on_a_version_refused_file(tmp_path: Path):
    # Unlike the corrupt case, a version-refused file is left untouched, not self-healed.
    record_report(tmp_path)
    poisoned = tcip_store.get_descriptor(PROJECT_STATUS_STORE).codec.encode(
        {"reports_since_last_retrospective": 1, "schema_version": 99})
    _damage_project_status(tmp_path, poisoned)

    record_report(tmp_path)  # must not raise
    assert read_project_status(tmp_path) == {"_version_refused": True}


def test_record_report_increments_both_counters(tmp_path: Path):
    record_report(tmp_path)
    status = read_project_status(tmp_path)
    assert status["reports_since_last_retrospective"] == 1
    assert status["reports_since_last_distillation"] == 1
    assert status["last_activity"]

    record_report(tmp_path)
    status = read_project_status(tmp_path)
    assert status["reports_since_last_retrospective"] == 2
    assert status["reports_since_last_distillation"] == 2


def test_record_retrospective_resets_retro_counter_not_distillation(tmp_path: Path):
    record_report(tmp_path)
    record_report(tmp_path)
    record_retrospective(tmp_path, "proj-a")

    status = read_project_status(tmp_path)
    assert status["reports_since_last_retrospective"] == 0  # reset
    assert status["retrospectives_since_last_distillation"] == 1  # bumped, not reset
    assert status["last_retrospective"]["project_id"] == "proj-a"
    assert "content" not in status["last_retrospective"]  # pointer only, no cached text
    assert "path" not in status["last_retrospective"]  # backend-dependent, never persisted
    assert status["last_retrospective"]["modified_at"]


def test_record_report_after_retrospective_does_not_reset_distillation_counters(tmp_path: Path):
    record_retrospective(tmp_path, "proj-a")
    record_report(tmp_path)

    status = read_project_status(tmp_path)
    assert status["reports_since_last_retrospective"] == 1  # counts from zero after the reset
    assert status["reports_since_last_distillation"] == 1
    assert status["retrospectives_since_last_distillation"] == 1  # untouched by a report


def test_record_distillation_resets_both_distillation_counters_only(tmp_path: Path):
    record_report(tmp_path)
    record_retrospective(tmp_path, "proj-a")
    record_report(tmp_path)

    record_distillation(tmp_path)
    status = read_project_status(tmp_path)
    assert status["reports_since_last_distillation"] == 0
    assert status["retrospectives_since_last_distillation"] == 0
    assert status["last_distillation_at"]
    # The retrospective counter (a different concern) is untouched by a distillation pass: it was
    # already reset to 0 by the record_retrospective call above, then bumped to 1 by the report
    # that followed it, and a distillation pass has no reason to touch it.
    assert status["reports_since_last_retrospective"] == 1


def test_record_functions_are_best_effort_on_a_corrupt_file(tmp_path: Path):
    # A pre-existing corrupt status record must not crash a record_* call: it's best-effort and
    # attached to a write (claude_reports/project_retrospective) that must not fail because of it.
    record_report(tmp_path)  # seed a real record so a damaged one has somewhere to overwrite
    _damage_project_status(tmp_path, b"{not valid json")

    record_report(tmp_path)  # must not raise
    status = read_project_status(tmp_path)
    assert status.get("_corrupt") is not True  # the write self-heals the corruption
    assert status["reports_since_last_retrospective"] == 1


def test_read_project_status_non_utf8_bytes_are_flagged_not_raised(tmp_path: Path):
    # UnicodeDecodeError is not a subclass of json.JSONDecodeError: a narrower except clause than
    # (OSError, ValueError) would let this raise straight through read_project_status.
    record_report(tmp_path)  # seed a real record so a damaged one has somewhere to overwrite
    _damage_project_status(tmp_path, b"\xff\xfe\x00\x01not valid utf-8")

    assert read_project_status(tmp_path) == {"_corrupt": True}
    record_report(tmp_path)  # must not raise past this either, same best-effort invariant


def test_concurrent_record_report_does_not_lose_updates(tmp_path: Path):
    # The increment must happen entirely inside the file_transaction lock: computing "old + 1"
    # from a read taken before the lock, then writing that absolute value, would let concurrent
    # callers race: all read the same old value, all write the same new one, updates lost.
    import threading

    n_threads = 25
    barrier = threading.Barrier(n_threads)

    def _call():
        barrier.wait()  # maximize actual overlap, not just "started around the same time"
        record_report(tmp_path)

    threads = [threading.Thread(target=_call) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    status = read_project_status(tmp_path)
    assert status["reports_since_last_retrospective"] == n_threads
    assert status["reports_since_last_distillation"] == n_threads
