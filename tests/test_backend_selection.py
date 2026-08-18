"""Which backend a process entry point binds, and what it does with a name it does not know.

Every entry point binds through one call, so the answer to "where did this process write" is one
fact rather than one per script. An unset environment is the ordinary case and is what the
platform runs on; naming the file backend is the deliberate opt-out an exported layout is read
under.
"""

from __future__ import annotations

import pytest

from tcip_store.binding import BACKEND_ENV, FILE_BACKEND, bind_default
from tcip_store.file_backend import FileBackend
from tcip_store.sqlite_backend import SqliteBackend


def test_an_unset_environment_binds_the_database_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The platform's own default: records and logs live in one database per root."""
    monkeypatch.delenv(BACKEND_ENV, raising=False)

    backend = bind_default()
    try:
        assert isinstance(backend, SqliteBackend)
    finally:
        backend.close()


def test_naming_the_file_backend_binds_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """The opt-out stays reachable: an exported layout is read by asking for it by name."""
    monkeypatch.setenv(BACKEND_ENV, FILE_BACKEND)

    backend = bind_default()
    try:
        assert isinstance(backend, FileBackend)
    finally:
        backend.close()


def test_a_name_no_backend_answers_to_refuses_and_lists_the_ones_that_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A misspelling that fell back would send a process's writes somewhere nobody chose."""
    monkeypatch.setenv(BACKEND_ENV, "sqlite3")

    with pytest.raises(ValueError) as caught:
        bind_default()

    message = str(caught.value)
    assert "'sqlite3'" in message
    assert "'file'" in message
    assert "'sqlite'" in message
