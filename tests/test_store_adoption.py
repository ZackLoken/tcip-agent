"""Moving an existing file layout into a database without losing or misattributing an entry.

Adoption is the one path that meets state written before any database existed, so the round
trip is the whole test: files in, database out, export back, byte for byte the same files. The
rest guards what adoption must refuse rather than guess, since every one of those guesses would
put a document under the wrong store or drop it entirely.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest

import tcip_store as ts
from tcip_store import export as store_export
from tcip_store.adoption import adopt_root, plan_root, unaccounted_files
from tcip_store.file_backend import FileBackend, RootedFileLocator, _is_bookkeeping
from tcip_store.layout_claims import STATE, literal, unconformed_files
from tcip_store.sqlite_backend import SqliteBackend, database_path
from tests._store_worker import (
    BLOB,
    CONTRACT_LAYOUT,
    LOG,
    LWW,
    NESTED,
    OVERLAP_DIR,
    OVERLAP_SERVED,
    OVERLAP_STRANDED,
    document_claim,
    register_contract_stores,
)

register_contract_stores()

ADOPT_ALPHA = "adopt_state_alpha"
ADOPT_BETA = "adopt_state_beta"
ADOPT_FREE = "adopt_state_free"
ADOPT_RIVAL = "adopt_state_rival"
_SHARED_SHAPE = RootedFileLocator(prefix=("documents",), suffix=".json")
_CLAIMS = {
    ADOPT_ALPHA: document_claim(literal("image_status")),
    ADOPT_BETA: document_claim(literal("gui")),
    ADOPT_FREE: document_claim(),
    ADOPT_RIVAL: document_claim(),
}
"""Four stores over one file shape: two naming their document, two claiming any of them."""

_declared = False


def _register_adoption_stores() -> None:
    """Declare the four, once per process, since a claim is in force while its store is."""
    global _declared
    if _declared:
        return
    _declared = True
    for name, claim in _CLAIMS.items():
        ts.register_store(
            ts.StoreDescriptor(
                name=name,
                kind="record",
                key_fields=("document",),
                codec=ts.RECORD_JSON,
                concurrency="last_writer_wins",
                locator=_SHARED_SHAPE,
                claim=claim,
            )
        )


_register_adoption_stores()

LAYOUT = CONTRACT_LAYOUT


@contextmanager
def bound(backend):
    """Bind one backend for a block: adoption hands a root from one to the other."""
    ts.bind(backend)
    try:
        yield backend
    finally:
        ts.unbind()
        backend.close()


def _key(store: str, root, *parts: str) -> ts.Key:
    return ts.Key(store, str(root), parts)


def _entries(root) -> dict[str, bytes]:
    """Every file under a root that is an entry rather than a backend's own bookkeeping."""
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and not _is_bookkeeping(path.name)
    }


def _write_a_layout(root) -> None:
    """A root written the way it would have been before any database existed."""
    ts.replace(_key(ADOPT_ALPHA, root, "image_status"), {"a_1.jpg": "negative", "ü": "complete"})
    ts.replace(_key(ADOPT_BETA, root, "gui"), {"active_tab": "annotate"})
    ts.replace(_key(LWW, root, "kept"), {"n": 1})
    ts.replace(_key(NESTED, root, "group", "member"), {"n": 2})
    ts.put_blob(_key(BLOB, root, "picture"), b"\x89PNG", expect=ts.Version.ABSENT)
    for epoch in (1, 2, 3):
        ts.append(_key(LOG, root, "metrics"), {"epoch": epoch, "note": "ü"})


def test_files_adopted_into_a_database_and_exported_again_are_the_same_files(tmp_path):
    """The round trip is the claim: nothing is dropped, reordered, re-encoded or re-placed."""
    with bound(FileBackend()):
        _write_a_layout(tmp_path)
    before = _entries(tmp_path)

    adopt_root(str(tmp_path), LAYOUT, report=lambda line: None)

    with bound(SqliteBackend()):
        assert ts.read(_key(ADOPT_ALPHA, tmp_path, "image_status")) == {
            "a_1.jpg": "negative", "ü": "complete"
        }
        assert ts.read(_key(NESTED, tmp_path, "group", "member")) == {"n": 2}
        page = ts.read_log(_key(LOG, tmp_path, "metrics"))
        assert [entry["epoch"] for entry in page.records] == [1, 2, 3]

    store_export.export_root(str(tmp_path), report=lambda line: None)
    assert _entries(tmp_path) == before


def test_an_adopted_root_is_stamped_current_so_a_file_reader_is_not_told_to_export(tmp_path):
    """Adoption loads exactly what the files already say, so the files are current the moment it
    returns; anything else would leave the doctor invalid until a redundant export ran."""
    with bound(FileBackend()):
        _write_a_layout(tmp_path)

    adopt_root(str(tmp_path), LAYOUT, report=lambda line: None)

    assert store_export.stale_stores(database_path(str(tmp_path))) == ()


def test_a_blob_keeps_its_file_and_never_becomes_a_row(tmp_path):
    """Blobs are files under every backend, so adoption must not take one in."""
    with bound(FileBackend()):
        _write_a_layout(tmp_path)

    adopt_root(str(tmp_path), LAYOUT, report=lambda line: None)

    assert (tmp_path / "blobs" / "picture.bin").read_bytes() == b"\x89PNG"
    states = store_export.read_store_states(database_path(str(tmp_path)))
    assert BLOB not in states


def test_a_file_that_will_not_decode_refuses_the_root_before_anything_is_written(tmp_path):
    """Bytes no reader can get back out are not something to load and find out about later, and
    the refusal is the whole root because a half-adopted root has no owner."""
    with bound(FileBackend()):
        _write_a_layout(tmp_path)
    (tmp_path / "lww" / "kept.json").write_bytes(b"{not json")

    with pytest.raises(ts.DecodeError) as raised:
        adopt_root(str(tmp_path), LAYOUT, report=lambda line: None)

    assert "kept" in str(raised.value)
    assert not database_path(str(tmp_path)).exists()


def test_a_log_line_that_will_not_decode_refuses_the_root(tmp_path):
    """A log is adopted entry by entry, so one unreadable line is one entry that would otherwise
    vanish between the file and the rows."""
    with bound(FileBackend()):
        for epoch in (1, 2):
            ts.append(_key(LOG, tmp_path, "metrics"), {"epoch": epoch})
    path = tmp_path / "logs" / "metrics.jsonl"
    path.write_bytes(path.read_bytes() + b"{torn\n")

    with pytest.raises(ts.DecodeError):
        adopt_root(str(tmp_path), LAYOUT, report=lambda line: None)

    assert not database_path(str(tmp_path)).exists()


def test_a_file_two_stores_claim_equally_refuses_rather_than_picking_one(tmp_path):
    """Thirteen shipped stores share the ``.tcip/state`` json shape. Attributing a document to
    the wrong one puts it under a store no reader asks, and counts its writes against that
    store's export stamp."""
    with bound(FileBackend()):
        ts.replace(_key(ADOPT_FREE, tmp_path, "anything"), {"n": 1})

    with pytest.raises(ts.StoreError) as raised:
        plan_root(str(tmp_path), LAYOUT)

    message = str(raised.value)
    assert ADOPT_FREE in message and ADOPT_RIVAL in message


def test_a_constant_key_wins_over_a_varying_one_claiming_the_same_file(tmp_path):
    """The partner of the tie: a store that says the document is called ``image_status`` says
    more about that file than one whose key is any name at all, so there is no tie to refuse."""
    with bound(FileBackend()):
        ts.replace(_key(ADOPT_ALPHA, tmp_path, "image_status"), {"n": 1})

    plan = plan_root(str(tmp_path), LAYOUT)

    assert [(entry.store, entry.parts) for entry in plan.entries] == [
        (ADOPT_ALPHA, ("image_status",))
    ]


def test_the_files_the_rail_refuses_are_exactly_the_files_a_plan_accounts_for(tmp_path):
    """One claim set answers both questions, so a file cannot be evidence a root is unconformed
    and at the same time be a file no plan takes in: that gap is a file read as absent once the
    database exists, which for a confirmed negative means an annotated image training as empty."""
    with bound(FileBackend()):
        _write_a_layout(tmp_path)

    plan = plan_root(str(tmp_path), LAYOUT)
    refused = set(unconformed_files(str(tmp_path), LAYOUT))

    assert refused == {entry.path for entry in plan.entries} | set(unaccounted_files((plan,)))
    assert refused == set(plan.claimed)


def test_a_layout_whose_files_are_all_planned_leaves_nothing_unaccounted(tmp_path):
    """The partner: the accounting must be quiet on a root that is fully described, or an
    operator learns to ignore it."""
    with bound(FileBackend()):
        _write_a_layout(tmp_path)

    plan = plan_root(str(tmp_path), LAYOUT)

    assert unaccounted_files((plan,)) == ()


def test_a_shard_whose_filename_cannot_spell_its_key_adopts_under_the_key_its_bytes_state(
    tmp_path,
):
    """A review shard's bucket and image both carry separators its filename sanitizes out, so
    the path cannot be inverted back to the key. The bytes carry the real key, and adoption
    reads them through the same recovery hook enumeration does, so the identity a caller gets
    back from ``keys`` is the one it can read with on either backend."""
    from tcip_annotation import review_engine

    bucket, image = "predictions/live/2026-03-04", "a/b.jpg"
    key = review_engine.review_verdict_key(tmp_path, bucket, image)
    payload = {"bucket": bucket, "img_name": image, "verdict": "accepted", "reviewed_by": "ü"}
    with bound(FileBackend()):
        ts.replace(key, payload, expect=ts.Version.ABSENT)
        as_files = ts.keys(review_engine.REVIEW_VERDICTS_STORE, str(tmp_path))
    assert [k.parts for k in as_files] == [(bucket, image)]

    adopt_root(str(tmp_path), STATE, report=lambda line: None)

    with bound(SqliteBackend()):
        as_rows = ts.keys(review_engine.REVIEW_VERDICTS_STORE, str(tmp_path))
        assert [k.parts for k in as_rows] == [(bucket, image)]
        assert ts.read(as_rows[0]) == payload


def test_adopting_a_root_whose_database_already_holds_everything_changes_nothing(tmp_path):
    """A root run through adoption twice is ordinary operator work, and the second run must not
    read a served store's own export back in on top of the rows it came from."""
    with bound(SqliteBackend()):
        ts.replace(_key(LWW, tmp_path, "fresh"), {"n": 1})
        store_export.export_root(str(tmp_path), report=lambda line: None)
    before = _entries(tmp_path)

    result = adopt_root(str(tmp_path), LAYOUT, report=lambda line: None)

    assert (result.records, result.log_entries) == ({}, {})
    assert _entries(tmp_path) == before
    with bound(SqliteBackend()):
        assert ts.keys(LWW, str(tmp_path)) == [_key(LWW, tmp_path, "fresh")]


def test_a_file_two_layouts_claim_refuses_rather_than_being_taken_in_under_one(tmp_path):
    """A directory serves whatever stores a caller roots there, so one path can be a legal entry
    of two stores under two layouts. Where the database holds state for one and none for the
    other, no marker says whose the file is, and adopting it under the planner's winner would
    count another store's document as this one's."""
    with bound(SqliteBackend()):
        ts.replace(_key(OVERLAP_SERVED, tmp_path, "served"), {"n": 1})
    overlap = tmp_path / OVERLAP_DIR
    overlap.mkdir(exist_ok=True)
    (overlap / "planted.json").write_text('{"n": 2}', encoding="utf-8")

    with pytest.raises(ts.StoreError) as raised:
        adopt_root(str(tmp_path), LAYOUT, report=lambda line: None)

    message = str(raised.value)
    assert OVERLAP_SERVED in message and OVERLAP_STRANDED in message
    assert "Nothing was written" in message


def test_a_store_whose_files_arrived_after_adoption_is_taken_in_on_a_second_run(tmp_path):
    """A store the database has never held is exactly what the rail refuses a conformed root
    for, so the remediation it names has to load those files and leave every served store's
    rows and export files alone."""
    with bound(FileBackend()):
        ts.replace(_key(LWW, tmp_path, "kept"), {"n": 1})
    adopt_root(str(tmp_path), LAYOUT, report=lambda line: None)
    with bound(SqliteBackend()):
        store_export.export_root(str(tmp_path), report=lambda line: None)
    exported = (tmp_path / "lww" / "kept.json").read_bytes()
    (tmp_path / "documents").mkdir(parents=True, exist_ok=True)
    (tmp_path / "documents" / "image_status.json").write_bytes(b'{\n  "a_1.jpg": "negative"\n}\n')

    with bound(SqliteBackend()):
        with pytest.raises(ts.StoreError) as refused:
            ts.read(_key(LWW, tmp_path, "kept"))
    assert "image_status.json" in str(refused.value)

    result = adopt_root(str(tmp_path), LAYOUT, report=lambda line: None)

    assert result.records == {ADOPT_ALPHA: 1}
    assert (tmp_path / "lww" / "kept.json").read_bytes() == exported
    with bound(SqliteBackend()):
        assert ts.read(_key(LWW, tmp_path, "kept")) == {"n": 1}
        assert ts.read(_key(ADOPT_ALPHA, tmp_path, "image_status")) == {"a_1.jpg": "negative"}
    assert store_export.stale_stores(database_path(str(tmp_path))) == ()


def test_a_root_holding_no_records_adopts_to_an_empty_database(tmp_path):
    """A root with nothing to move is still a root an operator may conform, and it must end up
    usable rather than refused."""
    with bound(FileBackend()):
        ts.put_blob(_key(BLOB, tmp_path, "picture"), b"\x89PNG", expect=ts.Version.ABSENT)

    adopt_root(str(tmp_path), LAYOUT, report=lambda line: None)

    with bound(SqliteBackend()):
        ts.replace(_key(LWW, tmp_path, "later"), {"n": 1})
        assert ts.read(_key(LWW, tmp_path, "later")) == {"n": 1}
