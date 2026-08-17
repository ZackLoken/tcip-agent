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
from tcip_store.adoption import (
    ANY,
    StoreSource,
    adopt_root,
    literal,
    plan_root,
    unaccounted_files,
)
from tcip_store.file_backend import FileBackend, RootedFileLocator, _is_bookkeeping
from tcip_store.sqlite_backend import SqliteBackend, database_path
from tests._store_worker import BLOB, LOG, LWW, NESTED, register_contract_stores

register_contract_stores()

ADOPT_ALPHA = "adopt_state_alpha"
ADOPT_BETA = "adopt_state_beta"
ADOPT_FREE = "adopt_state_free"
_SHARED_SHAPE = RootedFileLocator(prefix=(".tcip", "state"), suffix=".json")

_declared = False


def _register_adoption_stores() -> None:
    """Three stores over one file shape: two constants and one varying key."""
    global _declared
    if _declared:
        return
    _declared = True
    for name in (ADOPT_ALPHA, ADOPT_BETA, ADOPT_FREE):
        ts.register_store(
            ts.StoreDescriptor(
                name=name,
                kind="record",
                key_fields=("document",),
                codec=ts.RECORD_JSON,
                concurrency="last_writer_wins",
                locator=_SHARED_SHAPE,
            )
        )


_register_adoption_stores()

LAYOUT = "a_root_under_test"
SOURCES = {
    LWW: StoreSource(LAYOUT, (ANY,)),
    NESTED: StoreSource(LAYOUT, (ANY, ANY)),
    LOG: StoreSource(LAYOUT, (ANY,)),
    ADOPT_ALPHA: StoreSource(LAYOUT, (literal("image_status"),)),
    ADOPT_BETA: StoreSource(LAYOUT, (literal("gui"),)),
}
"""The inventory these cases adopt against, the shape scripts/_store_bootstrap.py carries."""


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

    adopt_root(str(tmp_path), LAYOUT, SOURCES, report=lambda line: None)

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

    adopt_root(str(tmp_path), LAYOUT, SOURCES, report=lambda line: None)

    assert store_export.stale_stores(database_path(str(tmp_path))) == ()


def test_a_blob_keeps_its_file_and_never_becomes_a_row(tmp_path):
    """Blobs are files under every backend, so adoption must not take one in."""
    with bound(FileBackend()):
        _write_a_layout(tmp_path)

    adopt_root(str(tmp_path), LAYOUT, SOURCES, report=lambda line: None)

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
        adopt_root(str(tmp_path), LAYOUT, SOURCES, report=lambda line: None)

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
        adopt_root(str(tmp_path), LAYOUT, SOURCES, report=lambda line: None)

    assert not database_path(str(tmp_path)).exists()


def test_a_file_two_stores_claim_equally_refuses_rather_than_picking_one(tmp_path):
    """Thirteen shipped stores share the ``.tcip/state`` json shape. Attributing a document to
    the wrong one puts it under a store no reader asks, and counts its writes against that
    store's export stamp."""
    with bound(FileBackend()):
        ts.replace(_key(ADOPT_FREE, tmp_path, "anything"), {"n": 1})
    ambiguous = dict(SOURCES)
    ambiguous[ADOPT_FREE] = StoreSource(LAYOUT, (ANY,))
    ambiguous[LWW] = StoreSource(LAYOUT, (ANY,))
    ambiguous["second_free"] = StoreSource(LAYOUT, (ANY,))
    ts.register_store(
        ts.StoreDescriptor(
            name="second_free",
            kind="record",
            key_fields=("document",),
            codec=ts.RECORD_JSON,
            concurrency="last_writer_wins",
            locator=_SHARED_SHAPE,
        )
    )

    with pytest.raises(ts.StoreError) as raised:
        plan_root(str(tmp_path), LAYOUT, ambiguous)

    message = str(raised.value)
    assert ADOPT_FREE in message and "second_free" in message


def test_a_constant_key_wins_over_a_varying_one_claiming_the_same_file(tmp_path):
    """The partner of the tie: a store that says the document is called ``image_status`` says
    more about that file than one whose key is any name at all, so there is no tie to refuse."""
    with bound(FileBackend()):
        ts.replace(_key(ADOPT_ALPHA, tmp_path, "image_status"), {"n": 1})
    with_free = dict(SOURCES)
    with_free[ADOPT_FREE] = StoreSource(LAYOUT, (ANY,))

    plan = plan_root(str(tmp_path), LAYOUT, with_free)

    assert [(entry.store, entry.parts) for entry in plan.entries] == [
        (ADOPT_ALPHA, ("image_status",))
    ]


def test_a_record_file_no_store_in_the_plan_owns_is_named_rather_than_left_behind(tmp_path):
    """A file left in the layout reads as absent once a database exists, which for a confirmed
    negative means an annotated image training as empty."""
    with bound(FileBackend()):
        ts.replace(_key(LWW, tmp_path, "kept"), {"n": 1})
        ts.replace(_key(ADOPT_FREE, tmp_path, "unclaimed"), {"n": 2})

    plan = plan_root(str(tmp_path), LAYOUT, SOURCES)
    left = unaccounted_files((plan,))

    assert [path.name for path in left] == ["unclaimed.json"]


def test_a_layout_whose_files_are_all_planned_leaves_nothing_unaccounted(tmp_path):
    """The partner: the accounting must be quiet on a root that is fully described, or an
    operator learns to ignore it."""
    with bound(FileBackend()):
        _write_a_layout(tmp_path)

    plan = plan_root(str(tmp_path), LAYOUT, SOURCES)

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

    sources = {review_engine.REVIEW_VERDICTS_STORE: StoreSource(LAYOUT, (ANY, ANY))}
    adopt_root(str(tmp_path), LAYOUT, sources, report=lambda line: None)

    with bound(SqliteBackend()):
        as_rows = ts.keys(review_engine.REVIEW_VERDICTS_STORE, str(tmp_path))
        assert [k.parts for k in as_rows] == [(bucket, image)]
        assert ts.read(as_rows[0]) == payload


def test_adopting_a_root_that_already_has_a_database_is_refused(tmp_path):
    """Its records are rows now, so a second adoption would build a database out of files the
    database no longer owns."""
    with bound(SqliteBackend()):
        ts.replace(_key(LWW, tmp_path, "fresh"), {"n": 1})

    with pytest.raises(ts.StoreError) as raised:
        adopt_root(str(tmp_path), LAYOUT, SOURCES, report=lambda line: None)

    assert "already exists" in str(raised.value)


def test_a_root_holding_no_records_adopts_to_an_empty_database(tmp_path):
    """A root with nothing to move is still a root an operator may conform, and it must end up
    usable rather than refused."""
    with bound(FileBackend()):
        ts.put_blob(_key(BLOB, tmp_path, "picture"), b"\x89PNG", expect=ts.Version.ABSENT)

    adopt_root(str(tmp_path), LAYOUT, SOURCES, report=lambda line: None)

    with bound(SqliteBackend()):
        ts.replace(_key(LWW, tmp_path, "later"), {"n": 1})
        assert ts.read(_key(LWW, tmp_path, "later")) == {"n": 1}
