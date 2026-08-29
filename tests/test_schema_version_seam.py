"""The version-field accept rule at the storage seam: absence reads as version 1, 1 accepts, a
present version above a frozen store's ceiling refuses naming the document, the ceiling and the
store, and so does a present version that is not a plain integer. The refusal is
SchemaVersionRefused, deliberately not a DecodeError subclass, so an existing corruption
softener cannot absorb it by accident.
"""

from __future__ import annotations

import pytest

import tcip_store as ts
from tcip_store.file_backend import FileBackend, RootedFileLocator
from tcip_store.layout_claims import ANY, Claim, Constant, Patterned
from tcip_store.schema_version import check_schema_version

_LAYOUT = "schema_version_seam_test_root"
_RECORD = "schema_version_seam_test_record"
_LOG = "schema_version_seam_test_log"
_UNSTABLE = "schema_version_seam_test_unstable"
_CANNOT_CARRY = "schema_version_seam_test_cannot_carry"


def _claim(directory: str, suffix: str) -> Claim:
    return Claim(_LAYOUT, ((Constant(directory), Patterned(ANY, tail=suffix)),))


def _register(name: str, **kwargs) -> None:
    if name in ts.registered_stores():
        return
    ts.register_store(ts.StoreDescriptor(name=name, **kwargs))


_register(
    _RECORD, kind="record", key_fields=("name",), frozen=True,
    codec=ts.RECORD_JSON, concurrency="last_writer_wins",
    locator=RootedFileLocator(prefix=("schema_version_seam_test",), suffix=".json"),
    claim=_claim("schema_version_seam_test", ".json"),
)
_register(
    _LOG, kind="log", key_fields=("name",), frozen=True,
    codec=ts.LOG_JSON,
    locator=RootedFileLocator(prefix=("schema_version_seam_test_log",), suffix=".jsonl"),
    claim=_claim("schema_version_seam_test_log", ".jsonl"),
)
_register(
    _UNSTABLE, kind="record", key_fields=("name",), frozen=False,
    codec=ts.RECORD_JSON, concurrency="last_writer_wins",
    locator=RootedFileLocator(prefix=("schema_version_seam_test_unstable",), suffix=".json"),
    claim=_claim("schema_version_seam_test_unstable", ".json"),
)
_register(
    _CANNOT_CARRY, kind="record", key_fields=("name",), frozen=True,
    cannot_carry_field="a plain marker with nowhere to hold a version",
    codec=ts.RECORD_JSON, concurrency="last_writer_wins",
    locator=RootedFileLocator(prefix=("schema_version_seam_test_cc",), suffix=".json"),
    claim=_claim("schema_version_seam_test_cc", ".json"),
)


def _key(store: str, root) -> ts.Key:
    return ts.Key(store, str(root), ("doc",))


def test_absence_and_version_one_accept_through_the_platforms_own_writer(tmp_path):
    key = _key(_RECORD, tmp_path)
    ts.replace(key, {"n": 1})
    assert ts.read(key) == {"n": 1}

    ts.replace(key, {"n": 2, "schema_version": 1})
    assert ts.read(key) == {"n": 2, "schema_version": 1}


def test_the_write_side_refuses_a_document_above_the_ceiling_naming_store_and_key(tmp_path):
    key = _key(_RECORD, tmp_path)
    with pytest.raises(ts.SchemaVersionRefused) as raised:
        ts.replace(key, {"n": 3, "schema_version": 2})
    message = str(raised.value)
    assert _RECORD in message
    assert "doc" in message
    assert "above the 1 this reader knows" in message
    assert ts.read(key, default=None) is None


def test_the_write_side_refuses_a_log_line_above_the_ceiling(tmp_path):
    key = _key(_LOG, tmp_path)
    with pytest.raises(ts.SchemaVersionRefused):
        ts.append(key, {"n": 1, "schema_version": 2})
    assert ts.read_log(key).records == []


def test_a_version_above_the_ceiling_refuses_naming_the_document_ceiling_and_store(tmp_path):
    # Planted directly on disk: the write side now refuses this same document, so a document
    # above the ceiling reaches a read only from a source other than this store's own writer.
    key = _key(_RECORD, tmp_path)
    path = FileBackend().path_for(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(ts.RECORD_JSON.encode({"n": 3, "schema_version": 2}))

    ts.bind(FileBackend())
    try:
        with pytest.raises(ts.SchemaVersionRefused) as raised:
            ts.read(key)
    finally:
        ts.unbind()
    message = str(raised.value)
    assert _RECORD in message
    assert "2" in message
    assert "1" in message


def test_a_malformed_version_refuses():
    descriptor = ts.get_descriptor(_RECORD)
    for bad in ("2", True, 0, -1, 1.5):
        with pytest.raises(ts.SchemaVersionRefused):
            check_schema_version(descriptor, {"schema_version": bad})


def test_an_unstable_or_cannot_carry_store_is_out_of_the_checks_scope():
    unstable = ts.get_descriptor(_UNSTABLE)
    check_schema_version(unstable, {"schema_version": 999})  # no raise: shape still moving

    cannot_carry = ts.get_descriptor(_CANNOT_CARRY)
    assert cannot_carry.cannot_carry_field
    check_schema_version(cannot_carry, {"schema_version": 999})  # no raise: nowhere to hold it


def test_read_log_reports_a_version_refused_line_separately_from_corrupt(tmp_path):
    key = _key(_LOG, tmp_path)
    ts.bind(FileBackend())
    try:
        ts.append(key, {"n": 1})
        poisoned = ts.get_descriptor(key.store).codec.encode({"n": 2, "schema_version": 99})
        with open(FileBackend().path_for(key), "ab") as handle:
            handle.write(poisoned + b"\n")
        ts.append(key, {"n": 3})

        page = ts.read_log(key)
    finally:
        ts.unbind()
    assert [r["n"] for r in page.records] == [1, 3]
    assert page.version_refused == (1,)
    assert page.corrupt == ()


def test_a_transaction_read_is_refused_the_same_way_as_a_plain_read(tmp_path):
    key = _key(_RECORD, tmp_path)
    path = FileBackend().path_for(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(ts.RECORD_JSON.encode({"n": 4, "schema_version": 5}))

    ts.bind(FileBackend())
    try:
        with pytest.raises(ts.SchemaVersionRefused):
            with ts.transaction(key) as txn:
                txn.read(key)
    finally:
        ts.unbind()


def test_the_registry_refuses_a_cannot_carry_declaration_on_a_store_that_is_not_frozen():
    with pytest.raises(ValueError) as bad:
        ts.register_store(
            ts.StoreDescriptor(
                name="schema_version_seam_test_bad_cc", kind="record", key_fields=("name",),
                frozen=False, cannot_carry_field="raw bytes",
                codec=ts.RECORD_JSON, concurrency="last_writer_wins",
                locator=RootedFileLocator(
                    prefix=("schema_version_seam_test_bad_cc",), suffix=".json"
                ),
            )
        )
    assert "cannot_carry_field" in str(bad.value)

    declared = ts.register_store(
        ts.StoreDescriptor(
            name="schema_version_seam_test_good_cc", kind="record", key_fields=("name",),
            frozen=True, cannot_carry_field="raw bytes",
            codec=ts.RECORD_JSON, concurrency="last_writer_wins",
            locator=RootedFileLocator(
                prefix=("schema_version_seam_test_good_cc",), suffix=".json"
            ),
        )
    )
    assert declared.cannot_carry_field == "raw bytes"


def test_the_registry_refuses_a_schema_version_ceiling_below_one():
    with pytest.raises(ValueError) as bad:
        ts.register_store(
            ts.StoreDescriptor(
                name="schema_version_seam_test_bad_ceiling", kind="record",
                key_fields=("name",), frozen=True, schema_version=0,
                codec=ts.RECORD_JSON, concurrency="last_writer_wins",
                locator=RootedFileLocator(
                    prefix=("schema_version_seam_test_bad_ceiling",), suffix=".json"
                ),
            )
        )
    assert "schema_version" in str(bad.value)


def test_a_store_that_states_nothing_defaults_to_not_frozen():
    declared = ts.register_store(
        ts.StoreDescriptor(
            name="schema_version_seam_test_unclassified", kind="record", key_fields=("name",),
            codec=ts.RECORD_JSON, concurrency="last_writer_wins",
            locator=RootedFileLocator(
                prefix=("schema_version_seam_test_unclassified",), suffix=".json"
            ),
        )
    )
    assert declared.frozen is False
