"""Adoption's preflight decodes below the storage seam's own decode, so it runs the
schema-version check itself; an unsupported version is folded into the same plan refusal an
undecodable file gets, naming the version, since adoption reads and reports rather than acting
on a document's content.
"""

from __future__ import annotations

import pytest

import tcip_store as ts
from tcip_store.adoption import adopt_root
from tcip_store.file_backend import FileBackend, RootedFileLocator
from tcip_store.layout_claims import Claim, Constant, Patterned, literal
from tests._store_worker import CONTRACT_LAYOUT, DOCUMENT_DIR

_STORE = "adoption_schema_version_test"


def _claim() -> Claim:
    return Claim(CONTRACT_LAYOUT, ((Constant(DOCUMENT_DIR), Patterned(literal("frozen_stamp"), tail=".json")),))


def _register() -> None:
    if _STORE in ts.registered_stores():
        return
    ts.register_store(
        ts.StoreDescriptor(
            name=_STORE,
            kind="record",
            key_fields=("document",),
            frozen=True,
            codec=ts.RECORD_JSON,
            concurrency="last_writer_wins",
            locator=RootedFileLocator(prefix=(DOCUMENT_DIR,), suffix=".json"),
            claim=_claim(),
        )
    )


_register()


def _key(root) -> ts.Key:
    return ts.Key(_STORE, str(root), ("frozen_stamp",))


def test_a_version_one_document_adopts_through_the_platforms_own_writer(tmp_path):
    ts.bind(FileBackend())
    try:
        ts.replace(_key(tmp_path), {"n": 1, "schema_version": 1})
    finally:
        ts.unbind()

    result = adopt_root(str(tmp_path), CONTRACT_LAYOUT, report=lambda line: None)
    assert result.records.get(_STORE) == 1


def test_a_document_above_the_ceiling_refuses_the_root_naming_the_version(tmp_path):
    ts.bind(FileBackend())
    try:
        ts.replace(_key(tmp_path), {"n": 1, "schema_version": 2})
    finally:
        ts.unbind()

    with pytest.raises(ts.DecodeError) as raised:
        adopt_root(str(tmp_path), CONTRACT_LAYOUT, report=lambda line: None)

    message = str(raised.value)
    assert "schema_version" in message or "2" in message
    assert "frozen_stamp" in message
