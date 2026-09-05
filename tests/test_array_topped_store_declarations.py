"""Three registered stores wrap a top-level JSON array of entries rather than a keyed record: the
project dataset registry and the web job registry declare ``cannot_carry_field`` naming the
array-top shape, since neither has an object to hold ``schema_version`` on. The model registry
index used to be the same shape; it now wraps into ``{entries: [...]}`` (the relative-paths
family) and declares a cleared ``cannot_carry_field``, covered separately below.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import tcip_store as ts
from tcip_mcp.model_registry import (
    MODEL_REGISTRY_STORE,
    ModelRegistry,
    read_registry_index,
    registry_index_key,
)
from tcip_mcp.tools.project_tools import DATASET_REGISTRY_STORE, read_datasets, register_dataset
from tcip_store.binding import BACKEND_ENV, DEFAULT_BACKEND, FILE_BACKEND
from tcip_store.store import _backend
from tcip_web import jobstore


def _damage_record(key: ts.Key, data: bytes) -> None:
    """Overwrite a record's raw bytes, wherever the bound backend keeps it, bypassing the seam's
    own write-side schema_version check: the reader's explicit-``1`` accept branch has no producer
    that stamps the field, so this is the only way to reach it. The record must already exist."""
    name = os.environ.get(BACKEND_ENV) or DEFAULT_BACKEND
    if name == FILE_BACKEND:
        _backend().path_for(key).write_bytes(data)
        return
    from tcip_store.sqlite_backend import database_path, encode_parts

    conn = sqlite3.connect(str(database_path(str(key.root))), isolation_level=None)
    try:
        conn.execute(
            "update records set value = ? where store = ? and parts = ?",
            (data, key.store, encode_parts(key.parts)),
        )
    finally:
        conn.close()


def _cannot_carry_stores() -> tuple[str, ...]:
    return (DATASET_REGISTRY_STORE, jobstore.JOB_REGISTRY_STORE)


def test_every_still_array_topped_store_declares_cannot_carry_with_the_array_top_wording():
    for name in _cannot_carry_stores():
        descriptor = ts.get_descriptor(name)
        assert descriptor.frozen
        assert descriptor.cannot_carry_field, name
        assert "array" in descriptor.cannot_carry_field


def test_model_registry_declares_a_cleared_cannot_carry_field_and_ceiling_one():
    descriptor = ts.get_descriptor(MODEL_REGISTRY_STORE)
    assert descriptor.frozen
    assert descriptor.cannot_carry_field == ""
    assert descriptor.schema_version == 1


def test_model_registry_document_composes_with_its_own_declaration(tmp_path: Path):
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"weights")
    ModelRegistry(str(tmp_path)).register_model("a", str(ckpt), {}, metrics_source=None)

    raw = ts.read(registry_index_key(tmp_path))
    assert "schema_version" not in raw
    ts.check_schema_version(ts.get_descriptor(MODEL_REGISTRY_STORE), raw)


def test_model_registry_index_reads_as_an_empty_list_for_a_fresh_project(tmp_path: Path):
    assert read_registry_index(tmp_path) == []


def test_model_registry_document_with_an_explicit_schema_version_one_reads_the_same(
    tmp_path: Path,
):
    """The reader's explicit-1 accept branch: a document stamped with the frozen ceiling's own
    number, rather than left absent, is not a version this reader has never seen. No production
    writer stamps the field (absence is the frozen default), so this plants it by rewriting the
    record's own raw bytes, the same technique a document from before this store's version-1
    reset is planted with elsewhere."""
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"weights")
    ModelRegistry(str(tmp_path)).register_model("a", str(ckpt), {}, metrics_source=None)

    key = registry_index_key(tmp_path)
    raw = ts.read(key)
    descriptor = ts.get_descriptor(MODEL_REGISTRY_STORE)
    _damage_record(key, descriptor.codec.encode({**raw, "schema_version": 1}))

    assert read_registry_index(tmp_path) == raw["entries"]


def test_dataset_registry_composes_with_its_own_declaration(tmp_path: Path):
    root = tmp_path / "dataset"
    root.mkdir()
    result = register_dataset(str(root), "chestnut", str(tmp_path))
    assert "error" not in result, result

    entries = read_datasets(tmp_path)
    assert entries and isinstance(entries, list)
    ts.check_schema_version(ts.get_descriptor(DATASET_REGISTRY_STORE), entries)


def test_job_registry_composes_with_its_own_declaration(tmp_path: Path):
    key = jobstore.job_registry_key("inference_jobs", root=tmp_path)
    jobstore.persist_to(key, [{"id": "job-1", "state": "completed"}])

    entries = jobstore.load("inference_jobs")
    assert entries and isinstance(entries, list)
    ts.check_schema_version(ts.get_descriptor(jobstore.JOB_REGISTRY_STORE), entries)
