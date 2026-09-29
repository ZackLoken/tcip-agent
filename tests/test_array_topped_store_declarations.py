"""The project dataset registry wraps a top-level JSON array of entries rather than a keyed
record, and declares ``cannot_carry_field`` naming the array-top shape, since it has no object to
hold ``schema_version`` on. The model registry index wraps into ``{entries: [...]}`` and declares a
cleared ``cannot_carry_field``, covered separately below.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

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


def test_the_array_topped_store_declares_cannot_carry_with_the_array_top_wording():
    descriptor = ts.get_descriptor(DATASET_REGISTRY_STORE)
    assert descriptor.frozen
    assert descriptor.cannot_carry_field
    assert "array" in descriptor.cannot_carry_field


def test_model_registry_declares_a_cleared_cannot_carry_field_and_ceiling_one():
    descriptor = ts.get_descriptor(MODEL_REGISTRY_STORE)
    assert descriptor.frozen
    assert descriptor.cannot_carry_field == ""
    assert descriptor.schema_version == 1


def test_model_registry_document_composes_with_its_own_declaration(tmp_path: Path):
    pytest.importorskip("torch")
    from tests._verified_checkpoint_fixtures import checkpoint_file

    ckpt = checkpoint_file(tmp_path / "m.pt", "weights")
    ModelRegistry(str(tmp_path)).register_model("a", str(ckpt), {})

    raw = ts.read(registry_index_key(tmp_path))
    assert "schema_version" not in raw
    ts.check_schema_version(ts.get_descriptor(MODEL_REGISTRY_STORE), raw)


def test_model_registry_index_reads_as_an_empty_list_for_a_fresh_project(tmp_path: Path):
    assert read_registry_index(tmp_path) == []


def test_dataset_registry_composes_with_its_own_declaration(tmp_path: Path):
    root = tmp_path / "dataset"
    root.mkdir()
    result = register_dataset(str(root), "chestnut", str(tmp_path))
    assert "error" not in result, result

    entries = read_datasets(tmp_path)
    assert entries and isinstance(entries, list)
    ts.check_schema_version(ts.get_descriptor(DATASET_REGISTRY_STORE), entries)
