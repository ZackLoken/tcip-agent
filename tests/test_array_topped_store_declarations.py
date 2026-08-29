"""The three registered stores whose frozen document is a top-level JSON array, not an object:
the model registry index, the project dataset registry, and the web job registry. None has an
object to hold ``schema_version`` on, so each declares ``cannot_carry_field`` naming the array-top
shape, rather than the version check silently no-opping on them with no stated reason. The
document bytes these stores write are unchanged; only the descriptor's own classification is new.
"""

from __future__ import annotations

from pathlib import Path

import tcip_store as ts
from tcip_mcp.model_registry import MODEL_REGISTRY_STORE, read_registry_index
from tcip_mcp.tools.project_tools import DATASET_REGISTRY_STORE, read_datasets, register_dataset
from tcip_web import jobstore


def _array_topped_stores() -> tuple[str, ...]:
    return (MODEL_REGISTRY_STORE, DATASET_REGISTRY_STORE, jobstore.JOB_REGISTRY_STORE)


def test_every_array_topped_store_declares_cannot_carry_with_the_array_top_wording():
    for name in _array_topped_stores():
        descriptor = ts.get_descriptor(name)
        assert descriptor.frozen
        assert descriptor.cannot_carry_field, name
        assert "array" in descriptor.cannot_carry_field


def test_model_registry_index_composes_with_its_own_declaration(tmp_path: Path):
    entries = read_registry_index(tmp_path)  # a real project with nothing registered
    assert entries == []
    ts.check_schema_version(ts.get_descriptor(MODEL_REGISTRY_STORE), entries)


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
