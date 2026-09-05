"""Every audit line a real writer produces carries the scope stamp ``_stamp_scope`` decides, and
none carries a ``schema_version`` field.

A dataset-scoped write, a project-scoped write, and a platform-default write each go through a
real production writer (a decorated tool, ``record_event_or_raise``, and ``record_event``), never
a hand-built entry, so what these prove is the shared helper's actual contract: ``scope`` is the
resolved root exactly when the caller passed one, and no line carries ``schema_version`` (absence
is the frozen version 1, ``frozen-formats.json``'s ceiling for this store). The project case
passes a genuinely non-canonical spelling (a ``..`` segment) that reaches the helper unresolved,
so it proves the helper's own resolution; the dataset case's spelling is canonicalized upstream by
``dataset_scope_of`` before the helper runs, so what it guards is the absent-field contract, and
the resolution claim rests on the project case alone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import tcip_mcp.audit as audit_module
import tcip_store as ts


@pytest.fixture
def platform_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "platform"
    root.mkdir()
    monkeypatch.setattr(audit_module, "AUDIT_ROOT", root)
    return root


def test_dataset_scoped_decorator_write_carries_the_version_stamp_and_resolved_scope(
    platform_root: Path, tmp_path: Path,
) -> None:
    """``write_class_map`` (``@audited(scope_arg="dataset_root")``) is a real production door,
    reached with a ``..``-carrying spelling of its own root."""
    from tcip_mcp.tools.annotation_tools import write_class_map

    dataset_root = tmp_path / "orchard_dataset"
    dataset_root.mkdir()
    noncanonical = str(dataset_root / "nested" / "..")
    assert ".." in noncanonical

    subjects = {"bud": {"description": "a currant bud"}}
    assert "error" not in write_class_map(noncanonical, subjects=subjects)

    rows = list(ts.read_log(audit_module.audit_log_key(dataset_root)).records)
    matches = [r for r in rows if r["tool"] == "write_class_map"]
    assert len(matches) == 1, rows
    assert "schema_version" not in matches[0]
    assert matches[0]["scope"] == str(dataset_root.resolve())


def test_project_scoped_writer_carries_the_version_stamp_and_resolved_scope(
    platform_root: Path, tmp_path: Path,
) -> None:
    """``persist_mapping`` writes its receipt through ``record_event_or_raise`` with the caller's
    ``project_root`` passed straight through, unresolved, so this exercises ``_stamp_scope``'s own
    resolution directly rather than one an upstream canonicalizer already performed."""
    from tcip_mcp.pipelines.postprocessing.plant_mapping import MappingBuild, persist_mapping

    project_root = tmp_path / "project"
    project_root.mkdir()
    noncanonical = str(project_root / "nested" / "..")
    assert ".." in noncanonical

    build = MappingBuild(
        name="mapping", project_root=noncanonical, dataset_root=str(tmp_path / "ds"),
        dataset_id="ds-1", built_by="build_plant_mapping", built_at="2026-02-11T00:00:00+00:00",
        dates_requested=None, dates=[], nn_tolerance_m={"value": 10.0, "source": "fallback"},
        plant_registry={"name": "unregistered", "digest": "0" * 64},
        capture_identity={}, capture_digests={}, unreadable={}, assignments={},
    )
    persist_mapping(build, noncanonical, "mapping")

    rows = list(ts.read_log(audit_module.audit_log_key(project_root)).records)
    matches = [r for r in rows if r["tool"] == "plant_mapping_built"]
    assert len(matches) == 1, rows
    assert "schema_version" not in matches[0]
    assert matches[0]["scope"] == str(project_root.resolve())


def test_platform_default_write_carries_the_version_stamp_and_no_scope(
    platform_root: Path, tmp_path: Path,
) -> None:
    """A model-registry replace (real production ``record_event`` caller, no ``scope`` of its
    own) stays a platform event: no ``schema_version`` field, and no ``scope`` either."""
    from tcip_mcp.model_registry import ModelRegistry

    reg = ModelRegistry(str(tmp_path))
    first = tmp_path / "a.pt"
    first.write_bytes(b"first")
    second = tmp_path / "b.pt"
    second.write_bytes(b"second, different")
    reg.register_model("exp1", str(first), {}, metrics_source=None)
    reg.register_model("exp1", str(second), {}, metrics_source=None)

    rows = list(ts.read_log(audit_module.audit_log_key(platform_root)).records)
    matches = [r for r in rows if r["tool"] == "model_registry_replace"]
    assert len(matches) == 1, rows
    assert "schema_version" not in matches[0]
    assert "scope" not in matches[0]
