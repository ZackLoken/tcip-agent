"""Every audit line a real writer produces lands in the log of the resolved root its caller named,
and none carries a ``schema_version`` or a ``scope`` field.

A dataset-scoped write and a project-scoped write each go through a real production writer (a
decorated tool and ``record_event_or_raise``), never a hand-built entry. Each is reached with a
non-canonical spelling of its root (a ``..`` segment), so the line is found under the resolved
root's key only if the writer resolved it. Absence of ``schema_version`` is the frozen version 1,
``frozen-formats.json``'s ceiling for this store; a line names no root of its own, since the log
it sits in is the root's.
"""

from __future__ import annotations

from pathlib import Path

import tcip_mcp.audit as audit_module
import tcip_store as ts


def test_dataset_scoped_decorator_write_lands_under_the_resolved_dataset(tmp_path: Path) -> None:
    """``write_subject_registry`` (``@audited(scope_arg="dataset_root")``) is a real production door,
    reached with a ``..``-carrying spelling of its own root."""
    from tcip_mcp.tools.annotation_tools import write_subject_registry

    dataset_root = tmp_path / "orchard_dataset"
    dataset_root.mkdir()
    noncanonical = str(dataset_root / "nested" / "..")
    assert ".." in noncanonical

    subjects = {"bud": {"description": "a currant bud"}}
    assert "error" not in write_subject_registry(tmp_path, noncanonical, subjects=subjects)

    rows = list(ts.read_log(audit_module.audit_log_key(dataset_root)).records)
    matches = [r for r in rows if r["tool"] == "write_subject_registry"]
    assert len(matches) == 1, rows
    assert "schema_version" not in matches[0]
    assert "scope" not in matches[0]


def test_project_scoped_writer_lands_under_the_resolved_project(tmp_path: Path) -> None:
    """``persist_mapping`` writes its receipt through ``record_event_or_raise`` with the caller's
    ``project_root`` passed straight through, unresolved."""
    from tcip_mcp.pipelines.postprocessing.plant_mapping import MappingBuild, persist_mapping

    project_root = tmp_path / "project"
    project_root.mkdir()
    noncanonical = str(project_root / "nested" / "..")
    assert ".." in noncanonical

    build = MappingBuild(
        name="mapping", dataset_root="ds",
        dataset_id="ds-1", built_by="build_plant_mapping", built_at="2026-02-11T00:00:00+00:00",
        dates_requested=None, dates=[], nn_tolerance_m={"value": 10.0, "source": "stated"},
        plant_registry={"name": "unregistered", "digest": "0" * 64},
        capture_identity={}, capture_digests={}, unreadable={}, assignments={},
    )
    persist_mapping(build, noncanonical, "mapping")

    rows = list(ts.read_log(audit_module.audit_log_key(project_root)).records)
    matches = [r for r in rows if r["tool"] == "plant_mapping_built"]
    assert len(matches) == 1, rows
    assert "schema_version" not in matches[0]
    assert "scope" not in matches[0]
