"""``tcip_mcp.model_registry.conform_registry_paths_on_disk``: the model registry's only
surviving conform, called by ``import_project`` against a staging tree's loose files before
accounting for it. ``conform_registry_paths`` (the seam-backed conform this function was cloned
from) has no caller anywhere under ``packages`` and is gone with its own tests; the four cases
below outlive it because they exercise the on-disk function directly, plus the real
archive/import round trip that is the door's only production caller.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tcip_mcp.model_registry import (
    RegistryVersionRefused,
    conform_registry_paths_on_disk,
    registry_index_path,
)


def _entry(name: str, checkpoint_path: str, sha256: str, size: int, **overrides) -> dict:
    base = {
        "name": name, "checkpoint_path": checkpoint_path, "kind": None, "sha256": sha256,
        "file_size_bytes": size, "registered_at": "2026-01-01T00:00:00+00:00",
        "config": {}, "metrics": {}, "metrics_source": None, "tags": [], "experiment_id": None,
    }
    base.update(overrides)
    return base


def test_on_disk_conform_wraps_and_respells_directly_against_the_extracted_files(tmp_path: Path):
    """Exercised directly against loose files on disk rather than through the storage seam (a
    staging tree is always loose files, whatever backend the process is bound to): a bare
    top-level array wraps into the entries mapping and a stored path that resolves under root
    with a matching digest respells relative.
    """
    import hashlib

    root = tmp_path / "proj"
    ckpt_dir = root / ".tcip" / "models"
    ckpt_dir.mkdir(parents=True)
    content = b"on-disk conform weights"
    ckpt = ckpt_dir / "m.pt"
    ckpt.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    index_path = registry_index_path(root)
    index_path.write_text(json.dumps([_entry("m", str(ckpt), digest, len(content))]))

    lines = conform_registry_paths_on_disk(root)

    assert any("wrapped the registry index" in ln for ln in lines)
    assert any("respelled" in ln for ln in lines)
    raw = json.loads(index_path.read_text())
    assert "schema_version" not in raw
    assert raw["entries"][0]["checkpoint_path"] == ".tcip/models/m.pt"


def test_on_disk_conform_reports_dropping_a_stray_schema_version_two(tmp_path: Path):
    """An index already an entries mapping (no bare-array wrap needed) but still carrying a
    dev-era ``schema_version: 2`` must earn its own outcome line, not be folded silently into
    "already wrapped, nothing to say": the field vanishes from the write either way, so the drop
    is the only trace this conform's caller has that anything changed.
    """
    import hashlib

    root = tmp_path / "proj"
    ckpt_dir = root / ".tcip" / "models"
    ckpt_dir.mkdir(parents=True)
    content = b"stray schema_version two weights"
    ckpt = ckpt_dir / "m.pt"
    ckpt.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    index_path = registry_index_path(root)
    entry = _entry("m", ".tcip/models/m.pt", digest, len(content))
    index_path.write_text(json.dumps({"schema_version": 2, "entries": [entry]}))

    lines = conform_registry_paths_on_disk(root)

    assert any("dropped a stray schema_version" in ln for ln in lines)
    assert not any("wrapped the registry index" in ln for ln in lines)
    raw = json.loads(index_path.read_text())
    assert "schema_version" not in raw


def test_on_disk_conform_over_an_absent_registry_answers_nothing(tmp_path: Path):
    root = tmp_path / "proj"
    root.mkdir()

    assert conform_registry_paths_on_disk(root) == []


def test_on_disk_conform_refuses_a_registry_that_will_not_decode(tmp_path: Path):
    root = tmp_path / "proj"
    index_path = registry_index_path(root)
    index_path.parent.mkdir(parents=True)
    index_path.write_bytes(b"not json at all")

    with pytest.raises(RegistryVersionRefused):
        conform_registry_paths_on_disk(root)


def test_archive_then_import_lands_a_registry_already_conformed(tmp_path: Path, monkeypatch):
    """The door's only production caller, end to end: ``import_project`` runs the on-disk conform
    against the staging tree before accounting for it, so a registry the real archive/import
    round trip produces is already wrapped and correctly spelled, with nothing left for a second
    conform pass to do.
    """
    from tcip_mcp.experiments import (
        complete_run, create_experiment, experiment_dir, register_model_from_experiment,
        update_status,
    )
    from tcip_mcp.model_registry import ModelRegistry, read_registry_index
    from tcip_mcp.tools.project_tools import archive_project, import_project, initialize_project

    src = tmp_path / "src_project"
    initialize_project(str(src), site="north orchard")
    monkeypatch.setenv("TCIP_STATE_ROOT", str(src))
    create_experiment("exp1", {"model_source": {"builder": "x:y"}})
    update_status("exp1", "running")
    ckpt_dir = experiment_dir("exp1")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    weights = ckpt_dir / "model_final.pt"
    weights.write_bytes(b"a real run's own weights")
    assert "error" not in complete_run("exp1", str(weights))
    assert "error" not in register_model_from_experiment(
        "exp1", str(weights), project_path=str(src))

    zip_path = tmp_path / "export.zip"
    assert "error" not in archive_project(str(src), str(zip_path), include_models=True)
    dest = tmp_path / "restored"
    imported = import_project(str(zip_path), str(dest))
    assert "error" not in imported, imported

    entries = read_registry_index(dest)
    assert len(entries) == 1
    assert not Path(entries[0]["checkpoint_path"]).is_absolute()
    assert ModelRegistry(str(dest)).get_model("exp1") is not None
