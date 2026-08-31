"""``scripts/conform_model_registry_paths.py`` and its shared implementation,
``tcip_mcp.model_registry.conform_registry_paths``: wrap a version-1 registry index to version 2
and respell every entry's ``checkpoint_path`` relative to the registry's own scope root.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import tcip_store as ts

from tcip_mcp.model_registry import (
    RegistryVersionRefused,
    conform_registry_paths,
    read_registry_index,
    registry_index_key,
)

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "conform_model_registry_paths.py"


def _run_script(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args], capture_output=True, text=True, timeout=60,
    )


def _seed_v1(root: Path, entries: list[dict]) -> None:
    ts.replace(registry_index_key(root), entries, expect=ts.Version.ABSENT)


def _entry(name: str, checkpoint_path: str, sha256: str, size: int, **overrides) -> dict:
    base = {
        "name": name, "checkpoint_path": checkpoint_path, "kind": None, "sha256": sha256,
        "file_size_bytes": size, "registered_at": "2026-01-01T00:00:00+00:00",
        "config": {}, "metrics": {}, "metrics_source": None, "tags": [], "experiment_id": None,
    }
    base.update(overrides)
    return base


def test_conform_wraps_a_legacy_bare_array_with_nothing_to_respell(tmp_path: Path):
    root = tmp_path / "proj"
    _seed_v1(root, [])

    lines = conform_registry_paths(root)

    assert any("schema_version 2" in ln for ln in lines)
    raw = ts.read(registry_index_key(root))
    assert raw == {"schema_version": 2, "entries": []}


def test_conform_respells_an_entry_whose_stored_path_already_resolves_under_root(tmp_path: Path):
    root = tmp_path / "proj"
    ckpt_dir = root / ".tcip" / "models"
    ckpt_dir.mkdir(parents=True)
    content = b"in-place weights"
    ckpt = ckpt_dir / "m.pt"
    ckpt.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    _seed_v1(root, [_entry("m", str(ckpt), digest, len(content))])

    lines = conform_registry_paths(root)

    assert any("respelled" in ln for ln in lines)
    entries = read_registry_index(root)
    assert entries[0]["checkpoint_path"] == "m.pt" or entries[0]["checkpoint_path"] == ".tcip/models/m.pt"
    assert not Path(entries[0]["checkpoint_path"]).is_absolute()


def test_conform_leaves_a_genuinely_external_entry_absolute(tmp_path: Path):
    root = tmp_path / "proj"
    root.mkdir()
    external = tmp_path / "outside"
    external.mkdir()
    content = b"external weights"
    ckpt = external / "m.pt"
    ckpt.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    _seed_v1(root, [_entry("m", str(ckpt), digest, len(content))])

    lines = conform_registry_paths(root)

    assert not any("respelled" in ln or "relocated" in ln for ln in lines)
    entries = read_registry_index(root)
    assert entries[0]["checkpoint_path"] == str(ckpt)


def test_conform_relocates_a_checkpoint_whose_stale_exporting_root_still_exists(tmp_path: Path):
    """The exporter tree being present is ordinary: a stale-but-existing absolute path is still
    relocated by content digest among the destination's own checkpoint files, never trusted just
    because something still answers at the old path."""
    root = tmp_path / "dest"
    (root / ".tcip" / "models").mkdir(parents=True)
    content = b"the run's own recorded bytes"
    digest = hashlib.sha256(content).hexdigest()
    relocated = root / ".tcip" / "models" / "model_final.pt"
    relocated.write_bytes(content)

    stale_exporter = tmp_path / "exporter" / ".tcip" / "models" / "model_final.pt"
    stale_exporter.parent.mkdir(parents=True)
    stale_exporter.write_bytes(b"a different run's bytes now living at the old path")

    _seed_v1(root, [_entry("m", str(stale_exporter), digest, len(content))])

    lines = conform_registry_paths(root)

    assert any("relocated" in ln for ln in lines)
    entries = read_registry_index(root)
    assert entries[0]["checkpoint_path"] == ".tcip/models/model_final.pt"


def test_conform_disambiguates_duplicate_digests_by_basename_then_sorted_path(tmp_path: Path):
    root = tmp_path / "dest"
    (root / ".tcip" / "models").mkdir(parents=True)
    (root / ".tcip" / "experiments" / "exp1").mkdir(parents=True)
    content = b"byte-identical checkpoint content"
    digest = hashlib.sha256(content).hexdigest()
    (root / ".tcip" / "models" / "alpha.pt").write_bytes(content)
    named_copy = root / ".tcip" / "experiments" / "exp1" / "model_best.pt"
    named_copy.write_bytes(content)

    _seed_v1(root, [_entry(
        "dup", "/exporting/root/.tcip/models/model_best.pt", digest, len(content),
    )])

    lines = conform_registry_paths(root)

    assert any("ambiguous" in ln for ln in lines)
    entries = read_registry_index(root)
    assert entries[0]["checkpoint_path"] == ".tcip/experiments/exp1/model_best.pt"


def test_conform_classifies_a_digest_found_nowhere_as_external_or_missing(tmp_path: Path):
    root = tmp_path / "dest"
    (root / ".tcip" / "models").mkdir(parents=True)
    _seed_v1(root, [_entry("gone", "/exporting/root/.tcip/models/gone.pt", "a" * 64, 5)])

    lines = conform_registry_paths(root)

    assert any("external-or-missing" in ln for ln in lines)
    entries = read_registry_index(root)
    assert Path(entries[0]["checkpoint_path"]).is_absolute()


def test_conform_is_idempotent(tmp_path: Path):
    root = tmp_path / "proj"
    ckpt_dir = root / ".tcip" / "models"
    ckpt_dir.mkdir(parents=True)
    content = b"idempotence weights"
    ckpt = ckpt_dir / "m.pt"
    ckpt.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    _seed_v1(root, [_entry("m", str(ckpt), digest, len(content))])

    conform_registry_paths(root)
    second = conform_registry_paths(root)

    assert second == []


def test_conform_over_a_real_import_produced_registry(tmp_path: Path, monkeypatch):
    """Coverage: the conform script over a registry the real archive/import doors produced,
    naming the exporting root's absolute path for a checkpoint the import door's own on-disk
    conform already relativized, proving the two conform paths (import-time, operator-time)
    agree once more state has moved around it."""
    from tcip_mcp.experiments import (
        complete_run, create_experiment, experiment_dir, register_model_from_experiment,
        update_status,
    )
    from tcip_mcp.tools.project_tools import archive_project, import_project, init_project

    src = tmp_path / "src_project"
    init_project(str(src), site="north orchard")
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(src))
    create_experiment("exp1", {"model_source": {"builder": "x:y"}})
    update_status("exp1", "running")
    ckpt_dir = experiment_dir("exp1")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    weights = ckpt_dir / "model_final.pt"
    weights.write_bytes(b"a real run's own weights")
    assert "error" not in complete_run("exp1", str(weights))
    assert "error" not in register_model_from_experiment("exp1", str(weights), project_path=str(src))

    zip_path = tmp_path / "export.zip"
    assert "error" not in archive_project(str(src), str(zip_path), include_models=True)
    dest = tmp_path / "restored"
    imported = import_project(str(zip_path), str(dest))
    assert "error" not in imported, imported

    lines = conform_registry_paths(dest)
    assert lines == []


def test_conform_hashes_each_candidate_at_most_once_per_run(tmp_path: Path, monkeypatch):
    """The hash cache exists to bound the conform's cost by one hash per candidate file; a
    ``dict.setdefault`` built with an eagerly-evaluated default would rehash every candidate on
    every lookup instead, defeating the cache it appears to use."""
    import tcip_mcp.model_registry as model_registry

    root = tmp_path / "dest"
    (root / ".tcip" / "models").mkdir(parents=True)
    content = b"one candidate, two entries wanting to match it"
    digest = hashlib.sha256(content).hexdigest()
    candidate = root / ".tcip" / "models" / "shared.pt"
    candidate.write_bytes(content)

    _seed_v1(root, [
        _entry("a", "/exporting/root/.tcip/models/a.pt", digest, len(content)),
        _entry("b", "/exporting/root/.tcip/models/b.pt", digest, len(content)),
    ])

    calls: list[Path] = []
    real_compute = model_registry._compute_sha256
    monkeypatch.setattr(
        model_registry, "_compute_sha256",
        lambda p: (calls.append(Path(p)), real_compute(p))[1],
    )

    conform_registry_paths(root)

    assert calls.count(candidate.resolve()) == 1


def test_conform_skips_a_directory_shaped_like_a_checkpoint_under_experiments(tmp_path: Path):
    """A directory named ``*.pt`` under ``.tcip/experiments`` (a run's own output directory can
    be named however a bespoke loop likes) must never reach ``stat``/hash as a relocation
    candidate; the real checkpoint elsewhere is still found and the entry still relocates."""
    root = tmp_path / "dest"
    (root / ".tcip" / "experiments" / "weird_dir.pt").mkdir(parents=True)
    (root / ".tcip" / "models").mkdir(parents=True)
    content = b"the real checkpoint bytes"
    digest = hashlib.sha256(content).hexdigest()
    real = root / ".tcip" / "models" / "real.pt"
    real.write_bytes(content)

    _seed_v1(root, [_entry(
        "m", "/exporting/root/.tcip/models/real.pt", digest, len(content),
        file_size_bytes=None,
    )])

    lines = conform_registry_paths(root)

    assert any("relocated" in ln for ln in lines)
    entries = read_registry_index(root)
    assert entries[0]["checkpoint_path"] == ".tcip/models/real.pt"


def test_conform_refuses_a_malformed_mapping():
    from tcip_mcp.model_registry import _document_entries_for_conform

    import pytest

    with pytest.raises(RegistryVersionRefused):
        _document_entries_for_conform({"schema_version": 3, "entries": []})
    with pytest.raises(RegistryVersionRefused):
        _document_entries_for_conform({"schema_version": 2, "entries": "not-a-list"})


def test_script_plan_reports_without_writing(tmp_path: Path):
    root = tmp_path / "proj"
    _seed_v1(root, [])

    result = _run_script("--plan", str(root))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "wrapping" in result.stdout
    raw = ts.read(registry_index_key(root))
    assert isinstance(raw, list)


def test_script_apply_then_nothing_to_conform(tmp_path: Path):
    root = tmp_path / "proj"
    _seed_v1(root, [])

    first = _run_script(str(root))
    second = _run_script(str(root))

    assert first.returncode == 0, first.stdout + first.stderr
    assert second.returncode == 0, second.stdout + second.stderr
    assert "nothing to conform" in second.stdout
