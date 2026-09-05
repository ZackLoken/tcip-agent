"""``scripts/conform_model_registry_paths.py`` and its shared implementation,
``tcip_mcp.model_registry.conform_registry_paths``: wrap a bare top-level array into the
entries-mapping shape and respell every entry's ``checkpoint_path`` relative to the registry's
own scope root.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

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

    assert any("wrapped the registry index" in ln for ln in lines)
    raw = ts.read(registry_index_key(root))
    assert raw == {"entries": []}


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
    assert entries[0]["checkpoint_path"] == ".tcip/models/m.pt"


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
    # Kept byte-for-byte: the stored value is the external claim, and a host-resolved
    # respelling would fabricate a path the writer never stated (drive-anchored on Windows).
    assert entries[0]["checkpoint_path"] == "/exporting/root/.tcip/models/gone.pt"


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


# ── conform_registry_paths_on_disk: the import door's own bypass of the storage seam ───────


def test_on_disk_conform_wraps_and_respells_directly_against_the_extracted_files(tmp_path: Path):
    """The import door's own conform, exercised directly against loose files on disk rather
    than through the storage seam (a staging tree is always loose files, whatever backend the
    process is bound to): a bare top-level array wraps into the entries mapping and a stored
    path that resolves under root with a matching digest respells relative, the identical
    outcome conform_registry_paths produces through the seam."""
    from tcip_mcp.model_registry import conform_registry_paths_on_disk, registry_index_path

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
    is the only trace this conform's caller has that anything changed."""
    from tcip_mcp.model_registry import conform_registry_paths_on_disk, registry_index_path

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
    from tcip_mcp.model_registry import conform_registry_paths_on_disk

    root = tmp_path / "proj"
    root.mkdir()

    assert conform_registry_paths_on_disk(root) == []


def test_on_disk_conform_refuses_a_registry_that_will_not_decode(tmp_path: Path):
    from tcip_mcp.model_registry import RegistryVersionRefused, conform_registry_paths_on_disk, registry_index_path

    root = tmp_path / "proj"
    index_path = registry_index_path(root)
    index_path.parent.mkdir(parents=True)
    index_path.write_bytes(b"not json at all")

    with pytest.raises(RegistryVersionRefused):
        conform_registry_paths_on_disk(root)


def test_conform_over_a_real_import_produced_registry(tmp_path: Path, monkeypatch):
    """Coverage: the conform script over a registry the real archive/import doors produced,
    naming the exporting root's absolute path for a checkpoint the import door's own on-disk
    conform already relativized, proving the two conform paths (import-time, operator-time)
    agree once more state has moved around it."""
    from tcip_mcp.experiments import (
        complete_run, create_experiment, experiment_dir, register_model_from_experiment,
        update_status,
    )
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

    def _counting_compute(p: Path) -> str:
        calls.append(Path(p))
        return real_compute(p)

    monkeypatch.setattr(model_registry, "_compute_sha256", _counting_compute)

    conform_registry_paths(root)

    assert calls.count(candidate.resolve()) == 1


def test_conform_skips_a_directory_shaped_like_a_checkpoint_under_experiments(tmp_path: Path):
    """A directory named ``*.pt`` under ``.tcip/experiments`` (a run's own output directory can
    be named however a bespoke loop likes) must never reach ``stat``/hash as a relocation
    candidate; the real checkpoint elsewhere is still found and the entry still relocates.

    Exercises ``_conform_entries`` directly rather than through ``conform_registry_paths``: the
    seam version holds the registry's own write transaction (and, under the file backend, its
    on-disk lock file) open across the whole enumeration, which is a second, unrelated way a
    non-checkpoint file can reach this same enumeration -- out of this fix's scope.
    """
    from tcip_mcp.model_registry import _conform_entries

    root = tmp_path / "dest"
    (root / ".tcip" / "experiments" / "weird_dir.pt").mkdir(parents=True)
    (root / ".tcip" / "models").mkdir(parents=True)
    content = b"the real checkpoint bytes"
    digest = hashlib.sha256(content).hexdigest()
    real = root / ".tcip" / "models" / "real.pt"
    real.write_bytes(content)

    entries = [_entry(
        "m", "/exporting/root/.tcip/models/real.pt", digest, len(content),
        file_size_bytes=None,
    )]

    conformed, lines = _conform_entries(entries, root, plan=False, hash_cache={})

    assert any("relocated" in ln for ln in lines)
    assert conformed[0]["checkpoint_path"] == ".tcip/models/real.pt"


def test_candidate_checkpoint_paths_derives_the_suffix_from_the_last_tcip_segment(tmp_path: Path):
    """A stored path carrying two ``.tcip`` segments (a stale export nested under a backup tree
    that is itself named ``.tcip``-something, or similar) must derive its version-1 suffix from
    the last one, the segment nearest the actual checkpoint, matching the entry's own claim
    rather than an unrelated ancestor that happens to share the name. Coverage: the basename
    fallback in the same enumeration already finds this file by name alone, so this does not by
    itself distinguish the last-segment rule from the first-segment one it replaces; it pins the
    suffix construction's own intent directly."""
    from tcip_mcp.model_registry import _candidate_checkpoint_paths

    root = tmp_path / "dest"
    (root / ".tcip" / "models").mkdir(parents=True)
    real = root / ".tcip" / "models" / "m.pt"
    real.write_bytes(b"weights")

    raw = "/backup/.tcip/exports/project/.tcip/models/m.pt"

    candidates = _candidate_checkpoint_paths(root, raw)

    assert real.resolve() in candidates


def test_conform_refuses_a_malformed_mapping():
    from tcip_mcp.model_registry import _document_entries_for_conform

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
