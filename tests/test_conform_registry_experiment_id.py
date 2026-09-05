"""``scripts/conform_registry_experiment_id.py``: the one-off conform step for a project whose
registry entries predate ``experiment_id``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

import tcip_store as ts

from tcip_mcp.model_registry import (
    ModelRegistry,
    load_registered_checkpoint,
    read_registry_index,
    registry_index_key,
)

SCRIPT = Path(__file__).parent.parent / "scripts" / "conform_registry_experiment_id.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("conform_registry_experiment_id_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _seed_index(root: Path, entries: list[dict]) -> None:
    ts.replace(registry_index_key(root), {"entries": entries}, expect=ts.Version.ABSENT)


def _entry(name: str, **overrides) -> dict:
    base = {
        "name": name, "checkpoint_path": f"{name}.pt", "kind": None, "sha256": "abc",
        "file_size_bytes": 1, "registered_at": "2026-01-01T00:00:00+00:00",
        "config": {}, "metrics": {}, "metrics_source": None, "tags": [],
    }
    base.update(overrides)
    return base


def test_plan_names_the_null_experiment_id_each_entry_would_get_without_writing(tmp_path):
    script = _load_script()
    root = tmp_path / "proj"
    _seed_index(root, [_entry("untagged")])

    lines = script._plan(root)
    assert any("untagged" in ln and "experiment_id=null" in ln for ln in lines)

    # --plan writes nothing.
    index = read_registry_index(root)
    assert "experiment_id" not in index[0]


def test_a_tagged_entry_with_no_matching_run_resolves_to_null_and_drops_the_tag(tmp_path, monkeypatch):
    """An unverified experiment: tag is not a recorded binding: no such run at all, and a run
    that exists but recorded a different digest, both resolve to experiment_id=null with the
    tag dropped, the reason printed."""
    root = tmp_path / "proj"
    monkeypatch.setenv("TCIP_STATE_ROOT", str(root))
    from tcip_mcp.experiments import complete_run, create_experiment, update_status

    create_experiment("exp1", {"model_source": {"builder": "x:y"}})
    update_status("exp1", "running")
    ckpt = root / "m.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(b"a different run's own bytes")
    assert "error" not in complete_run("exp1", str(ckpt))

    _seed_index(root, [
        _entry("no-such-run", tags=["experiment:does-not-exist"], sha256="abc"),
        _entry("mismatched-digest", tags=["experiment:exp1"], sha256="wrong-digest"),
    ])

    script = _load_script()
    lines = script._apply(root)
    assert len(lines) == 2
    for line in lines:
        assert "experiment_id=null" in line and "experiment:" in line

    index = read_registry_index(root)
    for entry in index:
        assert entry["experiment_id"] is None
        assert entry["tags"] == []


def test_a_tagged_entry_whose_run_recorded_the_same_digest_binds_and_drops_the_tag(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    monkeypatch.setenv("TCIP_STATE_ROOT", str(root))
    from tcip_mcp.experiments import complete_run, create_experiment, update_status
    from tcip_mcp.model_registry import _sha256_of_bytes

    create_experiment("exp1", {"model_source": {"builder": "x:y"}})
    update_status("exp1", "running")
    ckpt = root / "m.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(b"the run's own recorded bytes")
    completed = complete_run("exp1", str(ckpt))
    assert "error" not in completed, completed

    digest = _sha256_of_bytes(ckpt.read_bytes())
    _seed_index(root, [_entry("bound", tags=["experiment:exp1"], sha256=digest)])

    script = _load_script()
    applied = script._apply(root)
    assert "experiment_id='exp1'" in applied[0]

    index = read_registry_index(root)
    assert index[0]["experiment_id"] == "exp1"
    assert index[0]["tags"] == []


def test_an_already_conformed_root_has_nothing_to_conform(tmp_path):
    """A rail must admit valid work: a root every entry of which already carries
    experiment_id is reported unchanged, not refused or rewritten."""
    root = tmp_path / "proj"
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"weights")
    ModelRegistry(str(root)).register_model("a", str(ckpt), {}, metrics_source=None)

    script = _load_script()
    lines = script._plan(root)
    assert lines == []


def test_main_returns_2_for_an_unreadable_index_and_0_once_conformed(tmp_path, monkeypatch):
    """The exit code its own name promises: an index that will not decode returns 2 without
    conforming anything else named; a readable root with an entry to conform returns 0.

    Bound to the file backend throughout: ``main`` rebinds through ``bind_default`` on every
    call, so the corruption (raw bytes on disk) and the read both have to agree on which
    backend owns these roots.
    """
    monkeypatch.setenv("TCIP_STORE_BACKEND", "file")
    from tcip_store.binding import bind_default as _rebind

    _rebind()  # the autouse fixture already bound before this env var was set
    broken_root = tmp_path / "broken"
    broken_root.mkdir()
    ckpt = tmp_path / "model_best.pt"
    ckpt.write_bytes(b"weights")
    ModelRegistry(str(broken_root)).register_model("a", str(ckpt), {}, metrics_source=None)
    index_path = broken_root / ".tcip" / "models" / "registry.json"
    index_path.write_text(index_path.read_text(encoding="utf-8")[:-8], encoding="utf-8")

    conformable_root = tmp_path / "conformable"
    _seed_index(conformable_root, [_entry("untagged")])

    script = _load_script()
    import sys
    old_argv = sys.argv
    try:
        sys.argv = ["conform_registry_experiment_id.py", str(broken_root)]
        assert script.main() == 2

        sys.argv = ["conform_registry_experiment_id.py", str(conformable_root)]
        assert script.main() == 0
    finally:
        sys.argv = old_argv

    index = read_registry_index(conformable_root)
    assert index[0]["experiment_id"] is None


def test_conformed_entry_admits_the_load_a_missing_key_would_have_refused(tmp_path, monkeypatch):
    """The conform script's own admits partner (rail 7's second half): once conformed, the entry
    is no longer refused by the missing-key rail."""
    torch = pytest.importorskip("torch")
    root = tmp_path / "proj"
    root.mkdir()
    monkeypatch.setenv("TCIP_STATE_ROOT", str(root))
    ckpt = root / "m.pt"
    torch.save({"model_state_dict": {}}, ckpt)
    from tcip_mcp.model_registry import _sha256_of_bytes

    digest = _sha256_of_bytes(ckpt.read_bytes())
    entry = _entry("pre-field", checkpoint_path=str(ckpt), sha256=digest)
    _seed_index(root, [entry])

    script = _load_script()
    applied = script._apply(root)
    assert len(applied) == 1

    checkpoint = load_registered_checkpoint(ckpt, project_path=str(root))
    assert checkpoint.producer is None


def test_conform_then_register_model_walks_the_ordered_remedy_for_a_pre_field_entry(tmp_path, monkeypatch):
    """The remedy best_model, scripts/doctor.py and ARCHITECTURE.md name for an entry predating
    both metrics_source and experiment_id: conform first, then re-register through
    register_model, landing a ranked entry the eviction rail no longer blocks."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    root = tmp_path / "proj"
    root.mkdir()
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"a pre-field checkpoint's own bytes")
    from tcip_mcp.model_registry import _sha256_of_bytes

    digest = _sha256_of_bytes(ckpt.read_bytes())
    pre_field = {
        "name": "pre-field", "checkpoint_path": str(ckpt), "kind": None, "sha256": digest,
        "file_size_bytes": ckpt.stat().st_size, "registered_at": "2026-01-01T00:00:00+00:00",
        "config": {}, "metrics": {}, "tags": [],
    }
    _seed_index(root, [pre_field])

    registry = ModelRegistry(str(root))
    with pytest.raises(ValueError, match="predate"):
        registry.best_model("val_map50", higher_is_better=True)

    script = _load_script()
    applied = script._apply(root)
    assert len(applied) == 1

    registry.register_model(
        "pre-field", str(ckpt), {"trained": True}, {"val_map50": 0.5}, metrics_source="caller",
    )

    registry = ModelRegistry(str(root))
    best = registry.best_model("val_map50", higher_is_better=True, include_unverified=True)
    assert best is not None and best["name"] == "pre-field"
