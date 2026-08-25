"""``scripts/conform_registry_metrics_source.py``: the one-off conform step for a project whose
registry entries predate ``metrics_source``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import tcip_store as ts

from tcip_mcp.model_registry import ModelRegistry, registry_index_key

SCRIPT = Path(__file__).parent.parent / "scripts" / "conform_registry_metrics_source.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("conform_registry_metrics_source_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _seed_index(root: Path, entries: list[dict]) -> None:
    ts.replace(registry_index_key(root), entries, expect=ts.Version.ABSENT)


def _entry(name: str, **overrides) -> dict:
    base = {
        "name": name, "checkpoint_path": f"{name}.pt", "kind": None, "sha256": "abc",
        "file_size_bytes": 1, "registered_at": "2026-01-01T00:00:00+00:00",
        "config": {}, "metrics": {}, "tags": [],
    }
    base.update(overrides)
    return base


def test_plan_names_the_source_each_entry_would_get_without_writing(tmp_path):
    script = _load_script()
    root = tmp_path / "proj"
    _seed_index(root, [
        _entry("empty"),
        _entry("caller_like", metrics={"val_map50": 0.5}, tags=["detector"]),
    ])

    lines, refused = script._plan(root, {})
    assert not refused
    assert any("empty" in ln and "metrics_source=null" in ln for ln in lines)
    assert any("caller_like" in ln and "metrics_source='caller'" in ln for ln in lines)

    # --plan writes nothing.
    index = ts.read(registry_index_key(root))
    assert "metrics_source" not in index[0]
    assert "metrics_source" not in index[1]


def test_an_experiment_tagged_entry_is_refused_without_a_stated_source(tmp_path):
    script = _load_script()
    root = tmp_path / "proj"
    _seed_index(root, [
        _entry("trained", metrics={"val_map50": 0.5}, tags=["experiment:exp1"]),
    ])

    lines, refused = script._plan(root, {})
    assert refused
    assert "refused" in lines[0] and "experiment:exp1" in lines[0]
    assert "exists=False" in lines[0]  # no such experiment on this root


def test_a_stated_source_conforms_the_experiment_tagged_entry(tmp_path):
    script = _load_script()
    root = tmp_path / "proj"
    _seed_index(root, [
        _entry("trained", metrics={"val_map50": 0.5}, tags=["experiment:exp1"]),
    ])

    applied, refused = script._apply(root, {"trained": "training_source"})
    assert "metrics_source='training_source'" in applied[0]
    assert refused == []

    index = ts.read(registry_index_key(root))
    assert index[0]["metrics_source"] == "training_source"


def test_apply_called_directly_does_not_silently_conform_a_refused_entry(tmp_path):
    """A caller that reaches ``_apply`` without going through ``_plan`` first (the path
    ``main`` always takes) must still learn which entries it could not conform, by name,
    rather than have them written back unchanged and reported as if they succeeded."""
    script = _load_script()
    root = tmp_path / "proj"
    _seed_index(root, [
        _entry("trained", metrics={"val_map50": 0.5}, tags=["experiment:exp1"]),
    ])

    applied, refused = script._apply(root, {})
    assert refused == ["trained"]

    index = ts.read(registry_index_key(root))
    assert "metrics_source" not in index[0]


def test_main_refuses_source_given_with_more_than_one_root(tmp_path):
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    _seed_index(root_a, [_entry("trained", metrics={"val_map50": 0.5}, tags=["experiment:exp1"])])
    _seed_index(root_b, [])

    script = _load_script()
    import sys
    old_argv = sys.argv
    try:
        sys.argv = ["conform_registry_metrics_source.py", str(root_a), str(root_b),
                   "--source", "trained=training_source"]
        assert script.main() == 2
    finally:
        sys.argv = old_argv


def test_an_already_conformed_root_has_nothing_to_conform(tmp_path):
    """A rail must admit valid work: a root every entry of which already carries
    metrics_source is reported unchanged, not refused or rewritten."""
    root = tmp_path / "proj"
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"weights")
    ModelRegistry(str(root)).register_model(
        "a", str(ckpt), {}, metrics={"val_map50": 0.5}, metrics_source="trainer")

    script = _load_script()
    lines, refused = script._plan(root, {})
    assert lines == [] and not refused


def test_main_exits_2_when_a_root_is_refused_and_0_once_conformed(tmp_path, capsys):
    root = tmp_path / "proj"
    _seed_index(root, [_entry("trained", metrics={"val_map50": 0.5}, tags=["experiment:exp1"])])

    script = _load_script()

    import sys
    old_argv = sys.argv
    try:
        sys.argv = ["conform_registry_metrics_source.py", str(root)]
        assert script.main() == 2
        capsys.readouterr()

        sys.argv = ["conform_registry_metrics_source.py", str(root),
                   "--source", "trained=training_source"]
        assert script.main() == 0
    finally:
        sys.argv = old_argv

    index = ts.read(registry_index_key(root))
    assert index[0]["metrics_source"] == "training_source"


def test_main_refuses_a_source_value_outside_the_declared_vocabulary(tmp_path):
    script = _load_script()
    import sys
    old_argv = sys.argv
    try:
        sys.argv = ["conform_registry_metrics_source.py", str(tmp_path),
                   "--source", "x=not_a_real_source"]
        assert script.main() == 2
    finally:
        sys.argv = old_argv
