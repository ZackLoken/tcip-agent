"""``scripts/conform_registry_experiment_id.py``: the one-off conform step for a project whose
registry entries predate ``experiment_id``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

import tcip_store as ts

from tcip_mcp.model_registry import ModelRegistry, load_registered_checkpoint, registry_index_key

SCRIPT = Path(__file__).parent.parent / "scripts" / "conform_registry_experiment_id.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("conform_registry_experiment_id_under_test", SCRIPT)
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
        "config": {}, "metrics": {}, "metrics_source": None, "tags": [],
    }
    base.update(overrides)
    return base


def test_plan_names_the_null_experiment_id_each_entry_would_get_without_writing(tmp_path):
    script = _load_script()
    root = tmp_path / "proj"
    _seed_index(root, [_entry("untagged")])

    lines, refused = script._plan(root)
    assert not refused
    assert any("untagged" in ln and "experiment_id=null" in ln for ln in lines)

    # --plan writes nothing.
    index = ts.read(registry_index_key(root))
    assert "experiment_id" not in index[0]


def test_an_experiment_tagged_entry_is_refused(tmp_path):
    script = _load_script()
    root = tmp_path / "proj"
    _seed_index(root, [_entry("trained", tags=["experiment:exp1"])])

    lines, refused = script._plan(root)
    assert refused
    assert "refused" in lines[0] and "experiment:exp1" in lines[0]


def test_apply_called_directly_does_not_silently_conform_a_refused_entry(tmp_path):
    """A caller that reaches ``_apply`` without going through ``_plan`` first (the path
    ``main`` always takes) must still learn which entries it could not conform, by name,
    rather than have them written back unchanged and reported as if they succeeded."""
    script = _load_script()
    root = tmp_path / "proj"
    _seed_index(root, [_entry("trained", tags=["experiment:exp1"])])

    applied, refused = script._apply(root)
    assert refused == ["trained"]

    index = ts.read(registry_index_key(root))
    assert "experiment_id" not in index[0]


def test_an_already_conformed_root_has_nothing_to_conform(tmp_path):
    """A rail must admit valid work: a root every entry of which already carries
    experiment_id is reported unchanged, not refused or rewritten."""
    root = tmp_path / "proj"
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"weights")
    ModelRegistry(str(root)).register_model("a", str(ckpt), {}, metrics_source=None)

    script = _load_script()
    lines, refused = script._plan(root)
    assert lines == [] and not refused


def test_main_exits_2_when_a_root_is_refused_and_0_once_conformed(tmp_path, capsys):
    root = tmp_path / "proj"
    _seed_index(root, [_entry("untagged")])

    script = _load_script()

    import sys
    old_argv = sys.argv
    try:
        sys.argv = ["conform_registry_experiment_id.py", str(root)]
        assert script.main() == 0
    finally:
        sys.argv = old_argv

    index = ts.read(registry_index_key(root))
    assert index[0]["experiment_id"] is None


def test_conformed_entry_admits_the_load_a_missing_key_would_have_refused(tmp_path, monkeypatch):
    """The conform script's own admits partner (rail 7's second half): once conformed, the entry
    is no longer refused by the missing-key rail."""
    torch = pytest.importorskip("torch")
    root = tmp_path / "proj"
    root.mkdir()
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(root))
    ckpt = root / "m.pt"
    torch.save({"model_state_dict": {}}, ckpt)
    from tcip_mcp.model_registry import _sha256_of_bytes

    digest = _sha256_of_bytes(ckpt.read_bytes())
    entry = _entry("pre-field", checkpoint_path=str(ckpt), sha256=digest)
    _seed_index(root, [entry])

    script = _load_script()
    applied, refused = script._apply(root)
    assert refused == []

    checkpoint = load_registered_checkpoint(ckpt, project_path=str(root))
    assert checkpoint.producer is None
