"""``scripts/conform_region_completeness_attested_view.py``: the one-off conform step
write-forwarding an empty ``cells_attested_view`` map onto a dataset's stored
``region_completeness`` records written before the key existed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import tcip_store as ts
from tcip_store.binding import bind_default
from tcip_store.sqlite_backend import SqliteBackend

from tcip_mcp.dataset_layout import region_completeness_key

SCRIPT = Path(__file__).parent.parent / "scripts" / "conform_region_completeness_attested_view.py"

_GRID = {"width": 100, "height": 80, "tile_size": 50, "overlap": 0.0, "cols": 2, "rows": 2}


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "conform_region_completeness_attested_view_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bind_sqlite() -> SqliteBackend:
    backend = SqliteBackend()
    ts.bind(backend)
    return backend


def _old_shape_record() -> dict:
    return {
        "grid": _GRID, "cells_complete": ["A1"], "attested_by": "user:z",
        "attested_at": "2026-01-01T00:00:00+00:00", "stem": "plot", "date": "2026-03-01",
        "subject": "bush",
    }


def _seed(root: Path, bucket: str, record: dict) -> None:
    (root / ".tcip").mkdir(parents=True, exist_ok=True)
    ts.replace(region_completeness_key(root), {bucket: record}, expect=ts.Version.ABSENT)


def test_a_record_lacking_the_key_is_write_forwarded_to_an_empty_map(tmp_path: Path):
    _bind_sqlite()
    module = _load_script()
    _seed(tmp_path, "bush/plot", _old_shape_record())

    outcomes, refused = module.conform_root(tmp_path, plan=False)

    assert refused is False
    assert outcomes == [
        "bush/plot: write-forwarded cells_attested_view to {} (no scale provenance was "
        "recorded before the key existed)"
    ]
    stored = ts.read(region_completeness_key(tmp_path))["bush/plot"]
    assert stored["cells_attested_view"] == {}
    assert stored["cells_complete"] == ["A1"]


def test_a_record_already_carrying_the_key_is_a_no_op(tmp_path: Path):
    _bind_sqlite()
    module = _load_script()
    record = {**_old_shape_record(), "cells_attested_view": {"A1": {"view_scale": 0.5}}}
    _seed(tmp_path, "bush/plot", record)

    outcomes, refused = module.conform_root(tmp_path, plan=False)

    assert refused is False
    assert outcomes == ["bush/plot: already carries cells_attested_view, unchanged"]
    assert ts.read(region_completeness_key(tmp_path))["bush/plot"] == record


def test_plan_mode_writes_nothing(tmp_path: Path):
    _bind_sqlite()
    module = _load_script()
    record = _old_shape_record()
    _seed(tmp_path, "bush/plot", record)

    outcomes, refused = module.conform_root(tmp_path, plan=True)

    assert refused is False
    assert outcomes == [
        "bush/plot: would write-forward cells_attested_view to {} (no scale provenance was "
        "recorded before the key existed)"
    ]
    assert ts.read(region_completeness_key(tmp_path))["bush/plot"] == record


def test_a_corrupt_non_dict_store_is_refused_by_name_not_treated_as_empty(tmp_path: Path):
    """A stored document present but not the recognized dict shape must be reported as a
    refusal naming the store, never silently folded into "nothing to conform" the way an
    absent document reads."""
    _bind_sqlite()
    module = _load_script()
    (tmp_path / ".tcip").mkdir(parents=True, exist_ok=True)
    ts.replace(region_completeness_key(tmp_path), ["not", "a", "dict"], expect=ts.Version.ABSENT)

    outcomes, refused = module.conform_root(tmp_path, plan=False)

    assert refused is True
    assert any("not a dict" in line for line in outcomes)
    assert ts.read(region_completeness_key(tmp_path)) == ["not", "a", "dict"]


def test_main_a_corrupt_store_exits_2_and_names_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
):
    bind_default()
    module = _load_script()
    (tmp_path / ".tcip").mkdir(parents=True, exist_ok=True)
    ts.replace(region_completeness_key(tmp_path), ["not", "a", "dict"], expect=ts.Version.ABSENT)

    monkeypatch.setattr(
        sys, "argv", ["conform_region_completeness_attested_view.py", str(tmp_path)])
    exit_code = module.main()
    output = capsys.readouterr().out

    assert exit_code == 2
    assert "not a dict" in output


def test_a_root_with_no_tcip_directory_is_refused_by_name(tmp_path: Path):
    _bind_sqlite()
    module = _load_script()

    outcomes, refused = module.conform_root(tmp_path, plan=False)

    assert refused is True
    assert outcomes == ["refused, no .tcip directory found; not a project root"]


def test_main_conforms_two_roots_and_exits_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    bind_default()
    module = _load_script()
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    _seed(root_a, "bush/plot", _old_shape_record())
    _seed(root_b, "bush/plot", _old_shape_record())

    monkeypatch.setattr(
        sys, "argv",
        ["conform_region_completeness_attested_view.py", str(root_a), str(root_b)])
    exit_code = module.main()

    assert exit_code == 0
    for root in (root_a, root_b):
        record = ts.read(region_completeness_key(root))["bush/plot"]
        assert record["cells_attested_view"] == {}


def test_main_a_missing_root_does_not_block_a_second_roots_conform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
):
    bind_default()
    module = _load_script()
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_b.mkdir()
    _seed(root_b, "bush/plot", _old_shape_record())

    monkeypatch.setattr(
        sys, "argv",
        ["conform_region_completeness_attested_view.py", str(root_a), str(root_b)])
    exit_code = module.main()
    output = capsys.readouterr().out

    assert exit_code == 2
    assert f"{root_a.resolve()}: refused, no .tcip directory found" in output
    record = ts.read(region_completeness_key(root_b))["bush/plot"]
    assert record["cells_attested_view"] == {}
