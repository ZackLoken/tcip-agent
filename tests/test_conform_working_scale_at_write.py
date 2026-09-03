"""``scripts/conform_working_scale_at_write.py``: the one-off conform step renaming a dataset's
stored ``region_completeness`` attestations' ``working_scale_bar_at_write`` key to
``working_scale_at_write``, nulling the value since the annotation-derived bar it once held
cannot be reconstructed as the breeder-set zoom the new key carries.
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

SCRIPT = Path(__file__).parent.parent / "scripts" / "conform_working_scale_at_write.py"

_GRID = {"width": 100, "height": 80, "tile_size": 50, "overlap": 0.0, "cols": 2, "rows": 2}


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "conform_working_scale_at_write_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bind_sqlite() -> SqliteBackend:
    backend = SqliteBackend()
    ts.bind(backend)
    return backend


def _old_shape_record(**attested_view_overrides) -> dict:
    entry = {
        "view_scale": 0.5,
        "working_scale_bar_at_write": {
            "value": 4.6, "median_extent_native_px": 10.0, "annotation_count": 1,
            "judged_span_px": 46, "source": "s"},
        "seen_on_record": {"at_scale": None, "grid_matched": False},
    }
    entry.update(attested_view_overrides)
    return {
        "grid": _GRID, "cells_complete": ["A1"], "attested_by": "user:z",
        "attested_at": "2026-01-01T00:00:00+00:00", "stem": "plot", "date": "2026-03-01",
        "subject": "bush", "cells_attested_view": {"A1": entry},
    }


def _seed(root: Path, bucket: str, record: dict) -> None:
    (root / ".tcip").mkdir(parents=True, exist_ok=True)
    ts.replace(region_completeness_key(root), {bucket: record}, expect=ts.Version.ABSENT)


def test_an_old_key_entry_is_renamed_and_nulled(tmp_path: Path):
    _bind_sqlite()
    module = _load_script()
    _seed(tmp_path, "bush/plot", _old_shape_record())

    outcomes, refused = module.conform_root(tmp_path, plan=False)

    assert refused is False
    assert outcomes == [
        "bush/plot/A1: renamed working_scale_bar_at_write to working_scale_at_write, nulled "
        "(no working scale to carry forward)"
    ]
    stored = ts.read(region_completeness_key(tmp_path))["bush/plot"]
    entry = stored["cells_attested_view"]["A1"]
    assert entry["working_scale_at_write"] is None
    assert "working_scale_bar_at_write" not in entry
    assert entry["view_scale"] == 0.5
    assert entry["seen_on_record"] == {"at_scale": None, "grid_matched": False}


def test_an_already_conformed_entry_is_reported_unchanged(tmp_path: Path):
    _bind_sqlite()
    module = _load_script()
    record = _old_shape_record()
    del record["cells_attested_view"]["A1"]["working_scale_bar_at_write"]
    record["cells_attested_view"]["A1"]["working_scale_at_write"] = None
    _seed(tmp_path, "bush/plot", record)

    outcomes, refused = module.conform_root(tmp_path, plan=False)

    assert refused is False
    assert outcomes == ["bush/plot/A1: already conformed, unchanged"]
    assert ts.read(region_completeness_key(tmp_path))["bush/plot"] == record


def test_a_bucket_with_no_attested_view_map_is_left_alone(tmp_path: Path):
    _bind_sqlite()
    module = _load_script()
    record = {"grid": _GRID, "cells_complete": [], "attested_by": "user:z",
              "attested_at": "2026-01-01T00:00:00+00:00", "stem": "plot", "date": "2026-03-01",
              "subject": "bush", "cells_attested_view": {}}
    _seed(tmp_path, "bush/plot", record)

    outcomes, refused = module.conform_root(tmp_path, plan=False)

    assert refused is False
    assert outcomes == []
    assert ts.read(region_completeness_key(tmp_path))["bush/plot"] == record


def test_plan_mode_writes_nothing(tmp_path: Path):
    _bind_sqlite()
    module = _load_script()
    record = _old_shape_record()
    _seed(tmp_path, "bush/plot", record)

    outcomes, refused = module.conform_root(tmp_path, plan=True)

    assert refused is False
    assert outcomes == [
        "bush/plot/A1: would be renamed working_scale_bar_at_write to working_scale_at_write, "
        "nulled (no working scale to carry forward)"
    ]
    assert ts.read(region_completeness_key(tmp_path))["bush/plot"] == record


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
        sys, "argv", ["conform_working_scale_at_write.py", str(root_a), str(root_b)])
    exit_code = module.main()

    assert exit_code == 0
    for root in (root_a, root_b):
        entry = ts.read(region_completeness_key(root))["bush/plot"]["cells_attested_view"]["A1"]
        assert entry["working_scale_at_write"] is None


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
        sys, "argv", ["conform_working_scale_at_write.py", str(root_a), str(root_b)])
    exit_code = module.main()
    output = capsys.readouterr().out

    assert exit_code == 2
    assert f"{root_a.resolve()}: refused, no .tcip directory found" in output
    entry = ts.read(region_completeness_key(root_b))["bush/plot"]["cells_attested_view"]["A1"]
    assert entry["working_scale_at_write"] is None
