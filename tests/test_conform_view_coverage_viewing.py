"""``scripts/conform_view_coverage_viewing.py``: the one-off conform step for a dataset whose
``view_coverage`` records still carry the old string forms of ``stats_source`` and
``display_bounds``, a stored ``working_scale_bar``, or an old ``cells_swept`` name list.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import tcip_store as ts
from fastapi.testclient import TestClient
from tcip_store.binding import bind_default
from tcip_store.sqlite_backend import SqliteBackend

from tcip_mcp.dataset_layout import view_coverage_key
from tcip_web.app import app

SCRIPT = Path(__file__).parent.parent / "scripts" / "conform_view_coverage_viewing.py"

_GRID = {"width": 100, "height": 80, "tile_size": 50, "overlap": 0.0, "cols": 2, "rows": 2}


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "conform_view_coverage_viewing_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bind_sqlite() -> SqliteBackend:
    backend = SqliteBackend()
    ts.bind(backend)
    return backend


def _old_shape_record(*, cells_swept=(), **viewing_overrides) -> dict:
    viewing = {
        "stats_source": "sampled(seed=0, pixel_fraction=0.500000)",
        "display_bounds": "0,1000;5,20",
        "base_served_size": "100x80",
    }
    viewing.update(viewing_overrides)
    return {
        "grid": _GRID,
        "cells_served_at_native": ["A1"],
        "cells_swept": list(cells_swept),
        "viewing": viewing,
        "updated_at": "2026-01-01T00:00:00+00:00",
    }


def _seed(root: Path, bucket: str, image_name: str, record: dict) -> None:
    """Add one image's record to whatever the key already holds, versioned against the read."""
    key = view_coverage_key(root)
    versioned = ts.read_versioned(key, default={})
    stored = versioned.value if isinstance(versioned.value, dict) else {}
    stored.setdefault(bucket, {})[image_name] = record
    ts.replace(key, stored, expect=versioned.version)


def _seed_fresh(root: Path, bucket: str, image_name: str, record: dict) -> None:
    key = view_coverage_key(root)
    ts.replace(key, {bucket: {image_name: record}}, expect=ts.Version.ABSENT)


class TestParsing:
    def test_window_sample_and_overview_string_forms_map_to_the_structured_shape(self):
        module = _load_script()
        assert module._parse_stats_source("sampled(seed=0, pixel_fraction=0.500000)") == {
            "read": "window_sample", "seed": 0, "pixel_fraction": 0.5, "overview_scale": None,
        }
        assert module._parse_stats_source("overview(scale=0.2048)") == {
            "read": "overview", "seed": None, "pixel_fraction": None, "overview_scale": 0.2048,
        }
        for literal in ("none", "dtype_full_scale", "served_array"):
            assert module._parse_stats_source(literal) == {
                "read": literal, "seed": None, "pixel_fraction": None, "overview_scale": None,
            }
        assert module._parse_stats_source(None) is None

    def test_an_unparseable_stats_source_raises(self):
        module = _load_script()
        with pytest.raises(ValueError, match="does not match a known old shape"):
            module._parse_stats_source("garbage")

    def test_joined_display_bounds_map_to_pairs(self):
        module = _load_script()
        assert module._parse_display_bounds("0,1000;5,20") == [[0.0, 1000.0], [5.0, 20.0]]
        assert module._parse_display_bounds(None) is None

    def test_an_unparseable_display_bounds_raises(self):
        module = _load_script()
        with pytest.raises(ValueError, match="does not parse"):
            module._parse_display_bounds("not-a-pair")


def test_an_old_shape_record_conforms_and_the_route_serves_it(tmp_path: Path):
    _bind_sqlite()
    module = _load_script()
    _seed_fresh(tmp_path, "bush/2026-03-01", "plot.tif", _old_shape_record())

    outcomes, refused = module.conform_root(tmp_path, plan=False)

    assert refused is False
    assert outcomes == ["bush/2026-03-01/plot.tif: conformed viewing"]
    stored = ts.read(view_coverage_key(tmp_path))
    record = stored["bush/2026-03-01"]["plot.tif"]
    assert record["cells_seen_at_scale"] == {}
    viewing = record["viewing"]
    assert viewing["stats_source"] == {
        "read": "window_sample", "seed": 0, "pixel_fraction": 0.5, "overview_scale": None,
    }
    assert viewing["display_bounds"] == [[0.0, 1000.0], [5.0, 20.0]]
    assert "working_scale_bar" not in viewing

    client = TestClient(app, base_url="http://127.0.0.1")
    resp = client.get("/api/coverage", params={
        "path": str(tmp_path / "images" / "2026-03-01" / "plot.tif"),
        "subject": "bush", "date": "2026-03-01", "dataset_root": str(tmp_path)})
    assert resp.status_code == 200, resp.text
    assert resp.json()["coverage"]["viewing"]["base_served_size"] == "100x80"


def test_an_old_swept_list_is_dropped_and_its_count_reported(tmp_path: Path):
    _bind_sqlite()
    module = _load_script()
    _seed_fresh(tmp_path, "bush/2026-03-01", "plot.tif",
                _old_shape_record(cells_swept=["A1", "B2"]))

    outcomes, refused = module.conform_root(tmp_path, plan=False)

    assert refused is False
    assert outcomes == [
        "bush/2026-03-01/plot.tif: conformed viewing and dropped 2 old swept cell names: no "
        "scale can be anchored to a cell swept under a bar the record no longer holds"
    ]
    record = ts.read(view_coverage_key(tmp_path))["bush/2026-03-01"]["plot.tif"]
    assert record["cells_seen_at_scale"] == {}
    assert "cells_swept" not in record


def test_a_half_migrated_records_seen_at_scale_carries_through_unblanked(tmp_path: Path):
    """A record already carrying ``cells_seen_at_scale`` (written under the new shape, only its
    ``viewing`` still old) keeps those values rather than being reset to empty."""
    _bind_sqlite()
    module = _load_script()
    record = _old_shape_record()
    del record["cells_swept"]
    record["cells_seen_at_scale"] = {"A1": 0.6, "B1": 0.3}
    _seed_fresh(tmp_path, "bush/2026-03-01", "plot.tif", record)

    outcomes, refused = module.conform_root(tmp_path, plan=False)

    assert refused is False
    assert outcomes == ["bush/2026-03-01/plot.tif: conformed viewing"]
    stored = ts.read(view_coverage_key(tmp_path))["bush/2026-03-01"]["plot.tif"]
    assert stored["cells_seen_at_scale"] == {"A1": 0.6, "B1": 0.3}


def test_a_record_whose_stats_source_is_already_structured_still_conforms_its_other_keys(tmp_path: Path):
    """A record written between the structured stats_source landing and the cells_seen_at_scale
    rename carries the mapping already, beside an old working_scale_bar and a cells_swept list:
    the mapping passes through and the other two are dropped, rather than the whole root refusing
    because the source is not a string to map."""
    _bind_sqlite()
    module = _load_script()
    structured = {"read": "none", "seed": None, "pixel_fraction": None, "overview_scale": None}
    record = _old_shape_record(
        cells_swept=["A1"], stats_source=structured, display_bounds=None,
        working_scale_bar={"value": 0.63, "source": "minimum view scale this session"})
    _seed_fresh(tmp_path, "bush/2026-03-01", "plot.tif", record)

    outcomes, refused = module.conform_root(tmp_path, plan=False)

    assert refused is False
    assert outcomes == [
        "bush/2026-03-01/plot.tif: conformed viewing and dropped 1 old swept cell name: no "
        "scale can be anchored to a cell swept under a bar the record no longer holds"
    ]
    stored = ts.read(view_coverage_key(tmp_path))["bush/2026-03-01"]["plot.tif"]
    assert stored["viewing"]["stats_source"] == structured
    assert "working_scale_bar" not in stored["viewing"]
    assert "cells_swept" not in stored
    assert stored["cells_seen_at_scale"] == {}


def test_a_structured_stats_source_the_model_rejects_refuses_by_name():
    module = _load_script()
    with pytest.raises(ValueError, match="StatsSource does not accept"):
        module._parse_stats_source({"read": "guess", "seed": None})


def test_an_empty_viewing_conforms_to_all_nulls_and_the_route_serves_it(tmp_path: Path):
    _bind_sqlite()
    module = _load_script()
    record = {
        "grid": _GRID, "cells_served_at_native": [], "cells_swept": [], "viewing": {},
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    _seed_fresh(tmp_path, "bush/2026-03-01", "empty.tif", record)

    outcomes, refused = module.conform_root(tmp_path, plan=False)

    assert refused is False
    assert outcomes == ["bush/2026-03-01/empty.tif: conformed viewing"]
    stored = ts.read(view_coverage_key(tmp_path))
    viewing = stored["bush/2026-03-01"]["empty.tif"]["viewing"]
    assert viewing == {
        "bands": None, "stretch": None, "base_served_size": None,
        "stats_source": None, "display_bounds": None,
    }

    client = TestClient(app, base_url="http://127.0.0.1")
    resp = client.get("/api/coverage", params={
        "path": str(tmp_path / "images" / "2026-03-01" / "empty.tif"),
        "subject": "bush", "date": "2026-03-01", "dataset_root": str(tmp_path)})
    assert resp.status_code == 200, resp.text
    assert resp.json()["coverage"]["viewing"]["stats_source"] is None


def test_plan_mode_writes_nothing(tmp_path: Path):
    _bind_sqlite()
    module = _load_script()
    record = _old_shape_record()
    _seed_fresh(tmp_path, "bush/2026-03-01", "plot.tif", record)

    outcomes, refused = module.conform_root(tmp_path, plan=True)

    assert refused is False
    assert outcomes == ["bush/2026-03-01/plot.tif: would conform viewing"]
    assert ts.read(view_coverage_key(tmp_path))["bush/2026-03-01"]["plot.tif"] == record


def test_a_record_already_in_the_current_shape_is_reported_unchanged(tmp_path: Path):
    _bind_sqlite()
    module = _load_script()
    conformed_viewing = {
        "bands": None, "stretch": None, "base_served_size": "100x80",
        "stats_source": {"read": "none", "seed": None, "pixel_fraction": None,
                          "overview_scale": None},
        "display_bounds": None,
    }
    record = {
        "grid": _GRID, "cells_served_at_native": [], "cells_seen_at_scale": {},
        "viewing": conformed_viewing, "updated_at": "2026-01-01T00:00:00+00:00",
    }
    _seed_fresh(tmp_path, "bush/2026-03-01", "plot.tif", record)

    outcomes, refused = module.conform_root(tmp_path, plan=False)

    assert refused is False
    assert outcomes == ["bush/2026-03-01/plot.tif: record already validates, unchanged"]
    assert ts.read(view_coverage_key(tmp_path))["bush/2026-03-01"]["plot.tif"] == record


def test_a_viewing_carrying_an_unknown_key_refuses_by_image_name_naming_the_key(tmp_path: Path):
    _bind_sqlite()
    module = _load_script()
    good = _old_shape_record()
    bad = _old_shape_record(rogue_key="anything")
    _seed_fresh(tmp_path, "bush/2026-03-01", "good.tif", good)
    _seed(tmp_path, "bush/2026-03-01", "bad.tif", bad)

    outcomes, refused = module.conform_root(tmp_path, plan=False)

    assert refused is True
    assert any(
        o == "bush/2026-03-01/bad.tif: refused, viewing carries keys the current shape does "
             "not declare: ['rogue_key']"
        for o in outcomes
    )
    stored = ts.read(view_coverage_key(tmp_path))
    assert stored["bush/2026-03-01"]["good.tif"] == good
    assert stored["bush/2026-03-01"]["bad.tif"] == bad


def test_an_unparseable_viewing_refuses_and_leaves_the_dataset_untouched(tmp_path: Path):
    _bind_sqlite()
    module = _load_script()
    good = _old_shape_record()
    bad = _old_shape_record(stats_source="not-a-known-shape")
    _seed_fresh(tmp_path, "bush/2026-03-01", "good.tif", good)
    _seed(tmp_path, "bush/2026-03-01", "bad.tif", bad)

    outcomes, refused = module.conform_root(tmp_path, plan=False)

    assert refused is True
    assert any(o.startswith("bush/2026-03-01/bad.tif: refused") for o in outcomes)
    stored = ts.read(view_coverage_key(tmp_path))
    assert stored["bush/2026-03-01"]["good.tif"] == good
    assert stored["bush/2026-03-01"]["bad.tif"] == bad


def test_main_conforms_two_roots_and_exits_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    bind_default()
    module = _load_script()
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    _seed_fresh(root_a, "bush/2026-03-01", "plot.tif", _old_shape_record())
    _seed_fresh(root_b, "bush/2026-03-01", "plot.tif", _old_shape_record())

    monkeypatch.setattr(
        sys, "argv", ["conform_view_coverage_viewing.py", str(root_a), str(root_b)])
    exit_code = module.main()

    assert exit_code == 0
    for root in (root_a, root_b):
        viewing = ts.read(view_coverage_key(root))["bush/2026-03-01"]["plot.tif"]["viewing"]
        assert viewing["stats_source"]["read"] == "window_sample"


def test_main_a_refused_root_does_not_block_a_second_roots_conform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
):
    bind_default()
    module = _load_script()
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    _seed_fresh(root_a, "bush/2026-03-01", "bad.tif", _old_shape_record(stats_source="garbage"))
    _seed_fresh(root_b, "bush/2026-03-01", "plot.tif", _old_shape_record())

    monkeypatch.setattr(
        sys, "argv", ["conform_view_coverage_viewing.py", str(root_a), str(root_b)])
    exit_code = module.main()
    output = capsys.readouterr().out

    assert exit_code == 2
    assert f"{root_a.resolve()}: bush/2026-03-01/bad.tif: refused" in output
    assert ts.read(view_coverage_key(root_b))[
        "bush/2026-03-01"]["plot.tif"]["viewing"]["stats_source"]["read"] == "window_sample"


def test_main_plan_mode_reports_the_refusal_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
):
    bind_default()
    module = _load_script()
    bad = _old_shape_record(stats_source="garbage")
    _seed_fresh(tmp_path, "bush/2026-03-01", "bad.tif", bad)

    monkeypatch.setattr(sys, "argv", ["conform_view_coverage_viewing.py", "--plan", str(tmp_path)])
    exit_code = module.main()
    output = capsys.readouterr().out

    assert exit_code == 2
    assert f"{tmp_path.resolve()}: bush/2026-03-01/bad.tif: refused" in output
    assert "nothing is written" in output
    assert ts.read(view_coverage_key(tmp_path))["bush/2026-03-01"]["bad.tif"] == bad
