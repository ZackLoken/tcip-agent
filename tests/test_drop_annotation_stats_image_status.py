"""``scripts/drop_annotation_stats_image_status.py``: the one-off conform step for a project
whose ``annotation_stats`` record still carries the dead, always-empty ``image_status`` key."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import tcip_store as ts
from tcip_store.sqlite_backend import SqliteBackend

from tcip_web.routes.sessions import annotation_stats_key

SCRIPT = Path(__file__).parent.parent / "scripts" / "drop_annotation_stats_image_status.py"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "drop_annotation_stats_image_status_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bind_sqlite() -> SqliteBackend:
    backend = SqliteBackend()
    ts.bind(backend)
    return backend


def test_a_record_with_no_annotation_stats_document_is_left_alone(tmp_path: Path) -> None:
    _bind_sqlite()
    module = _load_script()

    assert module.conform_root(tmp_path, plan=False) == "nothing to conform"


def test_an_already_conformed_record_is_left_alone(tmp_path: Path) -> None:
    _bind_sqlite()
    module = _load_script()
    key = annotation_stats_key(str(tmp_path))
    ts.replace(key, {"sessions": []}, expect=ts.Version.ABSENT)

    assert module.conform_root(tmp_path, plan=False) == "nothing to conform"
    assert ts.read(key) == {"sessions": []}


def test_an_empty_image_status_is_dropped(tmp_path: Path) -> None:
    _bind_sqlite()
    module = _load_script()
    key = annotation_stats_key(str(tmp_path))
    ts.replace(key, {"sessions": [{"user": "alice"}], "image_status": {}}, expect=ts.Version.ABSENT)

    outcome = module.conform_root(tmp_path, plan=False)

    assert outcome == "dropped the empty image_status key"
    assert ts.read(key) == {"sessions": [{"user": "alice"}]}


def test_plan_reports_the_drop_and_writes_nothing(tmp_path: Path) -> None:
    _bind_sqlite()
    module = _load_script()
    key = annotation_stats_key(str(tmp_path))
    original = {"sessions": [], "image_status": {}}
    ts.replace(key, original, expect=ts.Version.ABSENT)

    outcome = module.conform_root(tmp_path, plan=True)

    assert outcome == "would drop the empty image_status key"
    assert ts.read(key) == original


def test_a_non_empty_image_status_is_refused_rather_than_conformed(tmp_path: Path) -> None:
    """Real data under image_status would mean a writer this script does not know about; the
    script's job is to drop a dead key, not decide what to do with one that turns out to be live."""
    _bind_sqlite()
    module = _load_script()
    key = annotation_stats_key(str(tmp_path))
    original = {"sessions": [], "image_status": {"IMG_0001.JPG": "complete"}}
    ts.replace(key, original, expect=ts.Version.ABSENT)

    outcome = module.conform_root(tmp_path, plan=False)

    assert outcome.startswith("refused")
    assert ts.read(key) == original
