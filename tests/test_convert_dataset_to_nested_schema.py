"""The one-time old->nested-schema converter (scripts/convert_dataset_to_nested_schema.py).

Pins the migration's correctness (segment polygons win over their derived detect boxes, provenance and
geometry survive, the registry is nested with no ids/colors, negatives re-key to (subject,date)) and
its refuse-loud behaviors (an already-converted dataset, a negative whose subject cannot be resolved,
a category_id it cannot map) — a migration that silently drops a human's work is the worst outcome.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from tcip_annotation import json_io
from tcip_annotation.state import BBox, Polygon

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "convert_dataset_to_nested_schema.py"


def _load_converter():
    spec = importlib.util.spec_from_file_location("convert_dataset_to_nested_schema", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so dataclasses can resolve the module's own annotations (its @dataclass
    # triggers a sys.modules lookup of the defining module).
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _old_project(root: Path, *, with_negative: bool = True) -> None:
    """A minimal project in the pre-K13.5 layout: flat per-subject registry, per-(subject,task) labels,
    flat image_status. IMG_1 has both a segment polygon and its derived detect box; IMG_2 is detect-only."""
    reg = root / ".tcip" / "state" / "classes"
    reg.mkdir(parents=True)
    (reg / "catkin.json").write_text(json.dumps({"0": {"name": "catkin", "color": "#FF0000"}}))

    imgs = root / "images" / "2026-02-11"
    imgs.mkdir(parents=True)
    for name in ("IMG_1.jpg", "IMG_2.jpg", "IMG_3.jpg"):
        (imgs / name).write_bytes(b"x")

    date = root / "annotations" / "catkin" / "2026-02-11"
    (date / "segment").mkdir(parents=True)
    (date / "detect").mkdir(parents=True)
    seg_obj = {"category_id": 0, "segmentation": [[0, 0, 10, 0, 10, 10, 0, 10]],
               "created_by": "sam", "accepted_by": "user:zack"}
    (date / "segment" / "IMG_1.json").write_text(
        json.dumps({"image": "IMG_1", "width": 100, "height": 100, "objects": [seg_obj]}))
    (date / "detect" / "IMG_1.json").write_text(  # the box derived from IMG_1's polygon
        json.dumps({"image": "IMG_1", "width": 100, "height": 100,
                    "objects": [{"category_id": 0, "bbox": [0, 0, 10, 10], "created_by": "sam"}]}))
    (date / "detect" / "IMG_2.json").write_text(  # detect-only, no polygon
        json.dumps({"image": "IMG_2", "width": 100, "height": 100,
                    "objects": [{"category_id": 0, "bbox": [5, 5, 20, 20], "created_by": "sam"}]}))

    status = {"IMG_1.jpg": "partial", "IMG_2.jpg": "unannotated"}
    if with_negative:
        status["IMG_3.jpg"] = "negative"
    (root / ".tcip" / "state" / "image_status.json").write_text(json.dumps(status))


def _run(conv, root: Path, monkeypatch, extra: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["convert", str(root), *extra])
    conv.main()


def test_converts_registry_labels_and_negatives(tmp_path, monkeypatch):
    conv = _load_converter()
    root = tmp_path / "proj"
    _old_project(root)

    _run(conv, root, monkeypatch, ["--negative-subject", "catkin"])

    # Registry: one nested classes.json, plain subject, no ids/colors/attributes.
    reg = json.loads((root / "classes.json").read_text())
    assert set(reg) == {"catkin"}
    assert "attributes" not in reg["catkin"] and "color" not in json.dumps(reg) and "0" not in reg

    # IMG_1: the segment polygon wins over its derived box -> ONE annotation, geometry a polygon,
    # provenance preserved.
    a1 = json_io.read_annotations(str(root / "annotations" / "2026-02-11" / "IMG_1.json"))
    assert len(a1) == 1
    assert a1[0].subject == "catkin" and isinstance(a1[0].geometry, Polygon)
    assert a1[0].created_by == "sam" and a1[0].accepted_by == "user:zack"

    # IMG_2: detect-only -> a box survives.
    a2 = json_io.read_annotations(str(root / "annotations" / "2026-02-11" / "IMG_2.json"))
    assert len(a2) == 1 and isinstance(a2[0].geometry, BBox)

    # Negatives re-key from the flat store to (subject,date) — recovering the negative the flat store
    # left subject-less.
    store = json.loads((root / ".tcip" / "state" / "image_status.json").read_text())
    assert store == {"catkin/2026-02-11": {"IMG_3.jpg": "negative"}}


def test_refuses_a_negative_it_cannot_attribute_atomically(tmp_path, monkeypatch):
    conv = _load_converter()
    root = tmp_path / "proj"
    _old_project(root, with_negative=True)
    # A negative exists but no --negative-subject: the old flat store lost which subject it was for,
    # so refuse rather than guess.
    with pytest.raises(SystemExit):
        _run(conv, root, monkeypatch, [])
    # Atomic: the refusal wrote NOTHING, so a corrected re-run is not blocked by a half-conversion.
    assert not (root / "classes.json").exists()
    assert not (root / "annotations" / "2026-02-11").exists()  # no merged labels written
    store = json.loads((root / ".tcip" / "state" / "image_status.json").read_text())
    assert store == {"IMG_1.jpg": "partial", "IMG_2.jpg": "unannotated", "IMG_3.jpg": "negative"}

    # The corrected re-run (with the subject) now succeeds — the refusal left it re-runnable.
    _run(conv, root, monkeypatch, ["--negative-subject", "catkin"])
    assert (root / "classes.json").exists()
    assert json.loads((root / ".tcip" / "state" / "image_status.json").read_text()) == {
        "catkin/2026-02-11": {"IMG_3.jpg": "negative"}}


def test_refuses_an_unmappable_category_id_atomically(tmp_path, monkeypatch):
    conv = _load_converter()
    root = tmp_path / "proj"
    _old_project(root, with_negative=False)
    bad = root / "annotations" / "catkin" / "2026-02-11" / "detect" / "IMG_2.json"
    bad.write_text(json.dumps({"image": "IMG_2", "width": 100, "height": 100,
                               "objects": [{"category_id": 1, "bbox": [5, 5, 20, 20]}]}))
    with pytest.raises(SystemExit):
        _run(conv, root, monkeypatch, [])
    # Validation happens before any write, so a bad category_id leaves no orphaned classes.json.
    assert not (root / "classes.json").exists()


def test_leaves_an_already_bucketed_store_untouched(tmp_path, monkeypatch):
    conv = _load_converter()
    root = tmp_path / "proj"
    _old_project(root, with_negative=False)
    # A store already in the nested (subject,date) shape is not flat — the converter must not crash
    # on it (dict values are not hashable) nor re-key it; it is already in the target form.
    bucketed = {"catkin/2026-02-11": {"IMG_3.jpg": "negative"}}
    (root / ".tcip" / "state" / "image_status.json").write_text(json.dumps(bucketed))
    _run(conv, root, monkeypatch, [])
    assert json.loads((root / ".tcip" / "state" / "image_status.json").read_text()) == bucketed


def test_refuses_an_already_converted_dataset(tmp_path, monkeypatch):
    conv = _load_converter()
    root = tmp_path / "proj"
    _old_project(root, with_negative=False)
    (root / "classes.json").write_text("{}")  # already converted
    with pytest.raises(SystemExit):
        _run(conv, root, monkeypatch, [])


def test_refuses_an_unmappable_category_id(tmp_path):
    conv = _load_converter()
    root = tmp_path / "proj"
    _old_project(root, with_negative=False)
    # A single-class subject with a category_id != 0 cannot be mapped without an attribute scheme.
    bad = root / "annotations" / "catkin" / "2026-02-11" / "detect" / "IMG_2.json"
    bad.write_text(json.dumps({"image": "IMG_2", "width": 100, "height": 100,
                               "objects": [{"category_id": 1, "bbox": [5, 5, 20, 20]}]}))
    with pytest.raises(SystemExit):
        conv._old_annotations(bad, "catkin")
