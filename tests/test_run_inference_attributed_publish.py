"""``run_inference``'s images regime, publishing a bucket end to end: the checkpoint's own recorded
``scope`` decodes every detection's attribute ids into the ground-truth shape and the bucket's
record states the same scope, read back through ``read_bucket``. A detector whose scope declares
no attribute writes its subject alone the same way.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tcip_mcp import subject_registry as cr  # noqa: E402
from tcip_mcp.pipelines.execution import Stated  # noqa: E402
from tests._predictor_fixtures import BOX, StubPredictor, install  # noqa: E402

SUBJECT = "bud"
COLOR = cr.Attribute("color", "categorical", ("red", "blue"))
GRADE = cr.Attribute("grade", "ordinal", ("low", "mid", "high"))


def _checkpoint(tmp_path: Path, *attributes: cr.Attribute) -> str:
    """A registered checkpoint whose completing run's dataset declares ``attributes`` on
    :data:`SUBJECT`."""
    from tests._verified_checkpoint_fixtures import foreign_checkpoint

    return foreign_checkpoint(
        tmp_path, data={"num_channels": 3, "scope": {"subject": SUBJECT}},
        registry=cr.SubjectRegistry(subjects=(cr.Subject(name=SUBJECT, attributes=attributes),)))


def _attributed_predictor() -> StubPredictor:
    """A predictor over :data:`COLOR` and :data:`GRADE`: two detections, different values."""
    return StubPredictor(boxes=(BOX, (40.0, 40.0, 60.0, 60.0)), scores=(0.9, 0.8),
                         attributes=[[0, 2], [1, 0]])


def _one_image(images_dir: Path) -> None:
    from PIL import Image

    images_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 100), (120, 120, 120)).save(images_dir / "img.png")


def test_an_attributed_run_writes_every_attribute_value_and_stamps_its_scope(
    tmp_path: Path, monkeypatch,
) -> None:
    from tcip_mcp.buckets import read_bucket

    images_dir = tmp_path / "images"
    _one_image(images_dir)
    checkpoint = _checkpoint(tmp_path, COLOR, GRADE)
    install(monkeypatch, _attributed_predictor())
    from tcip_mcp.tools.inference_tools import run_inference

    out = tmp_path / "out"
    result = run_inference(tmp_path, checkpoint, str(images_dir), output_dir=str(out),
                           stated=Stated(tile=False))

    assert "error" not in result, result
    anns = json.loads((out / "img.json").read_text())["annotations"]
    assert [a["attributes"] for a in anns] == [{"color": "red", "grade": "high"},
                                               {"color": "blue", "grade": "low"}]
    assert all(a["subject"] == SUBJECT for a in anns)

    assert read_bucket(out).scope.attributes == (COLOR, GRADE)


def test_a_detector_run_declaring_no_attribute_writes_the_ordinary_shape_and_stamps_its_scope(
    tmp_path: Path, monkeypatch,
) -> None:
    from tcip_mcp.buckets import read_bucket

    images_dir = tmp_path / "images"
    _one_image(images_dir)
    checkpoint = _checkpoint(tmp_path)
    install(monkeypatch, StubPredictor())
    from tcip_mcp.tools.inference_tools import run_inference

    out = tmp_path / "out"
    result = run_inference(tmp_path, checkpoint, str(images_dir), output_dir=str(out),
                           stated=Stated(tile=False))

    assert "error" not in result, result
    anns = json.loads((out / "img.json").read_text())["annotations"]
    assert len(anns) == 1
    assert anns[0]["subject"] == SUBJECT
    assert not anns[0].get("attributes")

    assert read_bucket(out).scope.attributes == ()
