"""``run_inference``'s images regime, publishing a classified bucket end to end: the checkpoint's
own recorded ``scope`` decodes every detection into the ground-truth shape and the bucket's record
states the same scope, read back through ``read_bucket``. A detector run (no attribute) decodes
through its own one-subject map the same way.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tcip_mcp.pipelines.data.selection import ClassScope  # noqa: E402
from tcip_mcp.pipelines.execution import Stated  # noqa: E402

SUBJECT = "bud"
ATTRIBUTE = "opening"
ID_MAP = {"open": 0, "closed": 1}
CLASSIFIED = ClassScope(SUBJECT, ATTRIBUTE, ID_MAP)
DETECTOR = ClassScope(SUBJECT, None, {SUBJECT: 0})


def _checkpoint(tmp_path: Path, scope: ClassScope) -> str:
    """A registered checkpoint whose completing run recorded ``scope``."""
    from dataclasses import asdict

    from tests._verified_checkpoint_fixtures import foreign_checkpoint

    return foreign_checkpoint(tmp_path, data={"num_channels": 3, "scope": asdict(scope)})


class _ClassifiedPredictor:
    """A classified run's predictor: two detections, one of each value."""

    def __init__(self, checkpoint_path=None, **kwargs):
        pass

    def predict_batch(self, paths, execution=None, **kw):
        return [{"image": p, "width": 100, "height": 100,
                 "boxes": [[10.0, 10.0, 30.0, 30.0], [40.0, 40.0, 60.0, 60.0]],
                 "scores": [0.9, 0.8], "labels": [1, 2], "count": 2, "cap_hit": False}
                for p in paths]


class _DetectorPredictor:
    """A detector run's predictor: one detection of its one class."""

    def __init__(self, checkpoint_path=None, **kwargs):
        pass

    def predict_batch(self, paths, execution=None, **kw):
        return [{"image": p, "width": 100, "height": 100,
                 "boxes": [[10.0, 10.0, 30.0, 30.0]], "scores": [0.9], "labels": [1], "count": 1,
                 "cap_hit": False}
                for p in paths]


def _one_image(images_dir: Path) -> None:
    from PIL import Image

    images_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 100), (120, 120, 120)).save(images_dir / "img.png")


def test_a_classifier_scoped_run_writes_the_ground_truth_shape_and_stamps_the_pair(
    tmp_path: Path, monkeypatch,
) -> None:
    from tcip_mcp.buckets import read_bucket

    images_dir = tmp_path / "images"
    _one_image(images_dir)
    checkpoint = _checkpoint(tmp_path, CLASSIFIED)
    monkeypatch.setattr(
        "tcip_mcp.pipelines.inference.generic_predictor.GenericPredictor", _ClassifiedPredictor)
    from tcip_mcp.tools.inference_tools import run_inference

    out = tmp_path / "out"
    result = run_inference(tmp_path, checkpoint, str(images_dir), output_dir=str(out),
                           stated=Stated(tile=False))

    assert "error" not in result, result
    data = json.loads((out / "img.json").read_text())
    anns = data["annotations"]
    assert len(anns) == 2
    by_value = {a["attributes"][ATTRIBUTE]: a for a in anns}
    assert set(by_value) == {"open", "closed"}
    assert all(a["subject"] == SUBJECT for a in anns)

    assert read_bucket(out).scope == CLASSIFIED


def test_a_detector_run_with_a_decoded_detection_writes_the_ordinary_shape_and_stamps_the_pair(
    tmp_path: Path, monkeypatch,
) -> None:
    from tcip_mcp.buckets import read_bucket

    images_dir = tmp_path / "images"
    _one_image(images_dir)
    checkpoint = _checkpoint(tmp_path, DETECTOR)
    monkeypatch.setattr(
        "tcip_mcp.pipelines.inference.generic_predictor.GenericPredictor", _DetectorPredictor)
    from tcip_mcp.tools.inference_tools import run_inference

    out = tmp_path / "out"
    result = run_inference(tmp_path, checkpoint, str(images_dir), output_dir=str(out),
                           stated=Stated(tile=False))

    assert "error" not in result, result
    data = json.loads((out / "img.json").read_text())
    anns = data["annotations"]
    assert len(anns) == 1
    assert anns[0]["subject"] == SUBJECT  # decoded through the run's own one-subject map
    assert not anns[0].get("attributes")

    assert read_bucket(out).scope == DETECTOR
