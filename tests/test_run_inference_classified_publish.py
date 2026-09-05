"""``run_inference``'s images regime, publishing a classified bucket end to end: the run's own
recorded ``(subject, attribute, id_map)`` scope decodes every detection into the ground-truth
shape and stamps the bucket's ``operating_point.json`` with the same pair, read back through
``bucket_scope``. A detector run (no attribute) is unaffected: its stamp gains the pair, but its
documents are exactly what a detector run always wrote.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

SUBJECT = "bud"
ATTRIBUTE = "opening"
ID_MAP = {"open": 0, "closed": 1}


@pytest.fixture(autouse=True)
def _stub_checkpoint_verification(monkeypatch):
    import tcip_mcp.model_registry as model_registry_mod
    from tests._verified_checkpoint_fixtures import stub_verified_checkpoint

    def _stub(path, *a, **kw):
        return stub_verified_checkpoint(str(path))

    monkeypatch.setattr(model_registry_mod, "load_registered_checkpoint", _stub)


def _ckpt(tmp_path) -> str:
    p = tmp_path / "m.pt"
    p.write_bytes(b"stub")
    return str(p)


class _ClassifiedPredictor:
    """A run whose checkpoint recorded a classified scope: two detections, one of each value."""

    config = {"data": {"subject": SUBJECT, "attribute": ATTRIBUTE, "id_map": ID_MAP}}

    def __init__(self, checkpoint_path=None, **kwargs):
        pass

    def predict_batch(self, paths, **kw):
        return [{"image": p, "width": 100, "height": 100,
                 "boxes": [[10.0, 10.0, 30.0, 30.0], [40.0, 40.0, 60.0, 60.0]],
                 "scores": [0.9, 0.8], "labels": [1, 2], "count": 2}
                for p in paths]


class _DetectorPredictor:
    """A run with no recorded scope at all: the pre-existing detector shape."""

    config = {"data": {}}

    def __init__(self, checkpoint_path=None, **kwargs):
        pass

    def predict_batch(self, paths, **kw):
        return [{"image": p, "width": 100, "height": 100,
                 "boxes": [[10.0, 10.0, 30.0, 30.0]], "scores": [0.9], "labels": [1], "count": 1}
                for p in paths]


def _one_image(images_dir: Path) -> None:
    from PIL import Image

    images_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 100), (120, 120, 120)).save(images_dir / "img.png")


def test_a_classifier_scoped_run_writes_the_ground_truth_shape_and_stamps_the_pair(
    tmp_path: Path, monkeypatch,
) -> None:
    from tcip_mcp.pipelines.resolution import bucket_scope

    images_dir = tmp_path / "images"
    _one_image(images_dir)
    monkeypatch.setattr(
        "tcip_mcp.pipelines.inference.generic_predictor.GenericPredictor", _ClassifiedPredictor)
    from tcip_mcp.tools.inference_tools import run_inference

    out = tmp_path / "out"
    result = run_inference(_ckpt(tmp_path), str(images_dir), output_dir=str(out), tile=False)

    assert "error" not in result, result
    data = json.loads((out / "img.json").read_text())
    anns = data["annotations"]
    assert len(anns) == 2
    by_value = {a["attributes"][ATTRIBUTE]: a for a in anns}
    assert set(by_value) == {"open", "closed"}
    assert all(a["subject"] == SUBJECT for a in anns)

    scope = bucket_scope(out)
    assert (scope.subject, scope.attribute) == (SUBJECT, ATTRIBUTE)


def test_a_detector_run_with_a_decoded_detection_writes_the_ordinary_shape_and_stamps_the_pair(
    tmp_path: Path, monkeypatch,
) -> None:
    from tcip_mcp.pipelines.resolution import bucket_scope

    images_dir = tmp_path / "images"
    _one_image(images_dir)
    monkeypatch.setattr(
        "tcip_mcp.pipelines.inference.generic_predictor.GenericPredictor", _DetectorPredictor)
    from tcip_mcp.tools.inference_tools import run_inference

    out = tmp_path / "out"
    result = run_inference(_ckpt(tmp_path), str(images_dir), output_dir=str(out), tile=False)

    assert "error" not in result, result
    data = json.loads((out / "img.json").read_text())
    anns = data["annotations"]
    assert len(anns) == 1
    assert anns[0]["subject"] == "0"  # no recorded id_map: the raw-index name, as always
    assert not anns[0].get("attributes")

    scope = bucket_scope(out)
    assert (scope.subject, scope.attribute) == (None, None)
