"""export_predictions: writes per-image prediction JSON, never overwrites a bucket with verdicts."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402


def test_export_predictions_writes_json(tmp_path, monkeypatch):
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"stub")  # only existence is checked
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (100, 100), (120, 120, 120)).save(images_dir / "img.png")

    class FakePredictor:
        def __init__(self, checkpoint_path=None, **kwargs):
            pass

        def predict_batch(self, paths, **kw):
            return [{"image": p, "width": 100, "height": 100,
                     "boxes": [[10.0, 10.0, 30.0, 30.0]], "scores": [0.9], "labels": [1], "count": 1}
                    for p in paths]

    monkeypatch.setattr(
        "tcip_mcp.pipelines.inference.generic_predictor.GenericPredictor", FakePredictor)
    from tcip_mcp.tools.inference_tools import export_predictions

    out = tmp_path / "out"
    export_predictions(str(ckpt), str(images_dir), str(out), tile=False)
    data = json.loads((out / "img.json").read_text())
    assert data["image"] == "img"
    assert (data["width"], data["height"]) == (100, 100)
    anns = data["annotations"]
    assert len(anns) == 1
    assert anns[0]["subject"] == "0"                   # label 1 -> id 0; no run id_map -> stringified id
    assert anns[0]["score"] == pytest.approx(0.9)      # confidence
    # COCO xywh (pixel) from pixel-xyxy box [10,10,30,30].
    assert anns[0]["bbox"] == pytest.approx([10.0, 10.0, 20.0, 20.0])


def _fake_predictor(monkeypatch):
    class FakePredictor:
        def __init__(self, checkpoint_path=None, **kwargs):
            pass

        def predict_batch(self, paths, **kw):
            return [{"image": p, "width": 100, "height": 100,
                     "boxes": [[10.0, 10.0, 30.0, 30.0]], "scores": [0.9], "labels": [1], "count": 1}
                    for p in paths]

    monkeypatch.setattr(
        "tcip_mcp.pipelines.inference.generic_predictor.GenericPredictor", FakePredictor)


def test_export_predictions_redirects_when_bucket_has_verdicts(tmp_path, monkeypatch):
    from pathlib import Path

    project_root = tmp_path / "proj"
    (project_root / ".tcip" / "state").mkdir(parents=True)
    # export_predictions resolves the review state via the pinned project root.
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(project_root))

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (100, 100), (120, 120, 120)).save(images_dir / "img.png")

    out = tmp_path / "preds"
    out.mkdir()
    # A prediction file already sits in the bucket, and a human verdict is recorded against it.
    (out / "img.json").write_text(
        json.dumps({"image": "img", "width": 100, "height": 100, "annotations": []}))
    from tcip_annotation.review_engine import ReviewContext, ReviewDetection, ReviewEngine
    from tcip_annotation.state import Annotation, BBox

    engine = ReviewEngine(project_root / ".tcip" / "state")
    ctx = ReviewContext(img_name="img.png", img_width=100, img_height=100,
                        preds=[Annotation(subject="catkin", geometry=BBox(10.0, 10.0, 30.0, 30.0),
                                          score=0.9)])
    det = ReviewDetection(det_type="fp", class_name="catkin", conf=0.9, iou=None, gt_idx=None,
                          pred_idx=0, bbox=(10.0, 10.0, 30.0, 30.0))
    engine.record_detection_action(det, ctx, action="accepted")

    _fake_predictor(monkeypatch)
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"stub")
    from tcip_mcp.tools.inference_tools import export_predictions

    # overwrite=True is refused with the verdict count and a suggested fresh bucket.
    res2 = export_predictions(str(ckpt), str(images_dir), str(out), overwrite=True, tile=False)
    assert "error" in res2 and res2["verdict_count"] == 1
    assert Path(res2["suggested_bucket"]).name == "preds@r2"

    # Default: redirect to a fresh @r2 bucket; the reviewed bucket is left intact.
    res = export_predictions(str(ckpt), str(images_dir), str(out), tile=False)
    assert res["bucket_redirected"] is True
    assert Path(res["output_dir"]).name == "preds@r2"
    assert (Path(res["output_dir"]) / "img.json").is_file()
    assert json.loads((out / "img.json").read_text())["annotations"] == []  # untouched
